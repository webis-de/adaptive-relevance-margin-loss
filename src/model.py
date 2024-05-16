from typing import Optional, Tuple, Dict, Union, Literal

import bitsandbytes as bnb
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalRecall
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.trainer_pt_utils import get_parameter_names
import pandas as pd

from .types import TrainBatch, InferenceBatch
from .loss import MarginLoss
from .util import BinaryWrapper


class MarginRankingModel(LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str = None,
        target: Optional[Union[Literal["adaptive", "distributed"], float]] = "distributed",
        in_batch: Optional[bool] = False,
        error_fn: Optional[Literal["l1", "l2", "soft"]] = "l2",
        learning_rate: Optional[float] = 1e-7,
        weight_decay: Optional[float] = 1e-6,
        run_path: Optional[str] = None,
    ) -> None:
        """Constructor."""
        super().__init__()
        self.save_hyperparameters()
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = "distilbert-base-uncased"
        # Loss Instantiation
        self.loss = MarginLoss(target=target, in_batch=in_batch, error_fn=error_fn)
        # Model instantiation
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        # Metrics
        self.metrics = MetricCollection(
            [
                MetricCollection(
                    {
                        "RetrievalNormalizedDCG": RetrievalNormalizedDCG(top_k=k),
                        "RetrievalHitRate": BinaryWrapper(RetrievalHitRate(top_k=k), 1),
                        "RetrievalRecall": BinaryWrapper(RetrievalRecall(top_k=k), 1),
                    },
                    postfix=f"@{k}",
                )
                for k in [10, 50, 100, 1000]
            ],
            prefix="val/",
        )

    def configure_optimizers(self):
        """Prepare optimizer."""
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder. Produces an embedding for given input_ids/attention mask.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input IDs to embed, shape (B, N).
        attention_mask : torch.Tensor
            Attention mask to embed, shape (B, N).

        Returns
        -------
        torch.Tensor
            The pooled sequence embedding of last hidden state for given inputs in the batch, shape (B, D)
        """
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if isinstance(outputs, BaseModelOutputWithPoolingAndCrossAttentions):
            # If pooling layer is available return processed output
            return outputs.pooler_output
        else:
            # Return plain CLS embedding otherwise
            return outputs.last_hidden_state[:, 0, :]

    def predict(self, batch: InferenceBatch) -> torch.Tensor:
        """
        Given a batch of query-document pairs, estimates their relevance scores.

        Parameters
        ----------
        batch : InferenceBatch
            A batch of query-document pairs to predict scores for.

        Returns
        -------
        torch.Tensor
            Instance-wise score values for the batch, shape (B,).
        """
        qry = self.forward(batch["query_input_ids"], batch["query_attention_mask"])
        doc = self.forward(batch["doc_input_ids"], batch["doc_attention_mask"])
        return pairwise_cosine_similarity(qry, doc).diag().flatten()

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, InferenceBatch],
        batch_idx: int,
        dataloader_idx: Optional[int] = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs a predict step. Internally calls `predict`.

        Parameters
        ----------
        batch : InferenceBatch
            A batch of query-document pairs to predict scores for.
        batch_idx : int
            TrainBatch index, unused.
        dataloader_idx
            DataLoader index, unused.

        Returns
        -------
        Dict[str, torch.Tensor]
            Key-value pairs of instance indices and their score values.
        """
        indices, inputs = batch
        return {"indices": indices, "scores": self.predict(inputs)}

    def test_step(self, batch: InferenceBatch, batch_idx: int) -> STEP_OUTPUT:
        """
        Performs a validation step. Internally calls `predict`.

        Parameters
        ----------
        batch : InferenceBatch
            A batch of query-document pairs to predict scores for.
        batch_idx : int
            TrainBatch index, unused.

        Returns
        -------
        STEP_OUTPUT
            Estimated and ground-truth relevance for all query-document-pairs in this batch.
        """
        scores = self.predict(batch).cpu()
        self.test_outputs.extend(
            [
                {"query_id": q, "doc_id": d, "score": s.numpy().tolist()}
                for q, d, s in zip(batch["query_id"], batch["doc_id"], scores)
            ]
        )
        return {"query_id": batch["query_id"], "doc_id": batch["doc_id"], "score": scores}

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def on_test_epoch_end(self) -> None:
        df = (
            pd.DataFrame(self.test_outputs)
            .sort_values(["query_id", "score"], ascending=False)
            .assign(rank=lambda df: df.groupby("query_id").cumcount() + 1, tag="model", q0="q0")
            .loc[:, ["query_id", "q0", "doc_id", "rank", "score", "tag"]]
        )
        df.to_csv(self.hparams.run_path, index=False, header=None, sep=" ")
        self.test_outputs = []

    def validation_step(self, batch: InferenceBatch, batch_idx: int) -> STEP_OUTPUT:
        """
        Performs a validation step. Internally calls `predict`.

        Parameters
        ----------
        batch : InferenceBatch
            A batch of query-document pairs to predict scores for.
        batch_idx : int
            TrainBatch index, unused.

        Returns
        -------
        STEP_OUTPUT
            Estimated and ground-truth relevance for all query-document-pairs in this batch.
        """
        res = {"query_id": batch["query_id"], "score": self.predict(batch), "label": batch["label"]}
        self.metrics.update(res["score"], res["label"], indexes=res["query_id"])
        return res

    def on_validation_epoch_end(self) -> None:
        """Aggregates validation metrics over all steps in the epoch."""
        metrics = self.metrics.compute()
        self.logger.log_metrics(metrics, step=self.global_step)
        for k, v in metrics.items():
            self.log(k, v)
        self.metrics.reset()

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        """
        Performs a training step.

        Given a contrastive batch, embeds queries, positive documents, and negative documents in the batch and computes
        the loss for the batch.

        Parameters
        ----------
        batch : TrainBatch
            A batch of contrastive triples of (query, positive_document, negative_document).
        batch_idx : int
            Batch identifier, not used.

        Returns
        -------
        torch.Tensor
            Loss value for the batch, shape (1,).
        """
        qry_emb = self.forward(batch["query_input_ids"], batch["query_attention_mask"])
        pos_emb = self.forward(batch["positive_doc_input_ids"], batch["positive_doc_attention_mask"])
        neg_emb = self.forward(batch["negative_doc_input_ids"], batch["negative_doc_attention_mask"])
        loss, target = self.loss(qry_emb, pos_emb, neg_emb, return_target=True)
        self.log("loss/train", loss)
        self.log("loss/target/mean", target.mean())
        self.log("loss/target/min", target.min())
        self.log("loss/target/max", target.max())
        return loss
