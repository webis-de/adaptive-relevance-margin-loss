from typing import Any, Union, Tuple, Dict, Optional, Type, Callable, List

import datasets
import git
import torch
import transformers
from lightning.pytorch.cli import LightningModule, SaveConfigCallback, Trainer
from lightning.pytorch.loggers import Logger
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.abstract import WrapperMetric
from transformers import AutoConfig, AutoModelForMaskedLM, PreTrainedTokenizer

from src.types import TrainBatch, InferenceBatch, TrainSample, InferenceSample


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            repo = git.Repo(search_parent_directories=True)
            config_str = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams(
                {
                    "commit_id": repo.head.object.hexsha,
                    "config_str": config_str,
                }
            )


class BinaryWrapper(WrapperMetric):
    """Wrapper class for torchmetrics metric input transformations.
    Input transformations are characterized by them applying a transformation to the input data of a metric, and then
    forwarding all calls to the wrapped metric with modifications applied.
    """

    def __init__(
        self,
        wrapped_metric: Union[Metric, MetricCollection],
        threshold: Optional[float] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(wrapped_metric, (Metric, MetricCollection)):
            raise TypeError(
                f"Expected wrapped metric to be an instance of `torchmetrics.Metric` or "
                f"`torchmetrics.MetricsCollection`but received {wrapped_metric}"
            )
        self.wrapped_metric = wrapped_metric
        self.threshold = threshold

    def transform_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return pred

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.threshold is not None:
            return target.gt(self.threshold).to(target.dtype)
        else:
            return target

    def _wrap_transform(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Wraps transformation functions to dispatch args to their individual transform functions."""
        if len(args) == 1:
            return (self.transform_pred(args[0]),)
        if len(args) == 2:
            return self.transform_pred(args[0]), self.transform_target(args[1])
        return self.transform_pred(args[0]), self.transform_target(args[1]), *args[2:]

    def update(self, *args: Tuple[torch.Tensor], **kwargs: Dict[str, Any]) -> None:
        """Wraps the update call of the underlying metric."""
        args = self._wrap_transform(*args)
        self.wrapped_metric.update(*args, **kwargs)

    def compute(self) -> Any:
        """Wraps the compute call of the underlying metric."""
        return self.wrapped_metric.compute()

    def forward(self, *args: torch.Tensor, **kwargs: Dict[str, Any]) -> Any:
        """Wraps the forward call of the underlying metric."""
        args = self._wrap_transform(*args)
        return self.wrapped_metric.forward(*args, **kwargs)


def load_model_from_checkpoint(
    base_model_name_or_path: str, checkpoint_path: str, device: Union[str, None] = "cpu"
) -> AutoModelForMaskedLM:
    config_dict = AutoConfig.from_pretrained(base_model_name_or_path)
    model = AutoModelForMaskedLM.from_config(config_dict)
    state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)["state_dict"]
    state_dict_model = {k[6:]: v for k, v in state_dict.items() if (k.startswith("model."))}
    model.load_state_dict(state_dict_model, strict=False)
    model.to(device)
    return model


def check_columns(
    dataset, check_type: Union[Type[TrainBatch], Type[InferenceBatch], Type[TrainSample], Type[InferenceSample]]
) -> None:
    for key in check_type.__required_keys__:
        if key not in dataset.column_names:
            raise ValueError(f"Required column {key} for batch type {check_type} not found in dataset")


def get_collate_train(
    tokenizer: transformers.PreTrainedTokenizerFast, max_query_length: int, max_doc_length: int, **tokenizer_kwargs
) -> Callable[[List[TrainSample]], Dict[str, torch.Tensor]]:
    def __inner__(examples: List[TrainSample]) -> TrainBatch:
        qry = tokenizer([ex["query_text"] for ex in examples], max_length=max_query_length, **tokenizer_kwargs)
        pos = tokenizer([ex["positive_doc_text"] for ex in examples], max_length=max_doc_length, **tokenizer_kwargs)
        neg = tokenizer([ex["negative_doc_text"] for ex in examples], max_length=max_doc_length, **tokenizer_kwargs)

        return {
            "query_text": [ex["query_text"] for ex in examples],
            "positive_doc_text": [ex["positive_doc_text"] for ex in examples],
            "negative_doc_text": [ex["negative_doc_text"] for ex in examples],
            "query_input_ids": qry["input_ids"],
            "query_attention_mask": qry["attention_mask"],
            "positive_doc_input_ids": pos["input_ids"],
            "positive_doc_attention_mask": pos["attention_mask"],
            "negative_doc_input_ids": neg["input_ids"],
            "negative_doc_attention_mask": neg["attention_mask"],
        }

    return __inner__


def get_collate_val(
    tokenizer: transformers.PreTrainedTokenizerFast, max_query_length: int, max_doc_length: int, **tokenizer_kwargs
) -> Callable[[List[InferenceSample]], InferenceBatch]:
    def __inner__(examples: List[InferenceSample]) -> InferenceBatch:
        qry = tokenizer([ex["query_text"] for ex in examples], max_length=max_query_length, **tokenizer_kwargs)
        doc = tokenizer([ex["doc_text"] for ex in examples], max_length=max_doc_length, **tokenizer_kwargs)
        return {
            "query_input_ids": qry["input_ids"],
            "query_attention_mask": qry["attention_mask"],
            "doc_input_ids": doc["input_ids"],
            "doc_attention_mask": doc["attention_mask"],
            "query_id": torch.Tensor([ex["query_id"] for ex in examples]).long(),
            "label": torch.Tensor([ex["label"] for ex in examples]).long(),
        }

    return __inner__


def get_collate_test(
    tokenizer: transformers.PreTrainedTokenizerFast, max_query_length: int, max_doc_length: int, **tokenizer_kwargs
) -> Callable[[List[InferenceSample]], InferenceBatch]:
    def __inner__(examples: List[InferenceSample]) -> InferenceBatch:
        qry = tokenizer([ex["query_text"] for ex in examples], max_length=max_query_length, **tokenizer_kwargs)
        doc = tokenizer([ex["doc_text"] for ex in examples], max_length=max_doc_length, **tokenizer_kwargs)
        return {
            "query_input_ids": qry["input_ids"],
            "query_attention_mask": qry["attention_mask"],
            "doc_input_ids": doc["input_ids"],
            "doc_attention_mask": doc["attention_mask"],
            "query_id": [ex["query_id"] for ex in examples],
            "doc_id": [ex["doc_id"] for ex in examples],
        }

    return __inner__
