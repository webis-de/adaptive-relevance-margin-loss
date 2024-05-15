from typing import Optional, Union, Literal, Tuple

import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity, pairwise_euclidean_distance


class MarginLoss(torch.nn.Module):
    """
    Margin loss function
    """

    def __init__(
        self,
        target: Optional[Union[Literal["adaptive", "distributed"], float]] = "distributed",
        in_batch: Optional[bool] = False,
        error_fn: Optional[Literal["l1", "l2", "soft"]] = "l2",
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        target: Optional[Union[Literal["adaptive", "distributed"], float]]
            Margin target computation. Can be either, "distributed", or a float.
            Default: "distributed".
        in_batch: Optional[bool]
            Whether to use in-batch negatives or not.
            Default: False
        error_fn: Optional[str]
            Error function to use. Can be either "l1", "l2", or "soft".
            Default: "l2".
        """
        super().__init__()
        # Set up error function
        if error_fn == "l1":
            self.error_fn = lambda t: t.abs()
        elif error_fn == "l2":
            self.error_fn = lambda t: t.square()
        elif error_fn == "soft":
            self.error_fn = lambda t: torch.log1p((-1 * t).exp())
        else:
            raise ValueError(f"invalid error function, expected 'l1', 'l2', or 'soft', got {error_fn}")
        # Set up cosine similarity measure
        self.similarity = pairwise_cosine_similarity
        # Set up target computation
        if (not isinstance(target, float)) and (target not in ["adaptive", "distributed"]):
            raise ValueError(f"Invalid target, expected 'adaptive', 'distributed', or float value, got {target}")
        self.target = target
        # Set up in-batch negatives
        if in_batch and target == "distributed":
            raise ValueError("Distributed margins are not supported for use with in-batch negatives")
        self.in_batch = in_batch

    def forward(
        self,
        qry_emb: torch.Tensor,
        pos_doc_emb: torch.Tensor,
        neg_doc_emb: torch.Tensor,
        return_target: Optional[bool] = False,
    ) -> Union[torch.Tensor, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the batch-wise margin loss value.

        Parameters
        ----------
        qry_emb : torch.Tensor
            Embeddings for queries, shape = `(batch_size, embedding_size)`
        pos_doc_emb : torch.Tensor
            Embeddings for positive documents, shape = `(batch_size, embedding_size)`
        neg_doc_emb : torch.Tensor
            Embeddings for negative documents, shape = `(batch_size, embedding_size)`
        return_target
            Whether to return the target values or not.

        Returns
        -------
        float
            Batch-wise loss value.`
        """
        # Similarity matrices
        pos = self.similarity(qry_emb, pos_doc_emb)
        neg = self.similarity(qry_emb, neg_doc_emb)
        # Compute margin
        if self.in_batch:
            margin = pos.diag() - neg
        else:
            margin = pos.diag() - neg.diag()
        # Compute target
        if isinstance(self.target, float):
            target = self.target
        elif self.target == "adaptive" and not self.in_batch:
            target = self.similarity(pos_doc_emb, neg_doc_emb).diag() / 2 + 0.5
        elif self.target == "adaptive" and self.in_batch:
            target = self.similarity(pos_doc_emb, neg_doc_emb) / 2 + 0.5
        elif self.target == "distributed":
            target = self.similarity(pos_doc_emb, neg_doc_emb) / 2 + 0.5
        else:
            target = 0.5
        # Apply error function
        loss = self.error_fn(margin - target).mean()
        # Return mean
        if return_target:
            if isinstance(target, float):
                # If we are using static margins, calculate the actual doc similarities we're interested in
                target = self.similarity(pos_doc_emb, neg_doc_emb) / 2 + 0.5
            # target = self.cayley_menger(qry_emb, pos_doc_emb, neg_doc_emb)
            return loss, target
        else:
            return loss

    def cayley_menger(
        self, qry_emb: torch.Tensor, pos_doc_emb: torch.Tensor, neg_doc_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the simplex volume spanned by query, positive, negative and origin.
        If this volume goes towards 0, all points are on a plane.

        https://en.wikipedia.org/wiki/Cayley–Menger_determinant
        """
        origin = torch.zeros_like(qry_emb)  # B x 768
        stack = torch.stack([qry_emb, pos_doc_emb, neg_doc_emb, origin], dim=1)  # Shape B x 4 x 768
        dist_squared = torch.cdist(stack, stack).square()  # Shape B x 4 x 4
        tmp = torch.ones(dist_squared.shape[0], 1, 4).to(dist_squared.device)
        m_det = torch.cat([dist_squared, tmp], dim=1)  # Shape B x 5 x 4
        tmp = torch.ones(dist_squared.shape[0], 5, 1).to(dist_squared.device)
        tmp[:, -1, 0] = 0
        cayley_menger_matrix = torch.cat([m_det, tmp], dim=2)  # Shape (B x 5 x 5)
        return cayley_menger_matrix.det()  # Shape B
