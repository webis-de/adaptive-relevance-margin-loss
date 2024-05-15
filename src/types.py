from typing_extensions import (  # for Python <3.11 with (Not)Required
    NotRequired,
    TypedDict,
)
import torch


class TrainSample(TypedDict):
    """Sample check_type for contrastive training."""

    query_text: str
    positive_doc_text: str
    negative_doc_text: str
    positive_score: NotRequired[float]
    negative_score: NotRequired[float]


class InferenceSample(TypedDict):
    """Sample check_type for contrastive training."""

    query_text: str
    doc_text: str
    query_id: NotRequired[int]
    doc_id: NotRequired[int]
    label: NotRequired[int]


class TrainBatch(TypedDict):
    """Batch check_type for contrastive training."""

    # Data signals
    query_input_ids: torch.Tensor
    query_attention_mask: NotRequired[torch.Tensor]
    positive_doc_input_ids: torch.Tensor
    positive_doc_attention_mask: NotRequired[torch.Tensor]
    negative_doc_input_ids: torch.Tensor
    negative_doc_attention_mask: NotRequired[torch.Tensor]


class InferenceBatch(TypedDict):
    """Batch check_type for inference, i.e., validation, test, and prediction."""

    query_input_ids: torch.Tensor
    query_attention_mask: NotRequired[torch.Tensor]
    doc_input_ids: torch.Tensor
    doc_attention_mask: NotRequired[torch.Tensor]
    query_id: NotRequired[torch.Tensor]
    doc_id: NotRequired[torch.Tensor]
    label: NotRequired[torch.Tensor]
