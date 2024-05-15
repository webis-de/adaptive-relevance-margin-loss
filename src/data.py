from typing import List, Optional, Union

from transformers import AutoTokenizer

import datasets
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.types import TrainSample, InferenceSample
from src.util import check_columns, get_collate_train, get_collate_val, get_collate_test


class MarginRankingDataModule(LightningDataModule):
    """Ranking data module."""

    def __init__(
        self,
        train_dataset_path: Union[str, List[str]],
        val_dataset_path: Optional[Union[str, List[str]]] = None,
        test_dataset_path: Optional[Union[str, List[str]]] = None,
        shuffle: Optional[bool] = True,
        binarize_labels: Optional[bool] = False,
        train_batch_size: Optional[int] = 32,
        val_batch_size: Optional[int] = 64,
        num_workers: Optional[int] = 0,
        pretrained_tokenizer_name_or_path: Optional[str] = None,
        max_query_length: Optional[int] = 30,
        max_doc_length: Optional[int] = 200,
    ) -> None:
        """Constructor for RankingDataModule class."""
        super().__init__()
        self.save_hyperparameters()
        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_tokenizer_name_or_path)
        self.tokenizer_kwargs = {
            "return_tensors": "pt",
            "return_attention_mask": True,
            "padding": "max_length",
            "truncation": True,
            "return_token_type_ids": False,
        }
        # Dummy objects for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Load training and validation data, tokenizing if needed."""
        # Read train data
        self.train_dataset = datasets.load_dataset("parquet", data_files=self.hparams.train_dataset_path)["train"]
        check_columns(self.train_dataset, TrainSample)
        self.train_dataset = self.train_dataset.class_encode_column("query_id")
        self.train_dataset.set_format("torch")
        # Read val data
        self.val_dataset = datasets.load_dataset("parquet", data_files=self.hparams.val_dataset_path)["train"]
        check_columns(self.val_dataset, InferenceSample)
        self.val_dataset = self.val_dataset.class_encode_column("query_id")
        self.val_dataset.set_format("torch")
        # Read test data
        if self.hparams.test_dataset_path is not None:
            self.test_dataset = datasets.load_dataset("parquet", data_files=self.hparams.test_dataset_path)["train"]
            check_columns(self.test_dataset, InferenceSample)
            self.test_dataset.set_format("torch")

    def train_dataloader(self) -> DataLoader:
        """Returns train instances of class TrainBatch in batches of size train_batch_size."""
        return DataLoader(
            self.train_dataset,
            collate_fn=get_collate_train(
                tokenizer=self.tokenizer,
                max_query_length=self.hparams.max_query_length,
                max_doc_length=self.hparams.max_doc_length,
                **self.tokenizer_kwargs,
            ),
            shuffle=self.hparams.shuffle,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns test instances of class InferenceBatch in batches of size val_batch_size."""
        return DataLoader(
            self.val_dataset,
            collate_fn=get_collate_val(
                tokenizer=self.tokenizer,
                max_query_length=self.hparams.max_query_length,
                max_doc_length=self.hparams.max_doc_length,
                **self.tokenizer_kwargs,
            ),
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns test instances of class InferenceBatch in batches of size val_batch_size."""
        return DataLoader(
            self.test_dataset,
            collate_fn=get_collate_test(
                tokenizer=self.tokenizer,
                max_query_length=self.hparams.max_query_length,
                max_doc_length=self.hparams.max_doc_length,
                **self.tokenizer_kwargs,
            ),
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
        )
