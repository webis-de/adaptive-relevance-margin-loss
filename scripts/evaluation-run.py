import os
from typing import List, Tuple, Callable, Any, Dict, Iterator, Optional
import argparse

import faiss
import torch
import ir_datasets
import datasets
import numpy as np
from transformers import AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from src.util import load_model_from_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_kwargs = {
    "return_tensors": "pt",
    "return_attention_mask": True,
    "padding": "max_length",
    "truncation": True,
    "return_token_type_ids": False,
}


def get_embed_fn(
    base_model_name_or_path: str, checkpoint_path: str, tokenizer_kwargs: Dict[str, Any], normalize: bool = True
) -> Callable[[List[str]], torch.Tensor]:
    model = load_model_from_checkpoint(base_model_name_or_path, checkpoint_path, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=True)
    max_length = min(tokenizer.model_max_length, 200)

    def __inner__(texts: List[str]) -> torch.Tensor:
        batch = tokenizer(texts, max_length=max_length, **tokenizer_kwargs)
        with torch.no_grad():
            outputs: MaskedLMOutput = model(
                batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                output_hidden_states=True,
            )
            emb = outputs.hidden_states[-1][:, 0, :].detach().cpu()
        if normalize:
            emb = emb / np.linalg.norm(emb, axis=1)[:, None]
        return emb.numpy()

    return __inner__


def load_queries(
    dataset_name_or_path: str, embed_fn: Callable[[List[str]], torch.Tensor], batch_size: int
) -> datasets.Dataset:
    dataset = ir_datasets.load(dataset_name_or_path)
    assert dataset.has_queries()

    def __qry_gen__(ds: ir_datasets.Dataset) -> Iterator[Dict[str, Any]]:
        for query_id, text in ds.queries_iter():
            yield {"id": query_id, "text": text}

    return datasets.Dataset.from_generator(__qry_gen__, gen_kwargs={"ds": dataset}).map(
        lambda example: {"embeddings": embed_fn(example["text"])}, batched=True, batch_size=batch_size
    )


def load_documents(
    dataset_name_or_path: str, embed_fn: Callable[[List[str]], torch.Tensor], batch_size: int
) -> datasets.Dataset:
    dataset = ir_datasets.load(dataset_name_or_path)
    assert dataset.has_docs()

    # Convert to canonical HF dataset with embeddings
    def __doc_gen__(ds: ir_datasets.Dataset) -> Iterator[Dict[str, Any]]:
        for doc_id, text in ds.docs_iter():
            yield {"id": doc_id, "text": text}

    docs = datasets.Dataset.from_generator(__doc_gen__, gen_kwargs={"ds": dataset}).map(
        lambda example: {"embeddings": embed_fn(example["text"])}, batched=True, batch_size=batch_size
    )
    return docs


def search(query: Dict[str, Any], doc_store: datasets.Dataset, top_k: int = 1000) -> Dict[str, Any]:
    qry_emb = np.array(query["embeddings"][0], dtype=np.float32)
    scores, retrieved_docs = doc_store.get_nearest_examples("embeddings", qry_emb, k=top_k)

    results = [
        {
            "qid": str(query["id"][0]),
            "docno": str(doc_id),
            "rank": rank,
            "score": score,
        }
        for rank, (score, doc_id) in enumerate(zip(scores, retrieved_docs["id"]))
    ]
    return {k: [dic[k] for dic in results] for k in results[0]}


def run(
    document_dataset_name_or_path: str,
    query_dataset_name_or_path: str,
    base_model_name_or_path: str,
    checkpoint_path: str,
    batch_size: int,
    output: str,
    tag: Optional[str] = None,
    embedding_cache: Optional[str] = None,
):
    if tag is None:
        tag = checkpoint_path.split("/")[-1].split(".")[0]
    # Load embed function
    embed_fn = get_embed_fn(base_model_name_or_path, checkpoint_path, tokenizer_kwargs)
    # Load data
    documents = None
    if embedding_cache is not None:
        if os.path.isfile(embedding_cache):
            documents = datasets.load_dataset(embedding_cache)
        else:
            documents = load_documents(document_dataset_name_or_path, embed_fn, batch_size)
            documents.save_to_disk(embedding_cache)
    else:
        documents = load_documents(document_dataset_name_or_path, embed_fn, batch_size)
    # Index data
    documents.add_faiss_index(
        column="embeddings",
        string_factory="OPQ32,IVF262144_HNSW32,PQ32,RFlat",
        train_size=10_000_000,
        faiss_verbose=True,
        device=-1,
        metric_type=faiss.METRIC_INNER_PRODUCT,
    )
    documents.save_faiss_index("documents", f"{output.split('.')[0]}.faiss")

    queries = load_queries(query_dataset_name_or_path, embed_fn, batch_size)
    queries.map(
        lambda example: search(example, documents, top_k=1000),
        batched=True,
        batch_size=1,
        remove_columns=["id", "text", "embeddings"],
    )

    # Return as pandas dataframe
    (
        queries.to_pandas()
        .assign(tag=tag, Q0="Q0")
        .loc[:, ["qid", "Q0", "docno", "rank", "score", "tag"]]
        .to_csv(output, sep=" ", index=False, header=False, encoding="utf-8")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="",
        description="Produces an evaluation run for the given model on the full MSMARCO dataset.",
    )
    parser.add_argument(
        "-d",
        "--document_dataset",
        type=str,
        required=True,
        dest="document_dataset",
        action="store",
        help="The document dataset to execute a run on.",
    )
    parser.add_argument(
        "-q",
        "--query_dataset",
        type=str,
        required=True,
        dest="query_dataset",
        action="store",
        help="The query dataset(s) to execute a run on. Comma-separated for multiple.",
    )
    parser.add_argument(
        "-m",
        "--base_model",
        type=str,
        required=True,
        dest="base_model",
        action="store",
        help="The base model name.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        dest="checkpoint",
        action="store",
        help="The path to the model checkpoint.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=True,
        dest="batch_size",
        action="store",
        help="The batch size for embedding inference.",
    )
    parser.add_argument(
        "-e",
        "--embedding_cache",
        type=str,
        required=False,
        default=None,
        dest="embedding_cache",
        action="store",
        help="The output path for to persist the calculated embeddings to for later use.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        action="store",
        help="The output path for the evaluation run file.",
    )
    parser.add_argument(
        "-t",
        "--tag",
        type=str,
        required=False,
        default=None,
        dest="tag",
        action="store",
        help="The tag to write to the runfile.",
    )

    args = parser.parse_args()
    run(
        args.document_dataset,
        args.query_dataset,
        args.base_model,
        args.checkpoint,
        args.batch_size,
        args.output,
        args.tag,
        args.embedding_cache,
    )
