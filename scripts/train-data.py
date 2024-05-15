import argparse
from typing import List

import ir_datasets
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def load_data_grouped(dataset_name_or_path: str, output_path: str, batch_size: int = 32):
    dataset = ir_datasets.load(dataset_name_or_path)
    assert dataset.has_docs()
    assert dataset.has_queries()
    assert dataset.has_docpairs()

    docstore = dataset.docs_store()

    def __get_docs__(doc_ids: List[str]) -> List[str]:
        docs = docstore.get_many(doc_ids)
        texts = []
        for idx in tqdm(doc_ids):
            texts.append(docs[idx].text)
        return texts

    queries = {}
    for query in tqdm(dataset.queries_iter(), total=dataset.queries_count(), desc="Queries"):
        queries[query.query_id] = query.text

    num_docs = {}
    docpairs = []
    for pair in tqdm(dataset.docpairs_iter(), total=dataset.docpairs_count(), desc="Document pairs"):
        query_id = pair.query_id
        if not num_docs.get(query_id, 0) > batch_size:
            docpairs.append(
                {
                    "query_id": query_id,
                    "doc_id_a": pair.doc_id_a,
                    "doc_id_b": pair.doc_id_b,
                }
            )
            num_docs[query_id] = num_docs.get(query_id, 0) + 1
    df = pd.DataFrame(docpairs).groupby("query_id").head(batch_size).reset_index()
    df = df.sort_values("query_id").groupby("query_id").filter(lambda x: len(x) == batch_size)
    df["positive_doc_text"] = __get_docs__(df["doc_id_a"])
    df["negative_doc_text"] = __get_docs__(df["doc_id_b"])
    df = df.drop(["doc_id_a", "doc_id_b"], axis=1)
    df["query_text"] = df["query_id"].progress_apply(lambda x: queries.get(x))
    df.to_parquet(output_path + ".parquet")


def load_data(dataset_name_or_path: str, output_path: str, max_samples: int = None):
    dataset = ir_datasets.load(dataset_name_or_path)
    assert dataset.has_docs()
    assert dataset.has_queries()
    assert dataset.has_docpairs()

    docstore = dataset.docs_store()

    queries = {}
    for query in tqdm(dataset.queries_iter(), total=dataset.queries_count(), desc="Queries"):
        queries[query.query_id] = query.text

    docpairs = []
    for i, pair in tqdm(
        enumerate(dataset.docpairs_iter()),
        total=dataset.docpairs_count(),
        desc="Document pairs",
    ):
        if max_samples is not None and i > max_samples:
            break
        query_id = pair.query_id
        docpairs.append(
            {
                "query_id": query_id,
                "query_text": queries[query_id],
                "positive_doc_text": docstore.get(pair.doc_id_a).text,
                "negative_doc_text": docstore.get(pair.doc_id_b).text,
            }
        )
    docpairs = pd.DataFrame(docpairs)
    docpairs.to_parquet(output_path + ".parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train Dataset Parser",
        description="Converts an ir-dataset into the format expected by the train dataloaders.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        dest="dataset",
        action="store",
        help="The dataset to parse.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        action="store",
        help="The path to save the parsed data file to.",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        required=False,
        dest="samples",
        action="store",
        help="The maximum number of samples to parse.",
        default=None,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        dest="batch_size",
        action="store",
        help="The batch size to parse grouped samples into.",
        default=None,
    )
    args = parser.parse_args()
    if args.batch_size is not None:
        load_data_grouped(args.dataset, args.output, args.batch_size)
    else:
        load_data(args.dataset, args.output, args.samples)
