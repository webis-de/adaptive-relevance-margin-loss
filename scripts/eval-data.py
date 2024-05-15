import argparse
from typing import Optional
import ir_datasets
import pandas as pd
from tqdm import tqdm


def load_data(dataset_name_or_path: str, output_path: str, run_path: Optional[str] = None):
    dataset = ir_datasets.load(dataset_name_or_path)
    assert dataset.has_docs()
    assert dataset.has_queries()
    assert dataset.has_qrels()
    if run_path is None:
        assert dataset.has_scoreddocs()

    docstore = dataset.docs_store()

    queries = {}
    for query in tqdm(dataset.queries_iter(), total=dataset.queries_count(), desc="Queries"):
        queries[query.query_id] = query.text

    if run_path is None:
        scoreddocs = []
        for scoreddoc in tqdm(dataset.scoreddocs_iter(), total=dataset.scoreddocs_count(), desc="Documents"):
            query_id = scoreddoc.query_id
            doc_id = scoreddoc.doc_id
            scoreddocs.append(
                {
                    "query_id": query_id,
                    "query_text": queries[query_id],
                    "doc_id": doc_id,
                    "doc_text": docstore.get(doc_id).text,
                    "score": scoreddoc.score,
                }
            )
        scoreddocs = pd.DataFrame(scoreddocs)
    else:
        run = (
            pd.read_csv(run_path, sep=" ", names=["query_id", "q0", "doc_id", "rank", "score", "tag"], header=None)
            .drop(["q0", "tag"], axis=1)
            .astype({"query_id": str, "doc_id": str})
        )
        scoreddocs = []
        for _, row in tqdm(run.iterrows(), desc="Documents"):
            query_id = row.query_id
            doc_id = row.doc_id
            scoreddocs.append(
                {
                    "query_id": query_id,
                    "query_text": queries[query_id],
                    "doc_id": doc_id,
                    "doc_text": docstore.get(doc_id).text,
                    "score": row.score,
                }
            )
        scoreddocs = pd.DataFrame(scoreddocs)

    qrels = []
    for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count(), desc="Qrels"):
        qrels.append({"query_id": qrel.query_id, "doc_id": qrel.doc_id, "label": qrel.relevance})
    qrels = pd.DataFrame(qrels)

    df = scoreddocs.merge(qrels, on=["query_id", "doc_id"], how="left").fillna(0).drop(["score"], axis=1)
    df.to_parquet(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Eval Dataset Parser",
        description="Converts an ir-dataset into the format expected by the eval dataloaders.",
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
        "-r",
        "--run",
        type=str,
        required=False,
        default=None,
        dest="run",
        action="store",
        help="The runfile to rerank, if applicable.",
    )
    args = parser.parse_args()
    load_data(args.dataset, args.output, args.run)
