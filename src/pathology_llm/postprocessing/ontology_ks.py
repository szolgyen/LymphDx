import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import kstest
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim


class OntologyMatcher:
    def __init__(
        self,
        ontology,
        embedding_model="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",  # "vinid/plip",  # "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",  # "neuml/pubmedbert-base-embeddings",
        reranker_model="ncbi/MedCPT-Cross-Encoder",
        retrieval_k=50,
        acceptance_threshold=0.55,
    ):
        self.ontology = ontology
        self.retrieval_k = retrieval_k
        self.acceptance_threshold = acceptance_threshold

        print("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)

        print("Loading reranker...")
        self.reranker = CrossEncoder(reranker_model)

        ontology_texts = [item["description"] for item in ontology]

        print("Computing ontology embeddings...")
        self.ontology_embeddings = self.embedder.encode(
            ontology_texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

    def match(self, text):

        if text is None:
            return None

        text = str(text).strip()

        if not text:
            return None

        try:
            query_embedding = self.embedder.encode(
                text,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
        except ValueError as e:
            print("Skipping diagnosis: %s", e)
            return None

        # similarity against ALL ontology entries
        similarities = cos_sim(
            query_embedding,
            self.ontology_embeddings,
        )[0]

        # rank every synonym from best to worst
        ranked_indices = torch.argsort(
            similarities,
            descending=True,
        ).tolist()

        # collect ranks per ontology concept
        concept_ranks = defaultdict(list)

        for rank, idx in enumerate(ranked_indices, start=1):
            label = self.ontology[idx]["label"]
            concept_ranks[label].append(rank)

        n_terms = len(ranked_indices)

        # KS enrichment score:
        # are this concept's synonyms concentrated near the top?
        concept_scores = {}

        for label, ranks in concept_ranks.items():
            # convert ranks to [0,1]
            # rank 1 -> close to 0 (top)
            values = [(r - 1) / (n_terms - 1) for r in ranks]

            # compare observed rank distribution
            # against uniform random distribution
            ks_stat, p_value = kstest(
                values,
                "uniform",
            )

            # We want high enrichment near the top.
            # KS gives magnitude; multiply by number of hits
            score = ks_stat * len(ranks)

            concept_scores[label] = {
                "score": score,
                "ranks": ranks,
            }

        # top concepts by KS enrichment
        top_labels = sorted(
            concept_scores,
            key=lambda x: concept_scores[x]["score"],
            reverse=True,
        )[: self.retrieval_k]

        # existing cross encoder reranking
        pairs = []
        labels = []

        for label in top_labels:
            concept = next(x for x in self.ontology if x["label"] == label)

            candidate_text = " ; ".join(concept["all_synonyms"])

            pairs.append(
                (
                    text,
                    candidate_text,
                )
            )

            labels.append(label)

        rerank_scores = self.reranker.predict(pairs)

        best_idx = int(np.argmax(rerank_scores))

        best_label = labels[best_idx]

        best_concept = next(x for x in self.ontology if x["label"] == best_label)

        best_score = float(rerank_scores[best_idx])

        if best_score < self.acceptance_threshold:
            return None

        return {
            "code": best_label,
            "name": best_concept["description"],
            "score": best_score,
        }


def load_ontology(excel_file):
    """
    Input Excel:

    Code | Diagnosis

    Multiple rows with same code are treated as synonyms.
    """

    df = pd.read_excel(excel_file)

    required = {"Code", "Diagnosis"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in ontology file: {missing}")

    grouped = defaultdict(list)

    for _, row in df.iterrows():
        code = row["Code"]
        diagnosis = row["Diagnosis"]

        if pd.isna(code) or pd.isna(diagnosis):
            continue

        grouped[code].append(str(diagnosis).strip())

    ontology = []

    for code, synonyms in grouped.items():
        synonyms = sorted(set(synonyms))

        for synonym in synonyms:
            ontology.append(
                {
                    "label": code,
                    "description": synonym,
                    "all_synonyms": synonyms,
                }
            )

    print(
        f"Loaded {len(ontology)} synonym entries for {len(grouped)} ontology concepts"
    )

    return ontology


def process_jsonl(
    input_file,
    output_file,
    matcher,
):

    n_cases = 0

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            primary = record.get("primary_diagnosis")

            result = matcher.match(primary)

            if result is None:
                record["valid_primary_diagnosis_code"] = None
                record["valid_primary_diagnosis_name"] = None
                record["valid_primary_diagnosis_score"] = None
            else:
                record["valid_primary_diagnosis_code"] = result["code"]
                record["valid_primary_diagnosis_name"] = result["name"]
                record["valid_primary_diagnosis_score"] = result["score"]

            containers = record.get("containers", [])

            for container in containers:
                diagnosis = container.get("diagnosis")

                result = matcher.match(diagnosis)

                if result is None:
                    container["valid_diagnosis_code"] = None
                    container["valid_diagnosis_name"] = None
                    container["valid_diagnosis_score"] = None
                else:
                    container["valid_diagnosis_code"] = result["code"]
                    container["valid_diagnosis_name"] = result["name"]
                    container["valid_diagnosis_score"] = result["score"]

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            n_cases += 1

            if n_cases % 100 == 0:
                print(f"Processed {n_cases} cases...")

    print(f"Finished. Processed {n_cases} cases.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl-file",
        required=True,
        help="Input JSONL file",
    )

    parser.add_argument(
        "--ontology-file",
        required=True,
        help="Ontology Excel file",
    )

    parser.add_argument(
        "--output-file",
        required=True,
        help="Output JSONL file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=-10.55,
    )

    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    ontology = load_ontology(args.ontology_file)

    matcher = OntologyMatcher(
        ontology=ontology,
        retrieval_k=args.retrieval_k,
        acceptance_threshold=args.threshold,
    )

    process_jsonl(
        args.jsonl_file,
        args.output_file,
        matcher,
    )


if __name__ == "__main__":
    main()
