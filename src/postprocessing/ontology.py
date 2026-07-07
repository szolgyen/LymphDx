import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
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
        use_prefilter=True,
    ):
        self.ontology = ontology
        self.retrieval_k = retrieval_k
        self.acceptance_threshold = acceptance_threshold
        self.use_prefilter = use_prefilter

        print("Loading reranker...")
        self.reranker = CrossEncoder(reranker_model)

        if self.use_prefilter:
            print("Loading embedding model...")
            self.embedder = SentenceTransformer(embedding_model)

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

        candidate_concepts = {}

        if self.use_prefilter:
            query_embedding = self.embedder.encode(
                text,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )

            similarities = cos_sim(
                query_embedding,
                self.ontology_embeddings,
            )[0]

            top_k = min(
                self.retrieval_k,
                len(self.ontology),
            )

            top_indices = torch.topk(
                similarities,
                k=top_k,
            ).indices.tolist()

            for idx in top_indices:
                concept = self.ontology[idx]

                label = concept["label"]

                similarity = float(similarities[idx])

                if (
                    label not in candidate_concepts
                    or similarity > candidate_concepts[label]["similarity"]
                ):
                    candidate_concepts[label] = {
                        "concept": concept,
                        "similarity": similarity,
                    }

        else:
            # no retrieval stage: send everything to cross encoder
            for concept in self.ontology:
                label = concept["label"]

                if label not in candidate_concepts:
                    candidate_concepts[label] = {
                        "concept": concept,
                        "similarity": None,
                    }

        pairs = []

        labels = []

        for label, item in candidate_concepts.items():
            concept = item["concept"]

            candidate_text = " ; ".join(concept["all_synonyms"])

            pairs.append((text, candidate_text))

            labels.append(label)

        rerank_scores = self.reranker.predict(pairs)

        adjusted_scores = []

        query_lower = text.lower()

        for score, label in zip(rerank_scores, labels):
            concept = candidate_concepts[label]["concept"]

            bonus = 0.0
            penalty = 0.0

            synonyms = [s.lower() for s in concept["all_synonyms"]]

            # exact phrase match bonus
            for synonym in synonyms:
                if synonym in query_lower:
                    bonus += 0.15

            # penalize candidate being more specific than query
            # e.g. "primary cutaneous marginal zone lymphoma"
            # vs "marginal zone lymphoma"
            best_synonym = max(
                synonyms,
                key=len,
            )

            extra_words = set(best_synonym.split()) - set(query_lower.split())

            if len(extra_words) >= 2:
                penalty -= 0.10

            adjusted_scores.append(float(score) + bonus + penalty)

        best_idx = int(np.argmax(adjusted_scores))

        best_score = float(adjusted_scores[best_idx])

        best_code = labels[best_idx]

        if best_score < self.acceptance_threshold:
            return None

        best_concept = candidate_concepts[best_code]["concept"]

        return {
            "code": best_code,
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
        required=False,
        default="configs/extraction/LN_Dx_dictionary_codes_20260625.xlsx",
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

    parser.add_argument(
        "--use-prefilter",
        action="store_true",
        help="Use embedding retrieval before reranking",
    )

    args = parser.parse_args()

    ontology = load_ontology(args.ontology_file)

    matcher = OntologyMatcher(
        ontology=ontology,
        retrieval_k=args.retrieval_k,
        acceptance_threshold=args.threshold,
        use_prefilter=args.use_prefilter,
    )

    process_jsonl(
        args.jsonl_file,
        args.output_file,
        matcher,
    )


if __name__ == "__main__":
    main()
