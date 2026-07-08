import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
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
        code_to_groups=None,
    ):
        self.ontology = ontology
        self.retrieval_k = retrieval_k
        self.acceptance_threshold = acceptance_threshold
        self.use_prefilter = use_prefilter
        self.code_to_groups = code_to_groups or {}

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

    def match(self, text, top_n=5):

        if text is None:
            return {}

        text = str(text).strip()

        if not text:
            return {}

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

        # Get top N results sorted by score (descending)
        top_indices = np.argsort(adjusted_scores)[::-1][:top_n]

        results = {}

        for rank, idx in enumerate(top_indices, start=1):
            score = float(adjusted_scores[idx])

            # Skip if below threshold
            if score < self.acceptance_threshold:
                continue

            code = labels[idx]
            concept = candidate_concepts[code]["concept"]

            result = {
                "code": code,
                "name": concept["description"],
                "score": score,
            }

            # Add diagnostic groups if available
            if code in self.code_to_groups:
                groups = self.code_to_groups[code]
                result["group_1"] = groups.get("Diagnostic group 1")
                result["group_2"] = groups.get("Diagnostic group 2")
                result["group_3"] = groups.get("Diagnostic group 3")

            results[f"top_{rank}"] = result

        return results


def load_ontology(excel_file):
    """
    Input Excel:

    Code | Diagnosis | Diagnostic group 1 | Diagnostic group 2 | Diagnostic group 3

    Multiple rows with same code are treated as synonyms.
    """

    df = pd.read_excel(excel_file)

    required = {"Code", "Diagnosis"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in ontology file: {missing}")

    grouped = defaultdict(list)
    code_to_groups = {}

    for _, row in df.iterrows():
        code = row["Code"]
        diagnosis = row["Diagnosis"]

        if pd.isna(code) or pd.isna(diagnosis):
            continue

        grouped[code].append(str(diagnosis).strip())

        # Store group information for each code (take first occurrence)
        if code not in code_to_groups:
            code_to_groups[code] = {
                "Diagnostic group 1": row.get("Diagnostic group 1"),
                "Diagnostic group 2": row.get("Diagnostic group 2"),
                "Diagnostic group 3": row.get("Diagnostic group 3"),
            }

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

    return ontology, code_to_groups


def process_jsonl(
    input_file,
    output_file,
    matcher,
    top_n=5,
):

    n_cases = 0

    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            primary = record.get("primary_diagnosis")

            results = matcher.match(primary, top_n=top_n)

            record["valid_primary_diagnoses"] = {}

            if not results:
                # Add placeholder structure for top_1 when no matches
                record["valid_primary_diagnoses"]["top_1"] = {
                    "valid_primary_diagnosis_code": None,
                    "valid_primary_diagnosis_name": None,
                    "valid_primary_diagnosis_score": None,
                    "valid_primary_diagnosis_group_1": None,
                    "valid_primary_diagnosis_group_2": None,
                    "valid_primary_diagnosis_group_3": None,
                }
            else:
                for rank_key, result in results.items():
                    record["valid_primary_diagnoses"][rank_key] = {
                        "valid_primary_diagnosis_code": result["code"],
                        "valid_primary_diagnosis_name": result["name"],
                        "valid_primary_diagnosis_score": result["score"],
                        "valid_primary_diagnosis_group_1": result.get("group_1"),
                        "valid_primary_diagnosis_group_2": result.get("group_2"),
                        "valid_primary_diagnosis_group_3": result.get("group_3"),
                    }

            containers = record.get("containers", [])

            for container in containers:
                diagnosis = container.get("diagnosis")

                results = matcher.match(diagnosis, top_n=top_n)

                container["valid_diagnoses"] = {}

                if not results:
                    # Add placeholder structure for top_1 when no matches
                    container["valid_diagnoses"]["top_1"] = {
                        "valid_diagnosis_code": None,
                        "valid_diagnosis_name": None,
                        "valid_diagnosis_score": None,
                        "valid_diagnosis_group_1": None,
                        "valid_diagnosis_group_2": None,
                        "valid_diagnosis_group_3": None,
                    }
                else:
                    for rank_key, result in results.items():
                        container["valid_diagnoses"][rank_key] = {
                            "valid_diagnosis_code": result["code"],
                            "valid_diagnosis_name": result["name"],
                            "valid_diagnosis_score": result["score"],
                            "valid_diagnosis_group_1": result.get("group_1"),
                            "valid_diagnosis_group_2": result.get("group_2"),
                            "valid_diagnosis_group_3": result.get("group_3"),
                        }

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


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    return config


def argparse_setup():
    parser = argparse.ArgumentParser(
        description="Match ontology terms in prediction JSONL using embeddings and reranking",
    )
    parser.add_argument(
        "--config",
        default="configs/ontology.yaml",
        help="Path to ontology configuration YAML file",
    )

    args = parser.parse_args()

    return args.config


def main():

    config_path = argparse_setup()

    config = load_config(config_path)

    # Load ontology
    ontology, code_to_groups = load_ontology(config["ontology_file"])

    matcher = OntologyMatcher(
        ontology=ontology,
        retrieval_k=config["matching"]["retrieval_k"],
        acceptance_threshold=config["matching"]["acceptance_threshold"],
        use_prefilter=config["matching"]["use_prefilter"],
        code_to_groups=code_to_groups,
    )

    process_jsonl(
        config.get("input_file"),
        config.get("output_file"),
        matcher,
        top_n=config["matching"]["top_n"],
    )
