import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)


class OntologyMatcher:
    def __init__(
        self,
        ontology,
        embedding_model,
        reranker_model,
        retrieval_k,
        acceptance_threshold,
        use_prefilter,
        code_to_groups,
    ):
        self.ontology = ontology
        self.retrieval_k = retrieval_k
        self.acceptance_threshold = acceptance_threshold
        self.use_prefilter = use_prefilter
        self.code_to_groups = code_to_groups or {}

        logger.info("Loading reranker model: %s", reranker_model)
        self.reranker = CrossEncoder(reranker_model)

        if self.use_prefilter:
            logger.info("Loading embedding model: %s", embedding_model)
            self.embedder = SentenceTransformer(embedding_model)

            ontology_texts = [item["description"] for item in ontology]

            logger.info(
                "Computing ontology embeddings for %d concepts", len(ontology_texts)
            )
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
    logger.info("Loading ontology from %s", excel_file)
    df = pd.read_excel(excel_file)

    required = {"Code", "Diagnosis"}

    missing = required - set(df.columns)

    if missing:
        logger.error("Missing columns in ontology file: %s", missing)
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

    logger.info(
        f"Loaded {len(ontology)} synonym entries for {len(grouped)} ontology concepts"
    )

    diagnosis_to_code = {}

    for _, row in df.iterrows():
        code = row["Code"]
        diagnosis = str(row["Diagnosis"]).strip()

        diagnosis_to_code[diagnosis] = code

    return ontology, code_to_groups, diagnosis_to_code


def iter_records(input_file):
    if input_file.endswith(".jsonl"):
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    elif input_file.endswith(".json"):
        with open(input_file) as f:
            yield from json.load(f)

    else:
        raise ValueError(f"Unsupported input file format: {input_file}")


def build_diagnosis_results(results, prefix):
    if not results:
        return {
            "top_1": {
                f"{prefix}_code": None,
                f"{prefix}_name": None,
                f"{prefix}_score": None,
                f"{prefix}_group_1": None,
                f"{prefix}_group_2": None,
                f"{prefix}_group_3": None,
            }
        }

    output = {}

    for rank_key, result in results.items():
        output[rank_key] = {
            f"{prefix}_code": result["code"],
            f"{prefix}_name": result["name"],
            f"{prefix}_score": result["score"],
            f"{prefix}_group_1": result.get("group_1"),
            f"{prefix}_group_2": result.get("group_2"),
            f"{prefix}_group_3": result.get("group_3"),
        }

    return output


def process_record(
    record,
    matcher,
    top_n,
    ontology_matching,
    code_to_groups,
    diagnosis_to_code,
):
    primary = record.get("primary_diagnosis")
    primary_code = diagnosis_to_code.get(primary)

    if ontology_matching:
        results = matcher.match(primary, top_n=top_n)
    else:
        groups = code_to_groups.get(primary_code, {})
        results = {
            "top_1": {
                "code": primary_code,
                "name": primary,
                "score": None,
                "group_1": groups.get("Diagnostic group 1"),
                "group_2": groups.get("Diagnostic group 2"),
                "group_3": groups.get("Diagnostic group 3"),
            }
        }

    record["valid_primary_diagnoses"] = build_diagnosis_results(
        results,
        prefix="valid_primary_diagnosis",
    )

    for container in record.get("containers", []):
        diagnosis = container.get("diagnosis")
        diagnosis_code = diagnosis_to_code.get(diagnosis)

        if ontology_matching:
            results = matcher.match(diagnosis, top_n=top_n)
        else:
            groups = code_to_groups.get(diagnosis_code, {})
            results = {
                "top_1": {
                    "code": diagnosis_code,
                    "name": diagnosis,
                    "score": None,
                    "group_1": groups.get("Diagnostic group 1"),
                    "group_2": groups.get("Diagnostic group 2"),
                    "group_3": groups.get("Diagnostic group 3"),
                }
            }

        container["valid_diagnoses"] = build_diagnosis_results(
            results,
            prefix="valid_diagnosis",
        )

    return record


def process_json(
    input_file,
    output_file,
    matcher,
    top_n,
    ontology_matching,
    code_to_groups,
    diagnosis_to_code,
):
    with open(output_file, "w") as fout:
        for n_cases, record in enumerate(iter_records(input_file), start=1):
            record = process_record(
                record=record,
                matcher=matcher,
                top_n=top_n,
                ontology_matching=ontology_matching,
                code_to_groups=code_to_groups,
                diagnosis_to_code=diagnosis_to_code,
            )

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if n_cases % 100 == 0:
                logger.info(f"Processed {n_cases} cases...")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error("Configuration file not found: %s", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def main(run_name: str, config: dict) -> None:
    """Main entry point for ontology matching.

    Args:
        run_name: Name of the run.
        config: Configuration dict with ontology settings.

    Raises:
        ValueError: If run_name or config is missing.
    """
    if not run_name:
        logger.error("run_name must be provided to construct input/output paths.")
        raise ValueError("run_name must be provided to construct input/output paths.")
    if not config:
        logger.error("config must be provided.")
        raise ValueError("config must be provided.")

    logger.info("Starting ontology matching for run: %s", run_name)

    # Check if we are in ontology matching mode
    ontology_matching = config.get("ontology_matching", False)

    input_file = f"outputs/{run_name}/predictions_{run_name}.jsonl"
    output_file = f"outputs/{run_name}/ontology_{run_name}.jsonl"

    ontology, code_to_groups, diagnosis_to_code = load_ontology(
        config["diagnosis_dictionary"]
    )

    matcher = None
    if ontology_matching:
        logger.info("Initializing ontology matcher")
        matcher = OntologyMatcher(
            ontology=ontology,
            embedding_model=config["models"]["embedding_model"],
            reranker_model=config["models"]["reranker_model"],
            retrieval_k=config["matching"]["retrieval_k"],
            acceptance_threshold=config["matching"]["acceptance_threshold"],
            use_prefilter=config["matching"]["use_prefilter"],
            code_to_groups=code_to_groups,
        )

    process_json(
        input_file,
        output_file,
        matcher,
        top_n=config["matching"]["top_n"],
        ontology_matching=ontology_matching,
        code_to_groups=code_to_groups,
        diagnosis_to_code=diagnosis_to_code,
    )
