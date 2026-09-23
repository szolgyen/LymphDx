#TODO: The module needs a major revision for performance and maintainability.
# It is only commited because it is functional and helps to track the development direction.

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


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
        classifier_model=None,
        use_classifier=False,
    ):
        self.ontology = ontology
        self.retrieval_k = retrieval_k
        self.acceptance_threshold = acceptance_threshold
        self.use_prefilter = use_prefilter
        self.code_to_groups = code_to_groups or {}
        self.use_classifier = use_classifier

        if self.use_classifier:
            logger.info("Loading classifier model: %s", classifier_model)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model)
            self.classifier = AutoModelForCausalLM.from_pretrained(
                classifier_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.classifier.eval()
        else:
            logger.info("Loading reranker model: %s", reranker_model)
            self.reranker = CrossEncoder(reranker_model)

        if self.use_prefilter:
            logger.info("Loading embedding model: %s", embedding_model)
            self.embedder = SentenceTransformer(embedding_model)

            self.ontology_texts = [item["description"] for item in ontology]

            logger.info(
                "Computing ontology embeddings for %d concepts", len(self.ontology_texts)
            )
            self.ontology_embeddings = self.embedder.encode(
                self.ontology_texts,
                normalize_embeddings=True,
                convert_to_tensor=True,
                show_progress_bar=True,
            )

    @staticmethod
    def split_output(text: str) -> tuple[str, str | None]:
        """Return (markdown_reasoning, answer). answer is None if no tag was found."""
        matches = list(ANSWER_RE.finditer(text))
        if not matches:
            return text.strip(), None

        last = matches[-1]                      # last match, in case the tag appears earlier
        answer = last.group(1).strip()
        markdown = (text[:last.start()] + text[last.end():]).strip()
        return markdown, answer

    def _classify_matches(self, candidate: str, ontology: list):
        """Select best matching ontology using MedGemma classifier.

        Returns: Best matching ontology code.
        """
        ontology_text = "\n".join(f"- {x}" for x in ontology)

        prompt = (
            f"Select the single ontology that is semantically the best match for the "
            f"'{candidate}' concept from the following list:\n"
            f"{ontology_text}\n\n"
            f"Provide your reasoning, and put your final answer at the end in the format "
            f"<answer>ontology name</answer>."
        )

        chat = self.classifier_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.classifier_tokenizer(
                chat,
                return_tensors="pt",
                truncation=False,
                max_length=4096,
            ).to(self.classifier.device)

        if inputs["input_ids"].shape[1] > 4096:
            logger.warning(
                "Input exceeds the maximum token limit of 4096 tokens!"
            )

        with torch.no_grad():
            output = self.classifier.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

        input_len = inputs["input_ids"].shape[1]

        response = self.classifier_tokenizer.decode(
            output[0][input_len:],
            skip_special_tokens=True,
        )

        response = response.strip()

        markdown, answer =self.split_output(response)

        return markdown, answer

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

        if self.use_classifier:
            ontology_terms = [self.ontology_texts[idx] for idx in top_indices]
            # similarity_scores = [similarities[idx] for idx in top_indices]
            reasoning, matching_term = self._classify_matches(text, ontology_terms)
        else:
            rerank_scores = self.reranker.predict(pairs)
            classifier_probs = None

        adjusted_scores = []

        query_lower = text.lower()

        if not self.use_classifier:
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
                    result["group_1"] = groups.get("WHO-like Subcategories")
                    result["group_2"] = groups.get("WHO-like Categories")
                    result["group_3"] = groups.get("WHO-like Major Sections/Lineages")
                    result["group_4"] = groups.get("Diagnostic group 4")

                results[f"top_{rank}"] = result

        else:
            results = {}

            if matching_term is not None:
                # Find the corresponding code for the matching term
                for concept in self.ontology:
                    if matching_term in concept["all_synonyms"]:
                        code = concept["label"]
                        result = {
                            "code": code,
                            "name": matching_term,
                            "score": None,  # Classifier does not provide a score
                        }

                        # Add diagnostic groups if available
                        if code in self.code_to_groups:
                            groups = self.code_to_groups[code]
                            result["group_1"] = groups.get("WHO-like Subcategories")
                            result["group_2"] = groups.get("WHO-like Categories")
                            result["group_3"] = groups.get("WHO-like Major Sections/Lineages")
                            result["group_4"] = groups.get("Diagnostic group 4")

                        results["top_1"] = result
                        break

        return results


def load_ontology(excel_file):
    """
    Input Excel:

    Code | Diagnosis | WHO-like Subcategories | WHO-like Categories | WHO-like Major Sections/Lineages | Diagnostic group 4

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
                "WHO-like Subcategories": row.get("WHO-like Subcategories"),
                "WHO-like Categories": row.get("WHO-like Categories"),
                "WHO-like Major Sections/Lineages": row.get(
                    "WHO-like Major Sections/Lineages"
                ),
                "Diagnostic group 4": row.get("Diagnostic group 4"),
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
                f"{prefix}_group_4": None,
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
            f"{prefix}_group_4": result.get("group_4"),
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
                "group_1": groups.get("WHO-like Subcategories"),
                "group_2": groups.get("WHO-like Categories"),
                "group_3": groups.get("WHO-like Major Sections/Lineages"),
                "group_4": groups.get("Diagnostic group 4"),
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
                    "group_1": groups.get("WHO-like Subcategories"),
                    "group_2": groups.get("WHO-like Categories"),
                    "group_3": groups.get("WHO-like Major Sections/Lineages"),
                    "group_4": groups.get("Diagnostic group 4"),
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
        use_classifier = config["matching"].get("use_classifier", False)
        matcher = OntologyMatcher(
            ontology=ontology,
            embedding_model=config["models"]["embedding_model"],
            reranker_model=config["models"]["reranker_model"],
            retrieval_k=config["matching"]["retrieval_k"],
            acceptance_threshold=config["matching"]["acceptance_threshold"],
            use_prefilter=config["matching"]["use_prefilter"],
            code_to_groups=code_to_groups,
            classifier_model=config["models"].get("classifier_model") if use_classifier else None,
            use_classifier=use_classifier,
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
