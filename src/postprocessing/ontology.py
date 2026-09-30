import json
import logging
import re
import string
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
LETTER_RE = re.compile(r"[\(\[]?([A-Z])(?![A-Za-z])")
LETTERS = string.ascii_uppercase  # options are labelled A..Z (single tokens)


def normalize_text(value) -> str:
    """None / NaN -> '', otherwise str() + strip."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


class OntologyMatcher:
    def __init__(
        self,
        ontology_list,
        embedding_model,
        reranker_model,
        retrieval_k,
        acceptance_threshold,
        use_prefilter,
        diagnosis_to_groups,
        classifier_model=None,
        use_classifier=False,
        classifier_mode="logits",
        classifier_batch_size=16,
        classifier_max_options=10,
        classifier_min_prob=0.0,
        classifier_max_new_tokens=384,
        encode_batch_size=64,
    ):
        if use_classifier and not use_prefilter:
            raise ValueError("use_classifier requires use_prefilter (needs a shortlist).")
        if classifier_mode not in ("logits", "generate"):
            raise ValueError("classifier_mode must be 'logits' or 'generate'.")
        if not 1 <= classifier_max_options <= len(LETTERS):
            raise ValueError(f"classifier_max_options must be in 1..{len(LETTERS)}.")

        self.ontology_list = ontology_list
        self.retrieval_k = retrieval_k
        self.acceptance_threshold = acceptance_threshold
        self.use_prefilter = use_prefilter
        self.diagnosis_to_groups = diagnosis_to_groups or {}
        self.use_classifier = use_classifier
        self.classifier_mode = classifier_mode
        self.classifier_batch_size = classifier_batch_size
        self.classifier_max_options = classifier_max_options
        self.classifier_min_prob = classifier_min_prob
        self.classifier_max_new_tokens = classifier_max_new_tokens
        self.encode_batch_size = encode_batch_size
        self._cache = {}
        self._reasoning_cache = {}

        # ---- lookup structures (built once) ----
        # normalized string -> concept index (exact-match shortcut)
        self.text_to_idx = {}
        for i, t in enumerate(self.ontology_list):
            self.text_to_idx.setdefault(_norm(t), i)

        # no-prefilter case: every concept is a candidate
        self._all_candidates = list(range(len(self.ontology_list)))

        # ---- models ----
        if self.use_classifier:
            logger.info("Loading classifier model: %s", classifier_model)
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model)
            tok = self.classifier_tokenizer
            tok.padding_side = "left"  # required for batched decoder-only inference
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            self.classifier = AutoModelForCausalLM.from_pretrained(
                classifier_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.classifier.eval()

            # Token ids of the option letters (used by the logits mode).
            self.letter_ids = []
            for letter in LETTERS[:classifier_max_options]:
                ids = tok.encode(letter, add_special_tokens=False)
                if len(ids) != 1:
                    raise ValueError(f"Letter {letter!r} is not a single token: {ids}")
                self.letter_ids.append(ids[0])
        else:
            logger.info("Loading reranker model: %s", reranker_model)
            self.reranker = CrossEncoder(reranker_model)

        if self.use_prefilter:
            logger.info("Loading embedding model: %s", embedding_model)
            self.embedder = SentenceTransformer(embedding_model)
            logger.info(
                "Computing ontology embeddings for %d concepts", len(self.ontology_list)
            )
            self.ontology_embeddings = self.embedder.encode(
                self.ontology_list,
                normalize_embeddings=True,
                convert_to_tensor=True,
                batch_size=encode_batch_size,
                show_progress_bar=True,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def match(self, text, top_n=5):
        """Single-text convenience wrapper around match_many."""
        key = normalize_text(text)
        if not key:
            return {}
        return self.match_many([key], top_n=top_n)[key]

    def match_many(self, texts, top_n=5):
        """Match many texts at once. Returns {normalized_text: results_dict}.

        Duplicates are collapsed and previously seen texts are served from cache,
        so each distinct string is matched (and sent to the LLM) at most once.
        """
        keys = list(dict.fromkeys(k for k in map(normalize_text, texts) if k))
        todo = [k for k in keys if (k, top_n) not in self._cache]
        if todo:
            for k, res in self._match_uncached(todo, top_n).items():
                self._cache[(k, top_n)] = res
        return {k: self._cache[(k, top_n)] for k in keys}

    def get_reasoning(self, text):
        """Get stored reasoning for a text (only available if classifier_mode='generate').
        Returns the reasoning string or None if not available.
        """
        key = normalize_text(text)
        return self._reasoning_cache.get(key)

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------
    def _match_uncached(self, texts, top_n):
        cands = self._retrieve(texts)
        if self.use_classifier:
            ranked = self._rank_with_classifier(texts, cands)
        else:
            ranked = self._rank_with_reranker(texts, cands)
        return {t: self._finalize(r, top_n) for t, r in zip(texts, ranked)}

    def _retrieve(self, texts):
        """Per text: ordered list of unique (code, best_entry_idx) candidates."""
        if not self.use_prefilter:
            return [self._all_candidates] * len(texts)

        q = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_tensor=True,
            batch_size=self.encode_batch_size,
            show_progress_bar=False,
        )
        sims = cos_sim(q, self.ontology_embeddings)
        k = min(self.retrieval_k, len(self.ontology_list))
        return torch.topk(sims, k=k, dim=1).indices.tolist()

    # ---- reranker path -------------------------------------------------
    @staticmethod
    def _adjust(score, candidate, query_lower):
        cand = candidate.lower()
        bonus = 0.15 if cand in query_lower else 0.0
        extra = set(cand.split()) - set(query_lower.split())
        penalty = -0.10 if len(extra) >= 2 else 0.0
        return float(score) + bonus + penalty

    def _rank_with_reranker(self, texts, cands):
        pairs, spans = [], []
        for text, c in zip(texts, cands):
            start = len(pairs)
            pairs.extend((text, self.ontology_list[i]) for i in c)
            spans.append((start, len(pairs)))

        scores = (
            self.reranker.predict(
                pairs, batch_size=self.encode_batch_size, show_progress_bar=False
            )
            if pairs
            else []
        )

        ranked = []
        for text, c, (a, b) in zip(texts, cands, spans):
            q = text.lower()
            adj = [
                self._adjust(s, self.ontology_list[i], q)
                for s, i in zip(scores[a:b], c)
            ]
            order = np.argsort(adj)[::-1]
            ranked.append([(c[j], adj[j]) for j in order])
        return ranked

    # ---- classifier path -----------------------------------------------
    @staticmethod
    def split_output(text: str) -> tuple[str, str | None]:
        """Return (markdown_reasoning, answer). answer is None if no tag was found."""
        matches = list(ANSWER_RE.finditer(text))
        if not matches:
            return text.strip(), None
        last = matches[-1]
        answer = last.group(1).strip()
        markdown = (text[: last.start()] + text[last.end():]).strip()
        return markdown, answer

    def _build_prompt(self, concept, shortlist, cot):
        options = "\n".join(
            f"{LETTERS[j]}. {self.ontology_list[i]}" for j, i in enumerate(shortlist)
        )
        prompt = (
            "You are mapping a diagnostic concept to the best matching ontology entry.\n\n"
            f"Concept: {concept}\n\n"
            f"Options:\n{options}\n\n"
            "Choose the option that is the closest semantic match to the concept. "
        )
        if cot:
            prompt += (
                "\n\nGive your reasoning in three sections:\n"
                "  1. Analysis of the concept.\n"
                "  2. Analysis of the options.\n"
                "  3. Final decision with explanation.\n\n"
                "Put the letter of your final answer between <answer> and </answer> tags.\n\n"
                "For example, if your final answer is Z, write as <answer>Z</answer>."
            )
        else:
            prompt += "Reply with the option letter only."
        return prompt

    def _encode(self, prompts):
        tok = self.classifier_tokenizer
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False)
        if enc["input_ids"].shape[1] > 4096:
            logger.warning("Prompt batch has %d tokens (>4096).", enc["input_ids"].shape[1])
        return enc.to(self.classifier.device)

    @torch.inference_mode()
    def _option_probs(self, prompts, n_options):
        """One forward pass per batch; softmax over the option-letter logits."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for s in range(0, len(order), self.classifier_batch_size):
            batch = order[s : s + self.classifier_batch_size]
            enc = self._encode([prompts[i] for i in batch])
            try:
                # only compute logits for the last position (vocab is huge)
                logits = self.classifier(**enc, logits_to_keep=1).logits[:, -1, :]
            except TypeError:  # older transformers
                logits = self.classifier(**enc).logits[:, -1, :]
            for row, i in zip(logits, batch):
                k = n_options[i]
                opt = row[self.letter_ids[:k]].float()
                out[i] = torch.softmax(opt, dim=-1).cpu().numpy()
        return out

    @torch.inference_mode()
    def _generate(self, prompts):
        """Batched greedy generation (sorted by length to minimise padding)."""
        tok = self.classifier_tokenizer
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        total_prompts = len(prompts)
        processed = 0

        logger.info(f"Starting generation for {total_prompts} prompts (with reasoning/CoT)...")

        for s in range(0, len(order), self.classifier_batch_size):
            batch = order[s : s + self.classifier_batch_size]
            enc = self._encode([prompts[i] for i in batch])
            gen = self.classifier.generate(
                **enc,
                max_new_tokens=self.classifier_max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            texts = tok.batch_decode(
                gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for i, t in zip(batch, texts):
                out[i] = t.strip()

            processed += len(batch)
            if processed % self.classifier_batch_size == 0 or processed == total_prompts:
                logger.info(f"Generation progress: {processed}/{total_prompts} prompts completed")

        logger.info(f"Generation completed for all {total_prompts} prompts")
        return out

    @staticmethod
    def _parse_letter(answer, n_options):
        if not answer:
            return None
        m = LETTER_RE.match(answer.strip().upper())
        if not m:
            return None
        j = LETTERS.find(m.group(1))
        return j if 0 <= j < n_options else None

    def _rank_with_classifier(self, texts, cands):
        ranked = [[] for _ in texts]
        pending = []

        # Cheap shortcut: an unambiguous exact synonym match needs no LLM.
        for t_i, text in enumerate(texts):
            idx = self.text_to_idx.get(_norm(text))
            if idx is not None:
                ranked[t_i] = [(idx, 1.0)]
            else:
                pending.append(t_i)

        logger.info("Classifier: %d/%d texts need the LLM", len(pending), len(texts))
        if not pending:
            return ranked

        shortlists = [cands[i][: self.classifier_max_options] for i in pending]
        cot = self.classifier_mode == "generate"
        prompts = [
            self._build_prompt(texts[i], sl, cot) for i, sl in zip(pending, shortlists)
        ]

        if not cot:
            probs = self._option_probs(prompts, [len(sl) for sl in shortlists])
            for i, sl, p in zip(pending, shortlists, probs):
                order = np.argsort(p)[::-1]
                ranked[i] = [(sl[j], float(p[j])) for j in order]
        else:
            for i, sl, out in zip(pending, shortlists, self._generate(prompts)):
                reasoning, answer = self.split_output(out)
                logger.debug("%s -> %s\n%s", texts[i], answer, reasoning)

                # Store complete model output (including <answer> tags) for reasoning
                self._reasoning_cache[texts[i]] = out

                # Parse the answer letter
                j = self._parse_letter(answer, len(sl))

                if j is not None:
                    ranked[i] = [(sl[j], None)]

        return ranked

    # ---- output --------------------------------------------------------
    def _make_result(self, idx, score):
        name = self.ontology_list[idx]
        info = self.diagnosis_to_groups.get(name, {})
        return {
            "code": info.get("Code"),
            "name": name,
            "score": score,
            "group_1": info.get("WHO-like Subcategories"),
            "group_2": info.get("WHO-like Categories"),
            "group_3": info.get("WHO-like Major Sections/Lineages"),
            "group_4": info.get("Diagnostic group 4"),
        }

    def _finalize(self, ranked, top_n):
        threshold = (
            self.classifier_min_prob if self.use_classifier else self.acceptance_threshold
        )
        results, rank = {}, 0
        for idx, score in ranked:
            if score is not None and score < threshold:
                continue
            rank += 1
            results[f"top_{rank}"] = self._make_result(idx, score)
            if rank >= top_n:
                break
        return results


def load_ontology(excel_file: str) -> tuple[list, dict]:
    """ Load ontology from an Excel file and return a list of concepts, and the
    mapping of diagnosis concepts to codes and groups.

    Args:
        excel_file (str): Path to the Excel file containing the ontology.

    Returns:
        tuple: A tuple containing:
            - ontology_list (list): A list of ontology concepts.
            - diagnosis_to_groups (dict): A dictionary mapping diagnosis concepts to
                their corresponding codes and groups
    """
    logger.info("Loading ontology from %s", excel_file)
    df = pd.read_excel(excel_file)

    ontology_list = []
    diagnosis_to_groups = {}

    # Iterate through the DataFrame rows and map codes to their corresponding diagnoses and group names.
    for _, row in df.iterrows():
        diagnosis = str(row["Diagnosis"]).strip()

        ontology_list.append(diagnosis)

        # Store code and group information for each diagnosis:
        diagnosis_to_groups[diagnosis] = {
            "Code": row.get("Code"),
            "WHO-like Subcategories": row.get("WHO-like Subcategories"),
            "WHO-like Categories": row.get("WHO-like Categories"),
            "WHO-like Major Sections/Lineages": row.get(
                "WHO-like Major Sections/Lineages"
            ),
            "Diagnostic group 4": row.get("Diagnostic group 4"),
        }

    logger.info(
        f"Loaded {len(ontology_list)} ontology concepts."
    )

    return ontology_list, diagnosis_to_groups


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


def chunked(iterable, size):
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


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


def collect_diagnoses(record):
    yield record.get("primary_diagnosis")
    for container in record.get("containers", []):
        yield container.get("diagnosis")


def resolve_diagnosis(diagnosis, matches, ontology_matching, diagnosis_to_groups):
    """Return a {top_k: result} dict for one diagnosis string."""
    key = normalize_text(diagnosis)

    if ontology_matching:
        return matches.get(key, {})

    if not key:
        return {}

    info = diagnosis_to_groups.get(key)
    if info is None:
        logger.warning("Diagnosis %r not found in ontology.", diagnosis)
        info = {}

    return {
        "top_1": {
            "code": info.get("Code"),
            "name": diagnosis,
            "score": None,
            "group_1": info.get("WHO-like Subcategories"),
            "group_2": info.get("WHO-like Categories"),
            "group_3": info.get("WHO-like Major Sections/Lineages"),
            "group_4": info.get("Diagnostic group 4"),
        }
    }


def process_record(record, matches, ontology_matching, diagnosis_to_groups):
    results = resolve_diagnosis(
        record.get("primary_diagnosis"),
        matches,
        ontology_matching,
        diagnosis_to_groups
    )
    record["valid_primary_diagnoses"] = build_diagnosis_results(
        results, prefix="valid_primary_diagnosis"
    )

    for container in record.get("containers", []):
        results = resolve_diagnosis(
            container.get("diagnosis"),
            matches,
            ontology_matching,
            diagnosis_to_groups
        )
        container["valid_diagnoses"] = build_diagnosis_results(
            results, prefix="valid_diagnosis"
        )

    return record


def _save_reasoning_files(record, matcher, reports_dir):
    """Save reasoning from classifier to a single markdown file per case.
    Combines primary diagnosis and all container diagnoses in one file.

    Args:
        record: The processed record with case_id
        matcher: OntologyMatcher instance with reasoning cache
        reports_dir: Directory to save reasoning files

    Returns:
        int: Number of reasoning files saved (0 or 1)
    """
    case_id = record.get("case_id")
    if not case_id:
        return 0

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    # Collect all diagnoses with reasoning
    diagnoses_with_reasoning = []

    # Primary diagnosis
    primary_diagnosis = record.get("primary_diagnosis")
    if primary_diagnosis:
        reasoning = matcher.get_reasoning(primary_diagnosis)
        if reasoning:
            diagnoses_with_reasoning.append(("Primary Diagnosis", primary_diagnosis, reasoning))

    # Container diagnoses
    for idx, container in enumerate(record.get("containers", [])):
        diagnosis = container.get("diagnosis")
        if diagnosis:
            reasoning = matcher.get_reasoning(diagnosis)
            if reasoning:
                diagnoses_with_reasoning.append((f"Container {idx + 1}", diagnosis, reasoning))

    # If no reasoning found, return 0
    if not diagnoses_with_reasoning:
        return 0

    # Save all reasoning to a single markdown file
    # Format case_id as 4-digit zero-padded
    formatted_case_id = f"{case_id:0>4}" if isinstance(case_id, (int, float)) else str(case_id).zfill(4)
    filename = f"case_{formatted_case_id}.md"
    filepath = reports_path / filename

    # Create markdown content with all diagnoses and their reasoning
    content = f"# Case {formatted_case_id}\n\n"

    for diagnosis_type, diagnosis_text, reasoning in diagnoses_with_reasoning:
        content += f"## {diagnosis_type}\n\n"
        content += f"**Diagnosis:** {diagnosis_text}\n\n"
        content += f"### Reasoning\n\n"
        # Preserve reasoning with proper line breaks for readability
        content += reasoning.strip() + "\n\n"

    try:
        with open(filepath, "w") as f:
            f.write(content)
        logger.info(f"Saved reasoning file: {filename} ({len(diagnoses_with_reasoning)} diagnosis/diagnoses)")
        return 1
    except Exception as e:
        logger.error(f"Failed to save reasoning to {filepath}: {e}")
        return 0


def process_json(
    input_file,
    output_file,
    matcher,
    top_n,
    ontology_matching,
    diagnosis_to_groups,
    chunk_size=256,
    reports_dir=None,
):
    """Stream records in chunks; match every distinct diagnosis in a chunk in one batch.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        matcher: OntologyMatcher instance
        top_n: Number of top results to keep
        ontology_matching: Whether to use ontology matching
        diagnosis_to_groups: Diagnosis to groups mapping
        chunk_size: Batch size for processing
        reports_dir: Optional directory to save reasoning files (only when classifier_mode='generate')
    """
    n_cases = 0
    reasoning_files_saved = 0

    if reports_dir:
        logger.info(f"Reasoning export enabled - saving to: {reports_dir}")

    with open(output_file, "w") as fout:
        for chunk in chunked(iter_records(input_file), chunk_size):
            matches = {}
            if ontology_matching:
                texts = [d for record in chunk for d in collect_diagnoses(record)]
                matches = matcher.match_many(texts, top_n=top_n)

            for record in chunk:
                record = process_record(
                    record,
                    matches,
                    ontology_matching,
                    diagnosis_to_groups,
                )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Save reasoning files if reports_dir is provided and matcher is in generate mode
                if reports_dir and matcher and matcher.classifier_mode == "generate":
                    files_saved = _save_reasoning_files(
                        record,
                        matcher,
                        reports_dir,
                    )
                    reasoning_files_saved += files_saved

            n_cases += len(chunk)
            logger.info(f"Processed {n_cases} cases... ({reasoning_files_saved} reasoning files saved)")

    if reports_dir:
        logger.info(f"Completed - Total reasoning files saved: {reasoning_files_saved}")


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

    ontology_matching = config.get("ontology_matching", False)

    date_str = run_name.split("_")[0] + "_" + run_name.split("_")[1]
    input_file = f"outputs/{run_name}/predictions_{date_str}.jsonl"
    output_file = f"outputs/{run_name}/ontology_{date_str}.jsonl"
    reports_dir = f"outputs/{run_name}/reports"

    ontology_list, diagnosis_to_groups = load_ontology(
        config["diagnosis_dictionary"]
    )

    m = config["matching"]
    matcher = None
    if ontology_matching:
        logger.info("Initializing ontology matcher")
        use_classifier = m.get("use_classifier", False)
        matcher = OntologyMatcher(
            ontology_list=ontology_list,
            embedding_model=config["models"]["embedding_model"],
            reranker_model=config["models"]["reranker_model"],
            retrieval_k=m["retrieval_k"],
            acceptance_threshold=m["acceptance_threshold"],
            use_prefilter=m["use_prefilter"],
            diagnosis_to_groups=diagnosis_to_groups,
            classifier_model=config["models"].get("classifier_model") if use_classifier else None,
            use_classifier=use_classifier,
            classifier_mode=m.get("classifier_mode", "logits"),
            classifier_batch_size=m.get("classifier_batch_size", 16),
            classifier_max_options=m.get("classifier_max_options", 10),
            classifier_min_prob=m.get("classifier_min_prob", 0.0),
            classifier_max_new_tokens=m.get("classifier_max_new_tokens", 384),
            encode_batch_size=m.get("encode_batch_size", 64),
        )

    # Create reports directory if reasoning export is enabled
    if ontology_matching and m.get("classifier_mode") == "generate":
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Reports directory created at: %s", reports_dir)

    process_json(
        input_file,
        output_file,
        matcher,
        top_n=m["top_n"],
        ontology_matching=ontology_matching,
        diagnosis_to_groups=diagnosis_to_groups,
        chunk_size=m.get("chunk_size", 256),
        reports_dir=reports_dir if ontology_matching and m.get("classifier_mode") == "generate" else None,
    )