import argparse
import logging
from pathlib import Path
from datetime import datetime

from postprocessing.ontology import main as ontology_main
from evaluation.evaluate import main as evaluation_main
from utils.logging_config import configure_logging
from evaluation.evaluate import io_utils as eval_io_utils

CONFIG = eval_io_utils.load_config("configs/evaluation.yaml")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ontology matching and evaluation pipeline"
    )
    parser.add_argument(
        "run_name",
        help="Name of the run (e.g., '20260101_120000').",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"outputs/{args.run_name}") / f"evaluate_{timestamp}.log"
    configure_logging("INFO", log_file=str(log_file))
    logger.info("Starting evaluation for run: %s", args.run_name)

    # Check if ontology file exists; if not, run ontology matching
    if not Path(f"outputs/{args.run_name}/ontology_{args.run_name}.jsonl").exists():
        try:
            ontology_main(run_name=args.run_name, config=CONFIG)
        except Exception as exc:
            logger.exception("Ontology matching failed: %s", exc)

    else:
        logger.warning(
            "Using existing ontology file: %s",
            f"outputs/{args.run_name}/ontology_{args.run_name}.jsonl",
        )

    # Run evaluation process
    try:
        evaluation_main(run_name=args.run_name)
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    raise SystemExit(main())
