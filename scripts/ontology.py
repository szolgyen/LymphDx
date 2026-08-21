import argparse
import logging
from pathlib import Path
from datetime import datetime

from postprocessing.ontology import main as ontology_main
from utils.logging_config import configure_logging
from evaluation.evaluate import io_utils as eval_io_utils

CONFIG = eval_io_utils.load_config("configs/evaluation.yaml")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ontology matching process")
    parser.add_argument(
        "run_name",
        help="Name of the run/experiment (e.g., '20230101_120000'). Used to construct input and output paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"outputs/{args.run_name}") / f"ontology_{timestamp}.log"
    configure_logging("INFO", log_file=str(log_file))
    logger.info("Starting ontology process")

    try:
        ontology_main(run_name=args.run_name, config=CONFIG)
    except Exception as exc:
        logger.exception("Ontology process failed: %s", exc)

    logger.info("Ontology process completed successfully")


if __name__ == "__main__":
    raise SystemExit(main())
