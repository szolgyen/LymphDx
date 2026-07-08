import logging
from pathlib import Path
from datetime import datetime

from evaluation.evaluate import main as eval_main
from utils.logging_config import configure_logging
from evaluation.evaluate import load_config

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        config = load_config("configs/evaluation.yaml")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = (
            Path(Path(config["input_jsonl_file"]).parent) / f"evaluate_{timestamp}.log"
        )
        configure_logging("INFO", log_file=str(log_file))
        logger.info("Starting evaluation")
        eval_main()
        logger.info("Evaluation completed successfully")
        return 0
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
