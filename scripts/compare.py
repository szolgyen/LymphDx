import logging
from pathlib import Path
from datetime import datetime

from evaluation.compare import main as compare_main
from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("outputs/compare") / f"compare_{timestamp}.log"
        configure_logging("INFO", log_file=str(log_file))
        logger.info("Starting comparison")
        compare_main()
        logger.info("Comparison completed successfully")
        return 0
    except Exception as exc:
        logger.exception("Evaluation failed: %s", exc)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
