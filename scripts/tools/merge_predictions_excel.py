import logging
from pathlib import Path
from datetime import datetime

from postprocessing.excel_merge import main as merge_main
from utils.logging_config import configure_logging
from postprocessing.excel_merge import load_config

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        config = load_config("configs/excel_merge.yaml")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = (
            Path(Path(config["predictions_jsonl"]).parent) / f"merge_{timestamp}.log"
        )
        configure_logging("INFO", log_file=str(log_file))
        logger.info("Starting predictions merge")
        result = merge_main()
        logger.info("Predictions merge completed successfully")
        return result
    except Exception as exc:
        logger.exception("Predictions merge failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
