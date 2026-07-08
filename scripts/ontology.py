import logging
from pathlib import Path
from datetime import datetime

from postprocessing.ontology import main as ontology_main
from utils.logging_config import configure_logging
from postprocessing.ontology import load_config

logger = logging.getLogger(__name__)


def main() -> int:
    try:
        config = load_config("configs/ontology.yaml")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(Path(config["input_file"]).parent) / f"ontology_{timestamp}.log"
        configure_logging("INFO", log_file=str(log_file))
        logger.info("Starting ontology process")
        ontology_main()
        logger.info("Ontology process completed successfully")
        return 0
    except Exception as exc:
        logger.exception("Ontology process failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
