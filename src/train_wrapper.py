import logging
import subprocess
import sys
import time

LOGGER = logging.getLogger(__name__)


def run() -> int:
    """Execute the original training module and log elapsed time.

    Returns the exit code from the training module.
    """
    LOGGER.info("Model training started")
    start = time.time()
    # Run the original training script as a subprocess to preserve its CLI behaviour
    exit_code = subprocess.call([sys.executable, "-m", "src.train_model"])
    elapsed = time.time() - start
    LOGGER.info(f"Model training completed in {elapsed:.2f} seconds")
    return exit_code


if __name__ == "__main__":
    sys.exit(run())
