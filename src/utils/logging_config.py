from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Gibt einen Logger zurück. Stellt sicher, dass basicConfig einmal gesetzt wurde,
    falls main() nicht bereits eine Logging-Konfiguration vorgenommen hat.
    """
    # Falls noch kein Handler existiert, basicConfig setzen
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    return logging.getLogger(name)
