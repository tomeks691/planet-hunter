import logging
import threading
import time
from planet_hunter import db
from planet_hunter.models import QueueSource
from planet_hunter.config import (
    SCANNER_INTERVAL,
    SCANNER_BATCH_SIZE,
    PRIORITY_AUTO,
    PAUSE_AUTO_SCANNER_WHEN_ML_BACKLOG,
)
from planet_hunter.scanner.tic_catalog import find_fresh_targets

log = logging.getLogger(__name__)


class AutoScanner:
    """Background thread that periodically adds TIC IDs to the queue."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Auto-scanner started (interval=%ds, batch=%d)",
                 SCANNER_INTERVAL, SCANNER_BATCH_SIZE)

    def stop(self):
        self._stop_event.set()
        log.info("Auto-scanner stop requested")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def toggle(self) -> bool:
        """Toggle scanner on/off. Returns new state."""
        if self.running:
            self.stop()
            return False
        else:
            self.start()
            return True

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                if PAUSE_AUTO_SCANNER_WHEN_ML_BACKLOG:
                    ml_active = db.count_active_by_source(QueueSource.ML_TRAINING)
                    if ml_active > 0:
                        log.info(
                            "Auto-scanner paused (ML_TRAINING backlog active: %d)",
                            ml_active,
                        )
                        tic_ids = []
                    else:
                        tic_ids = find_fresh_targets(SCANNER_BATCH_SIZE)
                else:
                    tic_ids = find_fresh_targets(SCANNER_BATCH_SIZE)

                for tic_id in tic_ids:
                    db.enqueue(tic_id, QueueSource.AUTO, PRIORITY_AUTO)
                    log.info("Auto-scanner queued TIC %d", tic_id)
            except Exception as e:
                log.error("Auto-scanner error: %s", e)

            # Wait for interval, checking stop event every second
            for _ in range(SCANNER_INTERVAL):
                if self._stop_event.is_set():
                    break
                time.sleep(1)
