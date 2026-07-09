"""
In-memory background task registry for the standalone Dedup Study app.

The long-running operations (dedup analysis, batch AI audit) run in daemon
threads; the UI polls /api/tasks for progress. Single-process, single-user
tooling -- no Celery/Redis needed. Task state does not survive a server
restart (the durable outputs -- snapshots, audits -- are in SQLite).
"""

import itertools
import logging
import threading
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)


class Task:
    def __init__(self, task_id: int, task_type: str, name: str, series_id: int = None):
        self.id = task_id
        self.type = task_type
        self.name = name
        self.series_id = series_id
        self.status = "PENDING"  # PENDING | RUNNING | COMPLETED | FAILED | CANCELLED
        self.total = 0
        self.processed = 0
        self.error = None
        self.result = None
        self.created_at = datetime.utcnow().isoformat()
        self._cancel = threading.Event()

    # --- called from inside the worker ---
    def set_total(self, n: int):
        self.total = n

    def tick(self, n: int = 1):
        self.processed += n

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "name": self.name,
            "series_id": self.series_id, "status": self.status,
            "total": self.total, "processed": self.processed,
            "error": self.error, "result": self.result,
            "created_at": self.created_at,
        }


class TaskRegistry:
    def __init__(self):
        self._tasks = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def start(self, task_type: str, name: str, fn, series_id: int = None) -> Task:
        """fn(task) runs in a daemon thread; its return value becomes
        task.result. Raising marks the task FAILED with the message."""
        with self._lock:
            task = Task(next(self._ids), task_type, name, series_id=series_id)
            self._tasks[task.id] = task

        def runner():
            task.status = "RUNNING"
            try:
                task.result = fn(task)
                task.status = "CANCELLED" if task.cancelled else "COMPLETED"
            except Exception as e:
                logger.error(f"Task {task.id} ({task.type}) failed: {e}\n{traceback.format_exc()}")
                task.status = "FAILED"
                task.error = str(e)

        threading.Thread(target=runner, daemon=True).start()
        return task

    def get(self, task_id: int) -> Task:
        with self._lock:
            return self._tasks.get(task_id)

    def cancel(self, task_id: int) -> bool:
        task = self.get(task_id)
        if task and task.status in ("PENDING", "RUNNING"):
            task._cancel.set()
            return True
        return False

    def active_for_series(self, series_id: int) -> Task:
        with self._lock:
            for task in self._tasks.values():
                if task.series_id == series_id and task.status in ("PENDING", "RUNNING"):
                    return task
        return None

    def list(self, active_only: bool = False) -> list:
        with self._lock:
            tasks = sorted(self._tasks.values(), key=lambda t: t.id, reverse=True)
        if active_only:
            tasks = [t for t in tasks if t.status in ("PENDING", "RUNNING")]
        return [t.to_dict() for t in tasks]


registry = TaskRegistry()
