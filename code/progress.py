"""进度记录小工具：把当前分析进度写到 ../outputs/ml_progress.json，供本地看板轮询显示。"""
import json
import time
from pathlib import Path

PROGRESS_PATH = Path(__file__).resolve().parent.parent / "outputs" / "ml_progress.json"

class Progress:
    def __init__(self, step_names):
        self.steps = [{"name": n, "status": "pending", "detail": ""} for n in step_names]
        self.status = "running"
        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.current = -1
        self._write()

    def start_step(self, idx, detail=""):
        self.current = idx
        self.steps[idx]["status"] = "running"
        self.steps[idx]["detail"] = detail
        self._write()

    def done_step(self, idx, detail=""):
        self.steps[idx]["status"] = "done"
        if detail:
            self.steps[idx]["detail"] = detail
        self._write()

    def fail(self, idx, err):
        self.steps[idx]["status"] = "error"
        self.steps[idx]["detail"] = str(err)
        self.status = "error"
        self._write()

    def finish(self):
        self.status = "done"
        self._write()

    def _write(self):
        payload = {
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_step_index": self.current,
            "total_steps": len(self.steps),
            "steps": self.steps,
        }
        tmp = PROGRESS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(PROGRESS_PATH)
