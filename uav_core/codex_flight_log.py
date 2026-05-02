"""
Codex 飞行复盘日志。

该日志面向飞行后诊断，不替代控制台实时日志。主链路只把关键状态低频写入
JSONL 文件；任何写入异常都只打印 warning，不影响飞控控制流程。
"""

import json
import math
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


def _git_short_sha(repo_root):
    """读取当前 Git commit 短哈希；失败时返回 nogit。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=1.0,
        )
    except Exception:
        return "nogit"

    sha = result.stdout.strip()
    return sha if sha else "nogit"


def _json_safe(value):
    """把 numpy/Path/dataclass 等对象转换成 JSON 可序列化值。"""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return str(value)


class CodexFlightLogger:
    """
    每次运行生成一个结构化 JSONL 飞行日志。

    记录格式固定为：
        t_wall: 本机时间 ISO 字符串
        t_rel : 相对 logger 启动时间，单位 s
        kind  : session_config / event / sample / session_end
        frame : 主循环帧号；无帧号时为 None
        data  : 具体数据字典
    """

    def __init__(
        self,
        *,
        enabled=True,
        log_dir="logs",
        sample_interval_s=0.2,
        event_min_interval_s=0.5,
        flush_interval_s=1.0,
        repo_root=None,
        session_prefix="flight",
    ):
        self.enabled = bool(enabled)
        self.sample_interval_s = float(sample_interval_s)
        self.event_min_interval_s = float(event_min_interval_s)
        self.flush_interval_s = float(flush_interval_s)
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
        self.git_sha = _git_short_sha(self.repo_root)
        self.start_time = time.time()
        self.path = None
        self._file = None
        self._closed = False
        self._last_flush_time = self.start_time
        self._last_sample_time = 0.0
        self._last_event_times = {}
        self._last_event_values = {}

        if not self.enabled:
            return

        try:
            output_dir = Path(log_dir)
            if not output_dir.is_absolute():
                output_dir = self.repo_root / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            stamp = datetime.fromtimestamp(self.start_time).strftime("%Y%m%d_%H%M%S")
            self.path = output_dir / f"{session_prefix}_{stamp}_{self.git_sha}.jsonl"
            self._file = self.path.open("a", encoding="utf-8")
            logger.info(f"Codex 飞行日志已启用: {self.path}")
        except Exception as err:
            self.enabled = False
            logger.warning(f"Codex 飞行日志初始化失败，已自动关闭: {err}")

    def _record(self, kind, data=None, frame=None):
        now = time.time()
        return {
            "t_wall": datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
            "t_rel": round(now - self.start_time, 3),
            "kind": kind,
            "frame": frame,
            "data": _json_safe(data or {}),
        }

    def write(self, kind, data=None, frame=None, force_flush=False):
        """写入一条记录；失败时返回 False，不向外抛异常。"""
        if not self.enabled or self._file is None or self._closed:
            return False

        try:
            record = self._record(kind, data=data, frame=frame)
            self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

            now = time.time()
            if force_flush or now - self._last_flush_time >= self.flush_interval_s:
                self._file.flush()
                self._last_flush_time = now
            return True
        except Exception as err:
            logger.warning(f"Codex 飞行日志写入失败，已忽略本条记录: {err}")
            return False

    def record_session_config(self, data):
        return self.write("session_config", data=data, frame=None, force_flush=True)

    def record_event(self, name, data=None, frame=None, dedupe_key=None, value=None, force=False):
        """
        记录状态变化事件。

        dedupe_key/value 用于“只在值变化时记录”；没有 value 的事件会受
        event_min_interval_s 限制，避免异常状态每帧刷日志。
        """
        if not self.enabled:
            return False

        event_key = dedupe_key or name
        now = time.time()

        if not force and value is not None:
            if self._last_event_values.get(event_key) == value:
                return False
        elif not force:
            last_time = self._last_event_times.get(event_key)
            if last_time is not None and now - last_time < self.event_min_interval_s:
                return False

        self._last_event_values[event_key] = value
        self._last_event_times[event_key] = now

        payload = {"event": name}
        if data:
            payload.update(data)
        return self.write("event", data=payload, frame=frame)

    def should_sample(self, now=None):
        if not self.enabled:
            return False
        if now is None:
            now = time.time()
        return now - self._last_sample_time >= self.sample_interval_s

    def record_sample(self, data, frame=None):
        self._last_sample_time = time.time()
        return self.write("sample", data=data, frame=frame)

    def close(self, data=None, frame=None):
        """写入 session_end 并关闭文件；可重复调用。"""
        if self._closed:
            return

        payload = dict(data or {})
        payload.setdefault("duration_s", round(time.time() - self.start_time, 3))
        self.write("session_end", data=payload, frame=frame, force_flush=True)

        if self._file is not None:
            try:
                self._file.close()
            except Exception as err:
                logger.warning(f"Codex 飞行日志关闭失败，已忽略: {err}")
        self._closed = True
