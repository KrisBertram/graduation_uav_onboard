"""
MAVLink 本地生成代码包。

`mavlink.py` 是生成代码，内部使用 `from mavcrc import x25crc`
导入同目录文件。工程入口重构后不再手动追加 mavlink 目录到 sys.path，
因此在包初始化时补上该目录，保持生成代码不变。
"""

import sys
from pathlib import Path


_MAVLINK_DIR = str(Path(__file__).resolve().parent)
if _MAVLINK_DIR not in sys.path:
    sys.path.insert(0, _MAVLINK_DIR)
