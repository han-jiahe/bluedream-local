"""
BlueDream Local v3 — 配置
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # BlueDream_basic/

# 各模块路径
DETECTOR_DIR = PROJECT_ROOT / "bExamples_detect" / "soccer"
DVI_DIR = PROJECT_ROOT / "dVI_statistics"
MATCH_DIR = PROJECT_ROOT / "cMatchdata_clean"
VIS_DIR = PROJECT_ROOT / "eVisual&Analysis"

# 存储
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 分析参数
FRAMES_PER_SEGMENT = 120
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts", ".csv"}
