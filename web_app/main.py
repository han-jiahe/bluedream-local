"""
BlueDream Local v3 — 一键启动
用法: python main.py
"""

import sys
import webbrowser
from pathlib import Path

# 确保可以导入原项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bExamples_detect" / "soccer"))
sys.path.insert(0, str(PROJECT_ROOT / "dVI_statistics"))
sys.path.insert(0, str(PROJECT_ROOT / "cMatchdata_clean"))

import uvicorn

if __name__ == "__main__":
    print(f"""
    BlueDream Local v3
    个人足球视频分析 Web 原型
    http://127.0.0.1:8000
    """)
    webbrowser.open("http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
