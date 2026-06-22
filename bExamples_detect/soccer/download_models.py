"""
从华为云 OBS 下载 AI 模型文件到 data/ 目录

用法 (在 Notebook 上):
    python download_models.py
"""

import os
import sys
from pathlib import Path

# 模型列表
MODELS = [
    "football-player-detection.pt",
    "football-pitch-detection.pt",
    "football-ball-detection.pt",
]

# OBS 配置 (从环境变量读取)
BUCKET = os.getenv("OBS_BUCKET", "bluedream-models")
ENDPOINT = os.getenv("OBS_ENDPOINT", "obs.cn-north-4.myhuaweicloud.com")


def main():
    from obs import ObsClient

    ak = os.getenv("OBS_ACCESS_KEY", "")
    sk = os.getenv("OBS_SECRET_KEY", "")

    if not ak or not sk:
        print("请设置 OBS_ACCESS_KEY 和 OBS_SECRET_KEY 环境变量")
        sys.exit(1)

    # 目标目录: 脚本所在目录下的 data/
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    obs = ObsClient(access_key_id=ak, secret_access_key=sk, server=ENDPOINT)

    for model in MODELS:
        dest = data_dir / model
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"[SKIP] {model} 已存在 ({size_mb:.1f} MB)")
            continue

        obs_key = f"models/{model}"
        print(f"下载: {model} ...")
        try:
            resp = obs.getObject(BUCKET, obs_key, downloadPath=str(dest))
            if resp.status < 300:
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  完成: {model} ({size_mb:.1f} MB)")
            else:
                print(f"  失败: HTTP {resp.status}")
                if dest.exists():
                    dest.unlink()
        except Exception as e:
            print(f"  失败: {model} - {e}")
            if dest.exists():
                dest.unlink()

    obs.close()

    # 验证
    print("\n模型文件检查:")
    for model in MODELS:
        p = data_dir / model
        if p.exists():
            print(f"  [OK] {model} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"  [MISSING] {model}")


if __name__ == "__main__":
    main()
