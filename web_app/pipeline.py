"""
BlueDream Local v3 — 本地分析管线
封装原项目的 CSV转换 + VI计算 + 可视化
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# 确保能导入原项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dVI_statistics"))
sys.path.insert(0, str(PROJECT_ROOT / "cMatchdata_clean"))

from config import OUTPUT_DIR, PITCH_LENGTH, PITCH_WIDTH


def csv_to_samples(csv_path: Path, output_path: Optional[Path] = None) -> Path:
    """将检测追踪 CSV 转换为 samples.json"""
    from csv2json2 import build_samples

    if output_path is None:
        output_path = csv_path.parent / "samples.json"

    # Read CSV rows
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Default field corners (based on standard pitch dimensions)
    field_corners = [
        [192, 108], [1728, 108], [1728, 972], [192, 972],
    ]

    samples = build_samples(rows=rows, fps=25.0, field_corners=field_corners)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    return output_path


def compute_vi_distribution(
    samples_json: Path,
    output_path: Optional[Path] = None,
    window_size: int = 40,
) -> Path:
    """计算 VI 分布，返回 JSON 路径"""
    from vi_analysis import compute_vi_distribution as _compute

    if output_path is None:
        output_path = samples_json.parent / "VI_distribution.json"

    _compute(
        input_json=str(samples_json),
        output_json=str(output_path),
        window_size=window_size,
    )
    return output_path


def compute_voronoi_vi(csv_path: Path, output_path: Optional[Path] = None) -> Path:
    """Voronoi 空间 VI 分析"""
    from voronoi_vi import compute_voronoi_vi_from_csv

    if output_path is None:
        output_path = csv_path.parent / "voronoi_vi.json"

    compute_voronoi_vi_from_csv(
        csv_path=str(csv_path),
        output_json_path=str(output_path),
        include_referee=False,
    )
    return output_path


def generate_vi_chart(vi_json: Path, output_path: Optional[Path] = None) -> Path:
    """VI 分布时序图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(vi_json, "r", encoding="utf-8") as f:
        vi_data = json.load(f)

    times = [d["window_time"] for d in vi_data]
    vi_vals = [d["average_vi"] for d in vi_data]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, vi_vals, color="#FF1493", linewidth=2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Average VI")
    ax.set_title("Information Flow (VI) Over Time")
    ax.grid(True, alpha=0.3)

    if output_path is None:
        output_path = vi_json.parent / "vi_distribution_chart.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_player_ranking(voronoi_json: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """球员 VI 排名柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(voronoi_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    player_vi_sum: Dict[str, float] = {}
    player_vi_count: Dict[str, int] = {}
    for fn, fdata in data.get("frames", {}).items():
        for pid, vi_val in fdata.get("player_vi", {}).items():
            player_vi_sum[pid] = player_vi_sum.get(pid, 0.0) + vi_val
            player_vi_count[pid] = player_vi_count.get(pid, 0) + 1

    if not player_vi_sum:
        return None

    player_avg = sorted(
        [(f"P{pid}", player_vi_sum[pid] / player_vi_count[pid])
         for pid in player_vi_sum],
        key=lambda x: -x[1],
    )[:15]

    pids, vavgs = zip(*player_avg)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(pids)))
    ax.barh(list(reversed(pids)), list(reversed(vavgs)), color=list(reversed(colors)))
    ax.set_xlabel("Average Voronoi VI")
    ax.set_title("Player Ranking by Voronoi Influence (Top 15)")
    ax.invert_yaxis()

    if output_path is None:
        output_path = voronoi_json.parent / "player_vi_ranking.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_heatmap(
    csv_path: Path,
    player_id: int,
    output_path: Optional[Path] = None,
) -> Path:
    """球员热力图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("球员ID", -1)) == player_id:
                x = float(row.get("真实X(米)", 0))
                y = float(row.get("真实Y(米)", 0))
                positions.append((x, y))

    if not positions:
        positions = [(0, 0)]

    xs, ys = zip(*positions)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.hist2d(xs, ys, bins=(105, 68), range=[[0, 105], [0, 68]], cmap="hot", cmin=1)
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(0, PITCH_WIDTH)
    ax.set_title(f"Player {player_id} Heatmap")
    ax.set_aspect("equal")

    if output_path is None:
        output_path = csv_path.parent / f"heatmap_p{player_id}.png"
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def split_video(video_path: Path, output_dir: Path, frames_per_segment: int = 120) -> List[Path]:
    """将视频按帧数切分为片段"""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    segments = []
    seg_idx = 0
    frame_count = 0
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_segment == 0:
            if writer is not None:
                writer.release()
            seg_path = output_dir / f"segment_{seg_idx:04d}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(seg_path), fourcc, fps, (width, height))
            segments.append(seg_path)
            seg_idx += 1

        writer.write(frame)
        frame_count += 1

    if writer is not None:
        writer.release()
    cap.release()

    return segments


def merge_csvs(csv_paths: List[Path], output_path: Path) -> Path:
    """合并多个检测 CSV"""
    if len(csv_paths) == 1:
        import shutil
        shutil.copy(csv_paths[0], output_path)
        return output_path

    all_rows = []
    header = None
    global_frame_offset = 0
    last_max_frame = 0

    for i, cp in enumerate(csv_paths):
        with open(cp, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            rows = list(reader)

        seg_max_frame = 0
        for row in rows:
            fn = int(row["帧号"])
            if fn > seg_max_frame:
                seg_max_frame = fn

        offset = global_frame_offset
        for row in rows:
            row["帧号"] = str(int(row["帧号"]) + offset)
        all_rows.extend(rows)
        global_frame_offset += seg_max_frame + 1

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    return output_path
