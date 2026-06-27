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
        samples_json_path=str(samples_json),
        output_json_path=str(output_path),
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
    h = ax.hist2d(xs, ys, bins=(105, 68), range=[[0, 105], [0, 68]], cmap="hot", cmin=1)
    cbar = plt.colorbar(h[3], ax=ax, label="Frame Count")
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


# ── 单球员 VI 时序图 ────────────────────────────────────

def generate_player_timeseries(voronoi_json: Path, player_id: int,
                                output_path: Optional[Path] = None) -> Path:
    """单球员 VI 时序图: 上=时序曲线+平滑, 下=分布直方图+KDE"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from scipy.ndimage import uniform_filter1d

    with open(voronoi_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    pid = str(player_id)
    frames_data = {}
    for fn, fdata in data.get("frames", {}).items():
        pvi = fdata.get("player_vi", {}).get(pid)
        if pvi is not None:
            frames_data[int(fn)] = float(pvi)
    if not frames_data:
        raise ValueError(f"Player {player_id} not found")
    sorted_fns = sorted(frames_data.keys())
    vi_vals = [frames_data[fn] for fn in sorted_fns]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(sorted_fns, vi_vals, alpha=0.4, color="#FF1493", lw=0.8, label="Frame VI")
    if len(vi_vals) >= 10:
        sm = uniform_filter1d(vi_vals, size=min(10, len(vi_vals)))
        ax1.plot(sorted_fns, sm, color="#FF1493", lw=2, label="Smoothed")
    mv = np.mean(vi_vals)
    ax1.axhline(mv, color="green", ls="--", lw=1, label=f"Mean: {mv:.4f}")
    ax1.set_xlabel("Frame"); ax1.set_ylabel("VI")
    ax1.set_title(f"Player {player_id} — VI Time Series"); ax1.legend(); ax1.grid(alpha=0.3)

    arr = np.array(vi_vals)
    ax2.hist(arr, bins=20, density=True, alpha=0.6, color="steelblue", edgecolor="black")
    try:
        kde = gaussian_kde(arr, bw_method="scott")
        xr = np.linspace(arr.min(), arr.max(), 200)
        ax2.plot(xr, kde(xr), "r-", lw=2, label="KDE")
    except Exception: pass
    ax2.axvline(np.mean(arr), color="green", ls="--", label=f"Mean: {np.mean(arr):.4f}")
    ax2.axvline(np.median(arr), color="orange", ls="--", label=f"Median: {np.median(arr):.4f}")
    ax2.set_xlabel("VI"); ax2.set_ylabel("Density")
    ax2.set_title(f"Player {player_id} — VI Distribution"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        output_path = voronoi_json.parent / f"player_{player_id}_vi_timeseries.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return output_path


# ── 全场 VI 密度分布 ────────────────────────────────────

def generate_vi_density(vi_json: Path, output_path: Optional[Path] = None) -> Path:
    """全场 VI 概率密度: 直方图 + KDE + 均值/中位数"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    with open(vi_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    vi_values = np.array([d["average_vi"] for d in data])
    vi_values = vi_values[vi_values >= 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vi_values, bins=20, density=True, alpha=0.6, color="steelblue",
            edgecolor="black", label="Histogram")
    try:
        kde = gaussian_kde(vi_values, bw_method="scott")
        ax.plot(np.linspace(vi_values.min(), vi_values.max(), 200),
                kde(np.linspace(vi_values.min(), vi_values.max(), 200)),
                "r-", lw=2, label="KDE")
    except Exception: pass
    ax.axvline(np.mean(vi_values), color="green", ls="--", label=f"Mean: {np.mean(vi_values):.3f}")
    ax.axvline(np.median(vi_values), color="orange", ls="--", label=f"Median: {np.median(vi_values):.3f}")
    ax.set_xlabel("Average VI"); ax.set_ylabel("Density")
    ax.set_title("VI Probability Density Distribution"); ax.legend(); ax.grid(alpha=0.6)
    plt.tight_layout()

    if output_path is None:
        output_path = vi_json.parent / "vi_distribution_density.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return output_path


# ── 轨迹可视化 (真实坐标 105×68m) ──────────────────────

_PLAYER_COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700", "#7B68EE",
                  "#00FA9A", "#FF8C00", "#1E90FF", "#DC143C", "#32CD32"]

_BALL_COLOR = "#FF0000"


def _draw_pitch(ax):
    """在 axes 上绘制 105×68m 标准足球场"""
    import matplotlib.patches as patches
    # 边线
    ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], c="black", lw=2)
    # 中线
    ax.plot([52.5, 52.5], [0, 68], c="black", lw=1.5)
    # 中圈
    ax.add_patch(patches.Arc((52.5, 34), 18.3, 18.3, angle=0, theta1=0, theta2=360,
                             edgecolor="black", lw=1.5, fill=False))
    # 禁区
    ax.plot([0, 16.5, 16.5, 0], [13.84, 13.84, 54.16, 54.16], c="black", lw=1.5)
    ax.plot([105, 88.5, 88.5, 105], [13.84, 13.84, 54.16, 54.16], c="black", lw=1.5)
    # 小禁区
    ax.plot([0, 5.5, 5.5, 0], [24.84, 24.84, 43.16, 43.16], c="black", lw=1.5)
    ax.plot([105, 99.5, 99.5, 105], [24.84, 24.84, 43.16, 43.16], c="black", lw=1.5)
    # 球门
    for gx, gy in [(0, 30.34), (105, 30.34)]:
        ax.add_patch(patches.Rectangle((gx - 0.5, gy - 2.67), 0.5, 5.34, fc="black", lw=1))
    ax.set_xlim(-2, 107); ax.set_ylim(-2, 70)
    ax.set_aspect("equal"); ax.axis("off")


def generate_multi_player_trajectory(
    csv_path: Path,
    player_ids: List[int],
    task_id: str,
    include_ball: bool = False,
    output_path: Optional[Path] = None,
) -> Path:
    """多球员轨迹 + 可选足球轨迹 (真实坐标 105×68m 球场)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 读 CSV
    player_positions: Dict[int, List[tuple]] = {pid: [] for pid in player_ids}
    ball_positions: List[tuple] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row.get("球员ID", -1))
            rx = float(row.get("真实X(米)", 0))
            ry = float(row.get("真实Y(米)", 0))
            if pid in player_positions:
                player_positions[pid].append((rx, ry))
            if include_ball and pid == -3:
                ball_positions.append((rx, ry))

    fig, ax = plt.subplots(figsize=(12, 7.8))
    _draw_pitch(ax)

    # 球员轨迹
    for i, pid in enumerate(player_ids):
        pts = player_positions.get(pid, [])
        if pts:
            xs, ys = zip(*pts)
            color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]
            ax.plot(xs, ys, color=color, lw=1.5, alpha=0.7, label=f"P{pid}")
            ax.scatter(xs[0], ys[0], color=color, s=60, marker="o", edgecolors="white", lw=0.5, zorder=5)
            ax.scatter(xs[-1], ys[-1], color=color, s=80, marker="s", edgecolors="white", lw=0.5, zorder=5)
    # 标记起点/终点图例
    ax.scatter([], [], color="gray", s=60, marker="o", edgecolors="white", lw=0.5, label="Start")
    ax.scatter([], [], color="gray", s=80, marker="s", edgecolors="white", lw=0.5, label="End")

    # 足球轨迹
    if include_ball and ball_positions:
        bxs, bys = zip(*ball_positions)
        ax.plot(bxs, bys, color=_BALL_COLOR, lw=2.5, alpha=0.6, linestyle="--", label="Ball")
        ax.scatter(bxs[0], bys[0], color=_BALL_COLOR, s=70, marker="D", edgecolors="white", lw=0.5, zorder=5)

    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"Player Trajectories{' + Ball' if include_ball else ''}", fontsize=12)
    plt.tight_layout()

    if output_path is None:
        output_path = csv_path.parent / "player_trajectories.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return output_path


def generate_ball_trajectory(
    csv_path: Path,
    task_id: str,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """足球轨迹 (真实坐标 105×68m)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row.get("球员ID", -1))
            if pid == -3:
                rx = float(row.get("真实X(米)", 0))
                ry = float(row.get("真实Y(米)", 0))
                positions.append((rx, ry))

    if not positions:
        return None

    fig, ax = plt.subplots(figsize=(12, 7.8))
    _draw_pitch(ax)
    xs, ys = zip(*positions)
    ax.plot(xs, ys, color=_BALL_COLOR, lw=2.5, alpha=0.6, label="Ball")
    ax.scatter(xs[0], ys[0], color="green", s=80, marker="D", edgecolors="white", lw=0.5, zorder=5, label="Start")
    ax.scatter(xs[-1], ys[-1], color="red", s=80, marker="s", edgecolors="white", lw=0.5, zorder=5, label="End")
    ax.legend(fontsize=9)
    ax.set_title("Football Trajectory", fontsize=12)
    plt.tight_layout()

    if output_path is None:
        output_path = csv_path.parent / "ball_trajectory.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return output_path


# ── 球员/球队 VI 对比 + ratio ──────────────────────────

def _load_team_map(csv_path: Path) -> Dict[int, str]:
    """从 CSV 读取 player_id -> team (home/away) 映射"""
    team_map = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row.get("球员ID", -1))
            team_val = int(row.get("队伍", -1))
            if pid > 0 and team_val in (0, 1):
                team_map[pid] = "home" if team_val == 0 else "away"
    return team_map


def generate_player_with_team_vi(
    voronoi_json: Path,
    csv_path: Path,
    player_id: int,
    output_path: Optional[Path] = None,
) -> Path:
    """单球员 VI + 球队均值对比 + ratio 柱状图"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(voronoi_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    team_map = _load_team_map(csv_path)
    player_team = team_map.get(player_id, None)

    pid = str(player_id)
    frames_data: Dict[int, float] = {}
    home_vi: Dict[int, List[float]] = defaultdict(list)
    away_vi: Dict[int, List[float]] = defaultdict(list)

    for fn_str, fdata in data.get("frames", {}).items():
        fn = int(fn_str)
        player_vi_map = fdata.get("player_vi", {})
        if pid in player_vi_map:
            frames_data[fn] = float(player_vi_map[pid])
        # 按球队分组
        for opid, vi_val in player_vi_map.items():
            opid_int = int(opid)
            team = team_map.get(opid_int)
            if team == "home":
                home_vi.setdefault(fn, []).append(float(vi_val))
            elif team == "away":
                away_vi.setdefault(fn, []).append(float(vi_val))

    if not frames_data:
        raise ValueError(f"Player {player_id} not found")

    sorted_fns = sorted(frames_data.keys())
    player_vals = [frames_data[fn] for fn in sorted_fns]
    home_avg = [np.mean(home_vi.get(fn, [0])) for fn in sorted_fns]
    away_avg = [np.mean(away_vi.get(fn, [0])) for fn in sorted_fns]

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # Left: timeseries
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sorted_fns, player_vals, color="#FF1493", lw=2, alpha=0.8, label=f"P{player_id}")
    ax1.plot(sorted_fns, home_avg, color="#1E90FF", lw=1.5, alpha=0.6, linestyle="--", label="Home avg")
    ax1.plot(sorted_fns, away_avg, color="#FF6347", lw=1.5, alpha=0.6, linestyle="--", label="Away avg")
    ax1.set_xlabel("Frame"); ax1.set_ylabel("VI")
    ax1.set_title(f"Player {player_id} VI vs Team Averages"); ax1.legend(); ax1.grid(alpha=0.3)

    # Right: ratio bar
    ax2 = fig.add_subplot(gs[1])
    player_mean = np.mean(player_vals)
    team_vals = []
    team_labels = []
    if home_avg:
        team_vals.append(np.mean(home_avg))
        team_labels.append("Home")
    if away_avg:
        team_vals.append(np.mean(away_avg))
        team_labels.append("Away")

    all_means = [player_mean] + team_vals
    all_labels = [f"P{player_id}"] + team_labels
    colors = ["#FF1493"] + ["#1E90FF", "#FF6347"][:len(team_vals)]
    bars = ax2.bar(all_labels, all_means, color=colors)
    ax2.set_ylabel("Mean VI")
    ax2.set_title("Player vs Team VI")

    # 标注 ratio
    for i, (label, val) in enumerate(zip(all_labels, all_means)):
        if i == 0:
            for j, tv in enumerate(team_vals):
                ratio = player_mean / tv if tv > 0 else 0
                team_name = {0: "Home", 1: "Away"}.get(j, "T")
                ax2.text(i, val, f'{val:.4f}\nvs {team_name}: {ratio:.2f}x',
                         ha="center", va="bottom", fontsize=9, color="#333")

    plt.tight_layout()

    if output_path is None:
        output_path = voronoi_json.parent / f"player_{player_id}_team_vi.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return output_path


# ── Voronoi 多边形查看器 ────────────────────────────────

def generate_voronoi_frame(
    voronoi_json: Path,
    csv_path: Path,
    percentage: float = 50,
    output_path: Optional[Path] = None,
) -> Path:
    """按百分比选择帧，可视化 Voronoi 多边形"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, Point

    with open(voronoi_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames_dict = data.get("frames", {})
    if not frames_dict:
        raise ValueError("No voronoi data")

    frame_keys = sorted(frames_dict.keys(), key=int)
    idx = max(0, min(len(frame_keys) - 1, int(len(frame_keys) * percentage / 100)))
    selected_fn = frame_keys[idx]
    fdata = frames_dict[selected_fn]

    # 读取球员位置
    positions = {}
    team_map = _load_team_map(csv_path)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row.get("帧号", -1))
            pid = int(row.get("球员ID", -1))
            if fn == int(selected_fn) and pid > 0 and pid in team_map:
                positions[pid] = (
                    float(row.get("真实X(米)", 0)),
                    float(row.get("真实Y(米)", 0)),
                )

    if len(positions) < 4:
        raise ValueError(f"Not enough players at frame {selected_fn}")

    # Voronoi 剖分
    pts = np.array(list(positions.values()))
    # 添加镜像点扩展边界
    extended = np.vstack([pts,
        pts + [105, 0], pts - [105, 0],
        pts + [0, 68], pts - [0, 68],
        pts + [105, 68], pts - [105, 68],
    ])
    vor = Voronoi(extended)

    fig, ax = plt.subplots(figsize=(12, 7.8))
    _draw_pitch(ax)

    pid_list = list(positions.keys())
    player_vi = fdata.get("player_vi", {})

    for i, pid in enumerate(pid_list):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or not region:
            continue
        poly = Polygon([vor.vertices[j] for j in region])
        # 裁剪到球场
        pitch = Polygon([(0, 0), (105, 0), (105, 68), (0, 68)])
        clipped = poly.intersection(pitch)
        if not clipped.is_empty:
            team = team_map.get(pid, "unknown")
            color = "#1E90FF" if team == "home" else "#FF6347"
            alpha = 0.3
            try:
                xs, ys = clipped.exterior.xy
                ax.fill(xs, ys, color=color, alpha=alpha, edgecolor="gray", lw=0.5)
            except Exception:
                pass

    # 球员点 + VI 标注
    for pid in pid_list:
        x, y = positions[pid]
        team = team_map.get(pid, "unknown")
        color = "#1E90FF" if team == "home" else "#FF6347"
        ax.plot(x, y, "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=0.5)
        vi_label = f"{float(player_vi.get(str(pid), 0)):.3f}"
        ax.annotate(f"P{pid}\n{vi_label}", (x, y), fontsize=6, ha="center", va="bottom",
                    color=color, alpha=0.9)

    pct_label = f"{percentage:.0f}%"
    ax.set_title(f"Voronoi Tessellation — Frame {selected_fn} ({pct_label} of match)")

    if output_path is None:
        output_path = voronoi_json.parent / f"voronoi_frame_{int(percentage)}.png"
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight"); plt.close(fig)
    return output_path
