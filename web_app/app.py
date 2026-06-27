"""
BlueDream Local v3 — Web 应用
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from config import (
    BASE_DIR, UPLOAD_DIR, OUTPUT_DIR,
    ALLOWED_EXTENSIONS,
)
from pipeline import (
    merge_csvs,
    compute_voronoi_vi, compute_vi_distribution,
    generate_vi_chart, generate_player_ranking, generate_heatmap,
    generate_player_timeseries, generate_vi_density,
    generate_player_with_team_vi, generate_voronoi_frame,
    generate_multi_player_trajectory, generate_ball_trajectory,
    _load_team_map,
    csv_to_samples,
)


# ── 分析类型 ──────────────────────────────────────────

ANALYSIS_TYPES = {
    "full": "完整分析 (VI分布+排名+密度+Top3热力图)",
    "player": "单球员分析 (VI+球队对比+热力图)",
    "trajectory": "轨迹可视化 (多球员+可选足球)",
    "voronoi": "Voronoi 多边形查看器 (按百分比选帧)",
    "vi_only": "仅 VI 分布 (时序+密度+排名)",
    "heatmap": "仅热力图",
}

app = FastAPI(title="BlueDream Local v3")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/results-files", StaticFiles(directory=str(OUTPUT_DIR)), name="results_files")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(BASE_DIR / "templates" / "upload.html"))


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """上传整段视频（不切分，保证球员 ID 一致性）"""
    if not file.filename:
        return JSONResponse({"error": "No file"}, status_code=400)

    ext = Path(file.filename).suffix
    if ext.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}:
        return JSONResponse({"error": "不支持格式: " + ext}, status_code=400)

    video_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{video_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    return JSONResponse({
        "video_id": video_id,
        "original_name": file.filename,
        "size_mb": round(save_path.stat().st_size / 1e6, 1),
        "hint": "将视频传至 ModelArts Notebook，运行 RADAR 检测，下载生成的 CSV 后上传回来",
    })


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), video_id: str = Form("")):
    """上传 ModelArts 检测结果 CSV"""
    task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    save_path = task_dir / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    # If multiple CSV uploads, track the task
    csv_count = len(list(task_dir.glob("*.csv")))

    return JSONResponse({
        "task_id": task_id,
        "csv_file": file.filename,
        "csv_count": csv_count,
    })


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    video_id: str = Form(""),
    task_id: str = Form(""),
    analysis_type: str = Form("full"),
    player_id: str = Form(""),
    include_ball: str = Form("0"),
):
    """运行 VI 分析"""
    if task_id:
        task_dir = OUTPUT_DIR / task_id
    elif video_id:
        task_dir = OUTPUT_DIR / video_id
    else:
        return HTMLResponse("<h3>缺少参数</h3>", status_code=400)

    if not task_dir.exists():
        return HTMLResponse(f"<h3>任务目录不存在: {task_dir}</h3>", status_code=404)

    csv_files = sorted(task_dir.glob("*.csv"))
    if not csv_files:
        return HTMLResponse("<h3>未找到 CSV 文件</h3>", status_code=400)

    merged_csv = task_dir / "merged_coords.csv"
    if len(csv_files) > 1:
        merge_csvs(csv_files, merged_csv)
    else:
        merged_csv = csv_files[0]

    try:
        charts = []

        # ── Voronoi VI (所有分析类型都需要) ──
        voronoi_json = compute_voronoi_vi(merged_csv)

        # ── Clustering VI ──
        samples_json = task_dir / "samples.json"
        vi_json = None
        try:
            csv_to_samples(merged_csv, samples_json)
            vi_json = compute_vi_distribution(samples_json)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARN] Clustering VI failed (non-fatal): {e}")

        # 加载球队映射
        team_map = _load_team_map(merged_csv)

        # ── 按分析类型生成图表 ──
        if analysis_type == "full":
            if vi_json:
                charts.append(("VI 分布时序", generate_vi_chart(vi_json).name))
                charts.append(("VI 概率密度", generate_vi_density(vi_json).name))
            rc = generate_player_ranking(voronoi_json)
            if rc:
                charts.append(("球员 VI 排名", rc.name))
            # Top 3 球员热力图（按总 VI 排序）
            with open(voronoi_json, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            player_total_vi = {}
            for fdata in vdata.get("frames", {}).values():
                for pid, vi_val in fdata.get("player_vi", {}).items():
                    player_total_vi[pid] = player_total_vi.get(pid, 0.0) + float(vi_val)
            top_players = sorted(player_total_vi, key=player_total_vi.get, reverse=True)[:3]
            for pid_str in top_players:
                pid = int(pid_str)
                team_tag = f"({team_map.get(pid, '?')})" if pid in team_map else ""
                charts.append((f"Player {pid} {team_tag} 热力图 (总VI: {player_total_vi[pid_str]:.3f})",
                              generate_heatmap(merged_csv, pid).name))

        elif analysis_type == "player" and player_id:
            pid = int(player_id.strip())
            team_tag = f"({team_map.get(pid, '?')})" if pid in team_map else ""
            charts.append((f"Player {pid} {team_tag} vs Team VI",
                          generate_player_with_team_vi(voronoi_json, merged_csv, pid).name))
            charts.append((f"Player {pid} {team_tag} 热力图",
                          generate_heatmap(merged_csv, pid).name))

        elif analysis_type == "vi_only":
            if vi_json:
                charts.append(("VI 分布时序", generate_vi_chart(vi_json).name))
                charts.append(("VI 概率密度", generate_vi_density(vi_json).name))
            rc = generate_player_ranking(voronoi_json)
            if rc:
                charts.append(("球员 VI 排名", rc.name))

        elif analysis_type == "voronoi":
            pct = float(player_id) if player_id else 50
            charts.append((f"Voronoi @ {pct:.0f}%",
                          generate_voronoi_frame(voronoi_json, merged_csv, pct).name))

        elif analysis_type == "trajectory" and player_id:
            ids = [int(x.strip()) for x in player_id.replace(",", " ").split() if x.strip()]
            show_ball = include_ball == "1"
            if ids:
                traj_path = generate_multi_player_trajectory(
                    merged_csv, ids, task_dir.name, include_ball=show_ball
                )
                charts.append(("球员轨迹", traj_path.name))
            if show_ball:
                ball_path = generate_ball_trajectory(merged_csv, task_dir.name)
                if ball_path:
                    charts.append(("足球轨迹", ball_path.name))

        elif analysis_type == "heatmap" and player_id:
            pid = int(player_id.strip())
            charts.append((f"Player {pid} 热力图",
                          generate_heatmap(merged_csv, pid).name))

        else:
            # default: same as full
            if vi_json:
                charts.append(("VI 分布时序", generate_vi_chart(vi_json).name))
            rc = generate_player_ranking(voronoi_json)
            if rc:
                charts.append(("球员 VI 排名", rc.name))

        # 纯 HTML 构建结果页
        task_name = task_dir.name
        n_files = len(csv_files)
        csv_name = merged_csv.name if hasattr(merged_csv, 'name') else str(merged_csv)
        chart_html = "".join(
            f'<div class="card"><h3>{label}</h3>'
            f'<img src="/results-files/{task_name}/{path}" class="chart-img"></div>'
            for label, path in charts
        )
        html = (
            '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Analysis Results</title>'
            '<link rel="stylesheet" href="/static/style.css"></head><body>'
            '<main class="container"><h2>Analysis Results</h2>'
            f'<p>Task: <code>{task_name}</code> | CSV: {csv_name} ({n_files} files)</p>'
            f'{chart_html}'
            '<a href="/" class="btn" style="margin-top:2rem">Back</a>'
            '</main></body></html>'
        )
        return HTMLResponse(content=html, status_code=200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            f"<h3>分析失败</h3><pre>{traceback.format_exc()}</pre>", status_code=500,
        )


@app.get("/results/{task_id}")
async def view_results(request: Request, task_id: str):
    """查看已有结果"""
    task_dir = OUTPUT_DIR / task_id
    if not task_dir.exists():
        return HTMLResponse("<h3>任务不存在</h3>", status_code=404)

    pngs = sorted(task_dir.glob("*.png"))
    csvs = sorted(task_dir.glob("*.csv"))
    jsons = sorted(task_dir.glob("*.json"))

    chart_html = "".join(
        f'<div class="card"><h3>Charts</h3>'
        f'<img src="/results-files/{task_id}/{p.name}" class="chart-img"></div>'
        for p in pngs
    )
    file_links = "".join(f'<li><a href="/results-files/{task_id}/{f}" download>{f}</a></li>'
                         for f in list(csvs) + [j.name for j in jsons])
    html = (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Results</title>'
        '<link rel="stylesheet" href="/static/style.css"></head><body>'
        '<main class="container">'
        f'<h2>Task: {task_id}</h2>'
        f'{chart_html}'
        f'<div class="card"><h3>Files</h3><ul>{file_links}</ul></div>'
        '<a href="/" class="btn">Back</a>'
        '</main></body></html>'
    )
    return HTMLResponse(content=html, status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "app": "BlueDream Local v3"}
