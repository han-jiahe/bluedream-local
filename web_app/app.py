"""
BlueDream Local v3 — Web 应用
"""

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
    ALLOWED_EXTENSIONS, FRAMES_PER_SEGMENT,
)
from pipeline import (
    split_video, merge_csvs,
    compute_voronoi_vi, compute_vi_distribution,
    generate_vi_chart, generate_player_ranking, generate_heatmap,
    csv_to_samples,
)

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
    """上传视频并自动切分片段"""
    if not file.filename:
        return JSONResponse({"error": "No file"}, status_code=400)

    ext = Path(file.filename).suffix
    if ext.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts"}:
        return JSONResponse({"error": "不支持格式: " + ext}, status_code=400)

    video_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{video_id}{ext}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f, length=8 * 1024 * 1024)

    # Auto-split
    seg_dir = OUTPUT_DIR / video_id / "segments"
    segments = split_video(save_path, seg_dir, FRAMES_PER_SEGMENT)

    return JSONResponse({
        "video_id": video_id,
        "original_name": file.filename,
        "size_mb": round(save_path.stat().st_size / 1e6, 1),
        "segments": len(segments),
        "seg_dir": str(seg_dir),
        "hint": f"将 {seg_dir} 中的 {len(segments)} 个片段传到 ModelArts Notebook, "
                f"运行: cd bExamples_detect/soccer && python main.py --mode RADAR --source_video_path segment_0000.mp4 ... "
                f"然后把生成的 CSV 文件上传回来",
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
):
    """运行 VI 分析，显示结果"""
    if task_id:
        task_dir = OUTPUT_DIR / task_id
    elif video_id:
        task_dir = OUTPUT_DIR / video_id
    else:
        return HTMLResponse("<h3>缺少参数</h3>", status_code=400)

    if not task_dir.exists():
        return HTMLResponse(f"<h3>任务目录不存在: {task_dir}</h3>", status_code=404)

    # Find CSV files
    csv_files = sorted(task_dir.glob("*.csv"))
    if not csv_files:
        return HTMLResponse("<h3>未找到 CSV 文件，请先上传检测结果</h3>", status_code=400)

    # Merge if multiple
    merged_csv = task_dir / "merged_coords.csv"
    if len(csv_files) > 1:
        merge_csvs(csv_files, merged_csv)
    else:
        merged_csv = csv_files[0]

    # VI Analysis
    try:
        voronoi_json = compute_voronoi_vi(merged_csv)
        ranking_chart = generate_player_ranking(voronoi_json)
        vi_chart_path = None

        # Try clustering VI if samples.json available
        samples_json = task_dir / "samples.json"
        try:
            csv_to_samples(merged_csv, samples_json)
            vi_json = compute_vi_distribution(samples_json)
            vi_chart_path = generate_vi_chart(vi_json)
        except Exception:
            vi_chart_path = None

        charts = []
        if vi_chart_path:
            charts.append(("VI 分布图", vi_chart_path.name))
        if ranking_chart:
            charts.append(("球员 VI 排名", ranking_chart.name))

        return templates.TemplateResponse("results.html", {
            "request": request,
            "task_id": task_dir.name,
            "charts": charts,
            "csv_count": len(csv_files),
            "merged_csv": merged_csv.name,
        })

    except Exception as e:
        return HTMLResponse(
            f"<h3>分析失败</h3><pre>{e}</pre>",
            status_code=500,
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

    return templates.TemplateResponse("results.html", {
        "request": request,
        "task_id": task_id,
        "charts": [("图表", p.name) for p in pngs],
        "csv_count": len(csvs),
        "csvs": [c.name for c in csvs],
        "jsons": [j.name for j in jsons],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "app": "BlueDream Local v3"}
