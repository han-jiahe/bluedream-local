#!/usr/bin/env python3
"""
GPU 批量并发检测器 — 多进程并行处理视频片段

用法:
    python batch_process.py --segments_dir /path/to/segments [--workers 3]

改进:
  - 每个 worker 独立加载模型 (避免跨进程 GPU tensor 共享问题)
  - 控制并发数, 充分利用 GPU 显存
  - 只输出 CSV, 不生成标注视频 (加速)
  - 自动跳过已有 CSV 结果的片段
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import Process, Queue, cpu_count
from pathlib import Path
from typing import List, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.team import TeamClassifier
from sports.common.pitch import PitchRegistrator
from sports.configs.soccer import SoccerPitchConfiguration

# ── Worker 进程 ───────────────────────────────────────────

class SegmentWorker(Process):
    """独立进程: 加载模型, 处理视频片段, 输出 CSV"""

    def __init__(self, task_queue, result_queue, device="cuda", gpu_id=0):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.device = device
        self.gpu_id = gpu_id

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        CONFIG = SoccerPitchConfiguration()
        PARENT_DIR = Path(__file__).resolve().parent
        PLAYER_MODEL = str(PARENT_DIR / "data" / "football-player-detection.pt")
        PITCH_MODEL = str(PARENT_DIR / "data" / "football-pitch-detection.pt")
        BALL_MODEL = str(PARENT_DIR / "data" / "football-ball-detection.pt")

        # 加载模型 (FP16)
        player_model = YOLO(PLAYER_MODEL).to(self.device)
        try:
            player_model.model.half()
        except Exception:
            pass
        pitch_model = YOLO(PITCH_MODEL).to(self.device)
        try:
            pitch_model.model.half()
        except Exception:
            pass

        # 球模型可选
        ball_model = None
        if Path(BALL_MODEL).exists():
            ball_model = YOLO(BALL_MODEL).to(self.device)
            try:
                ball_model.model.half()
            except Exception:
                pass

        while True:
            task = self.task_queue.get()
            if task is None:  # 停止信号
                break

            video_path, csv_output = task
            csv_path = Path(csv_output)
            if csv_path.exists() and csv_path.stat().st_size > 500:
                self.result_queue.put(
                    ("skip", str(video_path), str(csv_output))
                )
                continue

            try:
                self._process_segment(
                    player_model, pitch_model, ball_model,
                    CONFIG, video_path, csv_output,
                )
                self.result_queue.put(
                    ("ok", str(video_path), str(csv_output))
                )
            except Exception as e:
                self.result_queue.put(
                    ("error", str(video_path), str(e))
                )

    def _process_segment(
        self, player_model, pitch_model, ball_model,
        CONFIG, video_path: str, csv_output: str,
    ):
        # Phase 1: 收集 crops 训练球队分类器
        frame_gen = sv.get_video_frames_generator(
            source_path=video_path, stride=60
        )
        crops = []
        for frame in frame_gen:
            result = player_model(frame, imgsz=960, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            players = detections[detections.class_id == 2]
            for xyxy in players.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                c = frame[y1:y2, x1:x2]
                if c.size > 0:
                    crops.append(c)

        team_classifier = TeamClassifier(device=self.device)
        team_classifier.fit(crops)

        # Phase 2: 逐帧检测 + 追踪
        tracker = sv.ByteTrack(minimum_consecutive_frames=3)
        registrator = PitchRegistrator(CONFIG)
        frame_gen = sv.get_video_frames_generator(source_path=video_path)
        frame_number = 0

        Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["帧号","球员ID","画面X","画面Y","雷达X","雷达Y","真实X(米)","真实Y(米)","队伍"])

        for frame in frame_gen:
            frame_number += 1
            # Pitch (每 15 帧检测一次, 其余帧用缓存)
            if frame_number % 15 == 1 or not registrator.is_registered:
                presult = pitch_model(frame, verbose=False)[0]
                kpts = sv.KeyPoints.from_ultralytics(presult)
                registrator.register(kpts, frame.shape[:2])

            # Player
            player_result = player_model(frame, imgsz=960, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = tracker.update_with_detections(detections)

            players = detections[detections.class_id == 2]
            if len(players) == 0:
                continue

            # Team
            player_crops = []
            for xyxy in players.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                c = frame[y1:y2, x1:x2]
                player_crops.append(c if c.size > 0 else np.zeros((10,10,3), dtype=np.uint8))
            try:
                team_ids = team_classifier.predict(player_crops)
            except Exception:
                team_ids = [0] * len(players)

            # Pitch 坐标
            tracked_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_xy = registrator.transform(tracked_xy) if registrator.is_registered else None

            with open(csv_output, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                for i, tid in enumerate(players.tracker_id):
                    px, py = tracked_xy[i]
                    if pitch_xy is not None and i < len(pitch_xy):
                        rpx, rpy = pitch_xy[i]
                    else:
                        rpx, rpy = px, py
                    rx = (rpx / frame.shape[1]) * 105.0
                    ry = (rpy / frame.shape[0]) * 68.0
                    w.writerow([
                        frame_number,
                        int(tid) if tid is not None else i,
                        round(float(px), 1), round(float(py), 1),
                        round(float(rpx), 1), round(float(rpy), 1),
                        round(float(rx), 2), round(float(ry), 2),
                        int(team_ids[i]) if i < len(team_ids) else 0,
                    ])


# ── 主函数 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU 批量并发检测")
    parser.add_argument("--segments_dir", required=True, help="视频片段目录")
    parser.add_argument("--workers", type=int, default=2,
                        help="并发 worker 数 (默认 2, 32GB 显存建议 3-4)")
    parser.add_argument("--device", default="cuda", help="设备")
    parser.add_argument("--output_dir", default=None, help="CSV 输出目录 (默认 segments_dir)")
    parser.add_argument("--keep_video", action="store_true", help="保留标注视频输出 (较慢)")
    parser.add_argument("--pattern", default="segment_*.mp4",
                        help="视频文件名匹配模式 (默认: segment_*.mp4)")
    args = parser.parse_args()

    seg_dir = Path(args.segments_dir)
    if not seg_dir.exists():
        print(f"[ERROR] 目录不存在: {seg_dir}")
        sys.exit(1)

    videos = sorted(seg_dir.glob(args.pattern))
    if not videos:
        print(f"[ERROR] 未找到匹配 '{args.pattern}' 的视频文件于 {seg_dir}")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else seg_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 准备任务队列
    tasks = []
    for v in videos:
        csv_path = out_dir / f"player_final_real_coords_{v.stem}.csv"
        if csv_path.exists() and csv_path.stat().st_size > 500:
            print(f"[SKIP] {v.name} -> {csv_path.name}")
            continue
        tasks.append((str(v), str(csv_path)))

    if not tasks:
        print("所有片段已有结果, 无需处理。")
        return

    # 限制并发数
    n_workers = min(args.workers, len(tasks))
    print(f"\n待处理: {len(tasks)} 个片段 | 并发数: {n_workers} (GPU: {args.device})")
    print(f"输出目录: {out_dir}\n")

    task_queue: Queue = Queue()
    result_queue: Queue = Queue()

    # 启动 workers
    workers = []
    for i in range(n_workers):
        w = SegmentWorker(task_queue, result_queue, device=args.device, gpu_id=min(i, 0))
        w.start()
        workers.append(w)

    # 投放任务
    for t in tasks:
        task_queue.put(t)
    for _ in range(n_workers):
        task_queue.put(None)  # 停止信号

    # 收集结果
    t0 = time.time()
    completed = skipped = errors = 0
    for _ in range(len(tasks)):
        status, path, msg = result_queue.get()
        if status == "ok":
            completed += 1
            print(f"[OK] {Path(path).name} ({completed}/{len(tasks)})")
        elif status == "skip":
            skipped += 1
        elif status == "error":
            errors += 1
            print(f"[ERR] {Path(path).name}: {msg}")

    for w in workers:
        w.join()

    elapsed = time.time() - t0
    print(f"\n完成! {completed} 成功, {skipped} 跳过, {errors} 失败 | "
          f"耗时: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
