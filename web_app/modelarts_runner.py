"""
ModelArts Notebook 端批量检测脚本

用法（在 ModelArts Notebook 终端中运行）:
    cd /home/ma-user/work/BlueDream_basic/bExamples_detect/soccer
    python modelarts_runner.py --segments_dir /tmp/segments

此脚本遍历 segments_dir 中所有 segment_*.mp4 文件，
对每个运行 RADAR 检测，将 CSV 输出到同一目录。
"""

import argparse
import os
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "bExamples_detect" / "soccer"))
sys.path.insert(0, str(PROJECT_ROOT))


def run_radar_on_segment(video_path: str, csv_output: str, device: str = "cuda"):
    """对单个视频片段运行 RADAR 管线"""
    from ultralytics import YOLO
    import supervision as sv
    import csv
    import numpy as np
    from sports.common.team import TeamClassifier
    from sports.common.ball import BallTracker
    from sports.common.pitch import PitchRegistrator
    from sports.configs.soccer import SoccerPitchConfiguration
    from tqdm import tqdm

    CONFIG = SoccerPitchConfiguration()
    PARENT_DIR = str(Path(__file__).resolve().parent)
    PLAYER_MODEL = os.path.join(PARENT_DIR, "data", "football-player-detection.pt")
    PITCH_MODEL = os.path.join(PARENT_DIR, "data", "football-pitch-detection.pt")
    BALL_MODEL = os.path.join(PARENT_DIR, "data", "football-ball-detection.pt")

    print(f"Loading models on {device}...")
    player_model = YOLO(PLAYER_MODEL).to(device)
    pitch_model = YOLO(PITCH_MODEL).to(device)
    ball_model = YOLO(BALL_MODEL).to(device) if os.path.exists(BALL_MODEL) else None

    # Phase 1: Collect crops for team classification
    frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=60)
    crops = []
    for frame in tqdm(frame_gen, desc="Collecting crops"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == 2]  # class 2 = player
        for xyxy in players.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    # Phase 2: Fit team classifier
    print(f"Fitting team classifier on {len(crops)} crops...")
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Phase 3: Full pipeline
    print("Running detection + tracking + pitch registration...")
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker() if ball_model else None
    registrator = PitchRegistrator(CONFIG)
    frame_gen = sv.get_video_frames_generator(source_path=video_path)
    frame_number = 0

    Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["帧号", "球员ID", "画面X", "画面Y", "雷达X", "雷达Y", "真实X(米)", "真实Y(米)", "队伍"])

    for frame in tqdm(frame_gen, desc=f"Processing {Path(video_path).name}"):
        frame_number += 1
        # Pitch detect
        pitch_result = pitch_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

        # Player detect
        player_result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        # Ball detect
        if ball_model:
            ball_result = ball_model(frame, verbose=False)[0]
            ball_det = sv.Detections.from_ultralytics(ball_result)
            ball_det = ball_tracker.update(ball_det)

        players = detections[detections.class_id == 2]
        if len(players) == 0:
            continue

        # Team classification
        player_crops = []
        for xyxy in players.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            player_crops.append(crop if crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8))

        if player_crops:
            try:
                team_ids = team_classifier.predict(player_crops)
            except Exception:
                team_ids = [0] * len(players)

            # Pitch registration
            registrator.register(keypoints, frame.shape[:2])
            tracked_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_xy = registrator.transform(tracked_xy) if registrator.is_registered else None

            with open(csv_output, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                for i, tid in enumerate(players.tracker_id):
                    px, py = tracked_xy[i]
                    rpx, rpy = px, py  # fallback
                    rx, ry = 0.0, 0.0
                    if pitch_xy is not None and i < len(pitch_xy):
                        rpx, rpy = pitch_xy[i]
                        rx = (rpx / frame.shape[1]) * 105.0
                        ry = (ry / frame.shape[0]) * 68.0
                    writer.writerow([frame_number, int(tid), round(px, 1), round(py, 1),
                                     round(rpx, 1), round(rpy, 1), round(rx, 2), round(ry, 2),
                                     int(team_ids[i])])

    print(f"Done: {video_path} -> {csv_output}")


def main():
    parser = argparse.ArgumentParser(description="ModelArts 批量检测")
    parser.add_argument("--segments_dir", required=True, help="视频片段目录")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    args = parser.parse_args()

    seg_dir = Path(args.segments_dir)
    segments = sorted(seg_dir.glob("segment_*.mp4"))

    if not segments:
        print(f"未找到 segment_*.mp4 文件于 {seg_dir}")
        return

    print(f"找到 {len(segments)} 个片段")

    for seg in segments:
        csv_out = seg.parent / f"player_final_real_coords_{seg.stem}.csv"
        if csv_out.exists() and csv_out.stat().st_size > 100:
            print(f"  SKIP: {seg.name} (CSV exists)")
            continue
        run_radar_on_segment(str(seg), str(csv_out), device=args.device)

    print(f"\n全部完成! CSV 文件在: {seg_dir}")


if __name__ == "__main__":
    main()
