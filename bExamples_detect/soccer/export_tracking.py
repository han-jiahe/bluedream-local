import argparse
import csv
import os
from typing import Iterator
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# 配置路径
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_MODEL = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_MODEL = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')

def export_player_positions(source_video, output_csv, device='cpu'):
    """导出球员ID和图像坐标"""
    model = YOLO(PLAYER_MODEL).to(device)
    tracker = sv.ByteTrack()
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'player_id', 'center_x', 'center_y', 'class'])
        
        for frame_idx, frame in enumerate(sv.get_video_frames_generator(source_video)):
            results = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            
            for i, (xyxy, tid, cid) in enumerate(zip(
                detections.xyxy, detections.tracker_id, detections.class_id
            )):
                center_x = (xyxy[0] + xyxy[2]) / 2
                center_y = (xyxy[1] + xyxy[3]) / 2
                writer.writerow([frame_idx, tid, center_x, center_y, cid])
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='输入视频路径')
    parser.add_argument('--output', required=True, help='输出CSV路径')
    parser.add_argument('--device', default='cpu', help='设备 (cpu/cuda)')
    args = parser.parse_args()
    
    export_player_positions(args.source, args.output, args.device)
