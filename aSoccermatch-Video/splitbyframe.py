import cv2
import os
from pathlib import Path

def split_video_by_frames_opencv(input_video, frames_per_segment=10, output_dir="output_segments"):
    """
    使用OpenCV按帧数拆分视频
    
    Args:
        input_video: 输入视频路径
        frames_per_segment: 每个片段包含的帧数（默认10帧）
        output_dir: 输出目录
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"原始视频信息:")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  分辨率: {width}x{height}")
    print(f"  总帧数: {total_frames}")
    print(f"  将拆分为: {total_frames // frames_per_segment + (1 if total_frames % frames_per_segment else 0)} 个片段")
    print("-" * 50)
    
    # 定义视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    segment_count = 0
    frame_count = 0
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_buffer.append(frame)
        frame_count += 1
        
        # 达到指定帧数，保存片段
        if len(frames_buffer) == frames_per_segment:
            segment_count += 1
            output_path = os.path.join(output_dir, f"segment_{segment_count:04d}.mp4")
            
            # 创建VideoWriter
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 写入帧
            for f in frames_buffer:
                out.write(f)
            
            out.release()
            print(f"已保存: {output_path} ({len(frames_buffer)} 帧)")
            
            # 清空缓冲区
            frames_buffer = []
    
    # 处理剩余的帧（不足10帧）
    if frames_buffer:
        segment_count += 1
        output_path = os.path.join(output_dir, f"segment_{segment_count:04d}.mp4")
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for f in frames_buffer:
            out.write(f)
        out.release()
        
        print(f"已保存: {output_path} ({len(frames_buffer)} 帧 - 最后片段)")
    
    cap.release()
    print(f"\n拆分完成！共生成 {segment_count} 个视频片段")
    return segment_count

# 注意替换路径为实际视频文件和输出目录，下面只是使用示例
split_video_by_frames_opencv("D:\SRDP\sports\examples\data.mp4", frames_per_segment=120, output_dir="D:\SRDP\sports\examples\output_segments")
