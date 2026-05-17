import cv2

def get_video_fps_opencv(video_path):
    """使用OpenCV获取视频帧率"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return None
    
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 获取其他信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration_seconds': duration,
        'duration_formatted': f"{duration // 60:.0f}:{duration % 60:02.0f}",
        'resolution': f"{width}x{height}"
    }

# 使用示例
#video_info = get_video_fps_opencv("data/QQ20260321-144603-HD.mp4")
video_info = get_video_fps_opencv("data/out1.mp4")
print(f"帧率: {video_info['fps']:.2f} fps")
print(f"总帧数: {video_info['total_frames']}")
print(f"时长: {video_info['duration_formatted']}")
print(f"分辨率: {video_info['resolution']}")
