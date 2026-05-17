import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

def process_video(args):
    video_path, output_video, csv_suffix, device = args
    cmd = [
        sys.executable,  # 当前 Python 解释器
        r"D:\BlueDream_basic\bExamples_detect\soccer\main.py",
        "--source_video_path", str(video_path),
        "--target_video_path", str(output_video),
        "--mode", "RADAR",
        "--device", device,
        "--csv_suffix", csv_suffix
    ]
    print(f"Processing {video_path.name} -> {csv_suffix}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error processing {video_path.name}: {result.stderr}")
    return result.returncode

def main():
    seg_dir = Path(r"D:\BlueDream_basic\aSoccermatch-Video\output_segments")
    video_files = sorted(seg_dir.glob("segment_*.mp4"))
    if not video_files:
        print("No segment files found.")
        return

    # 并发数（根据 GPU 显存调整，建议 2~3）
    num_workers = min(3, cpu_count(), len(video_files))
    print(f"Using {num_workers} processes.")

    tasks = []
    for idx, video in enumerate(video_files, start=1):
        suffix = video.stem  # e.g. segment_0001
        output_video = Path(f"output_{suffix}.mp4")
        tasks.append((video, output_video, suffix, "cuda"))

    with Pool(num_workers) as pool:
        pool.map(process_video, tasks)

    print("All segments processed.")

if __name__ == "__main__":
    main()