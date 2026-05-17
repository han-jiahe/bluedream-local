import os
# from video_processor import FootballVideoAnalyzer
from vi_analysis import compute_vi_distribution

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "football_analysis_output")
    samples_json = os.path.join(base_dir, "final_merged_coords.json") 
    window_size = 117                            # 窗口大小 (帧)，对应4秒
    
    '''
    # 配置参数
    video_path = os.path.join(base_dir, "football_match.mp4")          # 输入视频路径
    
    sample_rate = 10                           # 采样率 (Hz)
    # 球队颜色 (BGR格式)
    home_color = [0, 0, 255]      # 红色
    away_color = [255, 0, 0]      # 蓝色
    left_goal_team = 'home'       # 左球门归属
    right_goal_team = 'away'      # 右球门归属

    # 步骤1: 视频处理与采样
    print("="*50)
    print("步骤1: 视频处理与采样")
    print("="*50)
    analyzer = FootballVideoAnalyzer(
        video_path=video_path,
        output_dir=output_dir,
        sample_rate=sample_rate,
        home_color=home_color,
        away_color=away_color,
        left_goal_team=left_goal_team,
        right_goal_team=right_goal_team
    )
    # 如果已经存在samples.json且不想重新处理，可以跳过
    samples_json = os.path.join(output_dir, 'data', 'samples.json')
    if not os.path.exists(samples_json):
        analyzer.process_video()
    else:
        print(f"采样文件已存在: {samples_json}，跳过视频处理")
'''
    # 步骤2: 计算VI分布
    print("\n" + "="*50)
    print("步骤2: 计算VI分布")
    print("="*50)
    vi_json = os.path.join(output_dir, 'data', 'VI_distribution.json')
    compute_vi_distribution(
        samples_json_path=samples_json,
        output_json_path=vi_json,
        window_size=window_size,
        save_cluster_images=False   # 是否保存每帧聚类图像，建议False以免产生大量文件
    )

    print("\n全部处理完成！")

if __name__ == "__main__":
    main()