# BlueDream Local v3 — ModelArts Notebook 部署手册

> 个人足球视频分析 Web 原型系统  
> GPU 推理: ModelArts T4 | 本地 Web: Windows FastAPI

---

## 1. 架构概览

```
本地 Windows (Web GUI)              ModelArts Notebook (GPU)
┌────────────────────────┐            ┌──────────────────────┐
│  python main.py        │            │  python main.py      │
│  http://127.0.0.1:8000 │  ──SSH──▶    --mode RADAR        │
│                        │   Jupyter  │  --device cuda       │
│  Video  to   results   |  transport |  batch_process.py    │
└────────────────────────┘            └──────────────────────┘
```

**工作流**: 本地 Web 上传视频 → 传到 Notebook → GPU 检测 → 下载 CSV → 本地 VI 分析 → 图表

---

## 2. 创建 ModelArts Notebook

### 2.1 配置

| 选项 | 值 |
|------|-----|
| 镜像 | `PyTorch-2.1.0-cuda12.1-py3.10.6-ubuntu22.04-x86_64` |
| GPU | Tesla T4 16GB |
| 存储 | 默认 (50GB) |
| VPC | 任意（不跟 ECS 通信） |

### 2.2 创建 SSH 密钥对（用于远程文件传输）

1. 控制台 → ModelArts → Notebook → 点击实例名称进入详情页
2. 找到 **"远程访问"** 或 **"SSH 密钥对"** → 点击 **"创建密钥对"**
3. 名称自定义（如 `notebook-key`），类型选 **SSH_RSA_2048**
4. 点击确定 → 自动下载 `.pem` 私钥文件到本地
5. 将 `.pem` 文件放到 `%USERPROFILE%\.ssh\` 目录下：

```powershell
copy C:\Users\HJH\Downloads\notebook-key.pem %USERPROFILE%\.ssh\
```

> ⚠️ **私钥托管**：不需要勾选"将私钥托管到华为云"。仅本地保存即可。
>
> ⚠️ **SSH 直连限制**：ModelArts Notebook 处于 HTTP 代理后（`proxy.modelarts.com:80`），公网 SSH 连接可能被拦截。如果 SSH 不通，用 JupyterLab 网页界面传输文件（见 §2.3）同样方便。

### 2.3 文件传输方式

**方式 A — JupyterLab 网页上传（推荐，无网络限制）**

1. 控制台点击"打开"进入 JupyterLab
2. 左侧文件浏览器 → 进入目标目录（如 `bluedream-local/bExamples_detect/soccer/`）
3. 点击工具栏 **↑ 上传按钮** → 选择本地视频文件 → 等待上传完成
4. 下载结果时：右键文件 → **"Download"** 保存到本地

> 适合 500MB 以内的视频，大文件建议用 obsutil 经 OBS 中转。

**方式 B — 本地建 SSH 隧道传输（需要 Notebook 公网可达）**

在本地 Windows 终端建立 SSH 会话：

```powershell
# 查看 Notebook 详情页的 SSH 连接地址，格式类似:
# ssh://ma-user@dev-modelarts-cnnorth4.huaweicloud.com:31248

ssh -i %USERPROFILE%\.ssh\notebook-key.pem ma-user@dev-modelarts-cnnorth4.huaweicloud.com -p 31248
```

连接成功后可通过 `scp` 传文件：

```powershell
# 上传视频
scp -i %USERPROFILE%\.ssh\notebook-key.pem -P 31248 video.mp4 ma-user@dev-modelarts-cnnorth4.huaweicloud.com:/home/ma-user/work/bluedream-local/

# 下载 CSV
scp -i %USERPROFILE%\.ssh\notebook-key.pem -P 31248 ma-user@dev-modelarts-cnnorth4.huaweicloud.com:/home/ma-user/work/bluedream-local/bExamples_detect/soccer/player_final_real_coords.csv .
```

### 2.4 创建后初次打开

状态变为"运行中"后，点击"打开"进入 JupyterLab，打开终端。

---

## 3. 环境配置（关键步骤）

> ⚠️ 以下全部在 Notebook 终端中执行，按顺序，不可跳步。

### 3.1 克隆项目

```bash
cd /home/ma-user/work
git clone https://github.com/han-jiahe/bluedream-local.git
cd bluedream-local/bExamples_detect/soccer
```

### 3.2 创建独立 conda 环境

> **为什么要新建环境**: 镜像自带的 PyTorch-2.1.0 使用 CUDA 12.1 toolkit，但 T4 显卡 driver 是 11.4，不兼容。需要降级到 torch 1.12.1+cu113。

```bash
conda create -n soccer python=3.10 -y
conda activate soccer
```

### 3.3 安装依赖（严格按顺序）

```bash
# 1. numpy 1.x (必须在 torch 之前装)
pip install "numpy<2"

# 2. PyTorch CUDA 11.3 (兼容 T4 driver 11.4)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html

# 3. 设置 CUDA 库路径 (每次开新终端都要执行)
export LD_LIBRARY_PATH=$(python -c \
  "import torch; print(torch.utils.cmake_prefix_path.replace('/share/cmake', '/lib'))"):$LD_LIBRARY_PATH

# 4. 验证 GPU
python -c "import torch; t=torch.zeros(1).cuda(); print('CUDA OK')"
# 必须输出: CUDA OK

# 5. YOLO + sports 库
pip install ultralytics
pip install git+https://github.com/roboflow/sports.git

# 6. transformers (SigLIP 球队分类)
pip install "transformers==4.37.0"

# 7. 再次锁定 numpy 版本 (防止被拉高)
pip install "numpy<2"
```

### 3.4 永久写入环境变量

以后每次开 Notebook 终端不用重新 export：

```bash
echo 'export LD_LIBRARY_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path.replace(\"/share/cmake\", \"/lib\"))"):$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

### 3.5 下载模型文件

```bash
# 回到项目根
cd /home/ma-user/work/bluedream-local/bExamples_detect/soccer

# 设置 OBS 密钥 (替换为实际值)
export OBS_ACCESS_KEY=你的AK
export OBS_SECRET_KEY=你的SK
export OBS_BUCKET=bluedream-models

python download_models.py
```

验证:

```bash
ls -lh data/*.pt
# 应显示 3 个文件: football-player-detection.pt, football-pitch-detection.pt, football-ball-detection.pt
```

---

## 4. 运行 GPU 检测

### 4.1 单视频检测（输出标注视频 + CSV）

```bash
conda activate soccer
export LD_LIBRARY_PATH=$(python -c \
  "import torch; print(torch.utils.cmake_prefix_path.replace('/share/cmake', '/lib'))"):$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

python main.py \
  --source_video_path worldcupF2_goal1.mp4 \
  --target_video_path annotated_output.mp4 \
  --device cuda \
  --mode RADAR
```

**输出**:
- `player_final_real_coords.csv` — 球员追踪数据 (帧号, 球员ID, 真实坐标)
- `annotated_output.mp4` — 标注视频 (带球员ID和球队颜色)

### 4.2 批量检测（仅 CSV，更快）

```bash
python batch_process.py \
  --segments_dir /path/to/videos \
  --workers 3 \
  --device cuda
```

| 参数 | 说明 |
|------|------|
| `--segments_dir` | 视频文件目录 |
| `--workers` | 并发数 (T4 16GB 建议 2-3, 32GB 建议 3-5) |
| `--keep_video` | 保留标注视频输出（较慢） |

---

## 5. 本地 Web 服务启动

### 5.1 安装依赖

```powershell
cd C:\Users\HJH\Bluedream-Local_v3\BlueDream_basic\web_app
C:\Users\HJH\venv_sports\Scripts\activate
pip install -r requirements.txt
```

### 5.2 启动

```bash
python main.py
```

浏览器自动打开 `http://127.0.0.1:8000`

### 5.3 使用流程

| 步骤 | 操作 |
|------|------|
| 1 | 上传视频 (完整不切分) |
| 2 | 将视频传至 Notebook (Jupyter 上传 / SCP) |
| 3 | Notebook 运行 `main.py --mode RADAR` |
| 4 | 下载生成的 CSV |
| 5 | 本地 Web 上传 CSV |
| 6 | 选择分析类型，开始分析 |
| 7 | 查看图表 |

---

## 6. 分析类型说明

| 类型 | 图表 | 需要额外参数 |
|------|------|-------------|
| **完整分析** | VI 分布时序 + 概率密度 + 球员排名 + Top3 热力图 | — |
| **单球员分析** | VI 时序 (双面板) + 热力图 | 球员 ID |
| **轨迹可视化** | 多球员轨迹 + 可选足球轨迹 (真实坐标 105×68m) | 球员 ID (逗号分隔) |
| **仅 VI 分布** | VI 时序 + 概率密度 + 球员排名 | — |
| **仅热力图** | 指定球员热力图 + colorbar | 球员 ID |

---

## 7. Notebook 重新激活后的操作

ModelArts Notebook 闲置 1-2 小时会自动停止。重新启动后：

```bash
conda activate soccer
cd /home/ma-user/work/bluedream-local/bExamples_detect/soccer
export LD_LIBRARY_PATH=$(python -c \
  "import torch; print(torch.utils.cmake_prefix_path.replace('/share/cmake', '/lib'))"):$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

# 拉最新代码 (可选)
git pull origin master

# 运行检测
python main.py --source_video_path <视频名> --target_video_path out.mp4 --device cuda --mode RADAR
```

> 如果 `~/.bashrc` 已写入环境变量，`LD_LIBRARY_PATH` 和 `HF_ENDPOINT` 会自动设置。

---

## 8. 性能优化总结

| 优化项 | 效果 |
|--------|------|
| FP16 推理 (`model.half()`) | GPU 2x 加速 |
| imgsz 1280→960 | 像素减44%，~1.5x 加速 |
| Pitch 每15帧检测一次 | 跳过冗余的关键点检测 |
| 仅输出 CSV (batch模式) | 跳过视频编码，节省 30% 时间 |
| 2-3 worker 并发 | 充分利用 GPU 显存 |

---

## 9. 常见问题

| 症状 | 原因 | 解决 |
|------|------|------|
| `CUDNN_STATUS_NOT_INITIALIZED` | conda CUDA 库冲突 | 删环境重建: `conda env remove -n soccer` 然后重新执行 §3.2-3.3 |
| `Numpy is not available` | numpy 2.x 不兼容 | `pip install "numpy<2"` |
| HuggingFace 超时 | 代理/VPN 不可用 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `CUDA driver too old (11040)` | torch CUDA 版本太高 | 确认 `torch==1.12.1+cu113` |
| `No module named 'sports'` | roboflow/sports 未安装 | `pip install git+https://github.com/roboflow/sports.git` |
| GPU 显存占用低 (<6GB) | 正常，YOLO 架构决定 | 通过并发 worker 提高利用率 |
