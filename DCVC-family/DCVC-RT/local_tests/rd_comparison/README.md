# MP4 / WebM 视频替换与 DCVC-RT 全 I 帧—H.264 RD 测试

历史完整实验：[2026-10-13 Starigrad RD 曲线](../results/dcvc_i_vs_h264_rd_20261013/README.md)。
兼容性验证：[2026-10-14 MP4 测试](../results/mp4_rd_compatibility_20261014/README.md)。

## 1. 已修正的问题

- 接收 MP4、WebM 等 FFmpeg 可解码的本地视频，取消旧 Starigrad SHA-256 白名单；仍记录新文件的 SHA-256。
- 自动读取实际分辨率和帧率；将真实尺寸传给 DCVC worker，避免按 720P 解释其他尺寸的 RGB。
- 所有视频转换都使用旧版支持的 `-vsync 0`，不传 `-fps_mode`。
- FFmpeg `-nostdin`、子进程 stdin 关闭，stdout/stderr 直接写普通文件，避免交互等待和未消费管道。
- 每 15 秒显示等待状态；FFmpeg 日志实时写盘。默认单命令最长 1800 秒、日志/输出连续 120 秒不增长即退出。
  DCVC 全部 QP 默认最长 7200 秒。超时会终止该次启动的子进程（Linux 同一进程组，Windows 进程树）。
- 限制解码线程和 filter 线程；先跑 H.264，再跑 DCVC，减少资源争用。
- 测试档位、标题、报告和校验器均按当前视频/参数生成，不再固定 31 个点或断言 H.264 总是胜出。

“卡住”的具体原因仍需你的日志确认。旧脚本 `subprocess.run(capture_output=True)` 使用 communicate，
本身不是已证实的管道死锁；新版解决无限等待和无可见进度，并把失败位置保留到日志中。

## 2. 下载并替换代码

联网机器可使用 [main 分支 ZIP](https://github.com/qinyishcn/DCVC/archive/refs/heads/main.zip)，
或在已有仓库中执行：

```bash
git pull --ff-only https://github.com/qinyishcn/DCVC.git main
```

离线服务器无需重装模型环境。把 ZIP 中以下目录完整复制到服务器仓库对应位置：

```text
DCVC-family/DCVC-RT/local_tests/rd_comparison/
    run_rd_comparison.py
    dcvc_i_worker.py
    verify_rd_results.py
    test_rd_comparison.py
    README.md
```

运行前使用你已装好 DCVC 的 Python 环境检查：

```bash
cd /path/to/DCVC/DCVC-family/DCVC-RT
python -c "import numpy, PIL, matplotlib, torch, MLCodec_extensions_cpp; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders
```

`torch.cuda.is_available()` 必须为 True，编码器列表必须有 `libx264`。
`ffprobe` 最好与 `ffmpeg` 来自同一套安装包。模型继续用 `checkpoints/cvpr2025_image.pth.tar`。
Python 依赖仍是 NumPy/Pillow/Matplotlib、PyTorch CUDA、SciPy、pybind11 和已编译 rANS；没有新增 Python 包。
如缺 Matplotlib，在联网的同平台 Python 环境执行 `python -m pip download matplotlib -d wheelhouse`，
把 wheelhouse 复制到离线机后执行 `python -m pip install --no-index --find-links wheelhouse matplotlib`。

如果使用先前的离线 Docker 镜像，更新宿主机文件不会自动改变镜像内源码；
把完整 `local_tests/rd_comparison` 挂载到容器的 `/opt/dcvc-rt/local_tests/rd_comparison`，
并挂载视频/输出/work 路径，确认容器内 Matplotlib 已安装。

## 3. 替换为自己的 MP4

例如 MP4 放在 `/data/input/my_video.mp4`。不用改扩展名，也不用先手工抽帧。
后续命令只需把 `--source` 改成真实完整路径；路径有空格时加引号。

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,color_space,color_range \
  -of json "/data/input/my_video.mp4"
```

默认保持源文件编码尺寸和平均帧率，统一为 CFR；不使用音频。
默认关闭 MP4 显示旋转，两种编码器都使用编码方向的相同帧。
如果要固定 720P/30 FPS，追加 `--width 1280 --height 720 --fps 30`。
指定尺寸会直接缩放，非 16:9 内容可能被拉伸，建议保持原尺寸。
颜色优先读取文件标记；缺标记时假定 BT.709 limited，可显式传
`--input-color-matrix bt709 --input-range tv`。HDR 需要先做明确的 SDR tone mapping。

## 4. 先跑短片段冒烟测试

在 `DCVC-family/DCVC-RT` 下运行；`--work` 和 `--output` 必须是尚不存在的新目录：

```bash
python local_tests/rd_comparison/run_rd_comparison.py \
  --source "/data/input/my_video.mp4" \
  --checkpoint checkpoints/cvpr2025_image.pth.tar \
  --work /data/rd_work/my_video_smoke_20261014 \
  --output local_tests/results/my_video_smoke_20261014 \
  --duration 2 --max-frames 30 \
  --qps 0 32 63 --crfs 38 26 14 \
  --command-timeout 300 --idle-timeout 60
```

看到 `COMPLETED 6 points ...` 后应已生成曲线，不需要额外出图程序。
验证真实码流、帧数、码率、PSNR，并用 FFmpeg 独立复算最后一个 H.264 档位：

```bash
python local_tests/rd_comparison/verify_rd_results.py \
  --output local_tests/results/my_video_smoke_20261014 \
  --work /data/rd_work/my_video_smoke_20261014
```

## 5. 正式测试并自动出图

以下测前 10 秒，默认 9 个 DCVC QP + 11 个 H.264 CRF，输出两条主曲线：

```bash
python local_tests/rd_comparison/run_rd_comparison.py \
  --source "/data/input/my_video.mp4" \
  --checkpoint checkpoints/cvpr2025_image.pth.tar \
  --work /data/rd_work/my_video_full_20261014 \
  --output local_tests/results/my_video_full_20261014 \
  --duration 10
```

想测试整个文件，去掉 `--duration 10`。想从第 20 秒开始，追加 `--start 20`。
如需旧实验的 H.264 4:4:4 第三条参考曲线，追加 `--h264-formats yuv420p yuv444p`。
实际码率按当前片段的 `实际帧数/目标FPS` 计算，不使用 MP4 容器时长或声明码率。
720P/30 FPS/10 秒一份 RGB 约 791 MiB；参考和重构至少约 1.6 GiB，另需全部码流空间。
长视频可调大 `--command-timeout` / `--worker-timeout`，或先缩短片段。

结果目录：

```text
rd_curve.png          # 一张线性码率对比图（优先显示 DCVC 工作范围）
rd_curve_log.png      # 对数码率完整范围图
rd_curve.svg          # 可编辑矢量图
rd_points.csv         # 每个 QP / CRF 的实际码率与 PSNR
results.json          # 逐帧 MSE、PSNR、I/P/B 数量及码流 SHA-256
environment.json      # 本视频名称、尺寸、FPS、源文件/模型/参考 SHA-256
commands.json         # 本次真正执行的命令
README.md             # 自动生成的本视频结果报告
*_sample.png          # 同一中间帧的参考和重构
*.log                 # 实时执行日志
```

PSNR 为 RGB 全像素/全帧 MSE 汇总后取 dB，另存逐帧 dB 均值。
H.264 `slow + tune=psnr`、GOP=30、最多3个 B 帧，优先比较 RD，不是零延迟配置。

## 6. 已测完，只重新生成图

不需要 MP4、checkpoint、GPU、FFmpeg 或 work 目录，只要结果目录的两个 JSON：

```bash
python local_tests/rd_comparison/run_rd_comparison.py \
  --plot-only --output local_tests/results/my_video_full_20261014
```

它只更新 PNG/SVG，不重跑编码器、不改原始数据和结果报告。

## 7. Windows + WSL 的运行方式

若 FFmpeg 在 Windows、CUDA Python 在 WSL，用 PowerShell（反引号续行）：

```powershell
python local_tests/rd_comparison/run_rd_comparison.py `
  --source "D:\videos\my video.mp4" `
  --checkpoint checkpoints/cvpr2025_image.pth.tar `
  --work "D:\rd_work\my_video_smoke_20261014" `
  --output local_tests/results/my_video_smoke_20261014 `
  --duration 2 --max-frames 30 --qps 0 32 63 --crfs 38 26 14 `
  --wsl --dcvc-python /home/qinyi/.local/share/dcvc-rt/venv/bin/python
```

全程在 Linux/离线服务器运行时不要加 `--wsl`。
FFmpeg 不在 PATH 时，传 `--ffmpeg /full/path/ffmpeg --ffprobe /full/path/ffprobe`，
Windows 对应使用 `ffmpeg.exe` / `ffprobe.exe` 完整路径。

## 8. 若仍然卡住/报错

- `START reference_decode.log`：在抽取/转换源视频；`START h264_*`：H.264 编/解码；
  `START dcvc_worker.log`：在跑神经网络，不能把这个阶段误认为 FFmpeg 卡住。
- 每 15 秒有 `WAIT` 心跳，日志仍在增长时通常正在处理；可用 `tail -f` 看日志内容。
- `TimeoutError: no log/output activity`：对应日志/输出在 idle-timeout 内未增长。
  重复写统计但没有真正处理帧的程序最终仍会被总 command-timeout 截止。
- `Unknown encoder libx264`：当前 FFmpeg 缺 x264，需要换带 libx264 的构建。
- `Unrecognized option fps_mode`：确认服务器真正执行的是更新后的脚本。新脚本的命令参数不含该选项。
- `decoded frame count/size differs`：保留对应 encode/decode 日志、source_probe 信息和 commands.json 排查，
  不要通过忽略尺寸错误来强行计算 PSNR。
- `source_probe.json` 和 `commands.json` 在执行过程中就会写盘，即使中途失败也能查看输入信息和最后启动的命令。
- `Use new --output and --work directories`：换新名字重新运行；新视频不可复用旧 RGB/重构文件。
- MP4 源文件和 work 目录不要放在同一文件路径。不要在运行中手动改脚本、视频或模型。

本次修复回归测试（无需 GPU）：

```bash
python -m unittest discover -s local_tests/rd_comparison -p test_rd_comparison.py -v
```
