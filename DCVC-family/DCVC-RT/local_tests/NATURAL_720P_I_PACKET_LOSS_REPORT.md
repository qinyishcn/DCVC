# DCVC-RT I 帧 720P 分层丢包与帧率实测

## 结论

在本机 NVIDIA GeForce RTX 3070 上，DCVC-RT 的 CVPR 2025 I 帧模型以 QP 63 编码 32 张真实 1280×720 自然视频帧：

- I 帧编码平均 **82.07 ms / 12.19 FPS**，解码平均 **65.69 ms / 15.22 FPS**。
- 无丢包平均 PSNR 为 **40.81 dB**，平均码流为 **48,321 bytes/帧**。
- 将 `z/y0/y1/y2/y3` 分成可独立解码的 rANS 流并按优先级放置后，尾部丢包表现为明显的分层台阶：约 5%～10% 丢包时完整保留到 `y2`，平均 PSNR 均为 **35.16 dB**；约 30% 时只保留到 `y1`，为 **31.47 dB**；约 50% 时只保留到 `y0`，为 **28.65 dB**；约 70% 时只剩 `z`，仍有 **24.41 dB**。
- 当 `z` 也不完整时，该帧不能由真实熵码流安全解码。80% 和 90% 目标丢包下的可解码帧比例分别为 **96.88%** 和 **84.38%**；95% 时全部不可解码。

这里的“容错”来自**分层码流的完整前缀仍能独立重构**，不是因为一个被截断的 rANS 流可以带错继续解码。某一分量只要缺少一个字节，本测试就丢弃该分量以及依赖它的后续分量。

## 测试输入与环境

- 视频：[`Starigrad 105323`](https://commons.wikimedia.org/wiki/File:Starigrad_105323.webm)，真实街景、树木和车辆的相机拍摄视频。
- 作者：Quahadi Añtó；许可证：CC BY-SA 3.0 / GFDL，详见源页面。
- 下载文件：1280×720 VP8 WebM，SHA-256 `aee660c4ef1aa45dc169a0bf12bf2525869136d8dba7c1cce6a9d7b0c9f21bff`。
- 抽帧：在全片时域均匀抽取 32 帧；每帧由 I 帧模型独立编码，因此不使用视频 P 帧的参考状态。
- 模型：`checkpoints/cvpr2025_image.pth.tar`，QP 63，`force_zero_thres=0.12`。
- 硬件：NVIDIA GeForce RTX 3070；PyTorch 2.4.1+cu121，CUDA 12.1。
- 性能路径：仓库的 PyTorch fallback；本机未使用 DCVC-RT 自定义优化 CUDA 扩展。因此这些 FPS 是当前安装的真实结果，不代表论文优化路径的上限。

## 码流和丢包方法

官方模型输出 `z` 与 `y`；`y` 在模型内部按 `y0/y1/y2/y3` 的渐进掩码分为四层。为了精确实现指定的 `先丢 y3，再丢 y2，再丢 y1，最后丢 z`，本测试将五个分量分别送入 rANS 熵编码器，构成：

```text
20-byte lengths header | z | y0 | y1 | y2 | y3
```

然后将该真实字节串切成最多 1400 bytes 的包载荷，并从帧尾部删除数据包。20-byte 头也计入码流和分包；IP/UDP/RTP 协议头开销未计入 1400 bytes。

关键规则：

1. 每个分量是独立且完整的 rANS 流。
2. 如果丢包在某分量中间结束，该分量被视为不可用，同时丢弃所有更高层。
3. 如果 `z` 完整，则用现存的完整 `y` 前缀解码；缺失层的潜变量置零。
4. 如果 `z` 不完整，标记为不可解码。为了让汇总 PSNR 仍有定义，采用中性灰图作为 concealment；这不是模型输出。
5. 32 帧共 1123 个包。每个目标丢包率采用最大余数法把整数丢包数分配到各帧，使全序列实际丢包率尽量贴近目标值。

这不是 DCVC-RT 官方文件容器，而是为了验证分层传输而构造的最小实验封装。原生单一 rANS 流被截断后不能据此安全恢复部分符号。

## I 帧 FPS

前 5 帧仅用于 GPU 预热，不计入计时；表中统计 32 帧。

| 指标 | 编码 | 解码 |
|---|---:|---:|
| 平均耗时 | 82.07 ms | 65.69 ms |
| P50 | 82.10 ms | 65.86 ms |
| P95 | 82.68 ms | 66.91 ms |
| 吞吐率 | 12.19 FPS | 15.22 FPS |

## 全部 32 帧的丢包结果

| 目标丢包率 | 实际包丢失 | 接收字节 | 平均 PSNR | 可解码帧 | 完整前缀分布 |
|---:|---:|---:|---:|---:|---|
| 0% | 0.000% | 100.000% | 40.808 dB | 100.000% | full: 32 |
| 1% | 0.980% | 99.624% | 38.637 dB | 100.000% | full: 21, through-y2: 11 |
| 2% | 1.959% | 99.086% | 36.550 dB | 100.000% | full: 10, through-y2: 22 |
| 5% | 4.987% | 96.539% | 35.158 dB | 100.000% | through-y2: 32 |
| 10% | 9.973% | 91.472% | 35.158 dB | 100.000% | through-y2: 32 |
| 20% | 20.036% | 81.248% | 34.454 dB | 100.000% | through-y2: 26, through-y1: 6 |
| 30% | 30.009% | 71.115% | 31.469 dB | 100.000% | through-y1: 32 |
| 40% | 39.982% | 60.982% | 31.208 dB | 100.000% | through-y1: 29, through-y0: 3 |
| 50% | 50.045% | 50.758% | 28.646 dB | 100.000% | through-y0: 32 |
| 60% | 60.018% | 40.624% | 28.511 dB | 100.000% | through-y0: 31, z-only: 1 |
| 70% | 69.991% | 30.491% | 24.406 dB | 100.000% | z-only: 32 |
| 80% | 79.964% | 20.357% | 23.798 dB | 96.875% | z-only: 31, concealed: 1 |
| 90% | 90.027% | 10.133% | 21.778 dB | 84.375% | z-only: 27, concealed: 5 |
| 95% | 95.013% | 5.067% | 10.840 dB | 0.000% | concealed: 32 |
| 98% | 98.041% | 1.990% | 10.840 dB | 0.000% | concealed: 32 |
| 100% | 100.000% | 0.000% | 10.840 dB | 0.000% | concealed: 32 |

`received bytes` 与包丢失率不完全互补，因为每帧最后一个包通常不足 1400 bytes。`concealed` 的 PSNR 只是中性灰基线，不能计作成功解码质量。

## 示例帧

示例帧为 `frame_016.png`，其封装码流 49,789 bytes，共 36 个包。完整前缀的单帧 PSNR：

| 可用分量 | PSNR |
|---|---:|
| `z+y0+y1+y2+y3` | 40.524 dB |
| `z+y0+y1+y2` | 34.802 dB |
| `z+y0+y1` | 30.684 dB |
| `z+y0` | 27.795 dB |
| `z` | 23.257 dB |
| `z` 不完整，中性灰 concealment | 10.744 dB |

示例重构保存在 [`results/natural_720p_i_packet_loss/reconstructions`](results/natural_720p_i_packet_loss/reconstructions)。由于单帧只有 36 个包，单帧实际丢包率是离散的；全序列汇总值才精确贴近目标比例。

## 对 y3 猜测的验证

- `y3` 作为一个完整独立 rANS 流时，少丢和多丢都不能“部分解码 y3”；只要它不完整，本实验都会退化到相同的 `z+y0+y1+y2` 重构。因此 5% 和 10% 两档平均 PSNR 完全相同。
- 1% 与 2% 的平均 PSNR 不同，不是残缺 `y3` 内部错误量不同，而是跨 32 帧统计时，分别有 11 帧和 22 帧首次失去完整 `y3`。
- 若希望 `y3` 内部也随丢包量平滑退化，需要把 `y3` 再切成多个可独立熵解码的块，并为每块提供长度、上下文重同步/概率状态和完整性校验。当前测试没有假装截断 rANS 仍可用。

## 复现

在仓库的 WSL 环境、模型 checkpoint、`curl` 和 `ffmpeg` 准备好后运行：

```bash
cd /mnt/c/Users/qinyi/Documents/AItest\ 2/DCVC/DCVC-family/DCVC-RT
mkdir -p local_tests/natural_720p/frames_32
curl -L 'https://upload.wikimedia.org/wikipedia/commons/f/f2/Starigrad_105323.webm' \
  -o local_tests/natural_720p/starigrad_105323_720p.webm
ffmpeg -i local_tests/natural_720p/starigrad_105323_720p.webm \
  -vf 'fps=32/14.187' -frames:v 32 \
  local_tests/natural_720p/frames_32/frame_%03d.png
/home/qinyi/.local/share/dcvc-rt/venv/bin/python \
  local_tests/natural_720p_i_packet_loss_test.py \
  --checkpoint checkpoints/cvpr2025_image.pth.tar \
  --source-video local_tests/natural_720p/starigrad_105323_720p.webm \
  --frames local_tests/natural_720p/frames_32 \
  --output local_tests/results/natural_720p_i_packet_loss
```

机器可读结果：

- [`results.json`](results/natural_720p_i_packet_loss/results.json)：环境、每帧码流、每档全序列结果与示例帧结果。
- [`loss_summary.csv`](results/natural_720p_i_packet_loss/loss_summary.csv)：全序列丢包率、PSNR 与可解码率汇总。
- [`sample_packet_manifest.json`](results/natural_720p_i_packet_loss/sample_packet_manifest.json)：示例帧分量范围、分包范围及 SHA-256，用于证明测试操作的是实际编码字节。

原视频和抽取的未压缩参考帧不提交到仓库；视频的来源、许可证、校验值及下载地址记录在结果和 `SOURCE_LICENSE.md` 中。
