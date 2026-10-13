# DCVC-RT all-I versus H.264 inter-frame RD measurement

Published results: [2026-10-13 experiment](../results/dcvc_i_vs_h264_rd_20261013/README.md).

`run_rd_comparison.py` decodes the original video once into a shared contiguous
RGB24 reference at 59/4 FPS. It measures 9 native DMCI QP points and 11 libx264
CRF points for each of yuv420p and yuv444p, using actual saved and decoded streams.
`dcvc_i_worker.py` runs in the CUDA-enabled DCVC Python environment; the runner
can use a separate Python environment with NumPy, Pillow and Matplotlib.

Run from `DCVC-family/DCVC-RT` on Linux:

```bash
python local_tests/rd_comparison/run_rd_comparison.py \
  --source local_tests/natural_720p/starigrad_105323_720p.webm \
  --checkpoint checkpoints/cvpr2025_image.pth.tar \
  --work /data/rd_work \
  --output local_tests/results/rd_reproduction
```

On Windows with a WSL DCVC environment, append:

```text
--wsl --dcvc-python /home/qinyi/.local/share/dcvc-rt/venv/bin/python
```

The runner needs `ffmpeg` with `libx264`, `ffprobe`, and enough disk space for a
shared raw RGB sequence and decoded work files (about 2 GB). Use a new output
directory for each run. The source SHA-256 is checked against the original
Starigrad file. Download and model links are in the
[offline deployment guide](../../offline_deployment/README_zh.md).

The RD ordinate is pooled RGB PSNR; per-frame-average PSNR is also saved. The
rate is complete elementary-stream size divided by the common video duration.
No audio, transport overhead or packet loss is simulated. H.264 `slow` and
B-frames measure compression efficiency, not a zero-latency configuration.

Audit all actual streams, paired-reference hashes and frame counts, then
independently cross-check RGB PSNR with FFmpeg:

```bash
python local_tests/rd_comparison/verify_rd_results.py \
  --output local_tests/results/rd_reproduction \
  --work /data/rd_work
```
