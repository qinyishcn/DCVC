"""Measure actual bitstreams on one CFR RGB sequence; supports a WSL DCVC worker."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log=None):
    result = subprocess.run(list(map(str, command)), capture_output=True, check=False)
    if log:
        Path(log).write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def linux_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}" if sys.platform == "win32" else str(path)


def summarize(codec, quality, stream, reference, reconstructed):
    assert reconstructed.shape == reference.shape
    rows = []
    for index, (ref, rec) in enumerate(zip(reference, reconstructed)):
        error = ref.astype(np.float64) - rec.astype(np.float64)
        mse = float(np.mean(error * error))
        rows.append({"frame_index": index, "mse_rgb": mse,
                     "psnr_rgb_db": float(10 * np.log10(255 ** 2 / mse))})
    mse = float(np.mean([row["mse_rgb"] for row in rows]))
    return {"codec": codec, "quality": quality, "frames": len(rows),
            "bytes": stream.stat().st_size, "stream_sha256": sha256(stream),
            "psnr_rgb_pooled_db": float(10 * np.log10(255 ** 2 / mse)),
            "psnr_rgb_mean_frame_db": float(np.mean([row["psnr_rgb_db"] for row in rows])),
            "mean_mse_rgb": mse, "per_frame": rows}


def make_plots(points, output):
    styles = {"DCVC-RT all-I": ("o", "-", "#1764ab"),
              "H.264 4:2:0 inter": ("s", "-", "#d54e25"),
              "H.264 4:4:4 inter": ("^", "--", "#18845e")}
    for logarithmic in (False, True):
        fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
        for codec, (marker, line, color) in styles.items():
            subset = sorted([p for p in points if p["codec"] == codec], key=lambda p: p["bitrate_kbps"])
            ax.plot([p["bitrate_kbps"] / 1000 for p in subset],
                    [p["psnr_rgb_pooled_db"] for p in subset],
                    marker=marker, linestyle=line, color=color, label=codec)
        ax.set(xlabel="Actual video bitrate (Mbit/s)", ylabel="RGB PSNR, pooled MSE (dB)",
               title="Starigrad 105323 | 1280 x 720 | 14.75 fps | complete CFR clip")
        if logarithmic:
            ax.set_xscale("log")
        else:
            # Show the shared operating range; high-rate CRF=0 controls remain
            # visible in the companion full-range logarithmic plot and CSV.
            upper = max(p["bitrate_kbps"] for p in points if p["codec"] == "DCVC-RT all-I") / 1000 * 1.1
            shared = [p["psnr_rgb_pooled_db"] for p in points if p["bitrate_kbps"] / 1000 <= upper]
            ax.set_xlim(0, upper)
            ax.set_ylim(min(shared) - 1, max(shared) + 1)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower right")
        fig.savefig(output / ("rd_curve_log.png" if logarithmic else "rd_curve.png"), dpi=180)
        if not logarithmic:
            fig.savefig(output / "rd_curve.svg")
        plt.close(fig)


def make_report(points, metadata, output):
    groups = {codec: sorted([p for p in points if p["codec"] == codec],
                            key=lambda p: p["bitrate_kbps"])
              for codec in ("DCVC-RT all-I", "H.264 4:2:0 inter", "H.264 4:4:4 inter")}
    rows = []
    for rate in (0.5, 1, 2, 4, 5):
        row = [f"{rate:.1f}"]
        for subset in groups.values():
            rates = [p["bitrate_kbps"] / 1000 for p in subset]
            if rates[0] <= rate <= rates[-1]:
                value = np.interp(np.log(rate), np.log(rates),
                                  [p["psnr_rgb_pooled_db"] for p in subset])
                row.append(f"{value:.2f}")
            else:
                row.append("超出实测范围")
        rows.append("| " + " | ".join(row) + " |")
    table = "\n".join(rows)
    measured = "\n".join(
        f"| {p['codec']} | {p['quality']} | {p['bitrate_kbps']:.1f} | "
        f"{p['psnr_rgb_pooled_db']:.3f} | {p['psnr_rgb_mean_frame_db']:.3f} |"
        for p in sorted(points, key=lambda p: (p["codec"], p["bitrate_kbps"])))
    count = metadata["reference_frames"]
    dcvc = groups["DCVC-RT all-I"]
    h264 = groups["H.264 4:2:0 inter"]
    deltas = []
    for p in dcvc:
        if h264[0]["bitrate_kbps"] <= p["bitrate_kbps"] <= h264[-1]["bitrate_kbps"]:
            score = np.interp(np.log(p["bitrate_kbps"]),
                              np.log([x["bitrate_kbps"] for x in h264]),
                              [x["psnr_rgb_pooled_db"] for x in h264])
            deltas.append(score - p["psnr_rgb_pooled_db"])
    text = f"""# DCVC-RT 全 I 帧与 H.264 视频编码 RD 对比

本次在同一份 `Starigrad 105323` 720P 自然视频上测试 31 个真实编码工作点。
DCVC-RT 每帧独立编码；H.264 使用帧间预测。无丢包，不含音频。

## 结果

在本视频的 DCVC-RT 实测范围（{dcvc[0]['bitrate_kbps']:.0f}～{dcvc[-1]['bitrate_kbps']:.0f} kbps）内，
H.264 4:2:0 的 RD 曲线高于 DCVC-RT 全 I 帧曲线。在 9 个 DCVC 工作点处，
按 log(码率) 对 H.264 曲线插值得到的 PSNR 优势为
{min(deltas):.2f}～{max(deltas):.2f} dB。这是本片段、下述参数的结果，不能外推为所有内容的结论。
H.264 利用了时间冗余，而 DCVC-RT 此处只用了图像模型；本实验不能用于评价 DCVC-RT 的 P 帧视频模型。

![线性码率坐标 RD 曲线](rd_curve.png)

![对数码率坐标 RD 曲线](rd_curve_log.png)

### 相同码率下的 PSNR

下表是对相邻实测点在 `log(码率)` 轴上做线性插值，**不是额外测量的定码率点**。

| 码率 Mbps | DCVC-RT 全 I | H.264 4:2:0 帧间 | H.264 4:4:4 帧间 |
|---:|---:|---:|---:|
{table}

### 所有实测点

DCVC 的 QP 是模型内部 0～63 索引，越大质量和码率越高；H.264 表中质量参数是
CRF，越小质量和码率越高。二者数字不能直接对齐。纵轴默认使用全序列 pooled RGB PSNR，
同时给出逐帧 PSNR 均值以便与旧实验核对。

| 模型/编码器 | QP 或 CRF | 实际码率 kbps | pooled RGB PSNR dB | mean-frame RGB PSNR dB |
|---|---:|---:|---:|---:|
{measured}

## 数据与公平性

- 同一源视频，SHA-256：`{metadata['source_sha256']}`。
- 源文件包含 187 个带时间戳的视频帧；容器还有音频，容器时长不等于视频时长。
  统一按 59/4 FPS 的 CFR 时间轴生成 {count} 帧，共 {metadata['duration_seconds']:.6f} 秒。
  FFmpeg `fps` 滤镜依据时间戳复制/丢弃帧；两种编码器接收完全相同的 RGB24 字节。
  旧丢包实验的 32 张稀疏抽帧未用于本次视频 RD 对比。
- 分辨率 1280×720；共享 RGB 原始序列 SHA-256：`{metadata['reference_sha256']}`。
  [`reference_frame_manifest.json`](reference_frame_manifest.json) 记录每帧的时间和 RGB SHA-256。
- 源视频本身是已压缩 VP8，不是无压缩相机母版；本次衡量相对于其解码图像的再编码失真。
  源视频没有完整色彩标记，本实验统一假定 BT.709 limited YUV，转换为 full-range RGB。
- DCVC 使用官方 `DMCI` + `cvpr2025_image.pth.tar`，FP16，`force_zero_thres=0.12`，
  单熵编码器，原生整体 rANS 码流。每个 QP 均写出原生 SPS + I 帧头 + 码流，然后从保存的文件
  重新解析并解码每一帧；没有使用编码器内部重构来代替实际码流解码。
- H.264：FFmpeg `libx264`，`preset=slow`、`tune=psnr`、闭合 GOP=30、最多 3 个 B 帧、
  `scenecut=0`、4 CPU 编码线程，8-bit。4:2:0 是常用视频配置，4:4:4 为色度采样参考。
  每个点的实际 I/P/B 数量保存在 [`results.json`](results.json)；未把 H.264 强制改成全 I。
- `slow` + B 帧允许前瞻和重排，本曲线优先比较 RD 效率，不是零延迟实时编码配置。
- 码率 = 完整视频基础码流字节 × 8 / ({count}/14.75)，包含 DCVC SPS/帧头和 H.264
  Annex-B/SPS/PPS/SEI；不含 MP4/WebM 容器、音频、IP/UDP/RTP、FEC 或模型权重。
  完整 rANS 码流与旧容错测试的五个独立分量封装不同，不应直接复用旧封装码率。
- 主 PSNR = `10*log10(255² / 全部 RGB 像素与全部帧的平均平方误差)`；先实际解码、
  转换至 RGB24，再和共享参考比较。mean-frame PSNR 先分别计算每帧 dB 再平均，二者数值不完全相同。
  H.264 的 RGB→YUV→RGB 转换和 4:2:0 色度采样损失包含在指标内；4:4:4 曲线用于检查这一影响。
  CRF=0 仅对编码输入 YUV 无损，RGB 往返仍可能有失真，不能将它解释为 RGB 无损。

## 文件与复现

- [`rd_points.csv`](rd_points.csv)：31 个汇总工作点。
- [`results.json`](results.json)：每点字节数、码流 SHA-256、逐帧 MSE/PSNR、I/P/B 统计。
- [`environment.json`](environment.json)：帧数、帧率、GPU/框架版本、checkpoint/source/reference SHA-256。
- [`commands.json`](commands.json)：本机实际执行的全部命令。
- [`validation.json`](validation.json)：31 份码流大小/哈希、帧数、共享参考核对，以及 FFmpeg 独立 PSNR 复算。
- [`rd_curve.svg`](rd_curve.svg)：可编辑矢量曲线。
- `*_sample.png`：同一中间帧的参考和不同质量重构。
- `bitstreams/`：QP=32 的完整 DCVC 基础码流和 CRF=30 的两种 H.264 基础码流，供实际解码复查。
  其他码流保留在本机 work 目录，均有 SHA-256，可通过脚本重新生成。
- [`SOURCE_LICENSE.md`](SOURCE_LICENSE.md)：视频和派生图像的来源、许可。

从 `DCVC-family/DCVC-RT` 执行（需要 checkpoint、FFmpeg、PyTorch CUDA 和 rANS 扩展）：

```bash
python local_tests/rd_comparison/run_rd_comparison.py \\
  --source local_tests/natural_720p/starigrad_105323_720p.webm \\
  --checkpoint checkpoints/cvpr2025_image.pth.tar \\
  --work /data/dcvc_h264_rd_work \\
  --output local_tests/results/dcvc_i_vs_h264_rd_reproduction
```

Windows 主控 + WSL 推理时，追加
`--wsl --dcvc-python /home/qinyi/.local/share/dcvc-rt/venv/bin/python`。
工作目录保存较大的原始 RGB 和全部码流，结果目录需为新路径。文件读取/编码失败或
解码帧数不一致时脚本会报错；两种编码器的参考 SHA-256 必须一致。

实验结束后执行独立审计（路径与上面相同）：

```bash
python local_tests/rd_comparison/verify_rd_results.py \\
  --output local_tests/results/dcvc_i_vs_h264_rd_reproduction \\
  --work /data/dcvc_h264_rd_work
```

本次设备：{metadata['dcvc_environment']['gpu']}，PyTorch {metadata['dcvc_environment']['torch']}，
CUDA {metadata['dcvc_environment']['cuda']}，自定义 CUDA 算子启用状态：
`{metadata['dcvc_environment']['custom_cuda']}`（False 表示项目 PyTorch fallback）。
H.264 和 DCVC 为节省实验时间并行运行；日志中的耗时仅供诊断，未据此比较 FPS。

官方参数参考：[FFmpeg libx264 文档](https://ffmpeg.org/ffmpeg-codecs.html#libx264_002c-libx264rgb)、
[DCVC-RT 上游](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-RT)。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main(args):
    output = Path(args.output).resolve()
    work = Path(args.work).resolve()
    if output.exists():
        raise ValueError(f"Choose a new results directory: {output}")
    output.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    source = Path(args.source).resolve()
    assert sha256(source) == "aee660c4ef1aa45dc169a0bf12bf2525869136d8dba7c1cce6a9d7b0c9f21bff"
    width, height = 1280, 720
    fps_text = "59/4"
    fps = float(Fraction(fps_text))
    raw = work / "reference.rgb"
    extraction = [args.ffmpeg, "-y", "-hide_banner", "-i", source, "-map", "0:v:0", "-an",
                  "-vf", "fps=59/4,scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                  "-pix_fmt", "rgb24", "-f", "rawvideo", raw]
    run(extraction, output / "reference_decode.log")
    frame_bytes = width * height * 3
    assert raw.stat().st_size % frame_bytes == 0
    count = raw.stat().st_size // frame_bytes
    reference = np.memmap(raw, dtype=np.uint8, mode="r", shape=(count, height, width, 3))
    Image.fromarray(reference[count // 2]).save(output / "reference_sample.png")
    manifest = [{"index": i, "time_seconds": i / fps,
                 "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()}
                for i, frame in enumerate(reference)]
    (output / "reference_frame_manifest.json").write_text(json.dumps(manifest, indent=2))
    worker = Path(__file__).with_name("dcvc_i_worker.py")
    dcvc_work = work / "dcvc"
    worker_args = [args.dcvc_python, linux_path(worker) if args.wsl else str(worker),
                   "--rgb", linux_path(raw) if args.wsl else str(raw),
                   "--checkpoint", linux_path(args.checkpoint) if args.wsl else str(Path(args.checkpoint).resolve()),
                   "--output", linux_path(dcvc_work) if args.wsl else str(dcvc_work)]
    if args.wsl:
        worker_args.insert(0, "wsl")
    worker_log = (output / "dcvc_worker.log").open("wb")
    process = subprocess.Popen(worker_args, stdout=worker_log, stderr=subprocess.STDOUT)
    points = []
    commands = [{"purpose": "canonical RGB reference", "argv": list(map(str, extraction))},
                {"purpose": "DCVC native all-I encode/decode", "argv": worker_args}]
    try:
        for pix_fmt, codec in [("yuv420p", "H.264 4:2:0 inter"),
                               ("yuv444p", "H.264 4:4:4 inter")]:
            for crf in [42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 0]:
                name = f"h264_{pix_fmt}_crf_{crf:02d}"
                stream = work / f"{name}.h264"
                command = [args.ffmpeg, "-y", "-hide_banner", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-video_size", f"{width}x{height}", "-framerate", fps_text, "-i", raw,
                           "-an", "-vf", f"scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format={pix_fmt}",
                           "-c:v", "libx264", "-preset", "slow", "-tune", "psnr", "-crf", str(crf),
                           "-pix_fmt", pix_fmt, "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                           "-bf", "3", "-threads", "4", "-x264-params", "open-gop=0:repeat-headers=1",
                           "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                           "-color_trc", "bt709", "-f", "h264", stream]
                start = time.perf_counter()
                run(command, output / f"{name}_encode.log")
                encode_seconds = time.perf_counter() - start
                decoded = work / f"decoded_{pix_fmt}.rgb"
                decode = [args.ffmpeg, "-y", "-hide_banner", "-i", stream, "-an",
                          "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                          "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", decoded]
                run(decode, output / f"{name}_decode.log")
                assert decoded.stat().st_size == raw.stat().st_size
                reconstruction = np.memmap(decoded, dtype=np.uint8, mode="r", shape=reference.shape)
                point = summarize(codec, crf, stream, reference, reconstruction)
                probe = json.loads(run([args.ffprobe, "-v", "error", "-show_frames", "-show_entries",
                                        "frame=pict_type", "-of", "json", stream]))
                point["frame_types"] = dict(Counter(f["pict_type"] for f in probe["frames"]))
                assert sum(point["frame_types"].values()) == count
                assert point["frame_types"].get("P", 0) > 0
                point["encode_wall_seconds"] = encode_seconds
                point["bitrate_kbps"] = point["bytes"] * 8 * fps / count / 1000
                points.append(point)
                if crf in (38, 26, 14):
                    Image.fromarray(reconstruction[count // 2]).save(output / f"{name}_sample.png")
                del reconstruction
                commands.extend([{"purpose": name + " encode", "argv": list(map(str, command))},
                                 {"purpose": name + " decode", "argv": list(map(str, decode))}])
                print(f"{name}: {point['bitrate_kbps']:.1f} kbps, "
                      f"{point['psnr_rgb_pooled_db']:.3f} dB", flush=True)
        returncode = process.wait()
        if returncode:
            raise RuntimeError(f"DCVC worker exited {returncode}; see dcvc_worker.log")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        worker_log.close()
    dcvc = json.loads((dcvc_work / "dcvc_results.json").read_text())
    assert dcvc["environment"]["reference_sha256"] == sha256(raw)
    for point in dcvc["points"]:
        point["bitrate_kbps"] = point["bytes"] * 8 * fps / count / 1000
        points.append(point)
    for sample in dcvc_work.glob("*_sample.png"):
        (output / sample.name).write_bytes(sample.read_bytes())
    probe_source = json.loads(run([args.ffprobe, "-v", "error", "-count_frames", "-show_format", "-show_streams",
                                   "-of", "json", source]))
    metadata = {"source_sha256": sha256(source), "source_probe": probe_source,
                "reference_sha256": sha256(raw), "reference_frames": count,
                "width": width, "height": height, "fps": fps, "fps_rational": fps_text,
                "duration_seconds": count / fps, "dcvc_environment": dcvc["environment"],
                "host": platform.platform(), "python": sys.version,
                "ffmpeg_version": run([args.ffmpeg, "-version"]).decode(errors="replace"),
                "git_commit_before_experiment": run(["git", "rev-parse", "HEAD"]).decode().strip(),
                "method": {"psnr": "10*log10(255^2 / mean RGB squared error over all frames and pixels)",
                           "bitrate": "8 * complete elementary stream bytes / (frame_count / fps)",
                           "h264": "libx264 slow tune=psnr GOP=30 Bframes=3 scenecut=0 threads=4 closed GOP",
                           "dcvc": "DMCI official checkpoint; all-I native monolithic rANS; SPS + per-frame headers",
                           "color": "shared RGB24; explicit BT.709 limited YUV/full RGB transforms",
                           "loss": "no packet loss", "audio": "excluded"}}
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    (output / "commands.json").write_text(json.dumps(commands, indent=2))
    (output / "results.json").write_text(json.dumps(points, indent=2))
    columns = ["codec", "quality", "frames", "bytes", "bitrate_kbps", "psnr_rgb_pooled_db",
               "psnr_rgb_mean_frame_db", "mean_mse_rgb", "stream_sha256"]
    with (output / "rd_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    make_plots(points, output)
    bitstreams = output / "bitstreams"
    bitstreams.mkdir()
    for source_stream in (dcvc_work / "dcvc_qp_32.bin", work / "h264_yuv420p_crf_30.h264",
                          work / "h264_yuv444p_crf_30.h264"):
        shutil.copyfile(source_stream, bitstreams / source_stream.name)
    make_report(points, metadata, output)
    print(f"Completed {len(points)} operating points on {count} frames: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--dcvc-python", default=sys.executable)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    main(parser.parse_args())
