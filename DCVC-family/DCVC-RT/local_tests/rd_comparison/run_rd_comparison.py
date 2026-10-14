"""Measure DCVC all-I/H.264 RD for a user video, including MP4 and old FFmpeg."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
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


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run(command, log=None, timeout=1800, idle_timeout=0, merge_output=False):
    """Use regular files instead of pipes; show heartbeats and bound waits."""
    command = list(map(str, command))
    label = Path(log).name if log else Path(command[0]).name
    print(f"START {label}" + (f" | log={log}" if log else ""), flush=True)
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
               else {"start_new_session": True})
    with tempfile.TemporaryFile() as stdout, (
            Path(log).open("w+b") if log else tempfile.TemporaryFile()) as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=stderr if merge_output else stdout,
                                   stderr=stderr, **options)
        start = last_activity = heartbeat = time.monotonic()
        last_size = 0
        try:
            while process.poll() is None:
                now = time.monotonic()
                size = os.fstat(stderr.fileno()).st_size + os.fstat(stdout.fileno()).st_size
                if size != last_size:
                    last_activity, last_size = now, size
                if now - start > timeout or (idle_timeout and now - last_activity > idle_timeout):
                    reason = "command timeout" if now - start > timeout else "no log/output activity"
                    raise TimeoutError(f"{label}: {reason}; elapsed={now-start:.1f}s; log={log}")
                if now - heartbeat >= 15:
                    print(f"WAIT {label}: {now-start:.0f}s, log/output={size} bytes", flush=True)
                    heartbeat = now
                time.sleep(0.1)
        except BaseException:
            stop_process(process)
            raise
        if process.returncode:
            stderr.seek(max(0, os.fstat(stderr.fileno()).st_size - 8192))
            raise RuntimeError(f"{label} exited {process.returncode}; log={log}\n"
                               + stderr.read().decode(errors="replace"))
        stdout.seek(0)
        return stdout.read()


def linux_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}/{path.as_posix()[3:]}" if sys.platform == "win32" else str(path)


def record_command(commands, output, purpose, command):
    commands.append({"purpose": purpose, "argv": list(map(str, command))})
    (output / "commands.json").write_text(json.dumps(commands, indent=2, ensure_ascii=False), encoding="utf-8")


def rgb_psnr(mse):
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def summarize(codec, quality, stream, reference, reconstructed):
    if reconstructed.shape != reference.shape:
        raise ValueError("Reference and reconstruction shapes differ")
    rows = []
    for index, (ref, rec) in enumerate(zip(reference, reconstructed)):
        error = ref.astype(np.float64) - rec.astype(np.float64)
        mse = float(np.mean(error * error))
        rows.append({"frame_index": index, "mse_rgb": mse, "psnr_rgb_db": rgb_psnr(mse)})
    mse = float(np.mean([row["mse_rgb"] for row in rows]))
    return {"codec": codec, "quality": quality, "frames": len(rows),
            "bytes": stream.stat().st_size, "stream_sha256": sha256(stream),
            "psnr_rgb_pooled_db": rgb_psnr(mse),
            "psnr_rgb_mean_frame_db": float(np.mean([row["psnr_rgb_db"] for row in rows])),
            "mean_mse_rgb": mse, "per_frame": rows}


def make_plots(points, output, metadata=None):
    metadata = metadata or {}
    styles = {"DCVC-RT all-I": ("o", "-", "#1764ab"),
              "H.264 4:2:0 inter": ("s", "-", "#d54e25"),
              "H.264 4:4:4 inter": ("^", "--", "#18845e")}
    title = (f"DCVC-RT all-I vs H.264 | {metadata.get('width', 1280)} x "
             f"{metadata.get('height', 720)} | {metadata.get('fps', 14.75):g} fps | "
             f"{metadata.get('reference_frames', points[0]['frames'])} frames")
    for logarithmic in (False, True):
        fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
        for codec, (marker, line, color) in styles.items():
            subset = sorted([p for p in points if p["codec"] == codec], key=lambda p: p["bitrate_kbps"])
            if subset:
                rates = np.asarray([p["bitrate_kbps"] / 1000 for p in subset])
                scores = np.asarray([p["psnr_rgb_pooled_db"] for p in subset])
                ax.scatter(rates, scores, marker=marker, color=color, label=codec)
                finite = np.isfinite(scores)
                if finite.sum() >= 2:
                    grid = np.geomspace(rates[finite].min(), rates[finite].max(), 200)
                    ax.plot(grid, np.interp(np.log(grid), np.log(rates[finite]), scores[finite]),
                            linestyle=line, color=color)
        ax.set(xlabel="Actual video bitrate (Mbit/s)", ylabel="RGB PSNR, pooled MSE (dB)", title=title)
        if logarithmic:
            ax.set_xscale("log")
        else:
            upper = max(p["bitrate_kbps"] for p in points if p["codec"] == "DCVC-RT all-I") / 1000 * 1.1
            shared = [p["psnr_rgb_pooled_db"] for p in points
                      if p["bitrate_kbps"] / 1000 <= upper and math.isfinite(p["psnr_rgb_pooled_db"])]
            ax.set_xlim(0, upper)
            if shared:
                ax.set_ylim(min(shared) - 1, max(shared) + 1)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="lower right")
        fig.text(0.5, 0.005, "Markers: measured points; lines: linear interpolation in log(bitrate)",
                 ha="center", va="bottom", fontsize=7)
        fig.savefig(output / ("rd_curve_log.png" if logarithmic else "rd_curve.png"), dpi=180)
        if not logarithmic:
            fig.savefig(output / "rd_curve.svg")
        plt.close(fig)


def make_report(points, metadata, output):
    measured = "\n".join(
        f"| {p['codec']} | {p['quality']} | {p['bitrate_kbps']:.1f} | "
        f"{p['psnr_rgb_pooled_db']:.3f} | {p['psnr_rgb_mean_frame_db']:.3f} |"
        for p in sorted(points, key=lambda p: (p["codec"], p["bitrate_kbps"])))
    groups = {name: sorted([p for p in points if p["codec"] == name], key=lambda p: p["bitrate_kbps"])
              for name in ("DCVC-RT all-I", "H.264 4:2:0 inter")}
    deltas = []
    for p in groups["DCVC-RT all-I"]:
        h264 = groups["H.264 4:2:0 inter"]
        if h264 and h264[0]["bitrate_kbps"] <= p["bitrate_kbps"] <= h264[-1]["bitrate_kbps"]:
            score = np.interp(np.log(p["bitrate_kbps"]), np.log([x["bitrate_kbps"] for x in h264]),
                              [x["psnr_rgb_pooled_db"] for x in h264])
            deltas.append(score - p["psnr_rgb_pooled_db"])
    comparison = (f"在 {len(deltas)} 个共同码率范围内的 DCVC 点上，H.264 4:2:0 减去 DCVC 的 PSNR 差值为 "
                  f"{min(deltas):.2f}～{max(deltas):.2f} dB（H.264 按 log(码率) 插值，正值表示 H.264 更高）。"
                  if deltas else "两条主曲线没有重叠码率工作点，未计算相同码率的插值差值。")
    method = metadata.get("method", {})
    text = f"""# DCVC-RT 全 I 帧与 H.264 视频 RD 对比

输入：`{metadata.get('source_name', 'original experiment source')}`。
共 {len(points)} 个真实编码工作点，{metadata['reference_frames']} 帧，
{metadata['width']}×{metadata['height']}，{metadata['fps_rational']} FPS，
时长 {metadata['duration_seconds']:.6f} 秒。两种编码器使用同一份解码后的 RGB24 参考。

{comparison}

![RD 曲线](rd_curve.png)

![对数码率 RD 曲线](rd_curve_log.png)

| 编码器 | QP / CRF | 实际码率 kbps | pooled RGB PSNR dB | mean-frame RGB PSNR dB |
|---|---:|---:|---:|---:|
{measured}

## 方法

- DCVC：官方 DMCI、FP16、force_zero_thres=0.12，每帧独立编码；实际保存 SPS、I 帧头及 rANS 码流并重新读取解码。
- H.264：libx264，slow + tune=psnr，闭合 GOP=30，最多 3 个 B 帧，scenecut=0，4 个编码线程。
  帧间预测利用时间冗余，本结果不评价 DCVC 的 P 帧模型，也不代表 H.264 零延迟配置。
- CFR 参考采用源帧率或用户指定帧率；依据时间戳复制/丢弃帧。尺寸、片段和色彩转换见 environment.json / commands.json。
- 色彩处理：{method.get('color', 'explicit BT.709 YUV/RGB conversion')}。
  输入 MP4 若已压缩，本次 PSNR 相对于其解码图像，不是相机无压缩母版。
- 码率按完整基础码流字节×8/共同片段时长计算，含 DCVC 帧头和 H.264 SPS/PPS/SEI，
  不含音频、容器、传输头、FEC 或模型权重。无丢包。
- 主 PSNR 对所有 RGB 像素和帧的 MSE 汇总后取 dB；另存逐帧 dB 均值。
  H.264 的色彩转换/色度采样失真也计入。CRF=0 对输入 YUV 无损，RGB 往返不一定无损。
- DCVC QP 越大质量越高；H.264 CRF 越小质量越高，数字不能跨模型比较。

## 证据与复现

- [rd_points.csv](rd_points.csv)、[results.json](results.json)：汇总及逐帧结果。
- [environment.json](environment.json)、[commands.json](commands.json)：软件、数据、参数和全部命令。
- [reference_frame_manifest.json](reference_frame_manifest.json)：每帧 RGB SHA-256。
- `bitstreams/` 和 `*_sample.png`：代表性真实码流与中间帧重构。
- [使用指南](https://github.com/qinyishcn/DCVC/blob/main/DCVC-family/DCVC-RT/local_tests/rd_comparison/README.md)：替换 MP4、运行、验证、仅重新出图的步骤。

源文件 SHA-256：`{metadata['source_sha256']}`。

共享 RGB SHA-256：`{metadata['reference_sha256']}`。

checkpoint SHA-256：`{metadata['dcvc_environment']['checkpoint_sha256']}`。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def probe_video(source, args, output):
    command = [args.ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", source]
    probe = json.loads(run(command, output / "source_probe.log", timeout=args.command_timeout))
    (output / "source_probe.json").write_text(json.dumps(probe, indent=2, ensure_ascii=False), encoding="utf-8")
    videos = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic", 0)]
    if not videos:
        raise ValueError("Input has no video stream")
    video = videos[0]
    if video.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise ValueError("HDR input: first create an SDR copy with deliberate tone mapping")
    width, height = args.width or int(video["width"]), args.height or int(video["height"])
    if width % 2 or height % 2:
        raise ValueError("Use even --width and --height for the H.264 4:2:0 comparison")
    fps_text = args.fps or video.get("avg_frame_rate", "0/0")
    if fps_text in ("0/0", "0", "N/A"):
        fps_text = video.get("r_frame_rate", "0/0")
    try:
        fps = Fraction(fps_text)
        if not 0 < fps <= 240:
            raise ValueError()
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid or unusual FPS {fps_text!r}; specify --fps, e.g. 30") from None
    return probe, video, width, height, str(fps)


def main(args):
    output = Path(args.output).resolve()
    if args.plot_only:
        points = json.loads((output / "results.json").read_text(encoding="utf-8"))
        metadata = json.loads((output / "environment.json").read_text(encoding="utf-8"))
        make_plots(points, output, metadata)
        print(f"Plots regenerated: {output / 'rd_curve.png'}")
        return
    work = Path(args.work).resolve()
    source = Path(args.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if output.exists() or work.exists():
        raise ValueError("Use new --output and --work directories to avoid mixing different videos/runs")
    output.mkdir(parents=True)
    work.mkdir(parents=True)
    commands = []
    preflight = [args.dcvc_python, "-c", "import torch, MLCodec_extensions_cpp; "
                 "assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"]
    if args.wsl:
        preflight.insert(0, "wsl")
    record_command(commands, output, "DCVC environment check", preflight)
    run(preflight, output / "dcvc_preflight.log", timeout=min(args.command_timeout, 120), merge_output=True)
    probe, video, width, height, fps_text = probe_video(source, args, output)
    fps = float(Fraction(fps_text))
    matrix = args.input_color_matrix or {"bt709": "bt709", "smpte170m": "bt601",
                                        "bt470bg": "bt601", "bt2020nc": "bt2020"}.get(video.get("color_space"), "bt709")
    color_range = args.input_range or ("pc" if video.get("color_range") == "pc" else "tv")
    raw = work / "reference.rgb"
    # Disable autorotation; the probe and raw decoder use the same coded dimensions.
    filters = (f"setpts=PTS-STARTPTS,fps={fps_text},"
               f"scale={width}:{height}:in_color_matrix={matrix}:in_range={color_range}:out_range=pc,"
               "setsar=1,format=rgb24")
    extraction = [args.ffmpeg, "-nostdin", "-y", "-hide_banner", "-threads", "4",
                  "-filter_threads", "1", "-ss", str(args.start), "-noautorotate", "-i", source,
                  "-map", f"0:{video['index']}", "-an", "-sn", "-dn", "-vf", filters,
                  "-vsync", "0"]
    if args.duration:
        extraction += ["-t", str(args.duration)]
    if args.max_frames:
        extraction += ["-frames:v", str(args.max_frames)]
    extraction += ["-pix_fmt", "rgb24", "-f", "rawvideo", raw]
    record_command(commands, output, "canonical RGB reference", extraction)
    run(extraction, output / "reference_decode.log", args.command_timeout, args.idle_timeout)
    frame_bytes = width * height * 3
    if not raw.stat().st_size or raw.stat().st_size % frame_bytes:
        raise ValueError("Empty/incomplete RGB reference; check source_probe.log and reference_decode.log")
    count = raw.stat().st_size // frame_bytes
    print(f"REFERENCE: {source.name} | {width}x{height} | {fps_text} fps | {count} frames | "
          f"{raw.stat().st_size / 2**20:.1f} MiB RGB", flush=True)
    reference = np.memmap(raw, dtype=np.uint8, mode="r", shape=(count, height, width, 3))
    Image.fromarray(reference[count // 2]).save(output / "reference_sample.png")
    manifest = [{"index": i, "time_seconds": args.start + i / fps,
                 "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest()} for i, frame in enumerate(reference)]
    (output / "reference_frame_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    points = []
    for pix_fmt in args.h264_formats:
        codec = "H.264 4:2:0 inter" if pix_fmt == "yuv420p" else "H.264 4:4:4 inter"
        for crf in args.crfs:
            name = f"h264_{pix_fmt}_crf_{crf:02d}"
            stream = work / f"{name}.h264"
            command = [args.ffmpeg, "-nostdin", "-y", "-hide_banner", "-filter_threads", "1",
                       "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
                       "-framerate", fps_text, "-i", raw, "-frames:v", str(count), "-an",
                       "-vf", f"scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format={pix_fmt}",
                       "-c:v", "libx264", "-preset", "slow", "-tune", "psnr", "-crf", str(crf),
                       "-pix_fmt", pix_fmt, "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                       "-bf", "3", "-threads", "4", "-x264-params", "open-gop=0:repeat-headers=1",
                       "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
                       "-color_trc", "bt709", "-vsync", "0", "-f", "h264", stream]
            start = time.perf_counter()
            record_command(commands, output, name + " encode", command)
            run(command, output / f"{name}_encode.log", args.command_timeout, args.idle_timeout)
            encode_seconds = time.perf_counter() - start
            decoded = work / f"decoded_{pix_fmt}.rgb"
            decode = [args.ffmpeg, "-nostdin", "-y", "-hide_banner", "-threads", "4",
                      "-filter_threads", "1", "-i", stream, "-an",
                      "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24",
                      "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", decoded]
            record_command(commands, output, name + " decode", decode)
            run(decode, output / f"{name}_decode.log", args.command_timeout, args.idle_timeout)
            if decoded.stat().st_size != raw.stat().st_size:
                raise ValueError(f"{name}: decoded frame count/size differs from reference; see decode log")
            reconstruction = np.memmap(decoded, dtype=np.uint8, mode="r", shape=reference.shape)
            point = summarize(codec, crf, stream, reference, reconstruction)
            frames = json.loads(run([args.ffprobe, "-v", "error", "-show_frames", "-show_entries",
                                     "frame=pict_type", "-of", "json", stream], timeout=args.command_timeout))
            point["frame_types"] = dict(Counter(f["pict_type"] for f in frames["frames"]))
            if sum(point["frame_types"].values()) != count:
                raise ValueError(f"{name}: ffprobe frame count differs")
            point["encode_wall_seconds"] = encode_seconds
            point["bitrate_kbps"] = point["bytes"] * 8 * fps / count / 1000
            points.append(point)
            Image.fromarray(reconstruction[count // 2]).save(output / f"{name}_sample.png")
            del reconstruction
            print(f"DONE {name}: {point['bitrate_kbps']:.1f} kbps, {point['psnr_rgb_pooled_db']:.3f} dB", flush=True)
    worker = Path(__file__).with_name("dcvc_i_worker.py")
    dcvc_work = work / "dcvc"
    convert = linux_path if args.wsl else lambda p: str(Path(p).resolve())
    worker_args = [args.dcvc_python, convert(worker), "--rgb", convert(raw),
                   "--checkpoint", convert(args.checkpoint), "--output", convert(dcvc_work),
                   "--width", str(width), "--height", str(height), "--qps", *map(str, args.qps)]
    if args.wsl:
        worker_args.insert(0, "wsl")
    record_command(commands, output, "DCVC native all-I encode/decode", worker_args)
    run(worker_args, output / "dcvc_worker.log", args.worker_timeout, merge_output=True)
    dcvc = json.loads((dcvc_work / "dcvc_results.json").read_text(encoding="utf-8"))
    reference_hash = sha256(raw)
    if dcvc["environment"]["reference_sha256"] != reference_hash:
        raise ValueError("DCVC/H.264 reference hashes differ")
    for point in dcvc["points"]:
        point["bitrate_kbps"] = point["bytes"] * 8 * fps / count / 1000
        points.append(point)
    for sample in dcvc_work.glob("*_sample.png"):
        shutil.copyfile(sample, output / sample.name)
    metadata = {"source_name": source.name, "source_sha256": sha256(source), "source_probe": probe,
                "reference_sha256": reference_hash, "reference_frames": count, "width": width, "height": height,
                "fps": fps, "fps_rational": fps_text, "duration_seconds": count / fps, "start_seconds": args.start,
                "dcvc_environment": dcvc["environment"], "qps": args.qps, "crfs": args.crfs,
                "h264_formats": args.h264_formats, "host": platform.platform(), "python": sys.version,
                "ffmpeg_version": run([args.ffmpeg, "-version"], timeout=args.command_timeout).decode(errors="replace"),
                "method": {"bitrate": "8 * complete elementary stream bytes / (frame_count / fps)",
                           "color": f"input {matrix}/{color_range} -> shared RGB24 -> codec; H.264 BT.709 limited YUV",
                           "resize": "coded orientation (noautorotate); explicit width/height scale; square pixels",
                           "ffmpeg_sync": "-vsync 0; no fps_mode option", "execution": "H.264 then DCVC, sequential"}}
    for filename, value in (("environment.json", metadata), ("commands.json", commands), ("results.json", points)):
        (output / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["codec", "quality", "frames", "bytes", "bitrate_kbps", "psnr_rgb_pooled_db",
               "psnr_rgb_mean_frame_db", "mean_mse_rgb", "stream_sha256"]
    with (output / "rd_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(points)
    make_plots(points, output, metadata)
    make_report(points, metadata, output)
    bitstreams = output / "bitstreams"
    bitstreams.mkdir()
    representative_qp = args.qps[len(args.qps) // 2]
    representative_crf = args.crfs[len(args.crfs) // 2]
    for source_stream in ([dcvc_work / f"dcvc_qp_{representative_qp:02d}.bin"] +
                          [work / f"h264_{fmt}_crf_{representative_crf:02d}.h264" for fmt in args.h264_formats]):
        shutil.copyfile(source_stream, bitstreams / source_stream.name)
    print(f"COMPLETED {len(points)} points, {count} frames | plot={output / 'rd_curve.png'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dcvc-python", default=sys.executable)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--fps", help="CFR rate; default: source avg_frame_rate, e.g. 30 or 30000/1001")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--input-color-matrix", choices=["bt709", "bt601", "bt2020"])
    parser.add_argument("--input-range", choices=["tv", "pc"])
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 8, 16, 24, 32, 40, 48, 56, 63])
    parser.add_argument("--crfs", type=int, nargs="+", default=[42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 0])
    parser.add_argument("--h264-formats", nargs="+", choices=["yuv420p", "yuv444p"], default=["yuv420p"])
    parser.add_argument("--command-timeout", type=float, default=1800, help="Total seconds per FFmpeg/ffprobe command")
    parser.add_argument("--idle-timeout", type=float, default=120, help="FFmpeg inactivity seconds; 0 disables idle check")
    parser.add_argument("--worker-timeout", type=float, default=7200, help="Total seconds for all DCVC points")
    args = parser.parse_args()
    if not args.plot_only:
        if not all((args.source, args.checkpoint, args.work)):
            parser.error("--source, --checkpoint and --work are required unless --plot-only is used")
        if bool(args.width) != bool(args.height):
            parser.error("Use --width and --height together")
        if args.start < 0 or any(v is not None and v <= 0 for v in (args.width, args.height, args.duration, args.max_frames)):
            parser.error("Sizes/duration/max-frames must be positive and start must be nonnegative")
        if any(q < 0 or q > 63 for q in args.qps) or any(c < 0 or c > 51 for c in args.crfs):
            parser.error("DCVC QP must be 0..63; H.264 CRF must be 0..51")
        if any(len(v) != len(set(v)) for v in (args.qps, args.crfs, args.h264_formats)):
            parser.error("Quality grids and H.264 formats must not contain duplicates")
        if args.command_timeout <= 0 or args.worker_timeout <= 0 or args.idle_timeout < 0:
            parser.error("Timeouts must be positive; idle-timeout can be 0")
    main(args)
