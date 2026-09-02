import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from compressai.ops import compute_padding
from compressai.zoo import bmshj2018_hyperprior, mbt2018_mean


MODEL_FACTORIES = {
    "bmshj2018-hyperprior": bmshj2018_hyperprior,
    "mbt2018-mean": mbt2018_mean,
}
CHECKPOINT_NAMES = {
    ("bmshj2018-hyperprior", 1): "bmshj2018-hyperprior-1-7eb97409.pth.tar",
    ("bmshj2018-hyperprior", 4): "bmshj2018-hyperprior-4-de1b779c.pth.tar",
    ("bmshj2018-hyperprior", 6): "bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
    ("mbt2018-mean", 1): "mbt2018-mean-1-e522738d.pth.tar",
    ("mbt2018-mean", 4): "mbt2018-mean-4-6dba02a3.pth.tar",
    ("mbt2018-mean", 6): "mbt2018-mean-6-a19628ab.pth.tar",
}
COMPONENT_NAMES = ("z", "y0", "y1", "y2", "y3")


def load_image(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0) / 255.0
    pad, unpad = compute_padding(rgb.shape[0], rgb.shape[1], min_div=64)
    return rgb, F.pad(tensor, pad).cuda(), unpad


def to_rgb(x_hat, unpad):
    cropped = F.pad(x_hat, unpad).clamp(0, 1)
    return (cropped[0].cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def psnr(reference, reconstructed):
    mse = np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_components(model, model_name, x):
    y = model.g_a(x)
    z = model.h_a(torch.abs(y)) if model_name == "bmshj2018-hyperprior" else model.h_a(y)
    z_stream = model.entropy_bottleneck.compress(z)[0]
    z_hat = model.entropy_bottleneck.decompress([z_stream], z.size()[-2:])

    params = model.h_s(z_hat)
    if model_name == "bmshj2018-hyperprior":
        scales_hat = params
        means_hat = None
    else:
        scales_hat, means_hat = params.chunk(2, 1)
    indexes = model.gaussian_conditional.build_indexes(scales_hat)

    channel_groups = [torch.arange(offset, y.shape[1], 4, device=y.device)
                      for offset in range(4)]
    y_streams = []
    for channels in channel_groups:
        y_group = torch.index_select(y, 1, channels)
        index_group = torch.index_select(indexes, 1, channels)
        mean_group = None if means_hat is None else torch.index_select(means_hat, 1, channels)
        stream = model.gaussian_conditional.compress(
            y_group, index_group, means=mean_group)[0]
        y_streams.append(stream)

    return {
        "z_shape": list(z.size()[-2:]),
        "z_stream": z_stream,
        "y_streams": y_streams,
        "channel_groups": channel_groups,
    }


def decode_prefix(model, model_name, state, available_groups):
    z_hat = model.entropy_bottleneck.decompress([state["z_stream"]], state["z_shape"])
    params = model.h_s(z_hat)
    if model_name == "bmshj2018-hyperprior":
        scales_hat = params
        means_hat = None
        y_hat = torch.zeros_like(scales_hat)
    else:
        scales_hat, means_hat = params.chunk(2, 1)
        y_hat = means_hat.clone()
    indexes = model.gaussian_conditional.build_indexes(scales_hat)

    for group_index in range(available_groups):
        channels = state["channel_groups"][group_index]
        index_group = torch.index_select(indexes, 1, channels)
        mean_group = None if means_hat is None else torch.index_select(means_hat, 1, channels)
        decoded = model.gaussian_conditional.decompress(
            [state["y_streams"][group_index]], index_group,
            dtype=z_hat.dtype, means=mean_group)
        y_hat[:, channels, :, :] = decoded
    return model.g_s(y_hat).clamp_(0, 1)


def frame_payload(component_streams, packet_payload_bytes):
    header = struct.pack(">5I", *(len(stream) for stream in component_streams))
    framed = header + b"".join(component_streams)
    packets = [framed[index:index + packet_payload_bytes]
               for index in range(0, len(framed), packet_payload_bytes)]
    component_ranges = []
    offset = len(header)
    for name, stream in zip(COMPONENT_NAMES, component_streams):
        component_ranges.append({
            "name": name,
            "start": offset,
            "end": offset + len(stream),
            "bytes": len(stream),
            "sha256": hashlib.sha256(stream).hexdigest(),
        })
        offset += len(stream)
    return framed, packets, component_ranges


def available_stage(retained_bytes, component_ranges, header_bytes=20):
    if retained_bytes < header_bytes or retained_bytes < component_ranges[0]["end"]:
        return "concealment", False, 0
    available_y = 0
    for component in component_ranges[1:]:
        if retained_bytes >= component["end"]:
            available_y += 1
        else:
            break
    stage = "z" if available_y == 0 else "z+" + "+".join(
        f"y{index}" for index in range(available_y))
    return stage, True, available_y


def round_half_up(value):
    return int(math.floor(value + 0.5))


def allocate_drop_counts(packet_counts, target_rate):
    raw = [count * target_rate / 100.0 for count in packet_counts]
    allocated = [min(count, int(math.floor(value)))
                 for count, value in zip(packet_counts, raw)]
    target_total = min(sum(packet_counts), round_half_up(sum(packet_counts) * target_rate / 100.0))
    remaining = target_total - sum(allocated)
    order = sorted(range(len(packet_counts)),
                   key=lambda index: (raw[index] - math.floor(raw[index]), -index),
                   reverse=True)
    for index in order[:remaining]:
        if allocated[index] < packet_counts[index]:
            allocated[index] += 1
    return allocated


def native_stream_bytes(encoded):
    return sum(len(stream) for group in encoded["strings"] for stream in group)


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main(args):
    frame_paths = sorted(Path(args.frames).glob("*.png"))[:args.frame_count]
    if len(frame_paths) != args.frame_count:
        raise ValueError(f"expected {args.frame_count} PNG frames, found {len(frame_paths)}")

    output_dir = Path(args.output)
    reconstruction_dir = output_dir / "reconstructions"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)

    model = MODEL_FACTORIES[args.model](
        quality=args.qp, metric="mse", pretrained=True, progress=True).eval().cuda()
    model.update(force=True)

    frame_records = []
    frame_caches = []
    encode_times = []
    decode_times = []
    sample_manifest = None
    sample_images = None

    for frame_index, frame_path in enumerate(frame_paths):
        reference, x, unpad = load_image(frame_path)
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.perf_counter()
            native = model.compress(x)
            torch.cuda.synchronize()
            encode_times.append(time.perf_counter() - start)

            torch.cuda.synchronize()
            start = time.perf_counter()
            native_hat = model.decompress(native["strings"], native["shape"])["x_hat"]
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - start)

            state = encode_components(model, args.model, x)
            streams = [state["z_stream"], *state["y_streams"]]
            framed, packets, component_ranges = frame_payload(streams, args.packet_payload_bytes)

            stage_psnr = {}
            stage_images = {}
            for available_groups in range(5):
                reconstructed = decode_prefix(model, args.model, state, available_groups)
                torch.cuda.synchronize()
                rgb = to_rgb(reconstructed, unpad)
                stage = "z" if available_groups == 0 else "z+" + "+".join(
                    f"y{index}" for index in range(available_groups))
                stage_psnr[stage] = psnr(reference, rgb)
                if frame_index == args.sample_frame_index:
                    stage_images[stage] = rgb

        native_rgb = to_rgb(native_hat, unpad)
        full_rgb = stage_images.get("z+y0+y1+y2+y3") if frame_index == args.sample_frame_index else None
        native_psnr = psnr(reference, native_rgb)
        full_psnr = stage_psnr["z+y0+y1+y2+y3"]
        full_psnr_delta = abs(native_psnr - full_psnr)
        if full_psnr_delta > 1e-4:
            raise RuntimeError(
                f"custom full-stream reconstruction differs from native decode: "
                f"{native_psnr} vs {full_psnr}")

        gray = np.full_like(reference, 128)
        concealment_psnr = psnr(reference, gray)
        retained_prefix_bytes = [0]
        for packet in packets:
            retained_prefix_bytes.append(retained_prefix_bytes[-1] + len(packet))

        frame_records.append({
            "frame": frame_path.name,
            "native_bytes": native_stream_bytes(native),
            "framed_bytes": len(framed),
            "packet_count": len(packets),
            "component_bytes": {item["name"]: item["bytes"] for item in component_ranges},
            "native_psnr_db": native_psnr,
            "full_custom_psnr_db": full_psnr,
            "full_custom_psnr_abs_delta_db": full_psnr_delta,
            "stage_psnr_db": stage_psnr,
            "concealment_psnr_db": concealment_psnr,
        })
        frame_caches.append({
            "stage_psnr": stage_psnr,
            "concealment_psnr": concealment_psnr,
            "packet_count": len(packets),
            "retained_prefix_bytes": retained_prefix_bytes,
            "component_ranges": component_ranges,
        })

        if frame_index == args.sample_frame_index:
            sample_images = {"reference": reference, "concealment": gray, **stage_images}
            sample_manifest = {
                "frame": frame_path.name,
                "header_bytes": 20,
                "framed_bytes": len(framed),
                "framed_sha256": hashlib.sha256(framed).hexdigest(),
                "component_ranges": component_ranges,
                "packets": [{
                    "index": index,
                    "start": index * args.packet_payload_bytes,
                    "end": index * args.packet_payload_bytes + len(packet),
                    "bytes": len(packet),
                    "sha256": hashlib.sha256(packet).hexdigest(),
                } for index, packet in enumerate(packets)],
            }

    loss_results = []
    sample_loss_results = []
    packet_counts = [cache["packet_count"] for cache in frame_caches]
    total_packets = sum(packet_counts)
    total_bytes = sum(record["framed_bytes"] for record in frame_records)

    for target_rate in args.loss_rates:
        drop_counts = allocate_drop_counts(packet_counts, target_rate)
        psnrs = []
        decoded_frames = 0
        retained_total = 0
        stage_counts = {}

        for frame_index, (cache, dropped) in enumerate(zip(frame_caches, drop_counts)):
            kept_packets = cache["packet_count"] - dropped
            retained_bytes = cache["retained_prefix_bytes"][kept_packets]
            retained_total += retained_bytes
            stage, decodable, _ = available_stage(retained_bytes, cache["component_ranges"])
            value = cache["concealment_psnr"] if stage == "concealment" else cache["stage_psnr"][stage]
            psnrs.append(value)
            decoded_frames += int(decodable)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

            if frame_index == args.sample_frame_index:
                file_stage = stage.replace("+", "_")
                image_path = reconstruction_dir / f"plr_{target_rate:05.1f}_{file_stage}.png"
                save_rgb(image_path, sample_images[stage])
                sample_loss_results.append({
                    "target_loss_percent": target_rate,
                    "packet_count": cache["packet_count"],
                    "dropped_packets": dropped,
                    "actual_packet_loss_percent": 100.0 * dropped / cache["packet_count"],
                    "retained_bytes": retained_bytes,
                    "stage": stage,
                    "psnr_db": value,
                    "image": str(image_path.relative_to(output_dir)),
                })

        actual_dropped = sum(drop_counts)
        loss_results.append({
            "target_packet_loss_percent": target_rate,
            "actual_packet_loss_percent": 100.0 * actual_dropped / total_packets,
            "received_bytes_percent": 100.0 * retained_total / total_bytes,
            "mean_psnr_db": float(np.mean(psnrs)),
            "minimum_psnr_db": float(np.min(psnrs)),
            "maximum_psnr_db": float(np.max(psnrs)),
            "decodable_frame_percent": 100.0 * decoded_frames / len(frame_paths),
            "stage_frame_counts": stage_counts,
        })

    save_rgb(reconstruction_dir / "reference.png", sample_images["reference"])
    (output_dir / "sample_packet_manifest.json").write_text(
        json.dumps(sample_manifest, indent=2), encoding="utf-8")

    warmup = min(args.warmup_frames, max(0, len(frame_paths) - 1))
    enc_steady = encode_times[warmup:]
    dec_steady = decode_times[warmup:]
    mean_framed_bytes = float(np.mean([record["framed_bytes"] for record in frame_records]))
    mean_native_bytes = float(np.mean([record["native_bytes"] for record in frame_records]))
    checkpoint_file = Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_NAMES[(args.model, args.qp)]

    result = {
        "source": {
            "title": "Starigrad 105323",
            "web_page": "https://commons.wikimedia.org/wiki/File:Starigrad_105323.webm",
            "source_video": args.source_video,
            "source_sha256": sha256_file(Path(args.source_video)),
            "resolution": [1280, 720],
            "source_fps": args.source_fps,
            "sampled_frames": len(frame_paths),
        },
        "test": {
            "model": args.model,
            "model_family": "CompressAI official model zoo",
            "qp_label": f"quality-{args.qp}",
            "qp": args.qp,
            "metric": "mse-rgb",
            "packet_payload_bytes": args.packet_payload_bytes,
            "framing": "20-byte component length header + independent z/y0/y1/y2/y3 rANS streams",
            "y_grouping": "four fixed interleaved channel groups: channel index modulo 4",
            "loss_policy": "drop packets from frame tail; discard incomplete component and all later components",
            "missing_z_policy": "neutral-gray concealment; frame marked undecodable",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "compressai": __import__("compressai").__version__,
            "checkpoint_file": str(checkpoint_file),
            "checkpoint_sha256": sha256_file(checkpoint_file),
        },
        "i_frame_fps": {
            "measured_frames": len(frame_paths),
            "warmup_frames_excluded": warmup,
            "mean_encode_ms": 1000.0 * float(np.mean(enc_steady)),
            "p50_encode_ms": 1000.0 * percentile(enc_steady, 50),
            "p95_encode_ms": 1000.0 * percentile(enc_steady, 95),
            "encode_fps": 1.0 / float(np.mean(enc_steady)),
            "mean_decode_ms": 1000.0 * float(np.mean(dec_steady)),
            "p50_decode_ms": 1000.0 * percentile(dec_steady, 50),
            "p95_decode_ms": 1000.0 * percentile(dec_steady, 95),
            "decode_fps": 1.0 / float(np.mean(dec_steady)),
        },
        "bitstream": {
            "mean_native_bytes": mean_native_bytes,
            "mean_framed_bytes": mean_framed_bytes,
            "mean_native_bpp": mean_native_bytes * 8.0 / (1280 * 720),
            "mean_framed_bpp": mean_framed_bytes * 8.0 / (1280 * 720),
            "mean_native_kbps": mean_native_bytes * 8.0 * args.source_fps / 1000.0,
            "mean_framed_kbps": mean_framed_bytes * 8.0 * args.source_fps / 1000.0,
            "mean_packets_per_frame": float(np.mean(packet_counts)),
            "mean_native_psnr_db": float(np.mean(
                [record["native_psnr_db"] for record in frame_records])),
            "maximum_full_custom_psnr_abs_delta_db": float(max(
                record["full_custom_psnr_abs_delta_db"] for record in frame_records)),
        },
        "loss_results": loss_results,
        "sample_frame": {
            "frame": sample_manifest["frame"],
            "reference_image": "reconstructions/reference.png",
            "loss_results": sample_loss_results,
        },
        "frames": frame_records,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "model": args.model,
        "qp": args.qp,
        "bitstream": result["bitstream"],
        "i_frame_fps": result["i_frame_fps"],
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES), required=True)
    parser.add_argument("--qp", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--source-fps", type=float, default=14.75)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--sample-frame-index", type=int, default=15)
    parser.add_argument("--packet-payload-bytes", type=int, default=1400)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--loss-rates", type=float, nargs="+",
                        default=(0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100))
    main(parser.parse_args())
