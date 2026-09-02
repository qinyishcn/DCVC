import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.cuda_inference import replicate_pad, restore_y_4x, round_and_to_int8
from src.models.image_model import DMCI
from src.utils.common import get_state_dict, set_torch_env
from src.utils.transforms import rgb2ycbcr, ycbcr2rgb


COMPONENT_NAMES = ("z", "y0", "y1", "y2", "y3")


def load_image(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).cuda().float().unsqueeze(0) / 255.0
    ycbcr = rgb2ycbcr(tensor).half()
    height, width = rgb.shape[:2]
    pad_r, pad_b = DMCI.get_padding_size(height, width, 16)
    return rgb, replicate_pad(ycbcr, pad_b, pad_r)


def to_rgb(x_hat, height, width):
    rgb = ycbcr2rgb(x_hat)[:, :, :height, :width].clamp(0, 1)
    return (rgb[0].float().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)


def psnr(reference, reconstructed):
    mse = np.mean((reference.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * math.log10((255.0 ** 2) / mse)


def save_rgb(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def get_params(model, z_hat, y_height, y_width):
    params = model.y_prior_fusion(model.hyper_dec(z_hat))
    return params[:, :, :y_height, :y_width].contiguous()


def collect_latents(model, x, qp):
    curr_q_enc = model.q_scale_enc[qp:qp + 1]
    y = model.enc(x, curr_q_enc)
    z = model.hyper_enc(model.pad_for_y(y))
    z_hat, z_write = round_and_to_int8(z)
    params = get_params(model, z_hat, y.shape[2], y.shape[3])
    q_enc, q_dec, scales, means = model.separate_prior(params, False)
    common = model.y_spatial_prior_reduction(params)
    masks = model.get_mask_4x(*y.shape, y.dtype, y.device)
    y_scaled = y * q_enc
    y_streams = []
    y_hat_so_far = None

    for index, mask in enumerate(masks):
        _, y_q, y_hat, s_hat = model.process_with_mask(y_scaled, scales, means, mask)
        y_streams.append((model.single_part_for_writing_4x(y_q),
                          model.single_part_for_writing_4x(s_hat)))
        y_hat_so_far = y_hat if y_hat_so_far is None else y_hat_so_far + y_hat
        if index < 3:
            adaptor = (model.y_spatial_prior_adaptor_1,
                       model.y_spatial_prior_adaptor_2,
                       model.y_spatial_prior_adaptor_3)[index]
            spatial_input = torch.cat((y_hat_so_far, common), dim=1)
            scales, means = model.y_spatial_prior(adaptor(spatial_input)).chunk(2, 1)

    return {
        "y": y,
        "z_hat": z_hat,
        "z_write": z_write,
        "y_streams": y_streams,
        "q_dec": q_dec,
    }


def reconstruct_prefix(model, latents, qp, available_slices):
    y = latents["y"]
    params = get_params(model, latents["z_hat"], y.shape[2], y.shape[3])
    _, q_dec, scales, means = model.separate_prior(params, False)
    common = model.y_spatial_prior_reduction(params)
    masks = model.get_mask_4x(*y.shape, y.dtype, y.device)
    y_hat_so_far = None

    for index, mask in enumerate(masks):
        y_q = latents["y_streams"][index][0]
        if index >= available_slices:
            y_q = torch.zeros_like(y_q)
        y_hat = restore_y_4x(y_q, means, mask)
        y_hat_so_far = y_hat if y_hat_so_far is None else y_hat_so_far + y_hat
        if index < 3:
            adaptor = (model.y_spatial_prior_adaptor_1,
                       model.y_spatial_prior_adaptor_2,
                       model.y_spatial_prior_adaptor_3)[index]
            spatial_input = torch.cat((y_hat_so_far, common), dim=1)
            scales, means = model.y_spatial_prior(adaptor(spatial_input)).chunk(2, 1)

    curr_q_dec = model.q_scale_dec[qp:qp + 1]
    return model.dec(y_hat_so_far * q_dec, curr_q_dec).clamp_(0, 1)


def load_official_model(checkpoint, force_zero_thres):
    set_torch_env()
    model = DMCI()
    model.load_state_dict(get_state_dict(checkpoint))
    model = model.cuda().eval()
    model.update(force_zero_thres=force_zero_thres)
    model.half()
    model.set_use_two_entropy_coders(False)
    return model


def encode_components(model, latents, qp):
    streams = []
    model.entropy_coder.reset()
    model.bit_estimator_z.encode_z(latents["z_write"], qp)
    model.entropy_coder.flush()
    streams.append(bytes(model.entropy_coder.get_encoded_stream()))
    for y_q, scales in latents["y_streams"]:
        model.entropy_coder.reset()
        model.gaussian_encoder.encode_y(y_q, scales)
        model.entropy_coder.flush()
        streams.append(bytes(model.entropy_coder.get_encoded_stream()))
    return streams


def frame_payload(component_streams, packet_payload_bytes):
    header = struct.pack(">5I", *(len(stream) for stream in component_streams))
    framed = header + b"".join(component_streams)
    packets = [framed[index:index + packet_payload_bytes]
               for index in range(0, len(framed), packet_payload_bytes)]
    component_ranges = []
    offset = len(header)
    for name, stream in zip(COMPONENT_NAMES, component_streams):
        component_ranges.append({"name": name, "start": offset, "end": offset + len(stream),
                                 "bytes": len(stream)})
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


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    frame_dir = Path(args.frames)
    frame_paths = sorted(frame_dir.glob("*.png"))[:args.frame_count]
    if len(frame_paths) != args.frame_count:
        raise ValueError(f"expected {args.frame_count} PNG frames, found {len(frame_paths)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir = output_dir / "reconstructions"
    model = load_official_model(args.checkpoint, args.force_zero_thres)

    references = []
    tensors = []
    for frame_path in frame_paths:
        reference, tensor = load_image(frame_path)
        if reference.shape[:2] != (720, 1280):
            raise ValueError(f"not 1280x720: {frame_path} -> {reference.shape[:2]}")
        references.append(reference)
        tensors.append(tensor)

    sps = {"height": 720, "width": 1280, "ec_part": 0, "use_ada_i": 0, "sps_id": 0}
    encoding_times = []
    decoding_times = []
    native_streams = []
    native_psnrs = []

    with torch.no_grad():
        for reference, tensor in zip(references, tensors):
            torch.cuda.synchronize()
            start = time.perf_counter()
            encoded = model.compress(tensor, args.qp)
            torch.cuda.synchronize()
            encoding_times.append(time.perf_counter() - start)

            stream = bytes(encoded["bit_stream"])
            native_streams.append(stream)
            start = time.perf_counter()
            decoded = model.decompress(stream, sps, args.qp)["x_hat"]
            torch.cuda.synchronize()
            decoding_times.append(time.perf_counter() - start)
            native_rgb = to_rgb(decoded, 720, 1280)
            native_psnrs.append(psnr(reference, native_rgb))

    frame_records = []
    sample_index = min(args.sample_frame_index, len(frame_paths) - 1)
    sample_cache = None
    concealment_rgb = np.full((720, 1280, 3), 128, dtype=np.uint8)

    for frame_index, (frame_path, reference, tensor, native_stream) in enumerate(
            zip(frame_paths, references, tensors, native_streams)):
        with torch.no_grad():
            latents = collect_latents(model, tensor, args.qp)
            streams = encode_components(model, latents, args.qp)
            framed, packets, ranges = frame_payload(streams, args.packet_payload_bytes)
            stage_psnr = {}
            stage_rgb = {}
            for available_y in range(5):
                reconstructed = reconstruct_prefix(model, latents, args.qp, available_y)
                torch.cuda.synchronize()
                rgb = to_rgb(reconstructed, 720, 1280)
                stage = "z" if available_y == 0 else "z+" + "+".join(
                    f"y{index}" for index in range(available_y))
                stage_psnr[stage] = psnr(reference, rgb)
                if frame_index == sample_index:
                    stage_rgb[stage] = rgb
            stage_psnr["concealment"] = psnr(reference, concealment_rgb)

        full_stage = stage_psnr["z+y0+y1+y2+y3"]
        if abs(full_stage - native_psnrs[frame_index]) > 1e-8:
            raise RuntimeError(f"full component reconstruction mismatch on {frame_path.name}")

        packet_hashes = [hashlib.sha256(packet).hexdigest() for packet in packets]
        record = {
            "frame": frame_path.name,
            "native_monolithic_bytes": len(native_stream),
            "component_bytes": dict(zip(COMPONENT_NAMES, map(len, streams))),
            "framed_component_bytes": len(framed),
            "packet_count": len(packets),
            "last_packet_bytes": len(packets[-1]),
            "stage_psnr_db": stage_psnr,
        }
        frame_records.append(record)
        if frame_index == sample_index:
            sample_cache = {
                "reference": reference,
                "stage_rgb": stage_rgb,
                "framed_bytes": len(framed),
                "packets": packets,
                "packet_hashes": packet_hashes,
                "component_ranges": ranges,
                "stage_psnr_db": stage_psnr,
                "frame": frame_path.name,
            }

    loss_results = []
    sample_results = []
    loss_rates = [float(value) for value in args.loss_rates]
    packet_counts = [record["packet_count"] for record in frame_records]
    for target_rate in loss_rates:
        psnrs = []
        total_packets = 0
        dropped_packets = 0
        total_bytes = 0
        received_bytes = 0
        decodable = 0
        stage_counts = {}
        sample_result = None
        allocated_drops = allocate_drop_counts(packet_counts, target_rate)

        for frame_index, (record, drop_count) in enumerate(zip(frame_records, allocated_drops)):
            packet_count = record["packet_count"]
            keep_count = packet_count - drop_count
            retained_bytes = min(record["framed_component_bytes"],
                                 keep_count * args.packet_payload_bytes)
            offset = 20
            ranges = []
            for name in COMPONENT_NAMES:
                length = record["component_bytes"][name]
                ranges.append({"name": name, "start": offset, "end": offset + length,
                               "bytes": length})
                offset += length
            stage, z_ok, _ = available_stage(retained_bytes, ranges)
            score = record["stage_psnr_db"][stage]
            psnrs.append(score)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            decodable += int(z_ok)
            total_packets += packet_count
            dropped_packets += drop_count
            total_bytes += record["framed_component_bytes"]
            received_bytes += retained_bytes

            if frame_index == sample_index:
                sample_result = {
                    "target_loss_percent": target_rate,
                    "packet_count": packet_count,
                    "dropped_packets": drop_count,
                    "actual_packet_loss_percent": 100.0 * drop_count / packet_count,
                    "retained_bytes": retained_bytes,
                    "retained_bytes_percent": 100.0 * retained_bytes /
                    record["framed_component_bytes"],
                    "stage": stage,
                    "psnr_db": score,
                }
                image_name = f"plr_{target_rate:05.1f}_{stage.replace('+', '_')}.png"
                if stage == "concealment":
                    image = concealment_rgb
                else:
                    image = sample_cache["stage_rgb"][stage]
                save_rgb(reconstruction_dir / image_name, image)
                sample_result["image"] = str(Path("reconstructions") / image_name)

        loss_results.append({
            "target_packet_loss_percent": target_rate,
            "actual_packet_loss_percent": 100.0 * dropped_packets / total_packets,
            "received_bytes_percent": 100.0 * received_bytes / total_bytes,
            "mean_psnr_db": float(np.mean(psnrs)),
            "minimum_psnr_db": float(np.min(psnrs)),
            "maximum_psnr_db": float(np.max(psnrs)),
            "decodable_frame_percent": 100.0 * decodable / len(frame_records),
            "stage_frame_counts": stage_counts,
        })
        sample_results.append(sample_result)

    warmup = min(args.warmup_frames, len(frame_paths) - 1)
    enc_steady = encoding_times[warmup:]
    dec_steady = decoding_times[warmup:]
    sample_reference_path = reconstruction_dir / "reference.png"
    save_rgb(sample_reference_path, sample_cache["reference"])
    packet_manifest = {
        "frame": sample_cache["frame"],
        "framed_bytes": sample_cache["framed_bytes"],
        "packet_payload_bytes": args.packet_payload_bytes,
        "component_ranges": sample_cache["component_ranges"],
        "packets": [
            {"index": index, "bytes": len(packet), "sha256": sha256}
            for index, (packet, sha256) in enumerate(zip(sample_cache["packets"],
                                                         sample_cache["packet_hashes"]))
        ],
    }
    (output_dir / "sample_packet_manifest.json").write_text(
        json.dumps(packet_manifest, indent=2), encoding="utf-8")

    source_video = Path(args.source_video)
    result = {
        "source": {
            "title": "Starigrad 105323",
            "web_page": "https://commons.wikimedia.org/wiki/File:Starigrad_105323.webm",
            "download_url": "https://upload.wikimedia.org/wikipedia/commons/f/f2/Starigrad_105323.webm",
            "license": "CC BY-SA 3.0 / GFDL (see source page)",
            "source_video": str(source_video),
            "source_sha256": sha256_file(source_video),
            "resolution": [1280, 720],
            "sampled_frames": len(frame_paths),
        },
        "test": {
            "checkpoint": args.checkpoint,
            "qp": args.qp,
            "force_zero_thres": args.force_zero_thres,
            "packet_payload_bytes": args.packet_payload_bytes,
            "framing": "20-byte big-endian component length header + independent z/y0/y1/y2/y3 rANS streams",
            "loss_policy": "drop packets from frame tail; incomplete component and all dependent later components are discarded",
            "missing_z_policy": "neutral-gray concealment; frame marked undecodable",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
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
            "mean_native_monolithic_bytes": float(np.mean(
                [record["native_monolithic_bytes"] for record in frame_records])),
            "mean_separately_framed_bytes": float(np.mean(
                [record["framed_component_bytes"] for record in frame_records])),
            "mean_packets_per_frame": float(np.mean(
                [record["packet_count"] for record in frame_records])),
            "mean_native_psnr_db": float(np.mean(native_psnrs)),
        },
        "loss_results": loss_results,
        "sample_frame": {
            "frame": sample_cache["frame"],
            "reference_image": str(sample_reference_path.relative_to(output_dir)),
            "stage_psnr_db": sample_cache["stage_psnr_db"],
            "loss_results": sample_results,
        },
        "frames": frame_records,
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "i_frame_fps": result["i_frame_fps"],
        "bitstream": result["bitstream"],
        "loss_results": loss_results,
        "sample_frame": result["sample_frame"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--sample-frame-index", type=int, default=15)
    parser.add_argument("--qp", type=int, default=63)
    parser.add_argument("--force-zero-thres", type=float, default=0.12)
    parser.add_argument("--packet-payload-bytes", type=int, default=1400)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--loss-rates", type=float, nargs="+",
                        default=(0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98, 100))
    main(parser.parse_args())
