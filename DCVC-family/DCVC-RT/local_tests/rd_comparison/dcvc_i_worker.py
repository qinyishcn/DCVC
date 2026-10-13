"""Encode/decode a contiguous RGB sequence with the native DCVC-RT I codec."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "local_tests"))
from natural_720p_i_packet_loss_test import load_official_model, to_rgb
from src.layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, replicate_pad
from src.models.image_model import DMCI
from src.utils.stream_helper import (
    NalType, read_header, read_ip_remaining, read_sps_remaining, write_ip, write_sps,
)
from src.utils.transforms import rgb2ycbcr


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw = Path(args.rgb)
    frame_bytes = args.width * args.height * 3
    assert raw.stat().st_size % frame_bytes == 0
    count = raw.stat().st_size // frame_bytes
    references = np.memmap(raw, dtype=np.uint8, mode="r",
                           shape=(count, args.height, args.width, 3))
    model = load_official_model(args.checkpoint, 0.12)
    sps = {"height": args.height, "width": args.width, "ec_part": 0,
           "use_ada_i": 0, "sps_id": 0}
    results = []
    with torch.no_grad():
        for qp in args.qps:
            stream_path = output / f"dcvc_qp_{qp:02d}.bin"
            rows = []
            with stream_path.open("wb") as handle:
                write_sps(handle, sps)
                for index, reference in enumerate(references):
                    x = torch.from_numpy(reference.transpose(2, 0, 1).copy())
                    x = rgb2ycbcr(x.cuda().float().unsqueeze(0) / 255).half()
                    pad_r, pad_b = DMCI.get_padding_size(args.height, args.width, 16)
                    x = replicate_pad(x, pad_b, pad_r)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    encoded = model.compress(x, qp)
                    torch.cuda.synchronize()
                    encode_ms = (time.perf_counter() - start) * 1000
                    stream = bytes(encoded["bit_stream"])
                    written = write_ip(handle, True, 0, qp, stream)
                    rows.append({"frame_index": index, "entropy_bytes": len(stream),
                                 "framed_bytes": written, "encode_ms": encode_ms})
                    if (index + 1) % 50 == 0:
                        print(f"DCVC QP={qp}: encoded {index + 1}/{count}", flush=True)
            # Decode the saved native stream, including its SPS and frame headers.
            with stream_path.open("rb") as handle:
                header = read_header(handle)
                assert header["nal_type"] == NalType.NAL_SPS
                decoded_sps = read_sps_remaining(handle, header["sps_id"])
                for index, reference in enumerate(references):
                    header = read_header(handle)
                    assert header["nal_type"] == NalType.NAL_I
                    decoded_qp, stream = read_ip_remaining(handle)
                    assert decoded_qp == qp
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    decoded = model.decompress(stream, decoded_sps, decoded_qp)["x_hat"]
                    torch.cuda.synchronize()
                    rows[index]["decode_ms"] = (time.perf_counter() - start) * 1000
                    rgb = to_rgb(decoded, args.height, args.width)
                    error = reference.astype(np.float64) - rgb.astype(np.float64)
                    mse = float(np.mean(error * error))
                    rows[index].update(mse_rgb=mse, psnr_rgb_db=10 * np.log10(255 ** 2 / mse))
                    if index == count // 2 and qp in (0, 32, 63):
                        Image.fromarray(rgb).save(output / f"dcvc_qp_{qp:02d}_sample.png")
                assert handle.read(1) == b"", "extra bytes after final frame"
            mean_mse = float(np.mean([row["mse_rgb"] for row in rows]))
            point = {"codec": "DCVC-RT all-I", "quality": qp, "frames": count,
                     "bytes": stream_path.stat().st_size, "stream_sha256": sha256(stream_path),
                     "entropy_bytes": sum(row["entropy_bytes"] for row in rows),
                     "psnr_rgb_pooled_db": float(10 * np.log10(255 ** 2 / mean_mse)),
                     "psnr_rgb_mean_frame_db": float(np.mean([row["psnr_rgb_db"] for row in rows])),
                     "mean_mse_rgb": mean_mse, "frame_types": {"I": count}, "per_frame": rows}
            results.append(point)
            (output / f"dcvc_qp_{qp:02d}.json").write_text(json.dumps(point, indent=2))
            print(f"DCVC QP={qp} complete: {point['bytes']} bytes, "
                  f"{point['psnr_rgb_pooled_db']:.3f} dB", flush=True)
    metadata = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0), "python": sys.version,
                "platform": platform.platform(), "custom_cuda": CUSTOMIZED_CUDA_INFERENCE,
                "checkpoint_sha256": sha256(args.checkpoint), "reference_sha256": sha256(raw)}
    (output / "dcvc_results.json").write_text(json.dumps({"environment": metadata,
                                                         "points": results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--qps", type=int, nargs="+", default=[0, 8, 16, 24, 32, 40, 48, 56, 63])
    main(parser.parse_args())
