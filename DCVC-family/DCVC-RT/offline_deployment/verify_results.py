#!/usr/bin/env python3
import json
import math
import sys
from pathlib import Path


def fail(message):
    raise SystemExit(f"FAILED: {message}")


if len(sys.argv) != 2:
    fail("usage: verify_results.py <results.json>")

result_path = Path(sys.argv[1])
if not result_path.is_file():
    fail(f"missing {result_path}")

data = json.loads(result_path.read_text(encoding="utf-8"))
if data["source"]["resolution"] != [1280, 720]:
    fail("source resolution is not 1280x720")
if data["source"]["sampled_frames"] != 32:
    fail("test did not process 32 frames")
if len(data["loss_results"]) != 16:
    fail("expected 16 packet-loss operating points")

loss_rates = [row["target_packet_loss_percent"] for row in data["loss_results"]]
if loss_rates != sorted(loss_rates) or loss_rates[0] != 0 or loss_rates[-1] != 100:
    fail("packet-loss points are incomplete or not ordered")

no_loss_psnr = data["loss_results"][0]["mean_psnr_db"]
encode_fps = data["i_frame_fps"]["encode_fps"]
decode_fps = data["i_frame_fps"]["decode_fps"]
if not math.isfinite(no_loss_psnr) or not 35 <= no_loss_psnr <= 45:
    fail(f"unexpected no-loss PSNR: {no_loss_psnr}")
if not math.isfinite(encode_fps) or encode_fps <= 0:
    fail(f"invalid encode FPS: {encode_fps}")
if not math.isfinite(decode_fps) or decode_fps <= 0:
    fail(f"invalid decode FPS: {decode_fps}")

print("PASS")
print(f"device: {data['test']['device']}")
print(f"torch/cuda: {data['test']['torch']} / {data['test']['cuda']}")
print(f"no-loss PSNR: {no_loss_psnr:.3f} dB")
print(f"I-frame encode/decode: {encode_fps:.3f} / {decode_fps:.3f} FPS")
