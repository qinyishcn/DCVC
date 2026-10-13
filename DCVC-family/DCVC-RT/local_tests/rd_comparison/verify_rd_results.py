"""Audit saved RD results and independently cross-check one PSNR with FFmpeg."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    output, work = Path(args.output), Path(args.work)
    points = json.loads((output / "results.json").read_text())
    env = json.loads((output / "environment.json").read_text())
    expected = {"DCVC-RT all-I": 9, "H.264 4:2:0 inter": 11, "H.264 4:4:4 inter": 11}
    assert Counter(p["codec"] for p in points) == expected
    assert env["dcvc_environment"]["checkpoint_sha256"] == (
        "555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9")
    assert env["reference_sha256"] == env["dcvc_environment"]["reference_sha256"]
    assert sha256(work / "reference.rgb") == env["reference_sha256"]
    count = env["reference_frames"]
    for p in points:
        assert p["frames"] == len(p["per_frame"]) == count
        assert sum(p["frame_types"].values()) == count
        if p["codec"] == "DCVC-RT all-I":
            assert p["frame_types"] == {"I": count}
            stream = work / "dcvc" / f"dcvc_qp_{p['quality']:02d}.bin"
        else:
            assert p["frame_types"].get("P", 0) > 0
            chroma = "yuv420p" if "4:2:0" in p["codec"] else "yuv444p"
            stream = work / f"h264_{chroma}_crf_{p['quality']:02d}.h264"
        assert stream.stat().st_size == p["bytes"]
        assert sha256(stream) == p["stream_sha256"]
        assert abs(p["bytes"] * 8 / env["duration_seconds"] / 1000 - p["bitrate_kbps"]) < 1e-8
        mse = np.mean([row["mse_rgb"] for row in p["per_frame"]])
        assert abs(10 * np.log10(255 ** 2 / mse) - p["psnr_rgb_pooled_db"]) < 1e-10
    for codec in expected:
        curve = sorted([p for p in points if p["codec"] == codec], key=lambda p: p["bitrate_kbps"])
        assert all(b["psnr_rgb_pooled_db"] > a["psnr_rgb_pooled_db"]
                   for a, b in zip(curve, curve[1:]))
    # Work files retain the final decoded CRF=0 YUV420 point.  FFmpeg's own
    # planar-RGB PSNR filter is independent of our NumPy scoring code.
    command = [args.ffmpeg, "-hide_banner"]
    for path in (work / "reference.rgb", work / "decoded_yuv420p.rgb"):
        command.extend(["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720",
                        "-framerate", "59/4", "-i", str(path)])
    command.extend(["-filter_complex", "[0:v]format=gbrp[r];[1:v]format=gbrp[d];[r][d]psnr",
                    "-f", "null", "-"])
    result = subprocess.run(command, capture_output=True, check=True)
    log = result.stderr.decode(errors="replace")
    (output / "independent_ffmpeg_psnr.log").write_text(log, encoding="utf-8")
    score = float(re.search(r"PSNR .*?average:([0-9.]+)", log).group(1))
    point = next(p for p in points if p["codec"] == "H.264 4:2:0 inter" and p["quality"] == 0)
    difference = abs(score - point["psnr_rgb_pooled_db"])
    assert difference < 1e-6, (score, point["psnr_rgb_pooled_db"])
    validation = {"status": "PASS", "operating_points": len(points), "frames_per_point": count,
                  "all_stream_sizes_and_hashes_match": True,
                  "paired_reference_hash_match": True, "curves_strictly_monotonic": True,
                  "independent_ffmpeg_psnr_db": score,
                  "numpy_psnr_db": point["psnr_rgb_pooled_db"],
                  "absolute_psnr_difference_db": difference, "ffmpeg_command": command}
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    main(parser.parse_args())
