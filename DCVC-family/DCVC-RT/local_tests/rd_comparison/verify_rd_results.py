"""Audit saved RD results and independently cross-check one PSNR with FFmpeg."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from run_rd_comparison import rgb_psnr, run


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args):
    output, work = Path(args.output), Path(args.work)
    points = json.loads((output / "results.json").read_text(encoding="utf-8"))
    env = json.loads((output / "environment.json").read_text(encoding="utf-8"))
    formats = env.get("h264_formats", ["yuv420p", "yuv444p"])
    qps = env.get("qps", [0, 8, 16, 24, 32, 40, 48, 56, 63])
    crfs = env.get("crfs", [42, 38, 34, 30, 26, 22, 18, 14, 10, 6, 0])
    expected = {"DCVC-RT all-I": len(qps)}
    expected.update({"H.264 4:2:0 inter" if f == "yuv420p" else "H.264 4:4:4 inter": len(crfs)
                     for f in formats})
    assert Counter(p["codec"] for p in points) == expected
    assert env["reference_sha256"] == env["dcvc_environment"]["reference_sha256"]
    assert sha256(work / "reference.rgb") == env["reference_sha256"]
    count = env["reference_frames"]
    assert (work / "reference.rgb").stat().st_size == count * env["width"] * env["height"] * 3
    assert sorted(p["quality"] for p in points if p["codec"] == "DCVC-RT all-I") == sorted(qps)
    for p in points:
        assert p["frames"] == len(p["per_frame"]) == count
        assert sum(p["frame_types"].values()) == count
        if p["codec"] == "DCVC-RT all-I":
            assert p["frame_types"] == {"I": count}
            stream = work / "dcvc" / f"dcvc_qp_{p['quality']:02d}.bin"
        else:
            if count > 30:
                assert p["frame_types"].get("P", 0) > 0
            chroma = "yuv420p" if "4:2:0" in p["codec"] else "yuv444p"
            stream = work / f"h264_{chroma}_crf_{p['quality']:02d}.h264"
        assert stream.stat().st_size == p["bytes"]
        assert sha256(stream) == p["stream_sha256"]
        assert abs(p["bytes"] * 8 / env["duration_seconds"] / 1000 - p["bitrate_kbps"]) < 1e-8
        mse = np.mean([row["mse_rgb"] for row in p["per_frame"]])
        assert np.isclose(rgb_psnr(mse), p["psnr_rgb_pooled_db"], rtol=0, atol=1e-10)
    monotonic = {}
    for codec in expected:
        curve = sorted([p for p in points if p["codec"] == codec], key=lambda p: p["bitrate_kbps"])
        assert sorted(p["quality"] for p in curve) == sorted(qps if codec == "DCVC-RT all-I" else crfs)
        monotonic[codec] = all(b["psnr_rgb_pooled_db"] >= a["psnr_rgb_pooled_db"]
                               for a, b in zip(curve, curve[1:]))
    # The last requested CRF is retained in each decoded work file.  This can
    # be a smoke grid without CRF=0 or without a 4:4:4 reference curve.
    fmt = formats[0]
    selected_codec = "H.264 4:2:0 inter" if fmt == "yuv420p" else "H.264 4:4:4 inter"
    command = [args.ffmpeg, "-nostdin", "-hide_banner", "-filter_complex_threads", "1"]
    for path in (work / "reference.rgb", work / f"decoded_{fmt}.rgb"):
        command.extend(["-threads", "4", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size",
                        f"{env['width']}x{env['height']}", "-framerate", env["fps_rational"], "-i", str(path)])
    command.extend(["-filter_complex", "[0:v]format=gbrp[r];[1:v]format=gbrp[d];[r][d]psnr",
                    "-vsync", "0", "-f", "null", "-"])
    log_path = output / "independent_ffmpeg_psnr.log"
    run(command, log_path, args.command_timeout, args.idle_timeout)
    log = log_path.read_text(encoding="utf-8", errors="replace")
    score = float(re.search(r"PSNR .*?average:([0-9.]+|inf)", log).group(1))
    point = next(p for p in points if p["codec"] == selected_codec and p["quality"] == crfs[-1])
    difference = 0.0 if score == point["psnr_rgb_pooled_db"] else abs(score - point["psnr_rgb_pooled_db"])
    assert difference < 1e-5, (score, point["psnr_rgb_pooled_db"])
    validation = {"status": "PASS", "operating_points": len(points), "frames_per_point": count,
                  "all_stream_sizes_and_hashes_match": True,
                  "paired_reference_hash_match": True, "curve_monotonicity": monotonic,
                  "crosscheck_codec": selected_codec, "crosscheck_crf": crfs[-1],
                  "independent_ffmpeg_psnr_db": score,
                  "numpy_psnr_db": point["psnr_rgb_pooled_db"],
                  "absolute_psnr_difference_db": difference, "ffmpeg_command": command}
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--command-timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=120)
    main(parser.parse_args())
