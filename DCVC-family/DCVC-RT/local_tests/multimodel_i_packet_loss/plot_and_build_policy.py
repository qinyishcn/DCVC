import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_LABELS = {
    "dcvc-rt": "DCVC-RT 2025",
    "bmshj2018-hyperprior": "Balle Hyperprior 2018",
    "mbt2018-mean": "Minnen Mean-Scale 2018",
}
MODEL_COLORS = {
    "dcvc-rt": "#1565c0",
    "bmshj2018-hyperprior": "#ef6c00",
    "mbt2018-mean": "#2e7d32",
}
LINE_STYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def normalize_result(path, source_fps):
    data = json.loads(path.read_text(encoding="utf-8"))
    test = data["test"]
    if "model" in test:
        model = test["model"]
        qp_label = test["qp_label"]
        qp = test["qp"]
        mean_bytes = data["bitstream"]["mean_framed_bytes"]
        mean_bpp = data["bitstream"]["mean_framed_bpp"]
    else:
        model = "dcvc-rt"
        qp = test["qp"]
        qp_label = f"index-{qp}"
        mean_bytes = data["bitstream"]["mean_separately_framed_bytes"]
        mean_bpp = mean_bytes * 8.0 / (1280 * 720)
    mean_kbps = mean_bytes * 8.0 * source_fps / 1000.0
    return {
        "source_file": str(path),
        "model": model,
        "model_label": MODEL_LABELS[model],
        "qp": qp,
        "qp_label": qp_label,
        "mean_framed_bytes": mean_bytes,
        "mean_framed_bpp": mean_bpp,
        "mean_framed_kbps": mean_kbps,
        "encode_fps": data["i_frame_fps"]["encode_fps"],
        "decode_fps": data["i_frame_fps"]["decode_fps"],
        "loss_results": data["loss_results"],
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(combos):
    rows = []
    for combo in combos:
        for point in combo["loss_results"]:
            rows.append({
                "model": combo["model"],
                "model_label": combo["model_label"],
                "qp": combo["qp"],
                "qp_label": combo["qp_label"],
                "mean_framed_kbps": combo["mean_framed_kbps"],
                "mean_framed_bpp": combo["mean_framed_bpp"],
                "encode_fps": combo["encode_fps"],
                "decode_fps": combo["decode_fps"],
                "target_packet_loss_percent": point["target_packet_loss_percent"],
                "actual_packet_loss_percent": point["actual_packet_loss_percent"],
                "received_bytes_percent": point["received_bytes_percent"],
                "mean_psnr_db": point["mean_psnr_db"],
                "decodable_frame_percent": point["decodable_frame_percent"],
            })
    return rows


def plot_psnr(combos, output):
    fig, axis = plt.subplots(figsize=(14, 9), dpi=180)
    for model in MODEL_LABELS:
        model_combos = sorted(
            (combo for combo in combos if combo["model"] == model),
            key=lambda combo: combo["mean_framed_kbps"])
        for level, combo in enumerate(model_combos):
            points = combo["loss_results"]
            axis.plot(
                [point["actual_packet_loss_percent"] for point in points],
                [point["mean_psnr_db"] for point in points],
                color=MODEL_COLORS[model],
                linestyle=LINE_STYLES[level % len(LINE_STYLES)],
                marker=MARKERS[level % len(MARKERS)],
                markersize=4.2,
                linewidth=1.8,
                label=(f'{combo["model_label"]} | QP {combo["qp_label"]} | '
                       f'{combo["mean_framed_kbps"]:.0f} kbps'),
            )
    axis.set_xlabel("Actual packet loss rate (%)")
    axis.set_ylabel("Mean RGB PSNR (dB)")
    axis.set_title("720p neural I-frame robustness with prioritized tail packet loss")
    axis.set_xlim(-1, 101)
    axis.grid(True, alpha=0.28)
    axis.legend(loc="best", fontsize=8.3, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_decodability(combos, output):
    fig, axis = plt.subplots(figsize=(14, 8), dpi=180)
    for model in MODEL_LABELS:
        model_combos = sorted(
            (combo for combo in combos if combo["model"] == model),
            key=lambda combo: combo["mean_framed_kbps"])
        for level, combo in enumerate(model_combos):
            points = combo["loss_results"]
            axis.plot(
                [point["actual_packet_loss_percent"] for point in points],
                [point["decodable_frame_percent"] for point in points],
                color=MODEL_COLORS[model],
                linestyle=LINE_STYLES[level % len(LINE_STYLES)],
                marker=MARKERS[level % len(MARKERS)],
                markersize=4,
                linewidth=1.6,
                label=(f'{combo["model_label"]} | QP {combo["qp_label"]} | '
                       f'{combo["mean_framed_kbps"]:.0f} kbps'),
            )
    axis.set_xlabel("Actual packet loss rate (%)")
    axis.set_ylabel("Decodable frames (%)")
    axis.set_title("Frame decodability when z becomes incomplete")
    axis.set_xlim(-1, 101)
    axis.set_ylim(-2, 102)
    axis.grid(True, alpha=0.28)
    axis.legend(loc="lower left", fontsize=8.3, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def build_policy(combos, psnr_targets, minimum_decodable):
    loss_rates = sorted({
        point["target_packet_loss_percent"]
        for combo in combos for point in combo["loss_results"]
    })
    rows = []
    for loss_rate in loss_rates:
        candidates = []
        for combo in combos:
            point = next(point for point in combo["loss_results"]
                         if point["target_packet_loss_percent"] == loss_rate)
            candidates.append({**combo, "point": point})
        for target in psnr_targets:
            feasible = [candidate for candidate in candidates
                        if candidate["point"]["mean_psnr_db"] >= target
                        and candidate["point"]["decodable_frame_percent"] >= minimum_decodable]
            choice = min(feasible, key=lambda item: item["mean_framed_kbps"]) if feasible else None
            rows.append({
                "target_packet_loss_percent": loss_rate,
                "minimum_psnr_db": target,
                "minimum_decodable_frame_percent": minimum_decodable,
                "status": "feasible" if choice else "no_measured_combo",
                "selected_model": choice["model"] if choice else "",
                "selected_qp_label": choice["qp_label"] if choice else "",
                "mean_framed_kbps": choice["mean_framed_kbps"] if choice else "",
                "predicted_mean_psnr_db": choice["point"]["mean_psnr_db"] if choice else "",
                "predicted_decodable_frame_percent": (
                    choice["point"]["decodable_frame_percent"] if choice else ""),
                "encode_fps": choice["encode_fps"] if choice else "",
                "decode_fps": choice["decode_fps"] if choice else "",
            })
    return rows


def build_pareto(combos):
    loss_rates = sorted({
        point["target_packet_loss_percent"]
        for combo in combos for point in combo["loss_results"]
    })
    rows = []
    for loss_rate in loss_rates:
        candidates = []
        for combo in combos:
            point = next(point for point in combo["loss_results"]
                         if point["target_packet_loss_percent"] == loss_rate)
            candidates.append({**combo, "point": point})
        for candidate in candidates:
            dominated = any(
                other["mean_framed_kbps"] <= candidate["mean_framed_kbps"]
                and other["point"]["mean_psnr_db"] >= candidate["point"]["mean_psnr_db"]
                and other["point"]["decodable_frame_percent"] >= candidate["point"]["decodable_frame_percent"]
                and (other["mean_framed_kbps"] < candidate["mean_framed_kbps"]
                     or other["point"]["mean_psnr_db"] > candidate["point"]["mean_psnr_db"]
                     or other["point"]["decodable_frame_percent"] > candidate["point"]["decodable_frame_percent"])
                for other in candidates if other is not candidate)
            if not dominated:
                rows.append({
                    "target_packet_loss_percent": loss_rate,
                    "model": candidate["model"],
                    "qp_label": candidate["qp_label"],
                    "mean_framed_kbps": candidate["mean_framed_kbps"],
                    "mean_psnr_db": candidate["point"]["mean_psnr_db"],
                    "decodable_frame_percent": candidate["point"]["decodable_frame_percent"],
                    "encode_fps": candidate["encode_fps"],
                    "decode_fps": candidate["decode_fps"],
                })
    return rows


def main(args):
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_paths = sorted(raw_dir.glob("*/results.json"))
    combos = [normalize_result(path, args.source_fps) for path in result_paths]
    if len(combos) != args.expected_combinations:
        raise ValueError(f"expected {args.expected_combinations} combinations, found {len(combos)}")

    summary = build_summary(combos)
    summary_fields = list(summary[0])
    write_csv(output_dir / "curve_data.csv", summary, summary_fields)
    plot_psnr(combos, output_dir / "psnr_vs_packet_loss.png")
    plot_decodability(combos, output_dir / "decodability_vs_packet_loss.png")

    policy = build_policy(combos, args.psnr_targets, args.minimum_decodable)
    write_csv(output_dir / "adaptive_min_bitrate_policy.csv", policy, list(policy[0]))
    pareto = build_pareto(combos)
    write_csv(output_dir / "pareto_frontier.csv", pareto, list(pareto[0]))

    payload = {
        "source_fps": args.source_fps,
        "combination_count": len(combos),
        "combinations": [{key: combo[key] for key in (
            "model", "model_label", "qp", "qp_label", "mean_framed_bytes",
            "mean_framed_bpp", "mean_framed_kbps", "encode_fps", "decode_fps")}
            for combo in combos],
        "policy_rule": {
            "objective": "minimum framed bitrate meeting both PSNR and decodability constraints",
            "minimum_decodable_frame_percent": args.minimum_decodable,
            "psnr_targets_db": args.psnr_targets,
        },
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-fps", type=float, default=14.75)
    parser.add_argument("--expected-combinations", type=int, default=9)
    parser.add_argument("--minimum-decodable", type=float, default=99.0)
    parser.add_argument("--psnr-targets", type=float, nargs="+", default=(24, 28, 30, 32, 35, 38))
    main(parser.parse_args())
