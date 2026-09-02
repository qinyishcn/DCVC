#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
rt_python="${RT_PYTHON:-/home/qinyi/.local/share/dcvc-rt/venv/bin/python}"
compressai_python="${COMPRESSAI_PYTHON:-/home/qinyi/.local/share/dcvc-compressai/venv/bin/python}"
frame_count="${FRAME_COUNT:-32}"
sample_frame_index="${SAMPLE_FRAME_INDEX:-15}"
source_fps="${SOURCE_FPS:-14.75}"
warmup_frames="${WARMUP_FRAMES:-5}"

source_video="$project_root/local_tests/natural_720p/starigrad_105323_720p.webm"
frames="$project_root/local_tests/natural_720p/frames_32"
experiment="$project_root/local_tests/results/multimodel_i_packet_loss"
raw="$experiment/raw"

mkdir -p "$raw"

for qp in 0 31 63; do
  "$rt_python" "$project_root/local_tests/natural_720p_i_packet_loss_test.py" \
    --checkpoint "$project_root/checkpoints/cvpr2025_image.pth.tar" \
    --source-video "$source_video" \
    --frames "$frames" \
    --output "$raw/dcvc_rt_qp_${qp}" \
    --frame-count "$frame_count" \
    --sample-frame-index "$sample_frame_index" \
    --warmup-frames "$warmup_frames" \
    --qp "$qp"
done

for model in bmshj2018-hyperprior mbt2018-mean; do
  for qp in 1 4 6; do
    output_model="${model//-/_}"
    "$compressai_python" \
      "$project_root/local_tests/multimodel_i_packet_loss/compressai_packet_loss_worker.py" \
      --model "$model" \
      --qp "$qp" \
      --source-video "$source_video" \
      --source-fps "$source_fps" \
      --frames "$frames" \
      --output "$raw/${output_model}_qp_${qp}" \
      --frame-count "$frame_count" \
      --sample-frame-index "$sample_frame_index" \
      --warmup-frames "$warmup_frames"
  done
done

"$compressai_python" \
  "$project_root/local_tests/multimodel_i_packet_loss/plot_and_build_policy.py" \
  --raw-dir "$raw" \
  --output "$experiment" \
  --source-fps "$source_fps"
