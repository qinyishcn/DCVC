#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <arm64-cu130|amd64-cu124> [output-dir]" >&2
    exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
PROFILE=$1
BUNDLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR=$(realpath -m "${2:-${BUNDLE_DIR}/output_${PROFILE}}")

case "${PROFILE}" in
    arm64-cu130)
        EXPECTED_UNAME=aarch64
        IMAGE=dcvc-rt-i:arm64-cu130
        IMAGE_ARCHIVE=dcvc-rt-i-arm64-cu130.tar.gz
        ;;
    amd64-cu124)
        EXPECTED_UNAME=x86_64
        IMAGE=dcvc-rt-i:amd64-cu124
        IMAGE_ARCHIVE=dcvc-rt-i-amd64-cu124.tar.gz
        ;;
    *) usage ;;
esac

[[ "$(uname -m)" == "${EXPECTED_UNAME}" ]] || {
    echo "Architecture mismatch: expected ${EXPECTED_UNAME}, got $(uname -m)" >&2
    exit 1
}
[[ -f "${BUNDLE_DIR}/${IMAGE_ARCHIVE}" ]] || {
    echo "Image archive not found: ${BUNDLE_DIR}/${IMAGE_ARCHIVE}" >&2
    exit 1
}

cd "${BUNDLE_DIR}"
sha256sum -c SHA256SUMS
docker image load -i "${IMAGE_ARCHIVE}"
mkdir -p "${OUTPUT_DIR}"

docker run --rm --gpus all --network none "${IMAGE}" python -c \
    "import MLCodec_extensions_cpp, torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

docker run --rm --gpus all --network none \
    -v "${BUNDLE_DIR}/payload:/data:ro" \
    -v "${OUTPUT_DIR}:/output" \
    "${IMAGE}" \
    python /opt/dcvc-rt/local_tests/natural_720p_i_packet_loss_test.py \
        --checkpoint /data/checkpoints/cvpr2025_image.pth.tar \
        --source-video /data/testdata/starigrad_105323_720p.webm \
        --frames /data/testdata/frames_32 \
        --output /output \
        --frame-count 32 \
        --sample-frame-index 15 \
        --qp 63 \
        --force-zero-thres 0.12 \
        --packet-payload-bytes 1400 \
        --warmup-frames 5

python3 "${BUNDLE_DIR}/verify_results.py" "${OUTPUT_DIR}/results.json"
echo "Completed successfully: ${OUTPUT_DIR}"
