#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <arm64-cu130|amd64-cu124> <cvpr2025_image.pth.tar> [output-dir]" >&2
    exit 2
}

[[ $# -ge 2 && $# -le 3 ]] || usage

PROFILE=$1
CHECKPOINT=$(realpath "$2")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_DIR=$(git -C "${PROJECT_DIR}" rev-parse --show-toplevel)
OUTPUT_DIR=$(realpath -m "${3:-${PROJECT_DIR}/offline_bundle_${PROFILE}}")
EXPECTED_CHECKPOINT_SHA=555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9
VIDEO_SHA=aee660c4ef1aa45dc169a0bf12bf2525869136d8dba7c1cce6a9d7b0c9f21bff
VIDEO_URL=https://upload.wikimedia.org/wikipedia/commons/f/f2/Starigrad_105323.webm

case "${PROFILE}" in
    arm64-cu130)
        PLATFORM=linux/arm64
        IMAGE=dcvc-rt-i:arm64-cu130
        DOCKERFILE=${SCRIPT_DIR}/Dockerfile.arm64-cu130
        IMAGE_ARCHIVE=dcvc-rt-i-arm64-cu130.tar.gz
        ;;
    amd64-cu124)
        PLATFORM=linux/amd64
        IMAGE=dcvc-rt-i:amd64-cu124
        DOCKERFILE=${SCRIPT_DIR}/Dockerfile.amd64-cu124
        IMAGE_ARCHIVE=dcvc-rt-i-amd64-cu124.tar.gz
        ;;
    *) usage ;;
esac

for command_name in docker curl ffmpeg git gzip sha256sum; do
    command -v "${command_name}" >/dev/null || {
        echo "Missing command: ${command_name}" >&2
        exit 1
    }
done

[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
echo "${EXPECTED_CHECKPOINT_SHA}  ${CHECKPOINT}" | sha256sum -c -

[[ ! -e "${OUTPUT_DIR}" ]] || {
    echo "Output path already exists; choose a new empty path: ${OUTPUT_DIR}" >&2
    exit 1
}
mkdir -p "${OUTPUT_DIR}/payload/checkpoints" "${OUTPUT_DIR}/payload/testdata/frames_32"

cp "${CHECKPOINT}" "${OUTPUT_DIR}/payload/checkpoints/cvpr2025_image.pth.tar"
curl -fL --retry 3 -o "${OUTPUT_DIR}/payload/testdata/starigrad_105323_720p.webm" "${VIDEO_URL}"
echo "${VIDEO_SHA}  ${OUTPUT_DIR}/payload/testdata/starigrad_105323_720p.webm" | sha256sum -c -
ffmpeg -hide_banner -loglevel error -y \
    -i "${OUTPUT_DIR}/payload/testdata/starigrad_105323_720p.webm" \
    -vf 'fps=32/14.187' -frames:v 32 \
    "${OUTPUT_DIR}/payload/testdata/frames_32/frame_%03d.png"

FRAME_COUNT=$(find "${OUTPUT_DIR}/payload/testdata/frames_32" -maxdepth 1 -name 'frame_*.png' | wc -l)
[[ "${FRAME_COUNT}" -eq 32 ]] || { echo "Expected 32 frames, found ${FRAME_COUNT}" >&2; exit 1; }

docker buildx build --platform "${PLATFORM}" --load \
    -f "${DOCKERFILE}" -t "${IMAGE}" "${PROJECT_DIR}"

ACTUAL_ARCH=$(docker image inspect "${IMAGE}" --format '{{.Architecture}}')
case "${PROFILE}:${ACTUAL_ARCH}" in
    arm64-cu130:arm64|amd64-cu124:amd64) ;;
    *) echo "Wrong image architecture: ${ACTUAL_ARCH}" >&2; exit 1 ;;
esac

docker image save "${IMAGE}" | gzip -1 > "${OUTPUT_DIR}/${IMAGE_ARCHIVE}"
cp "${SCRIPT_DIR}/README_zh.md" "${SCRIPT_DIR}/run_offline.sh" \
   "${SCRIPT_DIR}/verify_results.py" "${OUTPUT_DIR}/"
chmod +x "${OUTPUT_DIR}/run_offline.sh" "${OUTPUT_DIR}/verify_results.py"

git -C "${REPO_DIR}" archive --format=tar.gz \
    -o "${OUTPUT_DIR}/DCVC-source.tar.gz" HEAD DCVC-family/DCVC-RT
git -C "${REPO_DIR}" rev-parse HEAD > "${OUTPUT_DIR}/SOURCE_COMMIT.txt"
docker image inspect "${IMAGE}" > "${OUTPUT_DIR}/IMAGE_INSPECT.json"

(
    cd "${OUTPUT_DIR}"
    find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "Bundle ready: ${OUTPUT_DIR}"
echo "Transfer the whole directory to the matching offline server."
