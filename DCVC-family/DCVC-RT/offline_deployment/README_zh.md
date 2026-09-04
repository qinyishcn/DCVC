# DCVC-RT I 帧双服务器离线部署指南

本文只部署 DCVC-RT 的神经网络 I 帧模型，目标是让下面的脚本在完全断网的
服务器上跑通：

```text
local_tests/natural_720p_i_packet_loss_test.py
```

该目录已合入公开仓库的默认分支：[`main`](https://github.com/qinyishcn/DCVC/tree/main/DCVC-family/DCVC-RT/offline_deployment)
（也可[直接下载 main 分支 ZIP](https://github.com/qinyishcn/DCVC/archive/refs/heads/main.zip)）。

## 1. 方案结论

使用两份**按 CPU 架构分别构建**的 Docker 镜像：

| 目标服务器 | 离线镜像 | 镜像内软件栈 | 宿主机 Python 是否使用 |
|---|---|---|---|
| Ubuntu 24.04、ARM64、GB10、CUDA 13.0 | `dcvc-rt-i:arm64-cu130` | NVIDIA PyTorch 25.08、Python 3.12、CUDA 13.0 | 否 |
| Ubuntu 20.04、x86_64、RTX 3090、CUDA 12.4 | `dcvc-rt-i:amd64-cu124` | Python 3.12、PyTorch 2.4.1+cu124、CUDA 12.4.1 | 否 |

选择容器而不是直接使用宿主机 Python，原因如下：

- DCVC-RT 的 `src/cpp/setup.py` 声明需要 Python 3.12 及以上；第一台的
  Python 3.13 较新，第二台的 Python 3.10 又低于该声明。
- ARM64 与 x86_64 的 C++ 扩展二进制不通用，必须分别编译和打包。
- `MLCodec_extensions_cpp` 是实际 rANS 熵编码所必需的扩展，两个镜像在
  制作时都会编译它，而不是在离线服务器上临时下载或编译。
- 项目的优化 CUDA inference 扩展是可选项。本方案不编译它，运行时会看到
  `fallback to pytorch` 提示；这不影响码流、PSNR、丢包实验或结果文件，
  但 FPS 低于安装优化算子后的上限。

宿主机上的 CUDA Toolkit 和 Python 都不会被容器使用。宿主机真正必须满足的
是：可工作的 NVIDIA 驱动、Docker Engine、NVIDIA Container Toolkit。

## 2. 已知兼容性边界

### 2.1 驱动门槛

先在两台服务器上执行以下只读检查：

```bash
uname -m
cat /etc/os-release
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
docker --version || true
nvidia-ctk --version || true
```

验收要求：

| 服务器 | `uname -m` | 最低建议驱动 | 原因 |
|---|---|---:|---|
| GB10 | `aarch64` | 580 系列 | CUDA 13.x 的最低兼容驱动系列是 580 |
| RTX 3090 | `x86_64` | 550.54.15 | CUDA 12.4 Update 1 对应的 Linux 驱动门槛 |

`nvidia-smi` 显示的 `CUDA Version` 是驱动能支持的最高 CUDA 版本，不等于
宿主机已安装了同版本 Toolkit。本方案不依赖宿主机 `nvcc`，因此不要为了
“版本看起来一致”盲目更换已工作的驱动。

参考：[CUDA 13.0 驱动兼容表](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-driver)、
[CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。

### 2.2 操作系统风险

- NVIDIA Container Toolkit 当前明确测试 Ubuntu 24.04 的 ARM64 和 x86_64，
  因而第一台在官方支持矩阵内：
  [Platform Support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html)。
- Docker 当前官方文档列出的 Ubuntu 是 22.04、24.04、26.04，不再列出
  Ubuntu 20.04。第二台可以继续使用 Ubuntu 20.04 自带/现有的 Docker，
  但本指南把“GPU 容器 smoke test 通过”作为强制门槛；若新装仍不能通过，
  长期方案是先升级到 Ubuntu 22.04，而不是绕过验收。

## 3. 必备文件与可点击下载地址

所有下载都应在联网机器完成，之后只把离线包目录复制到服务器。

| 内容 | 固定版本/文件 | 官方下载 |
|---|---|---|
| 本部署代码 | GitHub main ZIP | [直接下载](https://github.com/qinyishcn/DCVC/archive/refs/heads/main.zip) |
| 上游 DCVC 源码 | Microsoft/DCVC | [GitHub 仓库](https://github.com/microsoft/DCVC) |
| DCVC-RT I 帧权重 | `cvpr2025_image.pth.tar` | [作者 OneDrive checkpoint 目录](https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8) |
| 720P 自然视频 | `Starigrad_105323.webm` | [直接下载](https://upload.wikimedia.org/wikipedia/commons/f/f2/Starigrad_105323.webm)；[来源与许可](https://commons.wikimedia.org/wiki/File:Starigrad_105323.webm) |
| ARM PyTorch 基础镜像 | `nvcr.io/nvidia/pytorch:25.08-py3` | [NGC PyTorch Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)；[25.08 说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-08.html) |
| x86 CUDA 基础镜像 | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04` | [NVIDIA CUDA 镜像标签](https://hub.docker.com/r/nvidia/cuda/tags) |
| x86 镜像内 Python | Miniforge 26.3.2-3 x86_64 | [直接下载安装器](https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-26.3.2-3-Linux-x86_64.sh)；[校验文件](https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-26.3.2-3-Linux-x86_64.sh.sha256) |
| x86 镜像内 PyTorch | 2.4.1+cu124、CPython 3.12 | [直接下载 wheel](https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp312-cp312-linux_x86_64.whl)；[cu124 索引](https://download.pytorch.org/whl/cu124/torch/) |
| NumPy | 1.26.4（x86 固定；ARM 使用 NGC 约束版本） | [PyPI 文件下载](https://pypi.org/project/numpy/1.26.4/#files) |
| SciPy | 1.14.1（x86 固定；ARM 使用 NGC 内置版本） | [PyPI 文件下载](https://pypi.org/project/scipy/1.14.1/#files) |
| Pillow | 10.4.0（x86 固定；ARM 使用 NGC 约束版本） | [PyPI 文件下载](https://pypi.org/project/pillow/10.4.0/#files) |
| pybind11 | 2.13.6（x86 固定；ARM 使用 NGC 约束版本） | [PyPI 文件下载](https://pypi.org/project/pybind11/2.13.6/#files) |
| Docker Engine | 与目标 Ubuntu/架构匹配 | [Docker Ubuntu 安装与包仓库](https://docs.docker.com/engine/install/ubuntu/) |
| NVIDIA Container Toolkit | 与目标 Ubuntu/架构匹配 | [安装文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)；[GitHub Releases](https://github.com/NVIDIA/nvidia-container-toolkit/releases) |
| Ubuntu 20.04 Docker 包 | `docker.io` | [Ubuntu focal 包页面](https://packages.ubuntu.com/focal/docker.io) |
| ffmpeg | 仅联网制包机需要 | [FFmpeg 下载页](https://ffmpeg.org/download.html) |

关键固定文件的 SHA-256：

```text
cvpr2025_image.pth.tar
555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9

starigrad_105323_720p.webm
aee660c4ef1aa45dc169a0bf12bf2525869136d8dba7c1cce6a9d7b0c9f21bff

Miniforge3-26.3.2-3-Linux-x86_64.sh
848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
```

说明：镜像内的 `numpy`、`scipy`、`Pillow`、`pybind11` 以及 PyTorch 的
CUDA 依赖会在联网构建阶段装进镜像，不需要在离线服务器单独下载 wheel。
制包脚本最终会为**整个交付目录**生成 `SHA256SUMS`。

x86 CUDA 基础镜像还固定到 amd64 manifest digest
`sha256:f18cf1a9ac2842e59f13b0d0729594da8cbd68cadd2379308cdd98c0374dbd80`，
避免同名标签以后指向不同内容。ARM NGC 镜像的实际 digest 会被记录在
`IMAGE_INSPECT.json` 和制包日志中。

## 4. 联网制包机：生成两份离线包

### 4.1 制包机要求

需要 Linux、Docker、Docker Buildx、Git、curl、ffmpeg、gzip 和足够磁盘。
建议至少预留 80 GB。最稳妥的方式是在同架构机器上构建：ARM64 镜像在
联网 ARM64 机器上构建，amd64 镜像在联网 x86_64 机器上构建。

如果只有一台 x86_64 制包机，也可以使用 Docker Buildx/QEMU 构建 ARM64，
但速度明显更慢，且必须先正确安装 binfmt。参考 Docker 官方的
[Multi-platform builds](https://docs.docker.com/build/building/multi-platform/)。

### 4.2 获取代码

```bash
git clone --branch main \
  https://github.com/qinyishcn/DCVC.git
cd DCVC/DCVC-family/DCVC-RT
chmod +x offline_deployment/prepare_bundle.sh \
  offline_deployment/run_offline.sh \
  offline_deployment/verify_results.py
```

从上表 OneDrive 目录下载 `cvpr2025_image.pth.tar`，例如放在：

```text
/data/downloads/cvpr2025_image.pth.tar
```

手工校验一次：

```bash
echo '555eff5f4026774f477bebdcbb3b52548e0da230803959dcebcea4d732a90dd9  /data/downloads/cvpr2025_image.pth.tar' | sha256sum -c -
```

### 4.3 构建 ARM64 / CUDA 13 离线包

```bash
./offline_deployment/prepare_bundle.sh \
  arm64-cu130 \
  /data/downloads/cvpr2025_image.pth.tar \
  /data/export/dcvc-offline-arm64-cu130
```

使用 NGC 公共镜像时通常可直接拉取；如果 registry 要求认证，按
[DGX Spark NGC 指南](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)登录后重试。
NVIDIA 25.08 镜像使用 Python 3.12、CUDA 13.0 和 PyTorch 2.8.0a0；NVIDIA
支持矩阵把该系列列为 x86/Arm SBSA multi-arch。

### 4.4 构建 x86_64 / CUDA 12.4 离线包

```bash
./offline_deployment/prepare_bundle.sh \
  amd64-cu124 \
  /data/downloads/cvpr2025_image.pth.tar \
  /data/export/dcvc-offline-amd64-cu124
```

### 4.5 制包阶段的自动检查

`prepare_bundle.sh` 会自动：

1. 校验 checkpoint SHA-256；
2. 下载并校验自然视频；
3. 用 ffmpeg 均匀抽取 32 张 1280×720 PNG；
4. 按指定 CPU 架构构建镜像并检查镜像架构；
5. 将镜像导出为 `tar.gz`；
6. 保存源码快照、Git commit、镜像元数据和全目录 `SHA256SUMS`。

最终目录类似：

```text
dcvc-offline-arm64-cu130/
├── dcvc-rt-i-arm64-cu130.tar.gz
├── DCVC-source.tar.gz
├── IMAGE_INSPECT.json
├── SOURCE_COMMIT.txt
├── README_zh.md
├── run_offline.sh
├── verify_results.py
├── SHA256SUMS
└── payload/
    ├── checkpoints/cvpr2025_image.pth.tar
    └── testdata/
        ├── starigrad_105323_720p.webm
        └── frames_32/frame_001.png ... frame_032.png
```

不要把两个架构的镜像混在同一个目录。复制到移动硬盘前执行：

```bash
cd /data/export/dcvc-offline-arm64-cu130
sha256sum -c SHA256SUMS
```

Docker 官方支持用 `docker save` 导出、`docker load` 导入 gzip 镜像：
[save 文档](https://docs.docker.com/reference/cli/docker/image/save/)、
[load 文档](https://docs.docker.com/reference/cli/docker/image/load/)。

## 5. 离线服务器：准备宿主机运行时

### 5.1 已装 Docker/Toolkit 时

DGX Spark 的官方系统通常预装并配置 Docker 与 NVIDIA Container Runtime；
GB10 合作伙伴服务器仍应实际检查，不能只按产品名假设。执行：

```bash
sudo systemctl status docker --no-pager
nvidia-ctk --version
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

`nvidia-ctk runtime configure` 会修改 `/etc/docker/daemon.json`。如果服务器已有
自定义 Docker 配置，先备份并审阅该文件。NVIDIA 官方步骤见
[Configuring Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)。

### 5.2 完全没有 Docker/Toolkit 时如何离线安装

不要把另一种 Ubuntu 版本或另一种 CPU 架构的 `.deb` 复制过来。准备两台
**临时联网、同 Ubuntu 版本、同 CPU 架构**的干净虚拟机/实体机：

- ARM 服务器：Ubuntu 24.04 ARM64；
- 3090 服务器：Ubuntu 20.04 amd64。

在联网临时机上先按 Docker 与 NVIDIA Container Toolkit 的官方安装文档配置
软件源，然后清空缓存并只下载包：

Ubuntu 24.04 ARM64：

```bash
sudo apt-get clean
sudo apt-get update
sudo apt-get install --download-only \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
  nvidia-container-toolkit
mkdir -p host-debs-noble-arm64
cp /var/cache/apt/archives/*.deb host-debs-noble-arm64/
```

Ubuntu 20.04 amd64：

```bash
sudo apt-get clean
sudo apt-get update
sudo apt-get install --download-only docker.io nvidia-container-toolkit
mkdir -p host-debs-focal-amd64
cp /var/cache/apt/archives/*.deb host-debs-focal-amd64/
```

这里要求使用“干净的同版本临时机”，是为了让 apt 同时下载依赖；如果临时机
已经装过依赖，缓存目录可能不完整。将对应 `host-debs-*` 目录复制到离线机后：

```bash
cd host-debs-noble-arm64    # 另一台使用 host-debs-focal-amd64
sudo apt install ./*.deb
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable --now docker
sudo systemctl restart docker
```

若 `apt install ./*.deb` 报缺依赖，停止并回到匹配的联网临时机补齐，不要使用
`dpkg --force-*`。NVIDIA 显卡驱动不放入通用离线包：它与具体 GB10/DGX OS
版本或 3090 内核版本绑定；当前 `nvidia-smi` 已正常时，不应为本实验更换驱动。

## 6. 在两台离线服务器上运行

### 6.1 ARM64 / GB10

将整个 `dcvc-offline-arm64-cu130` 目录复制到服务器，然后：

```bash
cd /data/dcvc-offline-arm64-cu130
chmod +x run_offline.sh verify_results.py
sudo ./run_offline.sh arm64-cu130 /data/dcvc-result-gb10
```

### 6.2 x86_64 / RTX 3090

```bash
cd /data/dcvc-offline-amd64-cu124
chmod +x run_offline.sh verify_results.py
sudo ./run_offline.sh amd64-cu124 /data/dcvc-result-3090
```

脚本会先做全目录 SHA-256 校验，再导入镜像；随后以 `--network none` 运行，
证明测试阶段没有联网依赖。它先检查 CUDA、GPU 名称和 rANS 扩展，再运行完整
32 帧、16 个丢包率工作点的实验，最后验证 `results.json`。

若不想用 `sudo`，可以按 Docker 官方建议配置用户权限；请注意 `docker` 组
等价于较高的宿主机权限，不应随意添加不受信用户。

## 7. 成功标准与结果位置

以下条件全部满足才算部署成功：

- 容器内 `torch.cuda.is_available()` 为 `True`；
- GPU 名称分别包含 GB10/对应设备名和 `RTX 3090`；
- `import MLCodec_extensions_cpp` 成功；
- 实际处理 32 张 1280×720 图像；
- `results.json` 有从 0% 到 100% 的 16 个丢包点；
- 0% 丢包平均 PSNR 在 35～45 dB 的合理区间；同一数据集的已知参考值约为
  40.81 dB，跨架构/框架可能有小幅浮点差异；
- 编码和解码 FPS 都为正数；
- 终端最后输出 `PASS`。

结果目录包括：

```text
results.json
sample_packet_manifest.json
reconstructions/reference.png
reconstructions/plr_*.png
```

读取两台服务器的 FPS 和无丢包 PSNR：

```bash
python3 - <<'PY'
import json
with open('results.json', encoding='utf-8') as f:
    d = json.load(f)
print('GPU:', d['test']['device'])
print('Torch/CUDA:', d['test']['torch'], d['test']['cuda'])
print('encode FPS:', d['i_frame_fps']['encode_fps'])
print('decode FPS:', d['i_frame_fps']['decode_fps'])
print('0% loss PSNR:', d['loss_results'][0]['mean_psnr_db'])
PY
```

## 8. 常见错误

| 现象 | 含义 | 处理 |
|---|---|---|
| `could not select device driver ... [[gpu]]` | Docker 未接入 NVIDIA runtime | 重新执行 `nvidia-ctk runtime configure` 并重启 Docker |
| `CUDA driver version is insufficient` | 宿主机驱动低于镜像要求 | GB10 检查 580 系列；3090 检查至少 550.54.15 |
| `exec format error` | ARM/amd64 镜像用反了 | 核对 `uname -m`、离线包名和 `IMAGE_INSPECT.json` |
| `No module named MLCodec_extensions_cpp` | 镜像不是由本指南 Dockerfile 完整构建 | 重新制包；不能跳过 `src/cpp` 构建步骤 |
| `cannot import cuda implementation ... fallback to pytorch` | 可选优化扩展未安装 | 本方案的预期提示，不是失败 |
| SHA-256 校验失败 | 传输损坏或文件被改动 | 不要继续运行，重新复制对应文件 |
| 0% PSNR 明显不在 35～45 dB | 权重、帧、QP 或软件栈不一致 | 核对 checkpoint/video SHA、QP=63、32 张帧和 Git commit |
| Ubuntu 20.04 无法安装当前 Docker CE | 该系统已退出当前 Docker 官方支持列表 | 使用 Ubuntu focal 的 `docker.io`；仍失败则规划升级 Ubuntu 22.04 |

## 9. 可复现性记录

每次交付时保留以下文件，不要只复制镜像：

- `SHA256SUMS`：确认传输前后内容一致；
- `SOURCE_COMMIT.txt`：记录运行的源码 commit；
- `IMAGE_INSPECT.json`：记录镜像架构、层和配置；
- 两台服务器各自的 `results.json`；
- `nvidia-smi`、`docker --version`、`nvidia-ctk --version` 的文本输出。

两台机器的 FPS 不应直接用一次结果下结论。正式比较时建议每台连续运行三次，
保留三份结果，报告后三次中位数；运行期间不要有其他 GPU 作业。
