# Docker 使用说明

## 支持平台

- Linux：Docker Engine + Compose v2
- Windows：Docker Desktop + WSL2 Integration
- Intel Mac：Docker Desktop
- Apple Silicon：Docker Desktop，使用 `linux/amd64` 模拟

Windows 用户应在 WSL2 的 Linux 文件系统中 clone。不要把仓库放在
`/mnt/c` 或 `/mnt/e`；Mathlib 包含大量小文件，跨系统挂载会明显变慢。

## 启动

```bash
git clone <repository-url>
cd <repository-directory>
cp .env.example .env

docker compose config
docker compose build
docker compose run --rm prover-dev bash ./scripts/docker-smoke-test.sh
```

进入开发容器：

```bash
docker compose run --rm prover-dev bash
```

运行测试：

```bash
docker compose run --rm prover-dev python -m pytest -q
docker compose run --rm prover-dev \
  python -m scripts.pantograph_smoke_test
```

## 容器结构

```text
/workspace          bind mount 的项目源码
/opt/lean-project   镜像内构建好的 Lean/Mathlib 项目
/opt/venv           Python/Pantograph 环境
/opt/elan           Elan 和 Lean toolchain
```

Python 源码修改会立即反映到容器。修改以下内容后需要重新 build：

- `Dockerfile`
- `pyproject.toml` 或 `uv.lock`
- `lean-toolchain`
- `lakefile.lean` 或 `lake-manifest.json`
- `lean_project/` 中的 Lean 源码

## 固定版本

| 组件 | 版本 |
| --- | --- |
| Python | `3.12.3` |
| uv | `0.8.24` |
| Elan | `3.1.0` |
| Lean | `4.29.1` |
| Mathlib | `5e932f97dd25535344f80f9dd8da3aab83df0fe6` |
| PyPantograph | `0.3.15` / `ffa7f243824d2762825abddb1e9f6e939ede761f` |
| Pantograph | `842c0fe6e76b0771cc7f7939604c7a0b90e17433` |

Docker 构建不会复制宿主的 `.venv`、`.elan`、`.lake`、`.vendor`、
`.olean` 或 `.ilean`。Lean、Mathlib 和 Pantograph 均根据锁定配置重新构建。

## 环境变量

`.env.example` 包含：

- 模型配置：`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`MODEL_BASE_URL`、
  `MODEL_NAME`
- 权限：`HOST_UID`、`HOST_GID`
- 源码挂载：`PROJECT_DIR`
- 容器架构：`DOCKER_PLATFORM`

不要提交包含真实密钥的 `.env`。

Apple Silicon 保持：

```text
DOCKER_PLATFORM=linux/amd64
```

## Volumes

- `uv-cache`：Python 下载缓存
- `elan-cache`：Lean toolchain
- `mathlib-cache`：Mathlib 下载缓存
- `prover-data`、`prover-logs`、`prover-models`、`prover-checkpoints`：
  项目运行数据

删除本项目 containers 和 volumes：

```bash
docker compose down --volumes --remove-orphans
```

完全重新构建：

```bash
docker compose build --no-cache
```

## 常见问题

### Toolchain 不匹配

```bash
docker compose run --rm prover-dev \
  bash ./scripts/check-toolchain.sh
```

Lean 和 Lake 应报告 `4.29.1`。

### Mathlib 或 Git 下载失败

直接重试 `docker compose build`。构建脚本只恢复 manifest 中的固定
commit，不会升级依赖。

### Pantograph 无法导入

```bash
docker compose run --rm prover-dev \
  python -c "from pantograph.server import get_version; print(get_version())"
```

输出应为 `0.3.15`。

### 文件权限错误

在 `.env` 中设置当前用户 ID：

```bash
printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" >> .env
docker compose build
```

### Windows/WSL2 运行很慢

确认仓库位于 WSL2 Linux 文件系统，而不是 `/mnt/c`、`/mnt/e`。
