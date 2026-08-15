# Phase 1：DexJoCo 六任务数据路径与 normalization

## 阶段记录

- 日期：2026-08-15
- 阶段目标：接入 DexJoCo LeRobot v3 六任务数据，冻结并运行 22D action / 23D state 数据契约，生成可复用且可鉴别来源的 normalization statistics。
- 阶段边界：只修改数据配置、dataset/processor、statistics 工具、聚焦测试和项目记录；不修改模型、trainer、loss、checkpoint 或推理算法；不运行全量 statistics、训练、模型推理或 simulator。
- 计划 commit message：`feat(data): add DexJoCo 22d action pipeline`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `main` | `835c98b8c42471a5addf5666967706957844d7f4` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

开始前已阅读 `docs/dexjoco_22d_porting/` 的全部 Phase 0 记录。两个 DexJoCo 用户文件在本阶段未修改、未暂存。

## 真实数据兼容结论

DexJoCo 六任务数据的 `meta/info.json` 声明 `codebase_version: v3.0`。真实布局是：

```text
data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4
meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet
```

一个 Parquet/MP4 文件可连续保存多个 episode；`meta/episodes` 通过 `dataset_from_index/dataset_to_index` 和每路视频的 chunk/file/timestamp 字段给出 episode 映射。FastWAM 原有 `BaseLerobotDataset`/内置 LeRobot fork按 v2 的 `tasks.jsonl` 和每 episode 文件寻址；用真实 `water_plant` 初始化时首先在 `LeRobotDatasetMetadata.load_metadata()` 因缺少 `meta/tasks.jsonl` 失败。因此本阶段增加 DexJoCo 专用 v3 backend，不改变 LIBERO 的原 backend。

## 正式入口和输出

数据配置名：`data=dexjoco_6task_2cam`，文件为 `configs/data/dexjoco_6task_2cam.yaml`。

Hydra 公开 dataset 入口：

```text
fastwam.datasets.lerobot.dexjoco_robot_video_dataset.DexJoCoRobotVideoDataset
```

实际 v3 backend：

```text
fastwam.datasets.lerobot.dexjoco_v3_dataset.DexJoCoV3Dataset
fastwam.datasets.lerobot.dexjoco_v3_dataset.DexJoCoV3TaskSource
```

processor 入口：

```text
fastwam.datasets.lerobot.processors.dexjoco_processor.DexJoCoFastWAMProcessor
```

使用配置的 `num_frames=33`、`action_video_freq_ratio=4` 后，单个样本和 DataLoader batch 的正式布局为：

| 字段 | 单样本 | batch | 含义 |
| --- | --- | --- | --- |
| `action` | `[32,22]` | `[B,32,22]` | 归一化后的完整绝对 action chunk |
| `arm_action` | `[32,6]` | `[B,32,6]` | `action[..., :6]` |
| `hand_action` | `[32,16]` | `[B,32,16]` | `action[..., 6:22]` |
| `proprio` | `[32,23]` | `[B,32,23]` | 与 action 对齐的 state/proprio |
| `camera_videos` | `[2,3,9,224,224]` | `[B,2,3,9,224,224]` | `primary,wrist` 两路独立视频 |
| `video` | `[3,9,224,448]` | `[B,3,9,224,448]` | 现有模型接口使用的水平拼接视频 |
| `action_is_pad` | `[32]` | `[B,32]` | action chunk padding mask |
| `proprio_is_pad` | `[32]` | `[B,32]` | 与裁剪后 proprio 对齐 |
| `image_is_pad` | `[9]` | `[B,9]` | 与采样后视频时间轴对齐 |

源相机统一映射为公开 alias `primary,wrist`。`click_mouse.primary` 来自 `observation.images.ego_right`，其余五任务来自 `observation.images.front`；`wrist` 均来自 `observation.images.wrist`。metadata 中两路视频可独立切换 file，因此各自使用自身的 file index 和相对 timestamp；每路 episode duration 都必须等于 `length / 30`。

`DexJoCoFastWAMProcessor` 强制检查：

- `action_dim=22`、`arm_action_dim=6`、`hand_action_dim=16`、`proprio_dim=23`。
- image meta 必须按 `primary,wrist` 排列且正好两路。
- action/state 不执行维度变换。
- `delta_action_dim_mask` 必须为 `null/None`；非空配置直接报错，绝不沿用 LIBERO 7D delta-action mask。

## Statistics schema 和使用方式

生成入口：`scripts/compute_dexjoco_stats.py`。它只接受 `--split train`，默认处理六个固定任务。带 `--max-episodes-per-task` 的任何限量结果自动写成：

```json
{
  "schema_name": "fastwam.dexjoco.dataset_stats",
  "schema_version": 1,
  "statistics_mode": "smoke",
  "production": false,
  "split": "train"
}
```

不带 limit 才生成 `statistics_mode=production`、`production=true`。文件同时记录六任务及顺序、action/state ordering、Allegro actuator ordering、22/6/16/23 维、action horizon、每任务 episode/frame count、std floor，以及 FastWAM `LinearNormalizer` 需要的 global/stepwise `min/max/q01/q99/mean/std`。

std 小于 `1e-6` 的零方差或近零方差维统一 clamp 到 `1e-6`，原始触发维度写入 `std_floor_applied_dimensions`。DexJoCo processor 默认只接受 schema/version、任务/字段顺序、维度、tensor shape 均匹配的 production train statistics，因此 LIBERO statistics 和 Phase 1 smoke statistics 都不能静默用于训练或推理。`allow_non_production_stats=true` 只在 smoke 脚本中显式启用。

`RobotVideoDataset` 的既有行为会在训练未提供 stats 时由训练 split 计算并写入当前 work/output directory 的 `dataset_stats.json`；提供 production `pretrained_norm_stats` 时会复制同一个文件到输出目录。验证/推理应把训练输出的同一 `dataset_stats.json` 传给 DexJoCo 配置，从而复用相同归一化参数。

### 完整 training split 命令

本阶段未执行全量统计。准备好包含六个任务的持久数据根后，真实命令为：

```bash
conda run -n fastwam python scripts/compute_dexjoco_stats.py \
  --data-root /home/shared/ai/datasets/dexlewm/dexjoco \
  --output /path/to/training-output/dataset_stats.json \
  --split train \
  --action-horizon 32
```

当前 `/home/shared/ai/datasets/dexlewm/dexjoco` 只有四任务；运行完整命令前必须补齐官方 `click_mouse` 和 `pinch_tongs` 目录。脚本遇到缺任务会直接失败，不会生成部分任务 production 文件。

## 修改文件

- `configs/data/dexjoco_6task_2cam.yaml`
- `scripts/compute_dexjoco_stats.py`
- `scripts/smoke_test_dexjoco_data.py`
- `src/fastwam/datasets/lerobot/dexjoco_contract.py`
- `src/fastwam/datasets/lerobot/dexjoco_robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/dexjoco_stats.py`
- `src/fastwam/datasets/lerobot/dexjoco_v3_dataset.py`
- `src/fastwam/datasets/lerobot/processors/dexjoco_processor.py`
- `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `tests/test_dexjoco_data.py`
- `docs/dexjoco_22d_porting/README.md`
- `docs/dexjoco_22d_porting/01_data_and_normalization.md`
- `docs/dexjoco_22d_porting/decisions.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

通用 dataset 的改动仅有默认关闭的 backend 构造 hook、`return_camera_videos=false` 扩展和普通 dict shape meta 支持；LIBERO 配置及其 delta-action 配置未修改。`FastWAMProcessor` 的一处 shape 断言改为 tuple-to-tuple 比较，修复 `torch.Size` 与 list 内容相同却比较失败的问题，不改变张量处理。

## 执行命令与 smoke 结果

Phase 1 使用的关键命令如下：

```bash
conda run -n fastwam python tests/test_dexjoco_data.py

conda run -n fastwam python scripts/compute_dexjoco_stats.py \
  --data-root /tmp/dexjoco_phase1_smoke.Rlts3g/data \
  --output /tmp/dexjoco_phase1_smoke.Rlts3g/dataset_stats.smoke.v2.json \
  --split train --action-horizon 32 --max-episodes-per-task 1

conda run -n fastwam python scripts/smoke_test_dexjoco_data.py \
  --data-root /tmp/dexjoco_phase1_smoke.Rlts3g/data \
  --stats /tmp/dexjoco_phase1_smoke.Rlts3g/dataset_stats.smoke.v2.json \
  --num-workers 2

conda run -n fastwam python -m compileall -q \
  src/fastwam/datasets/lerobot scripts/compute_dexjoco_stats.py \
  scripts/smoke_test_dexjoco_data.py tests/test_dexjoco_data.py
```

限量 statistics 结果：6 tasks、每任务 1 episode、共 6 episodes / 2449 frames，`schema=fastwam.dexjoco.dataset_stats@1`、`mode=smoke`、`production=False`。六任务真实 DataLoader 输出都符合上表 shape；2 workers 路径通过。episode 尾部 padding 检查为 action `31/32`、state `31/32`、image `8/9`。少量真实 action/state 的 normalize -> denormalize 最大绝对误差分别为 `5.960e-08` 和 `1.192e-07`。

`click_mouse`、`pinch_tongs` 的 metadata、Parquet 和 episode 0 所在两路 `file-000.mp4` 从官方 `DexJoCo/DexJoCo-Datasets-LeRobot` 下载到 `/tmp`；其余四任务使用本地持久数据。smoke 脚本只为绕过本阶段无关的 T5 cache 前置条件在临时目录写入零值 context；像素、低维数据、processor、normalizer、padding 和 DataLoader 都走正式路径。临时 smoke stats 和 context 不进入提交。

## 已知风险、未运行项和下一阶段前置条件

- 本机持久 LeRobot 数据根仍缺 `click_mouse`、`pinch_tongs`，正式统计/训练前必须补齐。
- 未计算完整六任务 training split statistics；当前 smoke JSON 明确不可用于 production。
- 未预计算真实 DexJoCo T5 embeddings；正式训练前需按六任务指令生成 `text_embedding_cache_dir` 内容。
- 未运行训练、模型 forward/inference、checkpoint 或 simulator；这些不属于 Phase 1。
- `num_frames=33`、30 FPS 和 `action_video_freq_ratio=4` 保持当前 FastWAM action/video 接口；与 DexJoCo 50 Hz simulator 的 runtime 重采样仍是后续问题。
- 进入下一阶段前必须提交本阶段、补齐持久六任务数据、生成 production `dataset_stats.json` 和真实 text embeddings，并取得明确授权。
