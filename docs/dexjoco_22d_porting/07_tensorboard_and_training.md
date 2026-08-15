# Phase 7：TensorBoard 监控、收敛摘要与正式训练

## 阶段记录

- 日期：2026-08-15
- 阶段目标：为 DexJoCo joint post-training 增加可恢复的 TensorBoard 训练/验证监控和工程收敛摘要；生成 `rand_obj` 正式 training-subset statistics；执行一次有限预算 full-model joint post-training 并记录产物。
- 阶段边界：只运行 4+1 optimizer step tiny smoke，不运行完整 pytest；正式 run 是实验执行，不是测试；不运行六任务 simulator 评测，不提交 event、checkpoint、T5 cache、数据或大型日志。
- 监控实现 commit message：`feat(train): add DexJoCo TensorBoard monitoring`
- 实验记录 commit message：`docs: record DexJoCo joint post-training run`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `fastwam22d-train` | `34e1dd30158b05924a59496872930cf0ff01bb2a` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `3c85e48dc50c29e204259261eb97f3419e26e969` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

Phase 6 FastWAM hash 已回填为 `34e1dd30158b05924a59496872930cf0ff01bb2a`；DexJoCo client/fix hashes 为 `992abca3da2bb34475485658bf76b766c49b7efa` 和 `3c85e48dc50c29e204259261eb97f3419e26e969`。两个 DexJoCo 用户文件保持原样，未修改、未暂存。

## 真实入口和 Rank 语义

训练入口仍是：

```text
scripts/train.py
  -> fastwam.runtime.run_training()
  -> build_datasets()
  -> Wan22Trainer
  -> DexJoCoDualActionFastWAM.training_loss()
```

`Wan22Trainer.global_step` 只在未被 Accelerate 判定为 skipped 的 optimizer step 后递增。TensorBoard writer 在 checkpoint resume 完成后创建，resume 使用恢复后的 step 作为 `purge_step`，之后所有 scalar 的 step 都取真实 `global_step`。仅 `accelerator.is_main_process` 创建 writer 和写 event；其他 rank 不创建 event 文件。

writer 输出到 `${output_dir}/tensorboard`，按 `tensorboard.flush_every` 定期 flush，并由 `Wan22Trainer.train()` 的 `finally` 路径在正常返回和异常退出时 flush/close。默认 FastWAM 配置关闭 TensorBoard，DexJoCo task 配置显式开启；histogram 未实现且默认无高频昂贵监控。

Full DexJoCo 模型在各 rank 上先构造模块、再转换到配置指定的 bf16；`run_training()` 在模型实例化后、首个 distributed collective 前执行 Python GC 和 `torch.cuda.empty_cache()`，只释放 dtype 转换留下的非活跃 allocator cache。活跃 Video/Arm/Hand/VAE 参数、selective loading 结果和训练数学不受影响。这避免约 7.04B 参数构造期的临时 fp32 allocation 挤占 NCCL workspace。

### 冻结策略和显存语义

正式 run 没有全量训练架构中的每个模块。实际状态为：

| 模块 | 初始化 | 训练状态 | 训练进程中的作用 |
| --- | --- | --- | --- |
| Video DiT | pretrained | trainable | video/world objective |
| Arm/Action DiT backbone | pretrained | trainable | 6D Arm Expert |
| Hand Transformer backbone | 从 pretrained Action Expert remap | trainable | 16D Hand Expert |
| Arm/Hand projection、23D proprio encoder | random/new initialization | trainable | 新 DexJoCo action/state contract |
| T5/text encoder | 不加载权重；使用离线 cache | frozen/out of graph | 每个 prompt 直接读取 `[128,4096]` bf16 embedding |
| VAE | pretrained | frozen、eval | 只编码视频 latent |

冻结会减少 gradient、optimizer state 和 backward activation，但不会自动减少模型定义或 checkpoint 的参数总量；仍参与 forward 的冻结模块也可能占用显存。T5 因使用 cache 完全不驻留训练图，VAE 权重仍需驻留做编码。optimizer 实际覆盖：`action_new=145,430`、`action_backbone=2,041,741,312`、`video_backbone=4,999,787,712`，合计 `7,041,674,454` 个 trainable parameters。因此冻结 T5/VAE 后训练负担确实下降，但其余约 7.04B trainable parameters 的 Adam states 仍使 ZeRO-2 在首步 OOM；正式 run 改用 ZeRO-3 分片参数、梯度和 optimizer state，未改变冻结/训练策略。

## Episode 划分和 Statistics 一致性

正式数据根为 `/home/shared/ai/datasets/dexlewm/dexjoco`，包含六任务各 100 episodes，共 218,993 frames，数据组织符合 `rand_obj`。本阶段冻结以下策略：

- 每任务由 `per_task_seeded_shuffle_v1`、seed `42` 独立确定性划分；
- 90 episodes/task 用于 training 和 production statistics；
- 10 episodes/task 只用于 validation loss；
- `rand_full` 不混入训练，保留为后续独立泛化评测。

`select_dexjoco_episode_indices()` 同时被 `DexJoCoV3Dataset` 和 `compute_dexjoco_statistics_from_root()` 使用。statistics JSON 新增 `data_distribution=rand_obj` 和 `episode_split`，记录 policy、seed、比例、subset 及各任务实际 episode IDs。实测 train/validation 每任务为 `90/10`，交集为空且并集覆盖 100 episodes。

当 `pretrained_norm_stats` 已指向 `${output_dir}/dataset_stats.json` 时，dataset 初始化不会再读后写回同一文件；其他路径复制到 run directory 时使用临时文件加 `os.replace()` 原子发布。该规则避免多 rank 同时构建 train/validation dataset 时读到部分 JSON，也保留正式 statistics 的原始字节和 SHA256。

## TensorBoard Scalars

训练日志步不执行第二次 forward。DexJoCo `training_loss(..., return_outputs=True)` 返回现有 joint prediction/target，诊断只从这些 tensor 计算：

| 类别 | Tags |
| --- | --- |
| 训练 objective | `train/loss_total`、`train/loss_video`、`train/loss_action_22d` |
| 训练诊断 | `train/loss_arm_6d`、`train/loss_hand_16d` |
| 验证 | `val/loss_total`、`val/loss_video`、`val/loss_action_22d`、`val/loss_arm_6d`、`val/loss_hand_16d` |
| LR | `lr/action_new`、`lr/action_backbone`、`lr/video_backbone` |
| 梯度 | `grad_norm/video`、`grad_norm/arm`、`grad_norm/hand`、`grad_norm/action_new` |
| 性能/进度 | `train/step_time`、`train/data_time`、`train/epoch`、`train/samples_seen` |
| 数值稳定性 | `action_pred/mean`、`action_pred/std`、`action_target/mean`、`action_target/std` |

Arm/Hand diagnostic 使用与 joint objective 相同的 padding mask、diffusion scheduler weight 和 action loss lambda，但分别在 6D/16D 上求值，所以只用于定位问题。正式优化仍只反传原 `loss_video + loss_action_22d`；没有把两个 diagnostic mean 相加替代 22D objective。

分组梯度范数只在日志 step、AMP unscale 后和 clip 前计算。DexJoCo validation 使用单个确定性 per-rank sample 的 `training_loss`，不执行视频 rollout、action sampling、MP4 写入或 simulator。

## 工具脚本

TensorBoard 依赖固定为 `tensorboard==2.21.0`。`2.19.0` 在当前 `setuptools==83.0.0` 环境因已移除的 `pkg_resources` 无法启动，因此未采用。

启动命令：

```bash
conda run --no-capture-output -n fastwam python \
  scripts/launch_dexjoco_tensorboard.py \
  --logdir <run_dir>/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

启动器校验 logdir 和 port，默认只监听 `127.0.0.1`，拒绝通过 passthrough 重复覆盖 `--logdir`、`--host`、`--port` 或使用 `--bind_all`。实测访问地址为 `http://127.0.0.1:16006/`。

收敛摘要命令：

```bash
conda run -n fastwam python scripts/summarize_dexjoco_training.py \
  --logdir <run_dir>/tensorboard \
  --output-json <run_dir>/summaries/convergence.json \
  --output-markdown <run_dir>/summaries/convergence.md \
  --window 10 \
  --min-points 20
```

脚本为 total/video/action/arm/hand 分别输出首尾窗口 mean/median、relative reduction、末窗口 raw/relative slope、coefficient of variation、NaN/Inf count 和最后有效 step。状态为 `decreasing`、`plateau/converged`、`diverging`、`unstable` 或 `insufficient_data`。这是工程 loss 诊断，不替代 DexJoCo simulator success rate 或独立 `rand_full` 指标。

## TensorBoard Smoke

执行：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_tensorboard.py
```

范围：六任务各一个真实 `rand_obj` episode 的 low-dimensional statistics、真实双相机最小 clips、1-layer tiny Video/Arm/Hand MoT、CPU float32、batch size 2。先执行 4 optimizer steps，保存临时 state，再从 step 4 resume 到 step 5。

结果：

```text
optimizer_steps=5 initial=4 resumed_from=4 final=5
event_files=2 required_tags=25
train_loss_steps=[1,2,3,5]
no_step_zero=PASS resume_continuity=PASS
lr ratios action_new:action_backbone:video_backbone = 10:5:1 PASS
Video/Arm/Hand/action_new grad norms finite and nonzero PASS
T5/VAE gradients=None and parameters unchanged PASS
summary status for all five losses=insufficient_data
```

TensorBoard 的 `purge_step=4` 在 resume event 中移除旧 step 4，再由新 run 从 step 5 继续，因此聚合后 `[1,2,3,5]` 是预期去重结果。25 个必需 scalar tag 均至少有一个 finite 值。smoke 仅有 4 个聚合有效点，摘要在 `window=3,min_points=10` 下正确返回 `insufficient_data`。临时 events、checkpoint、statistics 和 summaries 均自动删除。

额外执行：

```bash
conda run -n fastwam python -m compileall -q \
  src/fastwam/trainer.py src/fastwam/runtime.py \
  src/fastwam/datasets/lerobot/dexjoco_v3_dataset.py \
  src/fastwam/datasets/lerobot/dexjoco_stats.py \
  scripts/compute_dexjoco_stats.py \
  scripts/launch_dexjoco_tensorboard.py \
  scripts/summarize_dexjoco_training.py \
  scripts/smoke_test_dexjoco_tensorboard.py
conda run -n fastwam python scripts/smoke_test_dexjoco_dual_action_model.py
conda run -n fastwam python scripts/smoke_test_dexjoco_optimizer.py
conda run -n fastwam python scripts/smoke_test_dexjoco_training_loop.py
git diff --check
```

均 PASS。Phase 2 dual-action/legacy instantiate、Phase 4 optimizer/scheduler 和 Phase 5 单步 loss/checkpoint 数值保持原结果；未运行完整 pytest。

## 正式 Statistics

正式 statistics 使用 `/home/shared/ai/datasets/dexlewm/dexjoco` 的 `rand_obj` training subset；`rand_full` 未混入训练。六任务各 100 episodes，经 seed 42 的确定性 90/10 划分后，只读取 90 episodes/task：

```bash
conda run --no-capture-output -n fastwam python scripts/compute_dexjoco_stats.py \
  --data-root /home/shared/ai/datasets/dexlewm/dexjoco \
  --output /home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/dataset_stats.json \
  --split train \
  --split-seed 42 \
  --val-set-proportion 0.1 \
  --action-horizon 32
```

结果为 `fastwam.dexjoco.dataset_stats@1`、`production=true`、action 22D、proprio 23D；540 episodes/197,528 training transitions，validation 的 60 episodes/21,465 transitions 未参与统计。所有 mean/std finite，`std_floor=1e-6`，action/state 的 `std_floor_applied_dimensions` 均为空。文件同时位于 run root 和 checkpoint state，字节一致：

```text
/home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/dataset_stats.json
SHA256 4b08421c418d7127f5c8f8e490e853791c095425dc657322dc21927f60a32e5f
```

六个正式 prompt 的 T5 cache 位于 `text_embeds_cache/dexjoco/`，共 6 个 `[128,4096]` bf16 文件，aggregate SHA256 为 `31aa571ea3e73786360131fcf91dc9edcae44f030f40617682f7a5a1dad8de8c`。训练过程没有实例化 T5。

## 正式 Joint Post-training

正式成功 run（Attempt 7）基于干净实现 commit `98c708c7e41e48e7b31c2ccf1bb388b2e23641b0`，于 2026-08-16 00:01:02 +08:00 完成。配置为 4 GPU、DeepSpeed ZeRO-3、bf16、每 GPU batch 1、global batch 4、gradient accumulation 1、seed 42、cosine scheduler、1 step warmup，共 20 optimizer steps；每 5 steps 做一次 loss-only validation，只保存最终 checkpoint。

真实启动命令：

```bash
NCCL_CUMEM_HOST_ENABLE=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
DEXJOCO_LEROBOT_ROOT=/home/shared/ai/datasets/dexlewm/dexjoco \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
conda run --no-capture-output -n fastwam accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero3_ds.yaml \
  --num_processes 4 \
  --main_process_port 29575 \
  scripts/train.py \
  task=dexjoco_joint_2cam224_1e-4 \
  output_dir=/home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815 \
  data.train.pretrained_norm_stats=/home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/dataset_stats.json \
  data.train.text_embedding_cache_dir=/home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/text_embeds_cache/dexjoco \
  batch_size=1 num_workers=2 max_steps=20 log_every=1 eval_every=5 save_every=0 \
  gradient_accumulation_steps=1 mixed_precision=bf16 tensorboard.flush_every=5 \
  tensorboard.log_dir=/home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/tensorboard_attempt7
```

`NCCL_CUMEM_HOST_ENABLE=0` 是该节点的必要 preflight 条件：未设置时，四 rank 的无模型 collective 也会在 NCCL shared-memory allocation 失败。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 用于减少 allocator fragmentation。两者都记录在 run metadata，不改变模型数学。

Pretrained checkpoint 为 `/home/user/fastwam-22d/checkpoints/libero_uncond_2cam224.pt`，SHA256 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`。selective loader 结果为 loaded 1,645、copied-to-hand 820、skipped-policy 10、newly-initialized 10、skipped-shape/missing/unexpected 均为 0；报告 SHA256 `85651652ac241d780a860c58064cd536f0645a58480d1d150bf270bb4f74c9a1`。

### Loss 和收敛摘要

训练首末点：

| Step | total | video/world | action 22D |
| ---: | ---: | ---: | ---: |
| 1 | 16.6813 | 0.3375 | 16.3438 |
| 20 | 1.0431 | 0.2053 | 0.8378 |

验证只含 4 个单样本 loss points：

| Step | total | video/world | action 22D | arm diag | hand diag |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 2.4384 | 0.6421 | 1.7963 | 1.3448 | 1.9656 |
| 10 | 1.8374 | 0.4006 | 1.4368 | 0.8467 | 1.6581 |
| 15 | 1.3584 | 0.3108 | 1.0476 | 0.4345 | 1.2776 |
| 20 | 1.6485 | 0.4089 | 1.2396 | 0.7188 | 1.4348 |

以 window 5、min-points 15 分析 20 个训练点：

| Loss | 首窗口 mean | 末窗口 mean | 相对下降 | 末窗口 CV | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| total | 6.30747 | 1.37002 | 78.28% | 0.1722 | decreasing |
| video/world | 0.612897 | 0.433252 | 29.31% | 0.2928 | decreasing |
| action 22D | 5.69457 | 0.936764 | 83.55% | 0.1389 | decreasing |
| arm 6D diagnostic | 2.46835 | 0.475694 | 80.73% | 0.2940 | decreasing |
| hand 16D diagnostic | 6.90441 | 1.10967 | 83.93% | 0.1196 | decreasing |

五条曲线的 NaN/Inf 均为 0、最后有效 step 均为 20。它们仍在下降，不能标记为 converged。validation 从 step 5 改善到 step 15，step 20 回升；由于只有四个随机 diffusion 单样本点，这只是波动/早期过拟合风险信号，不足以证明过拟合。尚未运行 simulator success 或 `rand_full` 泛化评测。

摘要文件：

```text
summaries/convergence.json SHA256 0b580f9147111d7dcb80d9390bf7e06f1fa2148463765eebbd62382f998a7fd4
summaries/convergence.md   SHA256 0c308e412fded66c2ab8da07e7ec801ff128641993a031803897fb4efcc58220
```

### TensorBoard 和 Checkpoint

正式 event 目录与访问命令：

```bash
python scripts/launch_dexjoco_tensorboard.py \
  --logdir /home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/tensorboard_attempt7 \
  --host 127.0.0.1 \
  --port 6006
```

event 文件为 26,940 bytes，SHA256 `7d800390fbfcf75922eeb3211745e373530c8e14c40ec309de659d17773909df`；包含 25 个 scalar tags、20 个 finite train points、steps 5/10/15/20 的 4 个 finite validation points，三组 LR 比例始终为 10:5:1，Video/Arm/Hand/action-new grad norms 均 finite。

权威 checkpoint：

```text
/home/user/fastwam-22d/runs/dexjoco_phase7_rand_obj_20260815/checkpoints/state/step_000020
size 85,920,058,245 bytes (~81 GiB)
aggregate SHA256 e3b1c7de732be2a8906f4690b7ebe7d87d5e99bddcb21720bd46c31a4f741629
```

aggregate hash 使用 checkpoint 根下 `find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum`。manifest 确认 global step 20、22/6/16/23D、四 rank model/optimizer shard、scheduler、random state、resolved config、production stats 和 selective report 均存在。ZeRO-3 下 `weights_checkpoint=null` 是设计行为，分片 state 是唯一可 resume 的权威 checkpoint。

step 15 有最低的观测 validation loss，但本次 `save_every=0` 只保存最终 step 20，因此不能选择不存在的 step-15 checkpoint。正式选择 step 20 作为唯一可恢复 checkpoint，不宣称它是 best-validation checkpoint。

### 尝试审计

正式产物只来自 Attempt 7。之前六次均不是重复的成功训练：Attempt 1/2 在 0 step 暴露 NCCL workspace 和 statistics 并发发布问题；Attempt 3 的 ZeRO-2 在首次 Adam state 分配 OOM；Attempt 4/5 在 0 step 暴露 ZeRO-3 wrapper 和 optimizer group-name 兼容问题；Attempt 6 完成 1 step 后暴露 partition gradient norm 读取问题且未保存 checkpoint。各问题修复后才启动下一次，Attempt 7 单次连续完成 20 steps。

## 修改文件

- `pyproject.toml`
- `configs/train.yaml`
- `configs/task/dexjoco_joint_2cam224_1e-4.yaml`
- `configs/data/dexjoco_6task_2cam.yaml`
- `src/fastwam/trainer.py`
- `src/fastwam/runtime.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/dexjoco_v3_dataset.py`
- `src/fastwam/datasets/lerobot/dexjoco_stats.py`
- `scripts/compute_dexjoco_stats.py`
- `scripts/launch_dexjoco_tensorboard.py`
- `scripts/summarize_dexjoco_training.py`
- `scripts/smoke_test_dexjoco_tensorboard.py`
- `docs/dexjoco_22d_porting/07_tensorboard_and_training.md`
- `docs/dexjoco_22d_porting/decisions.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

## 已知风险和停止点

- 20 steps 是有限预算正式实验，证明 full-model bf16/ZeRO-3 训练、监控和 checkpoint 链路可运行，但不证明 loss 已收敛或策略已获得任务成功率。
- validation 只有四个随机 diffusion 单样本点；step 20 回升需要更稳定的 validation 和 simulator 指标确认，不能单独判为发散或过拟合。
- 训练日志出现 ZeRO-3 allocator cache-flush warnings，未导致数值错误，但代表高显存压力和吞吐风险；延长训练前应做显存/性能优化。
- `train/step_time` 是 trainer 可观测 wall time，不额外调用 `cuda.synchronize()`；避免监控本身增加每步 GPU barrier。
- 尚未运行 `rand_full` 泛化或六任务 simulator success 评测；loss 摘要只是工程诊断。
- checkpoint、events、statistics、T5 cache、日志、summary 和 run metadata 位于被忽略的 `runs/`，不提交 Git。
- Phase 7 实验记录 commit 后停止，不进入下一阶段或继续训练。
