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

## 正式 Statistics 和训练

第一笔监控实现提交时本节状态为 `PENDING`。production statistics、真实 T5 cache、full-model throughput probe、有限预算正式 run、最终/选定 checkpoint、hash 和 loss 摘要将在训练真正结束后补录，并由第二笔小型实验记录提交固化。在这些结果存在前，不声称 Phase 7 完成或模型已收敛。

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

## 已知风险和后续条件

- tiny CPU smoke 不证明 7B trainable parameter 图在 bf16、ZeRO2、4 GPU 上的显存、通信、吞吐或稳定性。
- validation loss 含 diffusion timestep/noise 随机性；短窗口波动不能直接解释为泛化变化或过拟合。
- `train/step_time` 是 trainer 可观测 wall time，不额外调用 `cuda.synchronize()`；避免监控本身增加每步 GPU barrier。
- 正式 run 前必须生成 90 episodes/task 的 production statistics、六 prompt T5 cache，并用真实 12 GB selective checkpoint 做 full-model step。
- Phase 7 最终必须等待正式 run 结束、checkpoint/event 落盘且摘要生成后，才能提交实验记录并停止。
