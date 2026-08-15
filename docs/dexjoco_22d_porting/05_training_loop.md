# Phase 5：DexJoCo 22D Joint Post-training 完整训练链路

## 阶段记录

- 日期：2026-08-15
- 阶段目标：打通六任务 dataloader、normalization、22D joint diffusion、Video/Arm/Hand forward、原 video/world objective、完整 22D action objective、backward、optimizer/scheduler step 和可恢复 checkpoint。
- 阶段边界：只运行 1 个 tiny CPU optimizer step；不加载 5B 权重到训练模型，不执行正式长时间训练、完整 pytest、DeepSpeed/多卡训练、推理或 simulator。
- 计划 commit message：`feat(train): close DexJoCo 22d training loop`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `main` | `43f450641504ee0b0609511e0ca372a7c910f433` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

开始前已读取 `docs/dexjoco_22d_porting/` 全部记录并检查两个工作树。Phase 4 实际 commit hash `43f450641504ee0b0609511e0ca372a7c910f433` 已回填 `commit_index.md`。DexJoCo 两个用户文件未修改、未暂存。

## 训练链路

真实训练入口保持为：

```text
DexJoCoRobotVideoDataset / DataLoader
  -> DexJoCoFastWAMProcessor normalization
  -> DexJoCoDualActionFastWAM.training_loss
  -> one [B,T,22] noise + one action timestep
  -> Arm [:6] / Hand [6:22] split
  -> Video + Arm + Hand MoT forward
  -> existing video/world MSE + one concatenated 22D action MSE
  -> backward
  -> named AdamW groups
  -> proportional scheduler step
```

`DexJoCoDualActionFastWAM.training_loss()` 的默认数学定义未改动。本阶段只在 `return_outputs=True` 时额外返回完整 `target_action` 和共享的 `timestep_action`，使 smoke 能直接复算并断言 action loss。默认训练返回仍是 `(loss_total, {loss_video, loss_action})`：console 中 `loss` 是 total，`loss_video`/`loss_action` 是已有加权分项。

Smoke 的 counting scheduler 观测到：

```text
sample_training_t calls = 1
add_noise calls = 1
add_noise input shape = [6,4,22]
```

Arm 和 Hand 的 `pre_dit()` 接收同一个 `timestep_action` tensor；预测先拼接为 `[arm_6d, hand_16d]`，再与完整 `[6,4,22]` target 做一次 `F.mse_loss(..., reduction="none").mean(dim=2)`。smoke 按相同 padding mask 和 scheduler weight 独立复算，结果与 `loss_action` 一致。没有分别对 Arm/Hand 求 mean 后相加。

## Checkpoint 和 Resume 契约

`Wan22Trainer.save_checkpoint()` 继续保存独立权重 `.pt` 和 Accelerate state；DexJoCo 路径额外在 state directory 写入：

| 文件 | 内容 |
| --- | --- |
| `pytorch_model.bin` | Accelerate 模型状态；PyTorch serialization 保留 MoT/expert 的共享参数别名 |
| `optimizer.bin` | 三个命名 AdamW group 及 state |
| `scheduler.bin` | scheduler state |
| `training_config.yaml` | 完整 resolve 后的训练配置 |
| `dexjoco_training_manifest.json` | `fastwam.dexjoco.training_checkpoint@1`、22/6/16/23D metadata、artifact 索引和 Accelerate state 文件清单 |
| `dataset_stats.json` | 本次训练实际使用的 statistics 副本 |
| `selective_loading_report.json` | Phase 3 selective loading report 副本 |
| `trainer_state.json` | global step、epoch 和 batch offset |

DexJoCo state 使用 `safe_serialization=False`，原因是 `video_expert`/`action_expert`/`hand_expert` 同时注册在 MoT 下；默认 safetensors 会把共享别名判为重复 key 并丢弃。原 FastWAM/LIBERO checkpoint 路径不改变默认序列化选项。

`Wan22Trainer.load_training_state()` 在调用 `accelerator.load_state()` 前执行 `validate_dexjoco_training_checkpoint()`。它 fail-fast 检查：

- manifest schema/version；
- action=22、Arm=6、Hand=16、proprio=23；
- saved config 的 `data.contract` 维度；
- `dataset_stats.json` 的 `fastwam.dexjoco.dataset_stats@1`、维度、任务、split 和 production/smoke policy；
- 当前 dataloader 实际使用的 statistics 与 checkpoint 副本逐字段完全一致；
- selective report 的 `fastwam.dexjoco_selective_checkpoint@1`；
- model/optimizer/scheduler 三类 state 声明及 manifest 所列 state 文件存在性。

Legacy FastWAM 没有 Hand Expert，不要求 DexJoCo manifest，原 resume 行为保持不变。

## 单步 Smoke Training

执行命令：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_training_loop.py
```

数据来源：

- data root：`/tmp/dexjoco_phase1_smoke.Rlts3g/data`；
- statistics：`/tmp/dexjoco_phase1_smoke.Rlts3g/hydra_work/dataset_stats.json`；
- 每任务取 episode 0 的首个训练 clip；四任务链接本地数据，两项本地缺失任务使用 Phase 1 下载的官方最小文件；
- image/action/state 来自真实数据和真实 loader/normalizer；
- T5 context 是临时全零 `[3,32]` cache，只用于避免 smoke 加载完整 text encoder；
- statistics 为六任务各 1 episode 的 `statistics_mode=smoke`、`production=false`，仅通过显式 `allow_non_production_stats=true` 使用，不是正式统计。

输入和输出：

```text
tasks   = water_plant, hammer_nail, click_mouse,
          pick_bucket, pinch_tongs, fold_glasses
action  = [6,4,22]
state   = [6,4,23]
cameras = [6,2,3,5,16,16]
video   = [6,3,5,16,32]
arm prediction  = [6,4,6]
hand prediction = [6,4,16]
joint prediction/target = [6,4,22]
```

本次确定性 seed 结果：

| Loss | 值 | 检查 |
| --- | ---: | --- |
| total | `4.11274576` | finite |
| video/world | `2.52383971` | finite；沿用原 FastWAM objective |
| action | `1.58890617` | finite；独立复算完整 22D MSE 一致 |

Gradient 检查在 `optimizer.step()` 前执行：

| 模块 | 有 grad tensor / 总 tensor | grad norm | step 后变化 tensor |
| --- | ---: | ---: | ---: |
| Video DiT | 42 / 42 | `4.70599127` | 42 |
| Arm backbone | 37 / 37 | `0.12315085` | 37 |
| Arm projection/head | 4 / 4 | `0.44669697` | 4 |
| Hand backbone | 37 / 37 | `0.14989901` | 37 |
| Hand projection/head | 4 / 4 | `0.49128401` | 4 |
| proprio encoder | 2 / 2 | `0.04934722` | 2 |

所有以上 gradient 都是 finite 且非零。T5/text encoder 和 VAE 的所有 gradient 始终为 `None`，其代表性及全部 probe 参数在 step 前后逐 tensor 完全相等。

Smoke checkpoint 只存在于自动清理的 `/tmp/fastwam_dexjoco_phase5_*` 临时目录。保存后用新 tiny model、optimizer 和 scheduler 完成实际 `load_training_state()`：模型所有 state tensor 精确相等，optimizer state 数量/group name/LR 相等，scheduler state 相等，global step/batch offset 恢复为 `1/1`。随后分别破坏 manifest action dimension、manifest proprio dimension 和 statistics schema；三者均在 `accelerator.load_state()` 前抛错。未向仓库添加 checkpoint。

## Targeted 回归

实际执行：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_dual_action_model.py
conda run -n fastwam python scripts/smoke_test_dexjoco_selective_checkpoint.py
conda run -n fastwam python scripts/smoke_test_dexjoco_optimizer.py

conda run -n fastwam python -m compileall -q \
  src/fastwam/trainer.py \
  src/fastwam/models/wan22/dexjoco_dual_action.py \
  scripts/smoke_test_dexjoco_training_loop.py
git diff --check
```

结果均 PASS。没有运行完整 pytest、正式 5B forward/backward、DeepSpeed 或多卡测试。

## 正式命令草案

当前长期数据根缺少 `click_mouse` 和 `pinch_tongs`，且 production statistics 尚未生成，因此以下仅是补齐完整六任务数据后的命令草案，未在本阶段执行：

```bash
# 1. 必须省略 --max-episodes-per-task，计算完整 training split production stats
conda run -n fastwam python scripts/compute_dexjoco_stats.py \
  --data-root /abs/path/to/complete/dexjoco_lerobot \
  --output /abs/path/to/dexjoco_6task_train_dataset_stats.json \
  --split train \
  --action-horizon 32

# 2. 预计算正式 T5 cache 后，从仓库根目录启动 joint post-training
DEXJOCO_LEROBOT_ROOT=/abs/path/to/complete/dexjoco_lerobot \
conda run -n fastwam accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  scripts/train.py \
  task=dexjoco_joint_2cam224_1e-4 \
  data.train.pretrained_norm_stats=/abs/path/to/dexjoco_6task_train_dataset_stats.json \
  output_dir=/abs/path/to/runs/dexjoco_joint_22d
```

不得为正式训练设置 `data.train.processor.allow_non_production_stats=true`。正式启动前还必须确认完整六任务文件、T5 cache、12 GB selective checkpoint、单机多卡/DeepSpeed 显存配置和 checkpoint 写入空间。

## 修改文件

- `src/fastwam/trainer.py`
- `src/fastwam/models/wan22/dexjoco_dual_action.py`
- `scripts/smoke_test_dexjoco_training_loop.py`
- `docs/dexjoco_22d_porting/05_training_loop.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

## 已知风险和停止点

- 本阶段 tiny smoke 使用真实 `WanVideoDiT`/两个 `ActionDiT`/MoT，但只有 1 layer、float32 CPU；不证明正式约 7B trainable parameter 图的显存、吞吐、数值稳定性或通信正确性。
- save/reload 验证为单进程 Accelerate；DeepSpeed ZeRO2、多 rank shard 和跨机器 resume 尚未验证。
- smoke text context 为合成零值，不能覆盖真实 T5 embedding 分布；T5 冻结和无梯度契约已覆盖。
- 当前 production statistics 不存在；smoke statistics 明确 `production=false`，不能用于正式训练。
- DexJoCo 双 Action Expert cache/inference 仍按 Phase 2 显式禁用；训练链路闭环不等于 simulator 推理闭环。
- Phase 5 commit 后停止，不进入正式训练或下一阶段。
