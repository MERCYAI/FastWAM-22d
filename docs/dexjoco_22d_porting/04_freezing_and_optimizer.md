# Phase 4：Freezing、Optimizer Groups 和 Scheduler 兼容

## 阶段记录

- 日期：2026-08-15
- 阶段目标：实现 DexJoCo joint post-training 的可审计冻结策略、三个命名 AdamW 参数组，以及保持组间学习率比例的 scheduler。
- 阶段边界：只处理参数训练策略和 optimizer/scheduler 构建；不执行 backward、正式训练、完整 pytest、分布式/DeepSpeed optimizer-state 恢复、推理或 simulator。
- 计划 commit message：`feat(train): add DexJoCo parameter groups`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `main` | `9854c30685985b371c890b6db7f777e50ed1e6d7` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

开始前已读取 `docs/dexjoco_22d_porting/` 全部记录、检查两个工作树，并把 Phase 3 实际 commit hash `9854c30685985b371c890b6db7f777e50ed1e6d7` 写入 `commit_index.md`。DexJoCo 两个用户文件未修改、未暂存。

## 真实 Checkpoint 前置审计

用户提供的 checkpoint 位于：

```text
/home/user/fastwam-22d/checkpoints/libero_uncond_2cam224.pt
size = 12,041,735,140 bytes
```

在正式 5B config 的 meta target 上执行只读 dry-run：

```bash
conda run -n fastwam python scripts/audit_dexjoco_selective_checkpoint.py \
  --checkpoint checkpoints/libero_uncond_2cam224.pt \
  --report /tmp/dexjoco_phase4_real_checkpoint_audit.json
```

结果：

```text
loaded=1645
copied_to_hand=820
skipped_shape=0
skipped_policy=10
missing_in_checkpoint=0
unexpected_in_checkpoint=0
newly_initialized=10
apply=false
training_started=false
```

因此 Phase 3 的关键门槛 `missing_in_checkpoint=0`、`skipped_shape=0` 已由真实 checkpoint 满足。报告只保存在 `/tmp`，12 GB checkpoint 由 `.gitignore` 排除，均不属于本提交。

## 冻结和 Module Mode 策略

真实入口为：

```text
fastwam.models.wan22.dexjoco_dual_action.DexJoCoDualActionFastWAM.configure_joint_post_training
fastwam.trainer.Wan22Trainer._apply_dit_only_train_mode
fastwam.trainer.Wan22Trainer._set_dit_only_train_mode
```

`configure_joint_post_training()` 先清除整个模型的梯度标志，再只打开明确允许训练的模块：

| 组件 | 初始化状态 | `requires_grad` | module mode |
| --- | --- | --- | --- |
| Video Expert / Video DiT | pretrained | true | train |
| Arm Expert Transformer backbone | pretrained | true | train |
| Hand Expert Transformer backbone | old Action Expert remap | true | train |
| Arm 6D `action_encoder/head` | newly initialized | true | train |
| Hand 16D `action_encoder/head` | newly initialized | true | train |
| 23D `proprio_encoder` | newly initialized | true | train |
| T5/text encoder | pretrained | false | eval |
| VAE | pretrained | false | eval |

Trainer 在 optimizer/Accelerate 初始化前、正式 `train()` 开始时，以及 evaluation 返回训练状态时都复用该入口。即使外部调用一次 `model.train()`，下一次策略应用也会把 T5/VAE 恢复到 frozen/eval；smoke test 对此做了显式模拟。

原 FastWAM/LIBERO 模型没有 `configure_joint_post_training()`，继续使用已有 DiT-only 策略和单一 `default` optimizer group。

## 参数组 Ownership

参数组构建入口：

```text
DexJoCoDualActionFastWAM.build_joint_post_training_parameter_groups
Wan22Trainer._build_optimizer
```

固定 ownership：

| Group | 模块 | 默认 LR |
| --- | --- | ---: |
| `action_new` | Arm/Hand `action_encoder`、Arm/Hand `head`、23D `proprio_encoder` | `1.0e-4` |
| `action_backbone` | Arm/Hand `ActionDiT` 中排除 `action_encoder.*` 和 `head.*` 的参数 | `5.0e-5` |
| `video_backbone` | 完整 Video Expert / Video DiT | `1.0e-5` |

模型侧根据 parameter object identity 校验：

- 参数组名称必须恰好为上述三个名称。
- LR 必须为 finite positive，weight decay 必须为 finite nonnegative。
- LR 顺序必须满足 `action_new > action_backbone >= video_backbone`。
- 同一个 parameter object 不得出现在两个组。
- frozen parameter 不得进入任何组。
- 所有 `requires_grad=True` parameter 必须恰好被覆盖一次，多出或缺失均 fail-fast。

Trainer 创建 AdamW 时保留每组 `name`、`lr` 和 `weight_decay`，启动日志逐组打印名称、LR、weight decay、tensor 数和参数量。组配置支持每组覆盖 weight decay；未提供时回退到全局 `weight_decay`。

## 配置和 Scheduler

新增正式任务配置：

```text
configs/task/dexjoco_joint_2cam224_1e-4.yaml
```

核心配置：

```yaml
optimizer_groups:
  action_new:
    lr: 1.0e-4
  action_backbone:
    lr: 5.0e-5
  video_backbone:
    lr: 1.0e-5
weight_decay: 1.0e-2
```

这些是可覆盖的 Hydra 值，不是模型中的唯一硬编码常量。通用 `configs/train.yaml` 新增 `optimizer_groups: null`，因此原任务默认行为不变。DexJoCo config 同时解析真实 checkpoint 相对路径、6D/16D/23D 维度，并保持 `data.train.processor.delta_action_dim_mask: null`；没有引用 LIBERO 7D statistics。

Scheduler 构建入口为 `fastwam.trainer.build_lr_scheduler()`：

- 多参数组 cosine 使用所有组共享的 multiplicative `LambdaLR` factor，最低比例默认 1%，所以不会把不同 base LR 覆盖成一个绝对值。
- warmup 使用共享的 multiplicative `LinearLR` factor，再通过 `SequentialLR` 进入主 scheduler。
- constant scheduler 使用 factor 1.0，同样保持每组 base LR。
- 原单组 cosine 保留 `CosineAnnealingLR` 和该组 base LR 1% 的 `eta_min` 行为。

训练日志继续提供兼容字段 `train/lr`，并新增 `train/lr/<group_name>`；控制台同时显示三组当前 LR。

## Optimizer Smoke Test

主命令：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_optimizer.py
```

Tiny 真实模块结果：

| Group | LR | Weight decay | Tensors | Parameters |
| --- | ---: | ---: | ---: | ---: |
| `action_new` | `1.0e-4` | `0.01` | 10 | 1,894 |
| `action_backbone` | `5.0e-5` | `0.01` | 74 | 26,928 |
| `video_backbone` | `1.0e-5` | `0.01` | 42 | 13,732 |

正式 5B 配置的 meta module graph 结果：

| Group | LR | Weight decay | Tensors | Parameters |
| --- | ---: | ---: | ---: | ---: |
| `action_new` | `1.0e-4` | `0.01` | 10 | 145,430 |
| `action_backbone` | `5.0e-5` | `0.01` | 1,640 | 2,041,741,312 |
| `video_backbone` | `1.0e-5` | `0.01` | 825 | 4,999,787,712 |

代表性归属：

```text
action.action_encoder.weight       -> action_new
hand.head.weight                   -> action_new
proprio_encoder.weight             -> action_new
action.blocks.0.self_attn.q.weight -> action_backbone
hand.blocks.0.ffn.0.weight         -> action_backbone
video.blocks.0.self_attn.q.weight  -> video_backbone
```

Scheduler 在 base LR `[1e-4,5e-5,1e-5]` 上构建 2-step warmup，执行一次 optimizer/scheduler step 后：

```text
LR = [7.5e-5, 3.75e-5, 7.5e-6]
common_factor = 0.75
```

三组比例和顺序保持。smoke 同时确认 trainable coverage、互斥、frozen exclusion、T5/VAE mode restore、错误 LR 顺序 fail-fast 和原 7D FastWAM 单组兼容。没有调用 backward 或开始训练。

## 回归和执行命令

除真实 checkpoint audit 和 optimizer smoke 外，实际执行：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_dual_action_model.py
conda run -n fastwam python scripts/smoke_test_dexjoco_selective_checkpoint.py

conda run -n fastwam python -m compileall -q \
  src/fastwam/models/wan22/dexjoco_dual_action.py \
  src/fastwam/trainer.py \
  scripts/smoke_test_dexjoco_optimizer.py

git diff --check
```

Hydra compose 另行断言 DexJoCo 三组 LR、checkpoint path、6/16/23D 和 absolute-action mask，以及 LIBERO `optimizer_groups=null`、action_dim=7。回归结果：dual-action forward 输出 Arm `[1,4,6]`、Hand `[1,4,16]`、完整 action `[1,4,22]`、attention mask `[16,16]`；selective synthetic loader 的成功分类、tensor equality 和故障 fail-fast 均通过。

## 修改文件

- `configs/train.yaml`
- `configs/task/dexjoco_joint_2cam224_1e-4.yaml`
- `src/fastwam/models/wan22/dexjoco_dual_action.py`
- `src/fastwam/trainer.py`
- `scripts/smoke_test_dexjoco_optimizer.py`
- `docs/dexjoco_22d_porting/04_freezing_and_optimizer.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

## 已知风险和后续前置条件

- 没有启动正式训练，没有调用 backward，也没有分配正式 AdamW state；meta 参数计数不能证明目标硬件的显存、吞吐或分布式通信行为。
- 没有运行完整 pytest，也没有验证 Accelerate/DeepSpeed 多 rank、ZeRO optimizer state 或从旧单组 optimizer state 恢复到三组结构。
- 真实 checkpoint 已通过只读分类，但本阶段没有把 12 GB tensor 应用到完整模型；正式训练启动仍需在目标硬件上验证 apply、单步 forward/backward 和 save/resume。
- 六任务 production `dataset_stats.json` 尚未生成，不能使用 smoke statistics 开始正式训练。
- DexJoCo 双 Action Expert cache/inference 仍按 Phase 2 显式禁用；不得把训练结构完成等同于 simulator 闭环可用。
- 进入下一阶段前必须获得明确授权，并先满足 production statistics 和目标分布式配置的一步训练/恢复 smoke 前置条件。
