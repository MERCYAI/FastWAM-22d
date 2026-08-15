# Phase 2：DexJoCo 双 Action Expert 模型结构

## 阶段记录

- 日期：2026-08-15
- 阶段目标：在不改变 FastWAM/LIBERO 路径的前提下，增加 DexJoCo 专用 Video/Arm/Hand 三 expert MoT，接入 6D Arm、16D Hand、23D proprio，并保持完整 22D diffusion/loss 契约。
- 阶段边界：只实现模型结构、运行时构造、独立配置和 tiny forward smoke；不实现正式 checkpoint 迁移、旧 Action Expert 到 Hand Expert 的 key remap、optimizer 分组、训练、正式推理或 simulator。
- 计划 commit message：`feat(model): add dual action experts for DexJoCo`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `main` | `8409c0fef83da8e88e185e8d5beb1f57b3d78131` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

开始前已阅读 `docs/dexjoco_22d_porting/` 全部记录，并把 Phase 1 实际 commit 回填到 `commit_index.md`。DexJoCo 两个用户文件未修改、未暂存。

## 真实模型入口

新增 Hydra 配置：

```text
model=dexjoco_dual_action
configs/model/dexjoco_dual_action.yaml
```

真实构造链为：

```text
fastwam.runtime.create_dexjoco_dual_action_fastwam
  -> fastwam.models.wan22.dexjoco_dual_action.DexJoCoDualActionFastWAM.from_wan22_pretrained
  -> MoT(mixtures={"video": WanVideoDiT, "action": ActionDiT(6), "hand": ActionDiT(16)})
```

`action` 名称特意保留给 Arm Expert，使未来旧 checkpoint 的 `mixtures.action.*` key 有稳定目标；Hand Expert 使用新名称 `hand`。通用 `FastWAM`、`FastWAMJoint`、`FastWAMIDM`、`ActionDiT`、`MoT` 和原 `configs/model/fastwam*.yaml` 均未改变。

## 结构和 action 契约

`DexJoCoDualActionFastWAM` 强制校验：

- expert 顺序必须是 `video,action,hand`。
- `action_expert` 必须为 6D `ActionDiT`，输入 projection 和输出 head 都是 6D。
- `hand_expert` 必须为 16D `ActionDiT`；排除 `action_encoder/head` 后的全部 backbone key 和 tensor shape 必须与 Arm `ActionDiT` 完全一致，同时满足 Video expert 的 MoT attention 约束。
- proprio dimension 必须为 23；沿用 `FastWAM` 的 `nn.Linear(23, text_dim)`，作为新增可训练参数。
- 完整 action 必须是 `[B,T,22]`，并且只按 `action[..., :6]`、`action[..., 6:22]` 拆分。

Phase 2 的 `from_wan22_pretrained()` 复用现有 loader 加载 Video DiT，并复用 `ActionDiT.from_pretrained()` 组装 Arm Expert；该现有 loader 自动排除 `action_encoder` 和 `head`，符合 6D projection 新初始化契约。Hand Expert 本阶段只按 `ActionDiT` 结构随机初始化。旧 Action Expert Transformer 到 Hand Transformer 的正式 remap、加载报告和 shape/key 检查属于后续 checkpoint 阶段，本阶段没有伪装成已迁移。

## Diffusion、forward 和 loss

训练 forward 对完整 action 只执行：

```text
noise_action = randn_like(action[B,T,22])
timestep_action = action_scheduler.sample_training_t(...)
noisy_action = action_scheduler.add_noise(action, noise_action, timestep_action)
target_action = action_scheduler.training_target(action, noise_action, timestep_action)
```

完成 scheduler 计算后才把 `noisy_action` 拆为 6D/16D。两个 ActionDiT 接收完全相同的 `timestep_action`、text/proprio context 和 context mask。输出固定重新拼接：

```text
pred_action = cat([pred_arm_action[...,6], pred_hand_action[...,16]], dim=-1)
```

action objective 保留原 FastWAM reduction：对 `pred_action[B,T,22]` 和 `target_action[B,T,22]` 只调用一次 elementwise MSE，再对完整 22 维 `mean(dim=2)`，随后应用原 `action_is_pad` 和 scheduler training weight。没有分别计算 Arm/Hand mean 后相加。Video noise、target、first-frame latent 处理、video head、padding reduction、scheduler weight 和 `loss_lambda_video` 路径保持原实现。

## Attention 契约

MoT 普通 `forward()` 已按 `expert_order` 泛化遍历任意 mixture，因此无需重复实现 Transformer block 或修改通用 MoT。DexJoCo 专用 mask 的 query/key 关系为：

| Query \ Key | Video | Arm `action` | Hand `hand` |
| --- | --- | --- | --- |
| Video | 现有 `build_video_to_video_mask()` | 禁止 | 禁止 |
| Arm | 完整 Video tokens | 允许 | 允许 |
| Hand | 完整 Video tokens | 允许 | 允许 |

因此 Video 保持世界模型因果方向，不读 Arm/Hand；两个 action expert 都读取完整 video token 序列，并在每层 mixed self-attention 中双向交互。Arm/Hand token 时间长度还必须一致，防止两个 expert 静默使用不同 action horizon。

## Video KV Cache 结论

现有 `MoT.prefill_video_cache()` 只预填 Video K/V，这部分本身与 expert 数量无关；但 `MoT.forward_action_with_video_cache()` 的参数、模块选择和 K/V 拼接都只包含单个 `action` expert：

```text
K/V = cached video + current action
queries = action only
```

它没有 Hand query/K/V，也无法表达 Arm/Hand 双向交互。Phase 2 不对这个共享 cache API 做未经验证的扩展。DexJoCo 模型设置 `supports_video_kv_cache=False`，构造时输出 warning；`infer_action()` 和 `_predict_action_noise_with_cache()` 显式抛出 `NotImplementedError`。继承的 joint sampler 还会根据 6D `action_expert.action_dim` 分配 latent，因此 `infer_joint()` 也显式禁用，避免静默丢弃 Hand。

## 修改文件

- `configs/model/dexjoco_dual_action.yaml`
- `scripts/smoke_test_dexjoco_dual_action_model.py`
- `src/fastwam/models/wan22/dexjoco_dual_action.py`
- `src/fastwam/runtime.py`
- `docs/dexjoco_22d_porting/README.md`
- `docs/dexjoco_22d_porting/02_dual_action_experts.md`
- `docs/dexjoco_22d_porting/decisions.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

## 执行命令和结果

实际执行的聚焦命令：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_dual_action_model.py

conda run -n fastwam python -m compileall -q \
  src/fastwam/models/wan22/dexjoco_dual_action.py \
  src/fastwam/runtime.py \
  scripts/smoke_test_dexjoco_dual_action_model.py

conda run -n fastwam python -c \
  'from hydra import compose, initialize_config_dir; from pathlib import Path; p=str(Path("configs").resolve()); ctx=initialize_config_dir(config_dir=p, version_base=None); ctx.__enter__(); c=compose(config_name="train", overrides=["model=dexjoco_dual_action", "data=dexjoco_6task_2cam"]); assert c.model._target_ == "fastwam.runtime.create_dexjoco_dual_action_fastwam"; assert (c.model.video_dit_config.action_dim, c.model.action_dit_config.action_dim, c.model.hand_dit_config.action_dim, c.model.proprio_dim) == (22, 6, 16, 23); print("dexjoco compose: dims=(22,6,16,23) target=PASS"); ctx.__exit__(None,None,None)'

conda run -n fastwam python -c \
  'from hydra import compose, initialize_config_dir; from pathlib import Path; p=str(Path("configs").resolve()); ctx=initialize_config_dir(config_dir=p, version_base=None); ctx.__enter__(); c=compose(config_name="train", overrides=["model=fastwam", "data=libero_2cam"]); assert c.model._target_ == "fastwam.runtime.create_fastwam"; assert c.model.action_dit_config.action_dim == 7; print("libero compose: action_dim=7 target=PASS"); ctx.__exit__(None,None,None)'
```

tiny CPU 模型使用 1 层真实 `WanVideoDiT`、两个 1 层真实 `ActionDiT`、真实 `MoT`/scheduler 和只替代大型 VAE 的 deterministic tiny encoder。一次 `training_loss(..., return_outputs=True)` 结果：

```text
input action=(1,4,22)
arm prediction=(1,4,6)
hand prediction=(1,4,16)
concatenated prediction=(1,4,22)
attention_mask=(16,16)  # 8 video + 4 arm + 4 hand
loss=3.457450
action timestep calls=1
action add_noise input=(1,4,22)
proprio_encoder.in_features=23, requires_grad=true
video_kv_cache=disabled-explicit
```

mask 的三个方向逐块断言通过，输出拼接逐值一致，loss 为有限标量。另用相同 tiny 组件实例化原 `FastWAM` 的 `video,action` 两 expert / 7D action 结构，结果通过。Hydra compose 与 `compileall` 通过。本阶段未运行完整 pytest、完整 5B checkpoint、训练或推理。

## 已知风险和后续前置条件

- Hand Transformer 仍是 Phase 2 随机初始化；进入训练前必须完成旧 Action Expert backbone 的显式 remap，且 Arm/Hand 的新 projection 不能截取旧 7D 权重。
- 未实现正式 checkpoint save/load 迁移、key/shape 报告或 optimizer 参数分组。
- DexJoCo cached action inference 与 joint inference 显式不可用，直到 cache/sampler 同时支持 22D latent 和 Arm/Hand 双向 token 交互。
- tiny smoke 不替代完整 5B 模型显存、dtype、distributed 或训练稳定性验证。
- 数据侧仍需补齐持久六任务、production `dataset_stats.json` 和真实 T5 embeddings。
- 进入下一阶段前必须先提交本阶段，并获得明确授权。
