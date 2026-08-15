# Phase 6：FastWAM Websocket Inference 与 DexJoCo Simulator 对接

## 阶段记录

- 日期：2026-08-15
- 阶段目标：建立双相机 + 23D state + cached T5 prompt 到反归一化 22D action chunk 的 versioned websocket 协议，并在 DexJoCo 中转换为 23D simulator command。
- 阶段边界：只运行 tiny CPU websocket smoke 和 `water_plant` reset + 2 steps；不加载正式 5B joint-trained 权重，不执行六任务完整评测，不修改 simulator dynamics。
- 计划 FastWAM commit message：`feat(inference): serve DexJoCo 22d actions`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `main` | `88c040066d6618dd24443ee0965673497df035c3` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

Phase 5 实际 commit 为 `88c040066d6618dd24443ee0965673497df035c3`，已回填 `commit_index.md`。DexJoCo 两个用户文件全程未修改、未 stage。

## 两仓库提交

DexJoCo 先完成并提交：

| Commit | Message | 内容 |
| --- | --- | --- |
| `992abca3da2bb34475485658bf76b766c49b7efa` | `feat(eval): add FastWAM DexJoCo client` | 六任务配置、strict websocket client、22D rotvec 到 23D wxyz command adapter、chunk/replan、reset/termination/success/video |
| `3c85e48dc50c29e204259261eb97f3419e26e969` | `fix(eval): lazy-load DexJoCo simulator adapter` | protocol-only client 不再因 package import 强制加载 SciPy/MuJoCo；public API 不变 |

FastWAM commit 由本文件所在提交确定。DexJoCo 第二个小提交是跨仓库 websocket smoke 发现的导入边界修复；没有改 OpenPI client 或环境动力学。

## 正式协议

Schema 固定为 `fastwam.dexjoco.websocket@1`，使用与现有 OpenPI client 兼容的 binary msgpack + NumPy encoding。连接后 server 先发送 metadata；每个连接可连续发送请求。

请求：

| 字段 | 契约 |
| --- | --- |
| `primary` | unbatched HWC RGB `uint8` |
| `wrist` | 与 primary 相同 shape 的 unbatched HWC RGB `uint8` |
| `state` | finite floating `[23]`，顺序 `xyz(3) + quaternion_wxyz(4) + Allegro(16)` |
| `prompt` | 非空文本；server 按训练模板格式化后只读取 cached T5 |
| `horizon` | 正整数，必须等于 server 和 checkpoint statistics 的 action horizon |
| schema metadata | `schema_name`、`schema_version`、`action_dim=22`、`proprio_dim=23` |

响应：

```text
actions: float32 [horizon,22], finite, denormalized
actions[..., :6]   = absolute TCP xyz(3) + rotvec(3)
actions[..., 6:22] = Allegro Hand 16D
```

`action_ordering` 必须逐字段等于 `DEXJOCO_ACTION_ORDERING`。Server 对 batch、HWC/CHW、dtype、camera shape、state dimension、horizon、NaN/Inf、Arm/Hand 拼接一致性和 schema/version fail-fast；不执行广播或维度截取。

真实入口：

- `fastwam.inference.websocket_protocol.validate_inference_request()`：协议请求校验。
- `fastwam.inference.websocket_protocol.build_action_response()`：22D finite 响应和 ordering metadata。
- `fastwam.inference.msgpack_numpy`：与 OpenPI 相同的无 pickle NumPy msgpack layout。
- `fastwam.inference.websocket_server.DexJoCoWebsocketServer`：metadata、持续 request/reply 和异常关闭。
- DexJoCo `dexjoco_fastwam_client.protocol.FastWAMWebsocketClient`：真实 client facade。

## FastWAM Policy

`DexJoCoInferencePolicy` 的处理链：

```text
primary/wrist HWC uint8
  -> each camera resize to configured HxW
  -> float CHW [-1,1]
  -> horizontal concat [1,3,H,2W]
state [23]
  -> checkpoint dataset_stats.json validation
  -> same FastWAM LinearNormalizer -> [1,23]
prompt
  -> DEFAULT_PROMPT training template
  -> exact sha256.t5_len{L}.wan22ti2v5b.pt lookup
  -> cached context/mask (T5 itself remains unloaded)
  -> DexJoCoDualActionFastWAM.infer_action()
  -> internal [1,horizon,22]
  -> one 22D action denormalizer
  -> wire [horizon,22]
```

`DexJoCoInferenceNormalizer` 先调用 `validate_dexjoco_statistics()`，因此 LIBERO、7D、错误 task/ordering/schema、validation/test stats 都会在建立 policy 时被拒绝。默认要求 `statistics_mode=production`/`production=true`；`--allow-non-production-statistics` 只供显式 smoke 使用。

`DexJoCoT5Cache` 只接受 `scripts/precompute_text_embeds.py` 的 exact filename 和 `{context,mask}` payload。缺文件、shape/dim 不匹配或 NaN/Inf 直接失败，不回退加载 T5。`precompute_text_embeds.py` 本阶段补充 LeRobot v3 `meta/tasks.parquet` 读取，仍兼容旧 `tasks.jsonl`。

`load_dexjoco_inference_model()` 只接受 Phase 5 versioned state directory：先读取 `training_config.yaml` 构造 22/6/16/23D 模型，再执行 `validate_dexjoco_training_checkpoint()`，然后读取 manifest 明确列出的单一 `pytorch_model.bin`。`load_state_dict()` 的 missing/unexpected 结果必须均为空；不会把 `checkpoints/libero_uncond_2cam224.pt` 当成 trained DexJoCo checkpoint。

## 无 Cache 双 Expert Sampler

`DexJoCoDualActionFastWAM.infer_action()`：

1. 分配一次 `[1,horizon,22]` action latent。
2. VAE 只编码一次当前双相机拼接帧。
3. 每个 diffusion step 对完整 22D latent 调用一次 `_forward_dual_experts()`。
4. Arm/Hand 使用同一个 timestep、scheduler 和 text/proprio conditioning。
5. prediction 先拼为 `[arm_6d,hand_16d]`，scheduler 对完整 `[1,horizon,22]` step 一次。
6. 返回 batched action/arm/hand 和最后一个 attention mask；server 只在形状检查和反归一化后移除 batch dimension。

现有 `_predict_action_noise_with_cache()` 继续显式 fail-fast，因为 `MoT.forward_action_with_video_cache()` 只处理旧 `action` expert，不能表达 Arm/Hand 双向交互。`infer_joint()` 继续禁用；Phase 6 server 不生成视频。代价是每个 denoising step 重算 video tokens，正确性优先于尚未实现的三 expert KV cache。

Attention 保持 Phase 2 契约：Arm/Hand 读取全部 Video tokens，Arm/Hand 双向交互，Video 不读 Arm/Hand。

## DexJoCo Adapter

DexJoCo 新增 `DexJoCoFastWAMActionAdapter.to_simulator_command()`：

```text
FastWAM [xyz(3), rotvec(3), hand(16)]
  -> scipy Rotation.from_rotvec(...).as_quat(scalar_first=True)
  -> [xyz(3), quaternion_wxyz(4), clipped_hand(16)]
  -> environment.step(command), command.shape == (23,)
```

`wxyz` 不是维度猜测：现有 DexJoCo wrapper 使用 `Rotation.as_quat(scalar_first=True)`，raw task environment 按 `w,qx,qy,qz` 解包。Hand clipping 从运行中环境的 `_allegro_ctrl_ids` 和 `model.actuator_ctrlrange` 读取；Cartesian API 没有声明通用 workspace，因此六任务配置默认 `xyz_min/xyz_max: null`，若部署方提供真实 workspace 才显式 clipping。

六任务 `configs/fastwam/*.yaml` 均记录 `action_horizon=32`、`replan_ratio=0.8`、`max_steps=1000`、`video_fps=30` 和 camera mapping。`run_episode()` 每次请求 action chunk，执行 `ceil(replan_ratio * horizon)` 后 replan，并处理 reset、environment `terminated`、`info["succeed"]`、外层 max steps 和两路 MP4。现有 DexJoCo wrapper 没有把 Gym `truncated` 单独映射到 `is_done`；当前六任务仍由显式 `max_steps` 保证 episode 上界。

## Smoke Test

执行命令：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_inference.py
MUJOCO_GL=egl conda run -n dexjoco \
  python scripts/smoke_test_fastwam_client.py --simulator

conda run -n fastwam python scripts/smoke_test_dexjoco_dual_action_model.py
conda run -n fastwam python scripts/smoke_test_dexjoco_selective_checkpoint.py
conda run -n fastwam python scripts/smoke_test_dexjoco_optimizer.py
conda run -n fastwam python scripts/smoke_test_dexjoco_training_loop.py
```

FastWAM websocket smoke 使用真实 tiny Video/Arm/Hand/MoT、真实提交后的 DexJoCo client 和临时 `production=false` stats/T5 payload：

```text
request cameras = primary/wrist [18,20,3] uint8
request state   = [23] float32
model video     = [1,3,16,32]
model action    = [1,4,22]
model Arm/Hand  = [1,4,6] / [1,4,16]
wire response   = [4,22] float32 finite, denormalized
attention mask  = [16,16]
diffusion steps = 2; each scheduler model_output [1,4,22]
```

Normalize/denormalize round-trip、LIBERO 7D stats rejection、T5 cache hit/miss、六任务 parquet task discovery、schema/batch/horizon/dtype/NaN/Arm-Hand 拼接不一致 invalid cases 均 PASS。临时 artifacts 自动删除。

DexJoCo simulator smoke：

```text
rotvec zero -> quaternion_wxyz=(1,0,0,0): PASS
rotvec (pi,0,0) -> abs(quaternion_wxyz)=(0,1,0,0): PASS
water_plant reset=PASS steps=2 success=False
final environment.step command shape=(23,)
action_chunk_horizon=4 replan_ratio=0.5 requests=1
runtime actuator_ctrlrange clipping=PASS
video files front.mp4/wrist.mp4=PASS (temporary)
```

任务成功不属于 smoke 要求。没有运行六任务完整 episode/evaluation。

Phase 2-5 targeted 回归全部 PASS；Phase 5 的 total/video/action loss 仍为 `4.11274576/2.52383971/1.58890617`，save/reload 与 resume fail-fast 保持通过。

## 正式命令

六任务目录当前均已存在于 `/home/shared/ai/datasets/dexlewm/dexjoco`。先生成 production statistics 和 T5 cache：

```bash
conda run -n fastwam python scripts/compute_dexjoco_stats.py \
  --data-root /home/shared/ai/datasets/dexlewm/dexjoco \
  --output /abs/path/to/dexjoco_6task_train_dataset_stats.json \
  --split train \
  --action-horizon 32

DEXJOCO_LEROBOT_ROOT=/home/shared/ai/datasets/dexlewm/dexjoco \
conda run -n fastwam python scripts/precompute_text_embeds.py \
  task=dexjoco_joint_2cam224_1e-4 \
  +overwrite=false \
  data.train.text_embedding_cache_dir=/abs/path/to/dexjoco_t5_cache
```

完成 joint post-training 后，使用其 Phase 5 state directory 启动 server：

```bash
conda run -n fastwam python scripts/serve_dexjoco.py \
  --checkpoint-dir /abs/path/to/run/checkpoints/state/step_XXXXXX \
  --text-cache-dir /abs/path/to/dexjoco_t5_cache \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda \
  --dtype bfloat16 \
  --horizon 32 \
  --num-inference-steps 20
```

DexJoCo client：

```bash
cd /home/shared/ai/datasets/DexJoCo/dexjoco
MUJOCO_GL=egl conda run -n dexjoco \
  python -m dexjoco_fastwam_client.cli.evaluate \
  --config configs/fastwam/water_plant.yaml \
  --host 127.0.0.1 \
  --port 8000 \
  --episodes 1
```

## 修改文件

FastWAM：

- `src/fastwam/models/wan22/dexjoco_dual_action.py`
- `src/fastwam/inference/__init__.py`
- `src/fastwam/inference/msgpack_numpy.py`
- `src/fastwam/inference/websocket_protocol.py`
- `src/fastwam/inference/dexjoco_policy.py`
- `src/fastwam/inference/websocket_server.py`
- `scripts/serve_dexjoco.py`
- `scripts/smoke_test_dexjoco_inference.py`
- `scripts/smoke_test_dexjoco_dual_action_model.py`
- `scripts/precompute_text_embeds.py`
- `pyproject.toml`
- 本目录 Phase 6 记录文件

DexJoCo：见上述两个提交；未修改 OpenPI client 和 simulator dynamics。

## 已知风险和停止点

- `checkpoints/libero_uncond_2cam224.pt` 是约 12 GB 的旧 selective initialization source，不是 joint-trained DexJoCo checkpoint；同目录 statistics 是 LIBERO 7D，server 会拒绝。
- 当前仓库没有持久化的 Phase 5 trained state directory、production DexJoCo statistics 或正式 T5 cache，因此未启动 5B server。tiny smoke 不代表 5B 显存、吞吐或 latency。
- 无 cache sampler 每一步重算 Video/Arm/Hand；三 expert KV cache 尚未实现，不能静默启用旧 single-action cache。
- 只验证单进程 websocket 和一个 `water_plant` 短 simulator episode；网络认证、并发、长连接压力、完整六任务成功率尚未验证。
- Phase 6 commit 后停止，不开始正式训练或完整评测。
