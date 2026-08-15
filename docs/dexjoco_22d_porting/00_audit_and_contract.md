# Phase 0：仓库审计与 22D 正式契约

## 阶段记录

- 日期：2026-08-15
- 阶段目标：定位两个仓库的真实实现，轻量抽样六个任务，确认 DexJoCo 单臂动作、状态、相机和控制契约，并初始化项目记录。
- 阶段边界：只新增本目录下的项目文档；不修改模型、数据、训练、推理或仿真代码；不运行训练、模型推理或仿真闭环；不扫描完整数据集。
- 计划 commit message：`docs: record DexJoCo 22d adaptation contract`

### 阶段开始时仓库状态

| 仓库 | Git 根目录 | 分支 | HEAD | `git status --short` | remote |
| --- | --- | --- | --- | --- | --- |
| FastWAM | `/home/user/fastwam-22d` | `main` | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | 干净 | `origin https://github.com/MERCYAI/FastWAM-22d.git` |
| DexJoCo | `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` | `origin https://github.com/brave-eai/dexjoco.git` |

两个 Git 根目录及其适用祖先目录均未发现 `AGENTS.md`。审计前已阅读 FastWAM 的 `README.md`、`README_zh.md` 和 DexJoCo 顶层 `README.md`。DexJoCo 两个已有修改分别放宽/约束依赖版本，未修改、未暂存。

## FastWAM 真实代码映射

下表路径均相对 `/home/user/fastwam-22d`。名称和行号来自 Phase 0 当前 HEAD 的真实代码，不使用规划文档中的假定模块名。

| 审计对象 | 真实路径、符号和行号 | 结论 |
| --- | --- | --- |
| 模型组装 | `src/fastwam/models/wan22/fastwam.py:18` `FastWAM`；`:43-47` 子模块字段；`:91` `from_wan22_pretrained`；`:147-150` 挂载 MoT/Video/Action | `FastWAM` 组装 video DiT、action DiT 和 MoT。 |
| Video DiT | `src/fastwam/models/wan22/wan_video_dit.py:310` `WanVideoDiT`；`:367-368` patch embedding；`:381-384` blocks；`:291` `Head`；`:385` head 实例；`:509` `pre_dit`；`:622` `post_dit`；`:628` `forward` | Video backbone 是 `WanVideoDiT`；视频输出 head 使用该文件的 `Head`。 |
| Action DiT、encoder、head | `src/fastwam/models/wan22/action_dit.py:32` `ActionDiT`；`:33` 排除前缀；`:74` `action_encoder`；`:86-97` blocks；`:98` `head`；`:18` `ActionHead`；`:112` `from_pretrained`；`:201-223` backbone 加载；`:233-240` 输入检查 | 实际 `ActionDiT.head` 是裸 `nn.Linear`；已声明的 `ActionHead` 类当前没有被 `ActionDiT` 实例化。预训练 backbone 加载明确排除 `action_encoder` 和 `head`。 |
| MoT | `src/fastwam/models/wan22/mot.py:14` `MoT`；`:77` `_mixed_attention`；`:257` `prefill_video_cache`；`:343` `forward_action_with_video_cache`；`:447` `forward` | MoT 负责 video/action token 的混合注意力和推理缓存。 |
| proprio/state encoder | `src/fastwam/models/wan22/fastwam.py:57-61` `nn.Linear(proprio_dim, text_dim)`；`:219` `_append_proprio_to_context`；`:233-240` token 追加；`:352-366` 训练输入处理 | proprio 经过单层线性投影后作为一个 context token 追加。当前训练路径接受 `[B,T,D]`，但只使用 `proprio[:,0,:]`。 |
| T5/text encoder | `src/fastwam/models/wan22/wan_video_text_encoder.py:223` `WanTextEncoder`；`:297` `HuggingfaceTokenizer`；`src/fastwam/models/wan22/helpers/loader.py:33-51` registry；`:141` `load_wan22_ti2v_5b_components`；`:191-209` text 加载 | T5/text 组件由 Wan text encoder 和 loader registry 加载。 |
| VAE | `src/fastwam/models/wan22/wan_video_vae.py:1057` `WanVideoVAE`；`:1075-1078` spatial/temporal factor；`:1355` `WanVideoVAE38`；`src/fastwam/models/wan22/helpers/loader.py:210` VAE 加载 | VAE 的空间压缩因子为 8，时间压缩因子为 4。 |
| checkpoint save/load | `src/fastwam/models/wan22/fastwam.py:1088` `save_checkpoint`；`:1100` `load_checkpoint`；`src/fastwam/trainer.py:265` resume；`:567` 权重保存；`:583-601` Accelerate 完整状态 | 模型 checkpoint 保存 `mot`、可选 proprio/optimizer，并兼容 legacy `dit` key；trainer 另存训练状态。 |
| dataset/dataloader | `src/fastwam/datasets/lerobot/robot_video_dataset.py:25` `RobotVideoDataset`；`:115` `_get`；`:199-203` action/proprio 对齐；`src/fastwam/datasets/lerobot/base_lerobot_dataset.py:17` `BaseLerobotDataset`；`:68-86` delta timestamps；`:179` `__getitem__`；`src/fastwam/datasets/lerobot/processors/fastwam_processor.py:14` `FastWAMProcessor`；`:182` `preprocess`；`src/fastwam/trainer.py:167` `_build_loader`；`:174` `DataLoader` | `RobotVideoDataset` 返回 action `[T-1,D]`，并把原 `[T,D]` proprio 裁为 `[:-1]`；processor 负责多相机和张量整理；trainer 构造 DataLoader。 |
| normalization/statistics | `src/fastwam/datasets/lerobot/base_lerobot_dataset.py:251` statistics；`src/fastwam/datasets/lerobot/utils/normalizer.py:19` `LinearNormalizer`；`:91` `SingleFieldLinearNormalizer`；`src/fastwam/datasets/lerobot/transforms/action_state_merger.py:7` `ConcatLeftAlign` | 统计、字段归一化和 action/state 合并均有独立实现。 |
| trainer/loss/optimizer/scheduler | `src/fastwam/trainer.py:28` `Wan22Trainer`；`:89-94` AdamW；`:220` scheduler；`:646` train loop；`src/fastwam/models/wan22/fastwam.py:448` `training_loss`；`:458-477` flow targets；`:539-561` MSE；`:563-568` total loss | 优化器是 AdamW，scheduler 由 trainer 创建；video/action flow matching 使用 MSE 后加权求和。 |
| inference/runtime | `src/fastwam/runtime.py:76` `create_fastwam`；`:333` `build_datasets`；`:359` `run_training`；`:383` `run_inference`；`src/fastwam/models/wan22/fastwam.py:906` `infer_action` | FastWAM 提供本地 runtime 和模型级 action inference。仓库中未发现通用 FastAPI/WebSocket/gRPC 模型 server。 |
| 任务部署入口 | `experiments/libero/eval_libero_single.py:359` `_predict_action_chunk`；`:445` `run_single_episode`；`experiments/robotwin/fastwam_policy/deploy_policy.py:138` `WorldActionRobotWinPolicy`；`:236` `_infer_action_chunk`；`:276` `step` | 当前实际入口是本地 LIBERO eval 和 RoboTwin deploy，不是统一服务端。 |

## Action chunk、KV cache 和 attention mask

### Action chunk 张量布局

- `ActionDiT` 在 `src/fastwam/models/wan22/action_dit.py:233-240` 强制 action 输入为 `[B,T,D]`。
- `RobotVideoDataset` 在 `src/fastwam/datasets/lerobot/robot_video_dataset.py:199-203` 输出一个 clip 内的 `[T-1,D]` action，并把 proprio 对齐为 `[T-1,Dp]`。
- `FastWAM.infer_action` 在 `src/fastwam/models/wan22/fastwam.py:952-958` 创建 `(1, horizon, action_dim)` 噪声，经推理后在 `:1046-1048` 返回 `[T,D]`。
- DexJoCo OpenPI eval 在 `dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py:124-131` 从 `result["actions"]` 接收 action chunk 并逐行放入执行队列；配置的 action horizon 为 30（`:249`）。单步送入环境的每一行应为 `[22]`。

因此，后续 22D 迁移的模型内部正式布局是 `[batch, horizon, 22]`，模型级 API 返回 `[horizon, 22]`，环境每步消费 `[22]`。这一定义不改变时间维和 batch 维的既有约定。

### Video KV cache

- `src/fastwam/models/wan22/mot.py:257` 的 `prefill_video_cache` 为每个 MoT block 预填充 video K/V。
- 每层缓存为 `{"k": ..., "v": ...}`；`:277-280` 和 `:300-341` 显示张量布局为 `[B,Sv,H*Dh]`。
- `forward_action_with_video_cache`（`:343`）在 `:425-433` 让 action query 读取缓存的 video K/V，并拼接当前 action K/V。
- `src/fastwam/models/wan22/fastwam.py:998-1022` 在 action denoising 前预填充缓存，并在各去噪步复用。

### Attention mask

- `src/fastwam/models/wan22/wan_video_dit.py:473` `build_video_to_video_mask` 实现 video-to-video mask；`:484` 为双向模式，`:487` 为逐帧 causal，`:501` 为 first-frame causal。
- `src/fastwam/models/wan22/fastwam.py:386` `_build_mot_attention_mask` 构造联合 mask；`:402-406` 的当前语义是 action-to-action 全可见、action-to-video 仅能看首帧 video、video-to-action 禁止。

后续若拆成 Arm/Hand 两个 action expert，必须显式决定两个 action token 段之间的可见性和 cache 复用方式，不能假设当前单 expert mask 自动适用。

## DexJoCo 真实执行路径

本节路径均相对 `/home/shared/ai/datasets/DexJoCo/dexjoco`。

### Client、环境 step 和成功判定

- `dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py:12` `DexJoCoOpenPIEnv` 是 OpenPI simulator client wrapper。
- `DexJoCoOpenPIEnv.step`（`:106`）执行 action，并在 `:120` 从 `info["succeed"]` 更新成功状态。
- `_process_action`（`:204`）在 `:219-221` 把单臂 22D action 切成 xyz、rotvec 和 hand；`:223` 使用 `R.from_rotvec(...).as_quat(scalar_first=True)`；`:225` 拼成模拟器接收的 `[xyz, quat_wxyz, hand]`。
- 该 wrapper 在 `:196` 保留 observation state 的前 23 维。
- `dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py:103` `inference_process` 创建 `WebsocketClientPolicy`（`:116`），接收 `result["actions"]`（`:124-127`），控制循环调用 `env.step(action)`（`:349`），并在 `:377-384` 读取 `env.is_done`/`env.is_success`。

六任务环境及其成功判断位置如下：

| 任务 | 环境类 | `step()` | 成功判定 |
| --- | --- | --- | --- |
| `water_plant` | `dexjoco/dexjoco/sim/envs/panda_water_plant_env.py:101` `PandaWaterPlantGymEnv` | `:475` | `:557` |
| `hammer_nail` | `dexjoco/dexjoco/sim/envs/panda_hammer_nail_env.py:31` `PandaHammerNailGymEnv` | `:487` | `:600` |
| `click_mouse` | `dexjoco/dexjoco/sim/envs/panda_click_mouse_env.py:38` `PandaClickMouseGymEnv` | `:529` | `:636` |
| `pick_bucket` | `dexjoco/dexjoco/sim/envs/panda_pick_bucket_env.py:32` `PandaPickBucketGymEnv` | `:499` | `:586` |
| `pinch_tongs` | `dexjoco/dexjoco/sim/envs/panda_pinch_tongs_env.py:32` `PandaPinchTongsGymEnv` | `:421` | `:582` |
| `fold_glasses` | `dexjoco/dexjoco/sim/envs/panda_fold_glasses_env.py:25` `PandaFoldGlassesGymEnv` | `:436` | `:512` |

六个 `step()` 都把 action 前 7 维 `[x,y,z,w,qx,qy,qz]` 设为绝对 mocap pose target，以 operational-space controller 控制 Panda，并把后 16 维写入 Allegro position actuators。`dexjoco/dexjoco/tasks/policy_wrappers.py:5` 的 `SingleArmPolicyWrapper` 在 `:8-16` 明确 simulator action 是 `[xyz(3), quaternion wxyz(4), Allegro(16)]`。

### 控制量、单位和 control mode

- TCP xyz：MuJoCo world frame 中的绝对位置目标，单位米。
- 数据 action 的 rotvec：SciPy rotation vector，单位弧度；client 执行前转成 quaternion。
- simulator quaternion：`wxyz`。
- hand：16 个 Allegro 关节位置目标，单位弧度。
- Panda arm：`dexjoco/dexjoco/sim/controllers/opspace.py:59` `opspace`，在 `:126-179` 通过位置/姿态 PD、Jacobian 和 task-space dynamics 计算 torque。
- Allegro hand：MuJoCo `<position>` actuator。
- XML 在 `dexjoco/dexjoco/sim/envs/xmls/panda_allegro_right.xml:3` 声明 `angle="radian"`。
- `dexjoco/dexjoco/sim/mujoco_gym_env.py:18` `MujocoGymEnv` 默认 control timestep 为 0.02 s、physics timestep 为 0.002 s，每控制步 10 个 physics substeps（`:25-36`），即环境默认控制频率 50 Hz。数据记录/视频 metadata 为 30 FPS；两者不是同一概念。

### 数据写入语义

`scripts/record_demos_zarr.py:224` 的 `convert_action_quat_to_rotvec` 在 `:226-254` 明确把单臂 `[pos3, quat_wxyz4, hand16]` 转为 `[pos3, rotvec3, hand16]`；`:371-374` 将结果写入 `action_rotvec`。state 在 `:322-324`、`:357-361` 原样写入。

六任务配置的 `proprio_keys` 都以 `tcp_pose` 后接 `gripper_pose`：`dexjoco/dexjoco/tasks/water_plant/config.py:16-22`、`dexjoco/dexjoco/tasks/hammer_nail/config.py:14-20`、`dexjoco/dexjoco/tasks/click_mouse/config.py:14`、`dexjoco/dexjoco/tasks/pick_bucket/config.py:14-20`、`dexjoco/dexjoco/tasks/pinch_tongs/config.py:14`、`dexjoco/dexjoco/tasks/fold_glasses/config.py:14-20`。这些原始列表在前 23D 后还包含任务物体姿态和/或桌面高度，不能把完整原始 flat state 称为 23D。

`dexjoco/dexjoco/tasks/obs_adapters.py:29-35` 的 `DexjocoObsAdapter.observation` 按配置 key 顺序 flatten。随后 `dexjoco-data-converter/configs/rand_obj/slice_config.yaml:16-32` 和 `dexjoco-data-converter/configs/rand_full/slice_config.yaml:16-32` 对六个单臂任务明确执行 `state: [null, 23]`。所以正式 LeRobot state 的前 23D 是 TCP pose 7D 后接 Allegro 16D，其后的任务特定原始状态被 converter 丢弃。这一结论来自字段顺序和 slice 代码，而非维度猜测。

`DexJoCoOpenPIEnv.stay` 在 `dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py:158-167` 再次把 23D state 解析成 arm 7D 和 hand 16D，并用 `R.from_quat(quat, scalar_first=True)` 读取 state quaternion；这直接确认 state quaternion 顺序为 `wxyz`。

### Allegro Hand 16D 顺序

关节/执行器顺序由任务源码和 XML 交叉确认：

```text
ffj0, ffj1, ffj2, ffj3,
mfj0, mfj1, mfj2, mfj3,
rfj0, rfj1, rfj2, rfj3,
thj0, thj1, thj2, thj3
```

对应 actuator 顺序为 `ffa0..ffa3, mfa0..mfa3, rfa0..rfa3, tha0..tha3`。证据包括 `dexjoco/dexjoco/sim/envs/panda_water_plant_env.py:33-69`，以及 `dexjoco/dexjoco/sim/envs/xmls/panda_allegro_right.xml:361-446` 的 joints、`:488-503` 的 actuators 和 `:534-549` 的 sensors。

## 六任务轻量数据抽样

### 方法和范围

- 每个任务仅读取 episode 0 的 metadata 和最低限度低维 Parquet 行，不遍历全部 episode。
- 本地数据根 `/home/shared/ai/datasets/dexlewm/dexjoco` 含 `water_plant`、`hammer_nail`、`pick_bucket` 和 `fold_glasses`。
- 本地缺少的 `click_mouse`、`pinch_tongs` 仅从官方 Hugging Face 数据集 `DexJoCo/DexJoCo-Datasets-LeRobot` 下载 `meta/info.json`、episode metadata 和 episode 0 低维 Parquet 到临时目录；临时目录已自动删除。
- 未直接解码 MP4。相机 shape、FPS 和 codec 来自 `meta/info.json`；时间对齐来自 episode metadata 与 converter 的 lockstep 代码。不得把这些结果表述为像素解码验证。

### 公共 schema 结论

六个任务的 episode 0 均确认：

- `action` 的 Arrow schema 是 `fixed_size_list<float>[22]`，数据集 info dtype 是 `float32`。
- `observation.state` 的 Arrow schema 是 `fixed_size_list<float>[23]`，数据集 info dtype 是 `float32`。
- dataset split 是 `train: 0:100`，每任务 100 episodes。
- 两个 camera feature 都是 `[640,640,3]`、AV1、30 FPS。
- episode 0 state quaternion 的范数约为 1；结合写入/执行代码，顺序确认是 `wxyz`。范数检查本身只验证 quaternion 数值形态，不单独证明顺序。
- 同一任务的两路 camera episode timestamp 起止范围一致。
- `dexjoco-data-converter/src/dexjoco_data_converter/to_lerobot/merge_episode_lerobot.py:332-350` 对每个 step 从所有相机队列读取相同 `t`，并断言 `video_msg.step == t`，这是时间锁步的代码证据。

### Episode 0 样本

| 任务 | episode length | 双相机 key | 两路 video timestamp range | action | state |
| --- | ---: | --- | --- | --- | --- |
| `water_plant` | 309 | `front`, `wrist` | 0.000000-10.300000 s | `[309,22]`, float32 | `[309,23]`, float32 |
| `hammer_nail` | 202 | `front`, `wrist` | 0.000000-6.733333 s | `[202,22]`, float32 | `[202,23]`, float32 |
| `click_mouse` | 500 | `ego_right`, `wrist` | 0.000000-16.666667 s | `[500,22]`, float32 | `[500,23]`, float32 |
| `pick_bucket` | 537 | `front`, `wrist` | 0.000000-17.900000 s | `[537,22]`, float32 | `[537,23]`, float32 |
| `pinch_tongs` | 341 | `front`, `wrist` | 0.000000-11.366667 s | `[341,22]`, float32 | `[341,23]`, float32 |
| `fold_glasses` | 560 | `front`, `wrist` | 0.000000-18.666667 s | `[560,22]`, float32 | `[560,23]`, float32 |

这里的 `[length,D]` 是 episode 低维表的样本形状；FastWAM dataset 形成训练 clip 后会按自身逻辑输出 `[T-1,D]` action。

### `rand_obj` / `rand_full` 组织

- `dexjoco-data-converter/configs/rand_obj/selected_data.yaml` 对 `click_mouse` 选择 `ego_right,wrist`，其余五任务选择 `front,wrist`；action 字段为 `action_rotvec`，state 字段为 `state`。
- `dexjoco-data-converter/configs/rand_full/selected_data.yaml` 对六任务均选择 `random_camera,wrist`。
- 两套 `slice_config.yaml` 对单臂 state 均裁前 23 维。
- `openpi/config.yaml` 将 rand_obj root 配为 `../datasets/dexjoco_lerobot_datasets`，rand_full root 配为 `../datasets/dexjoco_lerobot_datasets_rand_full`。
- 两套原始目录分别列在各自的 `dataset_paths.yaml`；rand_full 使用独立的 `dexjoco_raw_datasets_rand_full/<task>` 源目录。

## 冻结的正式契约

代码写入路径、client 转换路径、simulator wrapper、六任务样本 schema 和 state 配置没有发现与下列契约冲突。因此 Phase 0 正式冻结：

```text
action[..., :6]     = absolute TCP xyz(3) + rotvec(3)
action[..., 6:22]   = Allegro Hand 16D
state                = xyz(3) + quat wxyz(4) + hand(16) = 23D
```

进一步限定：

- `xyz` 是 MuJoCo world frame 的绝对 TCP 位置目标，单位米。
- `rotvec` 是绝对 TCP 姿态对应的 SciPy rotation vector，单位弧度。
- state 中 quaternion 是 `wxyz`，不是 `xyzw`。
- hand 16D 依次是 `ffj0..3, mfj0..3, rfj0..3, thj0..3`，是关节位置目标，单位弧度。
- simulator 入口不直接接受 rotvec；client 必须把 `[xyz,rotvec,hand]` 转成 `[xyz,quat_wxyz,hand]`。

如果后续任一代码或数据证据与以上契约冲突，工作必须停止并明确报告；不得自行更改维度、顺序、坐标语义或单位。

## 默认迁移策略

以下策略作为后续阶段的默认权重契约冻结。本阶段不实现它：

| 组件 | 权重策略 |
| --- | --- |
| Video DiT backbone | pretrained |
| Arm Transformer backbone | pretrained |
| Arm `action_encoder` / `head` | newly initialized |
| Hand Transformer backbone | 从旧 Action Expert remap |
| Hand `action_encoder` / `head` | newly initialized |
| 23D proprio encoder | newly initialized |

禁止截取或部分复制旧 7D projection。旧 projection 的输入/输出坐标语义与新 22D 分解不等价，即使部分 tensor shape 可机械匹配，也不能视为可迁移参数。

当前真实代码只有单个 `ActionDiT`/Action Expert，尚无独立 Arm/Hand 双专家结构。因此上述 Arm/Hand 策略是后续实现必须满足的迁移目标，而不是对当前代码结构的描述。

## 技术决策

本阶段的正式技术决策见 `decisions.md`。核心结论是：以代码和数据写入链路交叉验证语义；以 `[B,T,22]` 作为模型内部 action chunk；冻结 `wxyz` state quaternion；严格新建所有维度相关 projection；只提交 Phase 0 文档。

## 修改文件

本阶段只新增：

- `docs/dexjoco_22d_porting/README.md`
- `docs/dexjoco_22d_porting/00_audit_and_contract.md`
- `docs/dexjoco_22d_porting/decisions.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

## 执行命令记录

审计使用的关键只读命令/命令类别如下。数据抽样脚本为一次性只读脚本，未写入仓库。

```bash
git -C /home/user/fastwam-22d status --short
git -C /home/user/fastwam-22d branch --show-current
git -C /home/user/fastwam-22d rev-parse HEAD
git -C /home/user/fastwam-22d remote -v

git -C /home/shared/ai/datasets/DexJoCo/dexjoco status --short
git -C /home/shared/ai/datasets/DexJoCo/dexjoco branch --show-current
git -C /home/shared/ai/datasets/DexJoCo/dexjoco rev-parse HEAD
git -C /home/shared/ai/datasets/DexJoCo/dexjoco remote -v

rg --files -g AGENTS.md -g README.md -g README_zh.md
rg -n "class |def |action_encoder|proprio|cache|mask|checkpoint|optimizer|scheduler" src experiments
rg -n "action_rotvec|tcp_pose|gripper_pose|scalar_first|succeed|def step|proprio_keys" .

# 每任务只读取 episode 0：使用 pyarrow 读取低维 Parquet schema/最少行，
# 使用 JSON 解析 meta/info、episode length 和 video timestamp metadata。
# click_mouse / pinch_tongs 的最低限度文件来自：
# DexJoCo/DexJoCo-Datasets-LeRobot
```

提交前按要求执行：

```bash
git add docs/dexjoco_22d_porting/
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "docs: record DexJoCo 22d adaptation contract"
```

## Smoke test 结果

- PASS：两个真实 Git 根、branch、HEAD、remote 和开始状态已记录。
- PASS：六任务各抽样一个 episode，低维 action/state schema 分别为 float32 22D/23D。
- PASS：action 语义由 recorder、client 和 simulator wrapper 交叉确认，不仅依赖 shape。
- PASS：state 排列、quaternion `wxyz` 和 hand 顺序由 config、adapter、任务代码及 XML 交叉确认。
- PASS：双相机 key、metadata shape/FPS/codec、episode timestamp 范围和 converter 锁步逻辑已检查。
- NOT RUN：MP4 像素解码/逐帧视觉内容检查。
- NOT RUN：FastWAM 训练、模型推理、checkpoint round trip。
- NOT RUN：DexJoCo simulator 闭环和任务成功率。

详细 ledger 见 `smoke_test_ledger.md`。

## 已知风险和待确认项

- 当前 FastWAM 只有单 Action Expert；Arm/Hand 双 Transformer 的模块边界、forward、mask 和 checkpoint key 设计待后续阶段实现并评审。
- 当前 `ActionDiT.from_pretrained` 只支持单 backbone 且要求 shape 匹配；Hand backbone remap 规则尚未实现。
- 当前 proprio 路径对 `[B,T,23]` 只取第一个时间步。它是否满足 DexJoCo 训练意图需在后续阶段明确，但不能在 Phase 0 擅自改变。
- MP4 未直接解码；当前只确认 metadata 和 converter 的逻辑时间对齐。
- `click_mouse`、`pinch_tongs` 不在本地数据根，本阶段样本来自官方 Hugging Face 临时下载。
- 30 FPS 是数据/视频频率，环境默认 control dt 对应 50 Hz。采集/重采样如何连接两者仍待确认。
- 22D action 与 23D state 的 normalization statistics 尚未接入 FastWAM，也未做数值范围/outlier 检查。
- 通用 FastWAM model server 不存在；后续 DexJoCo runtime 是复用 OpenPI WebSocket 形态还是增加项目内 adapter，尚待设计。

## 后续阶段前置条件

进入下一阶段前必须满足：

- Phase 0 文档提交完成，FastWAM 工作树不遗留本阶段未提交文件。
- DexJoCo 两个用户修改保持原样、未被 stage。
- 后续设计明确 Arm/Hand 双专家的模块、attention mask、checkpoint key 和严格加载报告。
- 后续实现接受并断言 `[B,T,22]` action、`[B,T,23]` proprio/state，不复用旧 7D projection。
- 获得明确的下一阶段授权；Phase 0 完成后不得自动继续。
