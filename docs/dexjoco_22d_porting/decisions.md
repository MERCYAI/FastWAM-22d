# 技术决策记录

## D-0001：冻结 DexJoCo 单臂 22D 动作契约

- 日期：2026-08-15
- 状态：Accepted
- 决策：`action[..., :6]` 是绝对 TCP `xyz + rotvec`，`action[..., 6:22]` 是 Allegro Hand 16D。
- 证据：`scripts/record_demos_zarr.py:224-254` 的 quaternion-to-rotvec 写入；`dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py:204-225` 的逆转换；`dexjoco/dexjoco/tasks/policy_wrappers.py:5-16` 的 simulator action 定义；六任务 episode 0 的 22D float32 schema。
- 影响：模型内部 action chunk 使用 `[B,T,22]`，环境单步消费 `[22]`。若后续证据冲突，停止并报告，不能静默重排。

## D-0002：冻结 23D state 排列和 quaternion 顺序

- 日期：2026-08-15
- 状态：Accepted
- 决策：`state = xyz(3) + quat_wxyz(4) + hand(16)`。
- 证据：六任务 `proprio_keys` 的 `tcp_pose, gripper_pose` 起始顺序，`DexjocoObsAdapter.observation` 的按 key flatten，rand_obj/rand_full converter 的前 23D slice，以及 `DexJoCoOpenPIEnv.stay` 对 state quaternion 的 SciPy `scalar_first=True` 读取。
- 影响：后续 proprio encoder 输入维必须是 23；所有 quaternion adapter 必须显式声明顺序。

## D-0003：维度相关 projection 全部新初始化

- 日期：2026-08-15
- 状态：Accepted
- 决策：Arm 和 Hand 的 `action_encoder/head`、23D proprio encoder 都 newly initialized；不得截取或部分复制旧 7D projection。
- 理由：旧 7D projection 的特征坐标不对应新的 Arm 6D/Hand 16D 或 22D 联合语义，机械复制会制造无证据的参数含义。
- 影响：后续 checkpoint loader 应报告这些 key 为有意新建，而不是把 shape mismatch 隐藏为兼容加载。

## D-0004：backbone 权重迁移按组件分开

- 日期：2026-08-15
- 状态：Accepted
- 决策：Video DiT backbone pretrained；Arm Transformer backbone pretrained；Hand Transformer backbone 从旧 Action Expert remap。
- 约束：当前仓库只有一个 `ActionDiT`，因此 Arm/Hand 是后续目标结构。remap 必须有显式 key 映射、shape 检查和加载报告。

## D-0005：Phase 0 采用最小只读数据抽样

- 日期：2026-08-15
- 状态：Accepted
- 决策：每任务只读取 episode 0 的低维数据和必要 metadata；不扫描全数据集。
- 限制：没有可用的 MP4 解码验证，因此相机 shape/FPS/codec 是 metadata 结论，时间对齐是 metadata 加 converter 锁步代码结论。
- 影响：像素内容、丢帧和完整数据统计仍需后续 smoke test。

## D-0006：以真实运行入口描述 server 能力

- 日期：2026-08-15
- 状态：Accepted
- 决策：FastWAM 当前记录为本地 runtime、LIBERO eval 和 RoboTwin deploy；不宣称存在通用模型 server。DexJoCo 的现有网络 client 是 OpenPI `WebsocketClientPolicy`。
- 影响：后续集成需明确新增 adapter 还是复用已有服务协议。

## D-0007：Phase 0 的 Git 范围仅限文档目录

- 日期：2026-08-15
- 状态：Accepted
- 决策：只 stage `docs/dexjoco_22d_porting/` 下五个文件。
- 约束：不修改或 stage DexJoCo 的 `environment-dexjoco.yaml` 和 `openpi/packages/openpi-client/pyproject.toml`；保留所有用户改动。

## D-0008：DexJoCo 使用专用 LeRobot v3 backend

- 日期：2026-08-15
- 状态：Accepted
- 决策：新增 `DexJoCoV3Dataset`/`DexJoCoV3TaskSource`，按 `meta/episodes` 的行区间和每路视频 file/timestamp 映射读取；不改用 FastWAM 原有 v2 backend。
- 证据：六任务 `codebase_version` 均为 `v3.0`，共享 file-level Parquet/MP4；原 backend 对真实 `water_plant` 首先尝试读取不存在的 `meta/tasks.jsonl`。
- 影响：LIBERO/FastWAM 原路径默认行为不变；DexJoCo loader 对 v3 schema、30 FPS、任务目录顺序和 22D/23D feature 做 fail-fast 校验。

## D-0009：双相机对外统一为 `primary,wrist`

- 日期：2026-08-15
- 状态：Accepted
- 决策：公开 `camera_videos` 顺序固定为 `primary,wrist`；`click_mouse.primary=ego_right`，其余五任务 `primary=front`。
- 理由：直接把六目录交给原 `MultiLeRobotDataset` 会只保留公共 feature，`front`/`ego_right` 不可能同时成为公共 key；alias 保持固定训练接口且不伪造源 metadata。
- 约束：两路视频分别使用自己的 file index 和相对 timestamp，因为 v3 的视频文件可在不同 episode 边界独立切分；每路 duration 必须匹配 episode frame count。

## D-0010：DexJoCo statistics 使用独立、严格的 versioned schema

- 日期：2026-08-15
- 状态：Accepted
- 决策：schema 固定为 `fastwam.dexjoco.dataset_stats@1`，记录 training split、六任务/字段/执行器 ordering、22/6/16/23 维、action horizon、计数、std floor 和 FastWAM 所需 global/stepwise statistics。
- 约束：带 episode limit 的输出自动标为 `statistics_mode=smoke`、`production=false`；processor 默认拒绝 smoke、LIBERO、版本/顺序/shape 不匹配的文件。
- 数值策略：小于 `1e-6` 的 std clamp 到 `1e-6` 并记录维度，避免 z-score 对零方差或近零方差维放大。

## D-0011：DexJoCo 禁用 delta action 并显式拆分 arm/hand view

- 日期：2026-08-15
- 状态：Accepted
- 决策：`DexJoCoFastWAMProcessor` 强制 `delta_action_dim_mask=None`；dataset 在完整归一化 action 上提供 `arm_action=action[..., :6]` 和 `hand_action=action[..., 6:22]`。
- 理由：Phase 0 已由 recorder/client/simulator 交叉确认 action 是绝对 TCP target；LIBERO 的 7D delta mask 语义不适用。
- 影响：完整 action、拆分 view、proprio 及其 padding mask 均按 `[B,T,D]` 对齐；现有模型仍消费拼接 `video`，独立双相机张量作为附加字段返回。

## D-0012：Phase 1 不生成或提交伪 production statistics

- 日期：2026-08-15
- 状态：Accepted
- 决策：本阶段只运行每任务 1 episode 的 smoke statistics，文件留在 `/tmp` 且不能由默认 processor 加载。
- 理由：本机持久数据根缺 `click_mouse` 和 `pinch_tongs`；为 shape/video smoke 下载的官方 `file-000` 只覆盖最少文件，不等于完整持久训练集。
- 后续条件：补齐六任务后，使用无 `--max-episodes-per-task` 的正式命令生成训练输出目录中的 `dataset_stats.json`。

## D-0013：DexJoCo 使用 `video,action,hand` 三 expert MoT

- 日期：2026-08-15
- 状态：Accepted
- 决策：新增 `DexJoCoDualActionFastWAM`；Arm 继续使用 expert 名 `action` 和 6D `ActionDiT`，Hand 使用 expert 名 `hand` 和 16D `ActionDiT`。
- 理由：`MoT.forward()` 已按 `expert_order` 泛化，不需要复制 Transformer 或修改原两 expert 路径；保留 `action` 名称为后续旧 checkpoint key remap 提供稳定目标。
- 边界：本阶段 Hand Transformer 随机初始化；旧 Action Expert 到 Hand backbone 的正式 remap 延后处理。

## D-0014：Arm/Hand 共享完整 22D diffusion sample 和单一 action loss

- 日期：2026-08-15
- 状态：Accepted
- 决策：对 `[B,T,22]` 只生成一次 noise、采样一次 timestep 并调用一次 scheduler，之后拆分 6D/16D；预测拼接后对完整 22D 执行一次原 FastWAM MSE reduction。
- 理由：分别对 6D/16D mean 后相加会把 Arm 与 Hand 各赋相同总权重，改变原来逐 action dimension 平均的 objective。
- 影响：两个 expert 始终共享 timestep、scheduler、text/proprio conditioning 和 padding mask。

## D-0015：双 Action Expert mask 保持 Video 单向因果边界

- 日期：2026-08-15
- 状态：Accepted
- 决策：Video query 只读现有 video mask；Arm/Hand query 均读取所有 Video、Arm 和 Hand keys。
- 影响：Arm/Hand 可以双向交互并读取完整 Video tokens；Video 不读任何动作 token，world objective 的因果方向不变。

## D-0016：Phase 2 显式禁用 DexJoCo video KV cache 推理

- 日期：2026-08-15
- 状态：Accepted
- 决策：设置 `supports_video_kv_cache=False`，并让 DexJoCo cache/inference 入口输出 warning 后抛出 `NotImplementedError`。
- 证据：现有 `MoT.forward_action_with_video_cache()` 只构造 `cached video + action` K/V 和 action queries，没有 Hand K/V/query，无法保持 Arm/Hand 双向交互。
- 后续条件：只有在 cache 同时更新 Arm/Hand 且 sampler 分配完整 22D latent 后才能启用。

## D-0017：Selective loading 只使用精确 root 和精确 local key

- 日期：2026-08-15
- 状态：Accepted
- 决策：仅规范化声明的容器前缀，并只接受 `mot/dit.mixtures.video`、`mot/dit.mixtures.action`、`video_expert`、`action_expert`、`video`、`action`、`dit` 和 `proprio_encoder` 等精确 root；root 之后必须与目标模块 local state key 完全相等。
- 理由：用模糊 suffix 匹配 30 层 DiT 的重复 `blocks.N.*` 名称可能把 Video/Arm/Hand tensor 错装到另一个 expert。
- 影响：无支持 root 的 tensor 进入 `unexpected_in_checkpoint`；关键目标 key 不会由相似后缀补齐。

## D-0018：Projection 的 policy skip 和新初始化分别记账

- 日期：2026-08-15
- 状态：Accepted
- 决策：Arm/Hand 的 `action_encoder`、`head` 以及 23D `proprio_encoder` 同时出现在 `skipped_policy` 和 `newly_initialized`：前者记录不加载旧 tensor 的决定，后者记录实际调用 `torch.nn.Linear.reset_parameters`。
- 理由：旧 7D Action Expert 的 bias 可能与新 encoder bias 偶然同 shape，但 shape 一致不代表语义可迁移；同时只记录 skip 不能证明目标参数经过显式初始化。
- 影响：不执行 slice、插值或部分复制；机器报告明确记录 `initialization_applied`。

## D-0019：关键 backbone 分类完成后再原子式应用

- 日期：2026-08-15
- 状态：Accepted
- 决策：先分类 Video、Arm 和 Hand 的全部关键 backbone key，存在 `missing_in_checkpoint` 或 `skipped_shape` 时先写报告并抛 `SelectiveCheckpointError`，不执行 reset 或 tensor copy。
- 理由：逐 key 边检查边写入会在后段发现错误时留下无法安全训练的部分加载模型。
- 影响：`unexpected_in_checkpoint` 保留为可审计非致命项；关键缺失和 shape 冲突始终 fail-fast。

## D-0020：单一旧 checkpoint 是三个 backbone 的来源

- 日期：2026-08-15
- 状态：Accepted
- 决策：配置 `selective_checkpoint_path` 后跳过单独的 Wan Video DiT 和预处理 ActionDiT backbone 加载；同一 checkpoint 的 Video key 加载到 Video、Action key 加载到 Arm 并显式 remap 到 Hand。
- 理由：这保证审计报告能完整说明三个目标 backbone 的来源，也避免在 selective load 前依赖仓库中当前不存在的 `ActionDiT_linear_interp_*.pt`。
- 边界：VAE 和可选 text encoder 仍由现有 Wan 组件 loader 管理，不属于 Phase 3 selective policy。

## D-0021：Phase 6 action inference 使用显式无 cache 三 expert forward

- 日期：2026-08-15
- 状态：Accepted
- 决策：每个 denoising step 用完整 `[1,T,22]` latent 调用 `_forward_dual_experts()`，Arm/Hand 共享 timestep/scheduler，完整 22D prediction 只 step 一次；旧 `_predict_action_noise_with_cache()` 继续 fail-fast。
- 理由：现有 KV cache 只更新 `action` expert，无法表达 Hand token 或 Arm/Hand 双向 interaction。无 cache 虽较慢，但不会静默改变 Phase 2 attention 契约。
- 边界：`infer_joint()` 仍禁用；Phase 6 server 只提供 action chunk。

## D-0022：Websocket 和 statistics 使用双重 versioned contract

- 日期：2026-08-15
- 状态：Accepted
- 决策：wire schema 固定为 `fastwam.dexjoco.websocket@1`，response 强制 exact 22D ordering；normalizer 只接受 `fastwam.dexjoco.dataset_stats@1` 并默认要求 production training-split statistics。
- 理由：仅凭 output dimension 无法排除 LIBERO 7D statistics、错误 actuator ordering 或错误 action semantics。
- 影响：server 从 checkpoint state directory 使用其 `dataset_stats.json`；smoke stats 只有显式 flag 才可加载。

## D-0023：Inference 只读取预计算 T5 cache

- 日期：2026-08-15
- 状态：Accepted
- 决策：server 按训练 `DEFAULT_PROMPT` 和 exact SHA256 filename 读取 `scripts/precompute_text_embeds.py` payload，缺失即 fail-fast；不加载或回退到 T5 encoder。
- 理由：T5 在训练策略中 frozen，独立加载会增加 server 显存且可能因 prompt 模板/encoder id 不一致产生不同 conditioning。
- 影响：precompute script 同时支持 LeRobot v3 `tasks.parquet` 和旧 `tasks.jsonl`；部署前必须生成所有评测 prompt cache。

## D-0024：Simulator clipping 以运行时 actuator range 为准

- 日期：2026-08-15
- 状态：Accepted
- 决策：rotvec 用 SciPy 转为 `quaternion_wxyz`；Hand 16D 从 `_allegro_ctrl_ids` 对应的 MuJoCo `actuator_ctrlrange` clipping；未声明通用 Cartesian workspace 时 xyz 不做虚构范围 clipping。
- 证据：现有 DexJoCo wrapper 使用 `as_quat(scalar_first=True)`，raw environment 按 `w,qx,qy,qz` 解包。
- 影响：可在六任务 YAML 中显式设置 `xyz_min/xyz_max`，但默认均为 null。
