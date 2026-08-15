# Smoke Test Ledger

## Phase 0 - 2026-08-15

阶段目标：以只读、轻量检查验证仓库基线、真实模块位置和 DexJoCo 22D/23D 数据契约。阶段边界是不执行训练、推理、完整数据扫描或仿真闭环。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P0-01 | Git 根和基线 | 两仓库 `git status --short`、branch、HEAD、remote | PASS | FastWAM `main@45d8e145...` 干净；DexJoCo `main@8d23b0fa...` 保留两个用户修改。 |
| P0-02 | 适用开发说明 | 查找两个仓库及适用祖先的 `AGENTS.md`，阅读仓库 README | PASS | 未发现 `AGENTS.md`；已读 FastWAM 中英文 README 和 DexJoCo README。 |
| P0-03 | 六任务 action schema | 每任务读取 episode 0 最低限度 Parquet schema/行 | PASS | 六任务均为 `fixed_size_list<float>[22]` / info float32。 |
| P0-04 | 六任务 state schema | 每任务读取 episode 0 最低限度 Parquet schema/行 | PASS | 六任务均为 `fixed_size_list<float>[23]` / info float32。 |
| P0-05 | action 语义 | recorder、OpenPI client、policy wrapper、task `step()` 交叉审计 | PASS | 确认绝对 `xyz + rotvec + hand16`；client 转为 `xyz + quat_wxyz + hand16` 后执行。 |
| P0-06 | state 语义 | `proprio_keys`、obs adapter、slice config、client `stay()`、样本 quaternion norm | PASS | 原始 flat state 还有任务字段；converter 前 23D 确认为 `xyz + quat_wxyz + hand16`。quaternion norm 单独不用于推断顺序。 |
| P0-07 | Hand 顺序/单位 | 任务数组、MuJoCo XML joint/actuator/sensor | PASS | `ffj0..3,mfj0..3,rfj0..3,thj0..3`；position target，弧度。 |
| P0-08 | 双相机 metadata | 六任务 info 和 episode 0 timestamp range | PASS | 640x640x3、AV1、30 FPS；每任务两路起止范围一致。 |
| P0-09 | 相机逻辑时间对齐 | converter 多相机队列和 step assertion | PASS | `dexjoco-data-converter/src/dexjoco_data_converter/to_lerobot/merge_episode_lerobot.py:332-350` 按同一 `t` 取帧并断言 step。 |
| P0-10 | MP4 像素解码 | 未运行 | NOT RUN | 环境无可用 `ffprobe`/视频解码 backend；不能据此确认像素内容或逐帧无丢帧。 |
| P0-11 | Dataset split/FPS | 六任务 `meta/info.json` | PASS | 每任务 100 episodes，`train: 0:100`，30 FPS。 |
| P0-12 | rand_obj/rand_full 组织 | 两套 selected/slice/path config | PASS | rand_obj 为固定任务相机 + wrist；rand_full 为 random_camera + wrist；单臂 state 前 23D。 |
| P0-13 | Simulator control contract | wrapper、六任务 `step()`、opspace controller、XML | PASS | arm 为绝对 mocap target + operational-space torque；hand 为 position actuator。 |
| P0-14 | FastWAM action layout | dataset、`ActionDiT.forward`、`infer_action` | PASS | train/model `[B,T,D]`，模型 API 返回 `[T,D]`；迁移目标 D=22。 |
| P0-15 | Video cache/mask | `MoT` 和 `FastWAM` mask/cache 代码审计 | PASS | 每层 K/V `[B,Sv,H*Dh]`；当前 action 仅看首帧 video，video 不看 action。 |
| P0-16 | Training/inference | 未运行 | NOT RUN | Phase 0 禁止代码变更，且不做训练或模型推理。 |
| P0-17 | Checkpoint round trip | 未运行 | NOT RUN | 双专家和 22D projection 尚未实现。 |
| P0-18 | DexJoCo simulator closed loop | 未运行 | NOT RUN | 没有执行 environment reset/step 或任务成功率测试。 |

### 六任务 episode 0 取样账本

| 任务 | 数据来源 | length | camera keys | timestamp range |
| --- | --- | ---: | --- | --- |
| `water_plant` | 本地 LeRobot 数据 | 309 | `front,wrist` | 0.000000-10.300000 s |
| `hammer_nail` | 本地 LeRobot 数据 | 202 | `front,wrist` | 0.000000-6.733333 s |
| `click_mouse` | 官方 HF 最小临时下载 | 500 | `ego_right,wrist` | 0.000000-16.666667 s |
| `pick_bucket` | 本地 LeRobot 数据 | 537 | `front,wrist` | 0.000000-17.900000 s |
| `pinch_tongs` | 官方 HF 最小临时下载 | 341 | `front,wrist` | 0.000000-11.366667 s |
| `fold_glasses` | 本地 LeRobot 数据 | 560 | `front,wrist` | 0.000000-18.666667 s |

### Phase 0 风险和下一阶段门槛

- 风险：视频只验证 metadata/锁步逻辑；两任务样本不是本地持久副本；30 FPS 数据频率与 50 Hz 默认环境控制频率需明确衔接；22D/23D statistics 尚未验证。
- 下一阶段门槛：明确双专家结构、mask/cache、checkpoint remap 和新 projection 初始化规则；在获授权后增加针对 22D/23D 的单元及 runtime smoke tests。
- 计划 commit message：`docs: record DexJoCo 22d adaptation contract`

## Phase 1 - 2026-08-15

阶段目标：验证 DexJoCo 专用 LeRobot v3 路径、六任务 22D/23D batch、双相机像素解码、padding/mask 和 versioned normalization。阶段边界是不执行全量 statistics、训练、模型推理或 simulator。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P1-01 | 阶段 Git 基线 | 两仓库 `status`/branch/HEAD；读取 Phase 0 全部记录 | PASS | FastWAM `main@835c98b8...` 干净；DexJoCo `main@8d23b0fa...` 两个用户修改保持原样。 |
| P1-02 | 原 LeRobot reader 兼容性 | 用真实 `water_plant` 初始化内置 metadata reader | EXPECTED FAIL | reader 查找 v2 `meta/tasks.jsonl`，而数据为 v3 `meta/tasks.parquet` 和 file-level 分片；据此使用专用 backend。 |
| P1-03 | v3 低维映射 | `DexJoCoV3TaskSource.load_episode_low_dim()` 读取 episode 0 | PASS | `water_plant` 得到 action `[309,22]` float32、state `[309,23]` float32；episode_index 行区间一致。 |
| P1-04 | AV1 像素解码 | torchcodec 读取真实 `file-000.mp4` | PASS | 输出 `[N,3,640,640]` uint8；六任务正式 smoke 均实际解码两路视频。 |
| P1-05 | 六任务 DataLoader shape | `scripts/smoke_test_dexjoco_data.py --num-workers 2`，每任务 episode 0 首 clip | PASS | 每任务均为 action `[1,32,22]`、arm `[1,32,6]`、hand `[1,32,16]`、state `[1,32,23]`、cameras `[1,2,3,9,224,224]`、video `[1,3,9,224,448]`。 |
| P1-06 | 相机 alias | 六任务 batch + source metadata | PASS | `click_mouse` 映射 `ego_right,wrist`；其余为 `front,wrist`；公开顺序统一 `primary,wrist`。 |
| P1-07 | padding/mask 边界 | `water_plant` episode 0 最后一帧起始 clip | PASS | action pad `31/32`、state pad `31/32`、image pad `8/9`，mask 分别与输出时间轴对齐。 |
| P1-08 | 限量 statistics CLI | 六任务各 1 episode | PASS | 6 episodes / 2449 frames；`schema@1`、`mode=smoke`、`production=False`、`split=train`。 |
| P1-09 | normalization round trip | 真实 `water_plant` 少量 action/state | PASS | 最大绝对误差 action `5.960e-08`、state `1.192e-07`。 |
| P1-10 | 零/近零方差处理 | 标准库单元测试 synthetic constant dimension | PASS | std clamp 到 `1e-6`；触发维度被记录。 |
| P1-11 | stats 防误读 | schema/version/production/ordering/dim/shape validator 单元测试 | PASS | production 路径拒绝 smoke；缺 schema 的 LIBERO 形态直接报错。 |
| P1-12 | Hydra 真实入口 | `data=dexjoco_6task_2cam` compose + instantiate | PASS | `DexJoCoRobotVideoDataset` -> `DexJoCoV3Dataset`，临时六任务根共 218993 frames。 |
| P1-13 | 多 worker | 六任务 DataLoader `num_workers=2` | PASS | 六个 batch 全部成功，decoder/Parquet cache 可在 worker 路径使用。 |
| P1-14 | 聚焦测试/语法 | `python tests/test_dexjoco_data.py`、`compileall`、`git diff --check` | PASS | 3 个标准库测试通过；编译和 whitespace 检查通过。环境未安装 pytest，因此测试不依赖 pytest。 |
| P1-15 | 完整 training statistics | 未运行 | NOT RUN | 本机持久根缺两任务；本阶段明确不扫描完整 split。 |
| P1-16 | 训练/模型推理/checkpoint | 未运行 | NOT RUN | 超出 Phase 1 边界。 |
| P1-17 | DexJoCo simulator | 未运行 | NOT RUN | 超出 Phase 1 边界。 |

### Phase 1 六任务输出账本

| 任务 | 数据来源 | action | arm | hand | state | 两路 camera | 拼接 video |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `water_plant` | 本地持久数据 | `[1,32,22]` | `[1,32,6]` | `[1,32,16]` | `[1,32,23]` | `[1,2,3,9,224,224]` | `[1,3,9,224,448]` |
| `hammer_nail` | 本地持久数据 | `[1,32,22]` | `[1,32,6]` | `[1,32,16]` | `[1,32,23]` | `[1,2,3,9,224,224]` | `[1,3,9,224,448]` |
| `click_mouse` | 官方 HF 临时最小视频/低维文件 | `[1,32,22]` | `[1,32,6]` | `[1,32,16]` | `[1,32,23]` | `[1,2,3,9,224,224]` | `[1,3,9,224,448]` |
| `pick_bucket` | 本地持久数据 | `[1,32,22]` | `[1,32,6]` | `[1,32,16]` | `[1,32,23]` | `[1,2,3,9,224,224]` | `[1,3,9,224,448]` |
| `pinch_tongs` | 官方 HF 临时最小视频/低维文件 | `[1,32,22]` | `[1,32,6]` | `[1,32,16]` | `[1,32,23]` | `[1,2,3,9,224,224]` | `[1,3,9,224,448]` |
| `fold_glasses` | 本地持久数据 | `[1,32,22]` | `[1,32,6]` | `[1,32,16]` | `[1,32,23]` | `[1,2,3,9,224,224]` | `[1,3,9,224,448]` |

### Phase 1 风险和下一阶段门槛

- 风险：持久六任务数据和 production statistics 尚未就绪；真实 T5 cache 尚未生成；30 FPS 数据到 50 Hz simulator 的 runtime 连接仍待后续确认。
- 门槛：补齐 `click_mouse`/`pinch_tongs`，无 episode limit 生成并保存 production `dataset_stats.json`，生成真实 text embeddings；本阶段提交完成后需获得下一阶段明确授权。
- 计划 commit message：`feat(data): add DexJoCo 22d action pipeline`

## Phase 2 - 2026-08-15

阶段目标：验证 DexJoCo Video/Arm/Hand 三 expert MoT、22D action diffusion/loss、23D proprio 和联合 attention 方向。阶段边界是不加载完整 checkpoint、不实现迁移/optimizer 分组、不运行完整 pytest、训练或正式推理。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P2-01 | 阶段 Git 基线 | 两仓库 status/branch/HEAD；读取全部项目记录 | PASS | FastWAM `main@8409c0fe...` 干净；DexJoCo `main@8d23b0fa...` 两个用户修改保持原样。 |
| P2-02 | 三 expert 实例化 | 1 层真实 `WanVideoDiT` + 两个 1 层真实 `ActionDiT` + `MoT` | PASS | `expert_order=('video','action','hand')`；Arm=6D、Hand=16D；两者 backbone key/shape 完全一致。 |
| P2-03 | 完整 forward | tiny CPU `training_loss(..., return_outputs=True)` | PASS | loss 有限；Arm `[1,4,6]`、Hand `[1,4,16]`、拼接 `[1,4,22]`。 |
| P2-04 | 完整 action 契约 | `[1,4,22]` 输入及错误维度 fail-fast 代码路径 | PASS | `split_action()` 固定 `:6` / `6:22`，输入末维非 22 直接报错。 |
| P2-05 | 单次 diffusion sample | counting scheduler 包装真实 action scheduler | PASS | `sample_training_t` 1 次，`add_noise` 1 次且输入 shape `[1,4,22]`。 |
| P2-06 | Attention mask shape | 8 video + 4 Arm + 4 Hand tokens | PASS | 联合 mask `[16,16]`。 |
| P2-07 | Attention 方向 | 对 mask 三个 block 逐项断言 | PASS | Video->Arm/Hand 全 false；Arm/Hand->完整 Video 全 true；Arm/Hand 联合 block 全 true。 |
| P2-08 | 23D proprio encoder | tiny model 参数检查和一次 forward | PASS | `in_features=23` 且参数 `requires_grad=true`，proprio token 加入三个 expert 共用 context。 |
| P2-09 | Cache fail-fast | 调用 DexJoCo `infer_action()` | EXPECTED FAIL | warning + `NotImplementedError`；现有 cache 只处理 `action` expert，禁止静默忽略 Hand。 |
| P2-10 | 原 FastWAM 兼容 | tiny 原 `FastWAM` 两 expert / 7D ActionDiT 实例化 | PASS | `expert_order=('video','action')`、action dim=7；原类和配置未修改。 |
| P2-11 | Hydra 配置 | compose `model=dexjoco_dual_action data=dexjoco_6task_2cam` | PASS | runtime target 和 22/6/16/23D 配置均解析正确。 |
| P2-12 | 聚焦语法检查 | 新模型/runtime/smoke `compileall`、`git diff --check` | PASS | 编译和 whitespace 检查通过。 |
| P2-13 | 完整 checkpoint/5B forward | 未运行 | NOT RUN | 正式 checkpoint 迁移被 Phase 2 明确排除，tiny smoke 不下载/加载 5B 权重。 |
| P2-14 | optimizer/训练/完整 pytest | 未运行 | NOT RUN | 超出 Phase 2 边界。 |
| P2-15 | 正式 inference/simulator | 未运行 | NOT RUN | 双 expert cache/sampler 尚未实现，simulator 不属于本阶段。 |

### Phase 2 风险和下一阶段门槛

- 风险：Hand backbone 尚未 remap；新 projection 初始化和迁移报告尚未落地；DexJoCo cache/joint inference 显式禁用；tiny CPU 模型不覆盖完整 5B 显存和分布式行为。
- 门槛：实现并审计正式 checkpoint key remap/shape 报告，保持 Arm/Hand/proprio 新 projection 规则；另行设计能表达 Arm/Hand 交互的 cache/sampler；本阶段提交后需获得下一阶段明确授权。
- 计划 commit message：`feat(model): add dual action experts for DexJoCo`

## Phase 3 - 2026-08-15

阶段目标：实现可审计、可保存 JSON 报告的 selective pretrained checkpoint loading。阶段边界是不处理 optimizer、resume state、正式训练、完整 pytest 或 simulator。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P3-01 | 阶段 Git 基线 | 两仓库 status/branch/HEAD；读取全部记录 | PASS | FastWAM `main@03caff2a...` 干净；DexJoCo `main@8d23b0fa...` 两个用户修改保持原样。 |
| P3-02 | Phase 2 hash 回填 | `commit_index.md` | PASS | `03caff2af53632914ec7418716480b7a5ae6dbdc`。 |
| P3-03 | 七类 synthetic 分类 | tiny 旧 `video+7D action` FastWAM `mot` payload，经 `module.model.mot.*` 前缀从临时 `.pt` 加载 | PASS | 成功报告：loaded=79、copied_to_hand=37、skipped_shape=0、skipped_policy=10、missing=0、unexpected=1、newly_initialized=10。 |
| P3-04 | Tensor 相等抽查 | Video self-attn、Arm FFN、Hand cross-attn | PASS | Video/Arm 等于各自源 tensor；Hand 等于同一个旧 Action Expert local key。 |
| P3-05 | Projection policy | 检查报告 target 集合和实际 shape | PASS | 10 个 Arm/Hand/proprio 参数都在 `skipped_policy`/`newly_initialized`，都不在 loaded/copied；Arm 6D、Hand 16D、proprio 23D。 |
| P3-06 | 显式初始化 | 先把新 projection 全填 99，再 selective load | PASS | `Linear.reset_parameters` 已执行，报告 `initialization_applied=true`，结果不再为 sentinel。 |
| P3-07 | fail-fast 与无部分写入 | 删除 1 个 Video key，并让 1 个旧 Action backbone tensor shape 错误 | EXPECTED FAIL | missing=1、skipped_shape=2（Arm/Hand）；抛 `SelectiveCheckpointError`，backbone/projection 均保持调用前值。 |
| P3-08 | JSON schema | 读取临时 report JSON | PASS | `schema=fastwam.dexjoco_selective_checkpoint`、`version=1`，汇总与内存报告一致。 |
| P3-09 | 完整配置 meta target | 构造正式 Video/Arm/Hand 模块 shape 图 | PASS | Video=825 keys、Arm=824、Hand=824、proprio=2；维度 6/16/23，不分配 5B 参数存储。 |
| P3-10 | 新旧 Hydra compose | DexJoCo selective path/report override；原 FastWAM+LIBERO | PASS | 新 runtime 签名解析；原配置仍为 7D 且没有 selective 字段。 |
| P3-11 | Phase 2 forward 回归 | 原 tiny dual-action smoke | PASS | Arm `[1,4,6]`、Hand `[1,4,16]`、action `[1,4,22]`、mask `[16,16]`。 |
| P3-12 | 真实 checkpoint CPU dry-run | 搜索仓库、`outputs`、用户/root HF cache 和 `/home/shared/ai` | NOT RUN | 未找到 FastWAM weight checkpoint 或配置中的 ActionDiT 文件；不下载或伪造真实权重。 |
| P3-13 | 聚焦语法/whitespace | `compileall`、`git diff --check` | PASS | 新 loader、模型/runtime 接线和两个 scripts 编译通过。 |
| P3-14 | `ruff` | `conda run -n fastwam ruff check ...` | NOT RUN | `fastwam` 环境未安装 `ruff`，命令返回 127。 |
| P3-15 | 完整 pytest/训练/optimizer | 未运行 | NOT RUN | 超出 Phase 3 边界。 |

### Phase 3 风险和下一阶段门槛

- 风险：没有真实旧 FastWAM checkpoint，因此本阶段只确认 synthetic tensor 内容和正式 5B config 的 meta key/shape 图；真实权重的 key/count 仍需在获得 checkpoint 后用审计 CLI 验证。
- 门槛：真实报告必须满足 `missing_in_checkpoint=0`、`skipped_shape=0`，并人工复核 `unexpected_in_checkpoint`；之后才能设计 optimizer 分组或启动训练。
- 计划 commit message：`feat(checkpoint): add selective DexJoCo weight loading`

## Phase 4 - 2026-08-15

阶段目标：验证 DexJoCo joint post-training 的训练/冻结状态、互斥且完备的命名 optimizer groups，以及保持组间 LR 比例的 scheduler。阶段边界是不执行 backward、正式训练、完整 pytest、分布式/DeepSpeed optimizer-state 恢复或 simulator。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P4-01 | 阶段 Git 基线 | 两仓库 status/branch/HEAD；读取全部项目记录 | PASS | FastWAM `main@9854c306...` 干净；DexJoCo `main@8d23b0fa...` 两个用户修改保持原样；Phase 3 hash 已回填。 |
| P4-02 | 真实 checkpoint dry-run | 正式 5B meta target + `checkpoints/libero_uncond_2cam224.pt`，`apply=false` | PASS | loaded=1645、copied_to_hand=820、skipped_shape=0、skipped_policy=10、missing=0、unexpected=0、newly_initialized=10；未开始训练。 |
| P4-03 | Tiny optimizer groups | 1 层真实 Video/Arm/Hand expert + 23D proprio | PASS | `action_new` 10 tensors/1,894 params；`action_backbone` 74/26,928；`video_backbone` 42/13,732。 |
| P4-04 | 正式结构 group 审计 | 正式 5B 配置在 meta device 构造 modules/optimizer | PASS | `action_new` 10 tensors/145,430 params；`action_backbone` 1,640/2,041,741,312；`video_backbone` 825/4,999,787,712。 |
| P4-05 | 互斥和覆盖 | 比较所有 optimizer parameter object id 与全部 `requires_grad=True` parameter id | PASS | 每个 trainable parameter 恰好出现一次；overlap=0；frozen-in-optimizer=0。 |
| P4-06 | T5/VAE freeze | 检查 `requires_grad`、module mode，并模拟 `model.train()` 后重用 trainer freeze 入口 | PASS | T5/VAE 全部 `requires_grad=False` 且 `training=False`；Video/Arm/Hand/proprio 为 trainable/train。 |
| P4-07 | LR policy | 配置值、启动组属性和错误顺序 fail-fast | PASS | base LR=`1e-4,5e-5,1e-5`，weight decay 均 `0.01`；满足 `action_new > action_backbone >= video_backbone`；无效顺序抛错。 |
| P4-08 | Scheduler 比例 | cosine scheduler 含 2-step warmup，optimizer/scheduler 前进一步 | PASS | stepped LR=`7.5e-5,3.75e-5,7.5e-6`，三组 common factor=`0.75`，顺序/比例保持。 |
| P4-09 | 代表参数归属 | 按 parameter object 查 optimizer group | PASS | Arm/Hand encoder/head 与 proprio -> `action_new`；Arm/Hand blocks -> `action_backbone`；Video block -> `video_backbone`。 |
| P4-10 | 原 FastWAM 兼容 | 原 tiny 7D FastWAM 构造 legacy optimizer | PASS | 单一 `default` AdamW group；LIBERO Hydra 配置 `optimizer_groups=null`、action_dim=7。 |
| P4-11 | Phase 2/3 回归 | dual-action forward smoke、selective synthetic smoke | PASS | forward action `[1,4,22]`、mask `[16,16]`；selective 成功/故障分类与 tensor equality/fail-fast 均通过。 |
| P4-12 | Hydra/absolute action | compose DexJoCo joint task config | PASS | 三组 LR、真实 checkpoint 相对路径、6/16/23D 解析正确；`data.train.processor.delta_action_dim_mask=null`。 |
| P4-13 | 语法/whitespace | `compileall`、`git diff --check` | PASS | Phase 4 Python 文件编译和 whitespace 检查通过。 |
| P4-14 | 正式训练/backward/完整 pytest | 未运行 | NOT RUN | 超出 Phase 4 边界。 |
| P4-15 | 分布式和 resume | 未运行 | NOT RUN | 未验证 DeepSpeed/ZeRO optimizer state、Accelerate 多 rank 或旧单组 state 到三组 state 的恢复。 |

### Phase 4 风险和下一阶段门槛

- 风险：正式参数计数来自 meta modules，证明 ownership graph 但不证明训练硬件上的显存、通信和吞吐；没有执行 backward、optimizer state 分配或 checkpoint resume。
- 风险：当前 DexJoCo cached inference 仍按 Phase 2 fail-fast 禁用；六任务 production `dataset_stats.json` 仍未生成，不能启动正式训练。
- 门槛：完整 training split statistics 就绪并通过 schema 校验；在目标分布式配置上做单步 forward/backward、optimizer/scheduler 和 save/resume smoke；确认双 Action Expert inference/cache 路径后再进入正式训练或仿真。
- 计划 commit message：`feat(train): add DexJoCo parameter groups`

## Phase 5 - 2026-08-15

阶段目标：用真实六任务最小数据闭合 normalization -> joint diffusion/forward/loss -> backward/step -> checkpoint save/reload。边界是一个 tiny CPU optimizer step，不执行正式训练、完整 pytest、5B backward、DeepSpeed/多卡、推理或 simulator。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P5-01 | 阶段 Git 基线 | 两仓库 status/branch/HEAD；读取全部项目记录 | PASS | FastWAM `main@43f45064...` 干净；DexJoCo `main@8d23b0fa...` 两个用户修改保持原样；Phase 4 hash 已回填。 |
| P5-02 | 六任务 dataloader | 每任务 episode 0 首 clip，真实 image/action/state + smoke normalizer | PASS | action `[6,4,22]`、state `[6,4,23]`、cameras `[6,2,3,5,16,16]`、video `[6,3,5,16,32]`；任务顺序匹配正式列表。 |
| P5-03 | Statistics policy | Phase 1 六任务各 1 episode stats | PASS/SMOKE ONLY | `fastwam.dexjoco.dataset_stats@1`、`production=false`；仅显式 `allow_non_production_stats=true`，没有冒充正式统计。 |
| P5-04 | 一次完整 22D diffusion | counting wrapper + `return_outputs` | PASS | action timestep 只采样 1 次、noise 只加 1 次且 shape `[6,4,22]`；Arm/Hand 共用 timestep。 |
| P5-05 | Joint forward shapes | 1 层真实 Video/Arm/Hand + MoT | PASS | Arm `[6,4,6]`、Hand `[6,4,16]`、拼接 prediction/target `[6,4,22]`，全 finite。 |
| P5-06 | Loss 数学 | 独立复算 padding/scheduler-weighted action objective | PASS | total=`4.11274576`、video=`2.52383971`、action=`1.58890617`；完整 22D MSE 与 `loss_action` 一致；video/world objective 未改。 |
| P5-07 | Backward gradients | step 前逐模块审计所有 grad | PASS | Video 42/42、Arm backbone 37/37、Arm new 4/4、Hand backbone 37/37、Hand new 4/4、proprio 2/2 tensor 均 finite/nonzero。 |
| P5-08 | Trainable 更新 | step 前后逐 tensor 比较 | PASS | 上述六类分别变化 42、37、4、37、4、2 个 tensor；optimizer steps=1。 |
| P5-09 | Frozen 检查 | T5/VAE probe parameters | PASS | 所有 grad 为 `None`；step 前后逐 tensor 完全相等。 |
| P5-10 | Checkpoint 内容 | `Wan22Trainer.save_checkpoint()` 临时目录 | PASS | model、optimizer、scheduler、resolved config、22/6/16/23D manifest、stats、selective report、trainer progress 均存在。 |
| P5-11 | Save/reload | 新 tiny model/optimizer/scheduler 调用真实 `load_training_state()` | PASS | 全 model tensor 精确相等；optimizer group/state/LR、scheduler state、stats 内容和 step/batch offset 恢复一致。 |
| P5-12 | Resume fail-fast | 分别损坏 action dim、proprio dim、statistics schema | EXPECTED FAIL | 三者均在 `accelerator.load_state()` 前抛错；没有加载任何 tensor state。 |
| P5-13 | Phase 2–4 回归 | dual-action、selective checkpoint、optimizer scripts | PASS | forward/mask/cache fail-fast、七类 checkpoint 分类和三组 optimizer/scheduler 均保持通过。 |
| P5-14 | 语法/whitespace | `compileall`、`git diff --check` | PASS | Phase 5 Python 文件编译和 whitespace 检查通过。 |
| P5-15 | 完整 pytest/正式训练 | 未运行 | NOT RUN | 符合阶段边界；临时 checkpoint 自动删除，没有提交训练产物。 |

### Phase 5 风险和正式训练门槛

- 风险：tiny 单进程 CPU 结果不覆盖正式约 7B trainable parameters、bf16、ZeRO2/multi-rank shard、显存、通信或吞吐。
- 风险：T5 context 为合成零 cache；persistent 数据根仍缺 `click_mouse`/`pinch_tongs`；production statistics 未生成；双 expert inference/cache 仍禁用。
- 门槛：补齐六任务数据；不带 episode limit 计算 production training-split stats；生成真实 T5 cache；在目标 DeepSpeed 配置上先做 1-step 5B save/resume；之后才能授权正式训练。
- 计划 commit message：`feat(train): close DexJoCo 22d training loop`

## Phase 6 - 2026-08-15

阶段目标：建立 FastWAM 双相机/23D state/cached T5 websocket inference server，以及 DexJoCo 22D action client、23D simulator command adapter 和 action chunk/replan 闭环。边界是 tiny CPU websocket 请求与 `water_plant` reset + 2 steps，不加载正式 5B joint-trained 权重，不执行六任务完整评测。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P6-01 | 阶段 Git 基线 | 两仓库 status/branch/HEAD；补录 Phase 5 hash | PASS | FastWAM `main@88c04006...` 干净；DexJoCo `main@8d23b0fa...` 仅两个受保护用户修改；Phase 5 hash 已回填。 |
| P6-02 | DexJoCo client 提交 | 精确 stage client/adapter/config/smoke 文件 | PASS | `992abca3da2bb34475485658bf76b766c49b7efa`；后续 protocol-only lazy import fix 为 `3c85e48dc50c29e204259261eb97f3419e26e969`。 |
| P6-03 | Websocket 协议 | 真实 DexJoCo `FastWAMWebsocketClient` 请求本地 ephemeral server | PASS | `fastwam.dexjoco.websocket@1`；binary msgpack NumPy；request cameras HWC uint8、state `[23]`，response `[4,22]` float32 finite。 |
| P6-04 | 双相机预处理 | primary/wrist `[18,20,3]` resize/concat | PASS | 每路 resize `[3,16,16]`，模型输入 `[1,3,16,32]`，顺序 primary 后 wrist。 |
| P6-05 | State/statistics | 23D normalize + checkpoint stats contract | PASS | normalize finite；默认拒绝 production=false；伪造 LIBERO schema/action_dim=7 在 policy 建立前失败。 |
| P6-06 | Cached T5 | exact prompt template/SHA256/cache payload | PASS | cache hit shape `[1,3,32]`；缺 cache fail-fast，不加载 T5；六任务 `tasks.parquet` prompt discovery PASS。 |
| P6-07 | 双 Expert sampler | tiny Video/Arm/Hand/MoT，2 个无 cache denoising steps | PASS | model action `[1,4,22]`、Arm `[1,4,6]`、Hand `[1,4,16]`；scheduler 每步只接收 `[1,4,22]`。 |
| P6-08 | Attention/cache | mask 审计及旧 cache 入口 | PASS | mask `[16,16]`；Video 不读动作，Arm/Hand 读 Video 且双向交互；旧 single-action cache 继续显式 fail-fast。 |
| P6-09 | Server 输出 | 拼接一致性、22D denormalization、wire 校验 | PASS | Arm/Hand 必须精确拼回 joint action；wire `[4,22]` finite float32；没有使用 LIBERO 7D denormalizer。 |
| P6-10 | 异常输入 | schema/batch/horizon/image/state/dtype/NaN/拼接不一致 | EXPECTED FAIL | 所有错误均被协议或 policy 显式拒绝，不广播、不截断。 |
| P6-11 | Quaternion 数值 | SciPy rotvec -> `scalar_first=True` quaternion | PASS | 零旋转得到 wxyz `(1,0,0,0)`；x 轴 pi 得到绝对值 `(0,1,0,0)`。 |
| P6-12 | Simulator control | `water_plant` reset + 2 steps | PASS/NO SUCCESS REQUIRED | 最终 `environment.step()` command `[23]` finite；success=false；不属于任务成功评测。 |
| P6-13 | Clipping/chunk/video | runtime actuator ctrlrange；H=4，replan=0.5；临时 MP4 | PASS | 16D Hand 依据 MuJoCo actuator IDs/range clipping；1 次 request 执行 2 steps；`front.mp4`/`wrist.mp4` 均生成后随临时目录删除。 |
| P6-14 | Phase 2-5 回归 | dual action/selective checkpoint/optimizer/training smoke | PASS | Phase 5 loss 仍为 total/video/action `4.11274576/2.52383971/1.58890617`；save/reload 和 resume fail-fast PASS。 |
| P6-15 | 正式部署 | 5B checkpoint/server/六任务完整评测 | NOT RUN | 缺持久 joint-trained state directory、production DexJoCo stats 和 T5 cache；仓库现有 LIBERO checkpoint 必须被拒绝。 |

### Phase 6 风险和正式部署门槛

- 风险：无 cache sampler 每个 diffusion step 重算 Video/Arm/Hand，tiny CPU smoke 不代表 5B CUDA 显存、吞吐或 latency。
- 风险：只验证单连接 request/reply 和一个任务的两步模拟；认证、并发、长连接、完整 episode 与六任务成功率尚未覆盖。
- 门槛：完整 training split production statistics、真实六任务 T5 cache 和 joint-trained Phase 5 state directory 就绪；先进行目标 GPU 的短 server/client smoke，再授权完整评测。
- 计划 FastWAM commit message：`feat(inference): serve DexJoCo 22d actions`

## Phase 7 monitoring - 2026-08-15

阶段目标：增加 rank-0 TensorBoard、loss 收敛摘要和与 production statistics 一致的 90/10 train/validation episode split。边界是 4+1 个 tiny CPU optimizer steps，不运行完整 pytest、正式 full-model train 或 simulator 评测。

| ID | 检查 | 范围/方法 | 结果 | 证据或限制 |
| --- | --- | --- | --- | --- |
| P7-01 | 阶段 Git 基线 | 两仓库 status/branch/HEAD；读取全部项目记录；补 Phase 6 hashes | PASS | FastWAM `fastwam22d-train@34e1dd3...` 干净；DexJoCo `main@3c85e48...` 两个用户修改保持原样。 |
| P7-02 | 数据完整性 | `/home/shared/ai/datasets/dexlewm/dexjoco` 六任务 v3 metadata | PASS | 每任务 100 episodes；总 218,993 frames；双相机；识别为 `rand_obj`，未发现独立 `rand_full` LeRobot root。 |
| P7-03 | 90/10 split | shared `select_dexjoco_episode_indices()` + runtime train/val instantiate | PASS | 每任务 train=90、val=10；交集为空、并集覆盖 100；seed=42、task-local deterministic。 |
| P7-04 | Statistics parity | `compute_dexjoco_stats.py --max-episodes-per-task 1 --val-set-proportion 0.1` | PASS/SMOKE ONLY | schema v1、22D/23D、`rand_obj`、split policy/IDs 已记录；六任务共 6 episodes/2,518 frames；`production=false`。 |
| P7-05 | TensorBoard rank/step | actual `Wan22Trainer` 4 steps + save/resume to step 5 | PASS | rank 0 两个 event files；聚合 train steps `[1,2,3,5]`，resume purge step 4 后从 5 继续；无 step 0。 |
| P7-06 | 必需 scalars | EventAccumulator 审计 25 tags | PASS | train/val total/video/action/arm/hand、3 LR、4 grad norms、timing/progress、prediction/target mean/std 均存在且 finite。 |
| P7-07 | LR 和 gradient | 每个 event step 数值检查 | PASS | LR 比例始终 `10:5:1`；Video/Arm/Hand/action_new grad norm 均 finite/nonzero，且在 unscale 后、clip 前计算。 |
| P7-08 | Frozen modules | tiny T5/VAE 全参数 | PASS | 5 steps 后 grad 始终 `None`；第一段训练前后参数逐 tensor 不变。 |
| P7-09 | Writer cleanup | normal completion、resume completion | PASS | 两次 `train()` 返回后 writer 均为 `None`；flush/close 在 `finally`。 |
| P7-10 | Summary CLI | smoke event，window=3、min_points=10 | PASS | total/video/action/arm/hand 均 `insufficient_data`；没有把 4 个点伪称收敛。 |
| P7-11 | Launcher | 本地空 logdir、host 127.0.0.1、port 16006、timeout cleanup | PASS | 输出 `http://127.0.0.1:16006/`；默认不监听 0.0.0.0。 |
| P7-12 | Dependency | TensorBoard 2.19 vs 2.21 import probe | PASS | 2.19 与 setuptools 83 不兼容；声明并实装 `tensorboard==2.21.0` 后 launcher/EventAccumulator 正常。 |
| P7-13 | 语法/whitespace | `compileall`、`git diff --check` | PASS | Phase 7 Python 文件编译通过，无 whitespace error。 |
| P7-14 | Phase 2/4/5 targeted regression | dual-action、optimizer、training-loop scripts | PASS | legacy instantiate、22D mask/forward、三组 LR/覆盖和 Phase 5 loss `4.11274576`、save/reload 均保持通过。 |
| P7-15 | 完整 pytest/正式训练 | 未运行 | NOT RUN | 符合第一笔提交边界；正式 statistics/T5/full-model run 在监控实现 commit 后执行。 |

### Phase 7 monitoring 风险和正式 run 门槛

- 风险：tiny CPU 模型不覆盖 bf16/ZeRO2/4 GPU 或完整 7B trainable graph；validation diffusion loss 仍有 timestep/noise 随机波动。
- 门槛：从干净监控实现 commit 生成 90 episodes/task production statistics、六 prompt T5 cache；先实测一个 full-model step 的显存/吞吐，再启动唯一一次有限预算正式 run。
- 计划 commit message：`feat(train): add DexJoCo TensorBoard monitoring`
