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
