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
