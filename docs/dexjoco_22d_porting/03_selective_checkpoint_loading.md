# Phase 3：Selective Checkpoint Loading

## 阶段记录

- 日期：2026-08-15
- 阶段目标：从旧 FastWAM 两 expert checkpoint 选择性加载 Video/Action backbone，将同一个旧 Action backbone 显式 remap 到 Hand，并为每个参数生成可审计分类。
- 阶段边界：只处理 pretrained checkpoint 初始化、机器报告和 targeted smoke；不处理 optimizer、scheduler/resume state、正式训练、完整 pytest、推理或 simulator。
- 计划 commit message：`feat(checkpoint): add selective DexJoCo weight loading`

### 阶段开始时仓库状态

| 仓库 | 分支 | HEAD | `git status --short` |
| --- | --- | --- | --- |
| FastWAM `/home/user/fastwam-22d` | `main` | `03caff2af53632914ec7418716480b7a5ae6dbdc` | 干净 |
| DexJoCo `/home/shared/ai/datasets/DexJoCo/dexjoco` | `main` | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` | ` M environment-dexjoco.yaml`<br>` M openpi/packages/openpi-client/pyproject.toml` |

开始前已读取 `docs/dexjoco_22d_porting/` 全部记录、检查两个工作树，并把 Phase 2 实际 commit hash 写入 `commit_index.md`。DexJoCo 两个用户文件未修改、未暂存。

## 真实入口和配置

核心实现：

```text
fastwam.models.wan22.dexjoco_checkpoint.load_dexjoco_selective_checkpoint
fastwam.models.wan22.dexjoco_checkpoint.SelectiveCheckpointReport
fastwam.models.wan22.dexjoco_checkpoint.SelectiveCheckpointError
```

模型 API：

```text
DexJoCoDualActionFastWAM.load_selective_pretrained_checkpoint(...)
DexJoCoDualActionFastWAM.from_wan22_pretrained(
    selective_checkpoint_path=...,
    selective_checkpoint_report_path=...,
)
```

Hydra 配置新增两个默认不启用的字段：

```yaml
selective_checkpoint_path: null
selective_checkpoint_report_path: null
```

未设置 selective path 时，Phase 2 和原 FastWAM/LIBERO 构造行为不变。设置后，DexJoCo 构造器跳过独立 Video/Action DiT pretrained load，随机建立正式 target structure，再从同一个旧 checkpoint 完成 Video/Arm/Hand selective load。VAE 和可选 T5 不属于该 policy，仍使用现有 Wan loader。

## Checkpoint 格式和精确 key policy

支持以下容器：

- FastWAM `{"mot": state_dict, "proprio_encoder": ...}` 权重文件。
- `state_dict`、`model_state_dict`、`model_state`、`model`、`module`、`policy` mapping wrapper。
- legacy `{"dit": ...}` Video-only payload；由于没有旧 Action backbone，最终会按关键缺失 fail-fast。
- `{"backbone_state_dict": ...}` Action-only payload；由于没有 Video backbone，同样 fail-fast。
- flat PyTorch state dict 和 flat safetensors。

只循环去除精确 common container prefix：

```text
module.  _orig_mod.  model.  policy.  network.
```

之后只识别声明的 expert root，例如：

```text
mot.mixtures.video.*   dit.mixtures.video.*   video_expert.*
mot.mixtures.action.*  dit.mixtures.action.*  action_expert.*
video.*                action.*                dit.*
proprio_encoder.*
```

root 之后必须等于目标 module 的完整 local state key。不使用 `endswith`、模糊 suffix、层号猜测或跨 expert 搜索；prefix 规范化产生重复 exact key 时直接报错。

## 七类报告

JSON schema 为：

```text
schema  = fastwam.dexjoco_selective_checkpoint
version = 1
```

每个条目记录 `target`、`target_module`、`source`、`checkpoint_shape`、`model_shape`、`reason`，初始化条目还记录 strategy 和是否已执行。

| 分类 | 语义 |
| --- | --- |
| `loaded` | Video exact key 或旧 Action -> Arm exact key，shape 完全一致并加载。 |
| `copied_to_hand` | 旧 Action backbone local key 显式映射到 Hand 同名 local key。 |
| `skipped_shape` | 关键目标存在 source，但 shape 不一致；属于 fail-fast 条件。 |
| `skipped_policy` | 即使 shape 偶然相同也禁止加载的新 projection。 |
| `missing_in_checkpoint` | Video/Arm/Hand 关键 backbone 目标无 exact source；属于 fail-fast 条件。 |
| `unexpected_in_checkpoint` | source 无支持 root，或不被 policy 消费；保留为非致命审计项。 |
| `newly_initialized` | 通过明确 initialization strategy 新初始化的目标参数。 |

`skipped_policy` 和 `newly_initialized` 有意重叠：前者说明为什么旧 tensor 没有加载，后者说明目标参数如何产生。它们不是互斥的 state-dict 分区。

启动日志先打印七类汇总，再逐项打印参数名、目标模块、source key、checkpoint/model shape、reason 和 initialization。若给出 report path，同一信息以 JSON 保存。

## 加载和初始化契约

加载映射固定为：

```text
old mixtures.video.<local>  -> video.<local>   category=loaded
old mixtures.action.<local> -> action.<local>  category=loaded
old mixtures.action.<local> -> hand.<local>    category=copied_to_hand
```

Arm/Hand backbone 使用现有 `ActionDiT.backbone_key_set()`，因此排除 `action_encoder.*` 和 `head.*`。以下 10 个目标参数始终 `skipped_policy`，随后调用所属 `nn.Linear.reset_parameters()`：

```text
action.action_encoder.weight/bias
action.head.weight/bias
hand.action_encoder.weight/bias
hand.head.weight/bias
proprio_encoder.weight/bias
```

初始化策略是 PyTorch `nn.Linear.reset_parameters`：weight 使用该实现的 Kaiming-uniform 默认策略，bias 使用基于 fan-in 的 uniform 默认策略。没有旧 7D/1D projection 的截取、插值、padding、部分复制或偶然同 shape bias 复用。

loader 使用两阶段执行：先分类所有目标并收集待复制 tensor；只有在 `missing_in_checkpoint=0` 且 `skipped_shape=0` 后，才执行五个 `reset_parameters()` 和逐 tensor `copy_()`。失败报告中 `applied=false` 且 `initialization_applied=false`，防止返回部分加载模型。

## Synthetic Smoke Test

执行：

```bash
conda run -n fastwam python scripts/smoke_test_dexjoco_selective_checkpoint.py
```

synthetic checkpoint 使用 1 层真实 `WanVideoDiT` 和旧 7D `ActionDiT`，按仓库真实 `{"mot": ..., "proprio_encoder": ...}` 格式保存为临时 `.pt`，其中 expert key 带 `module.model.mot.*` 前缀。成功路径分类数量：

```text
loaded=79
copied_to_hand=37
skipped_shape=0
skipped_policy=10
missing_in_checkpoint=0
unexpected_in_checkpoint=1
newly_initialized=10
```

抽查结果：

```text
Video blocks.0.self_attn.q.weight == old Video source       PASS
Arm   blocks.0.ffn.0.weight       == old Action source      PASS
Hand  blocks.0.cross_attn.k.weight == old Action source     PASS
projection targets absent from loaded/copied_to_hand        PASS
Arm encoder/head shapes = (24,6)/(6,24)                     PASS
Hand encoder/head shapes = (24,16)/(16,24)                  PASS
JSON schema/version/summary round-trip                       PASS
```

故障路径删除一个 Video key，并把一个旧 Action self-attention weight 改成错误 shape。报告得到 `missing_in_checkpoint=1`、`skipped_shape=2`（同一旧 Action source 对 Arm/Hand 都不兼容），随后抛 `SelectiveCheckpointError`。调用前保存的 backbone 和 projection tensor 均未变化。

Phase 2 forward smoke 也重新通过：Arm `[1,4,6]`、Hand `[1,4,16]`、完整 action `[1,4,22]`、attention mask `[16,16]`。原 FastWAM/LIBERO Hydra compose 仍为 7D。

## 正式配置 Dry-run

正式配置的 meta target 构造通过：Video 825 keys、Arm 824、Hand 824、proprio 2，维度为 6/16/23。该目标只持有 key/shape，不分配 5B tensor storage，也不构造 VAE、scheduler、forward 或训练。

获得真实旧 FastWAM checkpoint 后执行：

```bash
conda run -n fastwam python scripts/audit_dexjoco_selective_checkpoint.py \
  --checkpoint /absolute/path/to/old_fastwam_checkpoint.pt \
  --report /absolute/path/to/dexjoco_selective_load_report.json
```

该命令读取真实 checkpoint tensor，在正式 config 的 meta target 上执行 `apply=false` 分类，关键缺失/shape 冲突仍返回非零，不开始训练。进入后续阶段前必须人工复核 JSON，并满足：

```text
missing_in_checkpoint = 0
skipped_shape = 0
```

本机实际搜索了仓库 `checkpoints/`、`outputs/`、`/home/user/.cache/huggingface/hub`、`/root/.cache/huggingface/hub` 和 `/home/shared/ai` 的候选路径，没有找到 FastWAM weight checkpoint，也没有找到配置引用的 `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`。因此真实 CPU dry-run 本阶段为 `NOT RUN`，没有下载或伪造权重。

## 修改文件

- `src/fastwam/models/wan22/dexjoco_checkpoint.py`
- `src/fastwam/models/wan22/dexjoco_dual_action.py`
- `src/fastwam/runtime.py`
- `configs/model/dexjoco_dual_action.yaml`
- `scripts/audit_dexjoco_selective_checkpoint.py`
- `scripts/smoke_test_dexjoco_selective_checkpoint.py`
- `docs/dexjoco_22d_porting/README.md`
- `docs/dexjoco_22d_porting/03_selective_checkpoint_loading.md`
- `docs/dexjoco_22d_porting/decisions.md`
- `docs/dexjoco_22d_porting/smoke_test_ledger.md`
- `docs/dexjoco_22d_porting/commit_index.md`

## 执行命令和未运行项

除上面的 smoke 和未来真实 audit 命令外，本阶段实际执行：

```bash
conda run -n fastwam python -m compileall -q \
  src/fastwam/models/wan22/dexjoco_checkpoint.py \
  src/fastwam/models/wan22/dexjoco_dual_action.py \
  src/fastwam/runtime.py \
  scripts/audit_dexjoco_selective_checkpoint.py \
  scripts/smoke_test_dexjoco_selective_checkpoint.py

conda run -n fastwam python scripts/smoke_test_dexjoco_dual_action_model.py
conda run -n fastwam python scripts/audit_dexjoco_selective_checkpoint.py --help
```

另用 Hydra compose 断言 DexJoCo selective path/report、6/16/23D 和 runtime target，并 compose 原 `model=fastwam data=libero_2cam` 断言 7D 兼容。`git diff --check` 通过。环境没有 `ruff`，相应命令返回 127 并记为 `NOT RUN`。未运行真实 checkpoint audit、完整 pytest、optimizer、训练、推理或 simulator。

## 已知风险和后续前置条件

- synthetic state dict 能证明分类、内容复制和 fail-fast 实现，但不能证明未知真实 checkpoint 的 wrapper/root/key/shape 全部符合预期。
- 正式 meta target 证明目标 key/shape 可构建，不证明完整 5B 权重能在目标训练硬件上加载或运行。
- JSON 中的 `unexpected_in_checkpoint` 非致命，但真实使用前必须逐项人工确认，不能只看汇总数量。
- 本阶段没有改变通用 `FastWAM.load_checkpoint(strict=False)`；DexJoCo pretrained 初始化必须使用新增 selective API，后续如需 DexJoCo resume/save 需要另行设计。
- 进入下一阶段前必须取得真实 checkpoint 并通过 dry-run 审计，或明确接受缺少真实审计的风险；还必须获得下一阶段授权。
