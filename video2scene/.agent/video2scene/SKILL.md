---
name: video2scene
description: 在本仓库把真实视频重建为可编辑的 Blender 初始场景、Genesis 纯仿真场景和可续跑的域随机化数据；用于新增场景、迁移或修复物理、校准渲染、扩展求解能力及批量生产，不用于源视频合成、逐帧轨迹回放或只改 JSON/颜色的伪随机化。
---

# Video2Scene

在包含 `pyproject.toml`、`video2scene/`、`scenes/` 和 `schemas/` 的仓库根目录工作。把原始视频当作**观测证据**，把 Blender 当作可编辑的 `t=0` 场景与资产交付层，把 Genesis 当作 `t>0` 的唯一运动来源，最后才扩展到并行域随机化数据。

本 skill 负责驱动仓库代码继续演进，而不是绕开代码临时写一次性脚本。当前 `ball-and-block-fall` 是经过验证的参考实现，不是可直接套用到所有视频的通用模板；新增场景时必须审计并消除相关的场景名、实体名、事件和阈值硬编码。

## 不可破坏的约束

1. **只对齐初态**：从源视频估计参考帧、实体 6D 位姿、线速度、角速度、相机和尺度；预松弛后必须显式恢复动态体的 `t=0` 状态。
2. **后续纯仿真**：`t>0` 禁止读取源帧或观测轨迹，禁止关键帧回放、逐帧位姿覆盖、按未来视频时刻触发修正力、末帧纠偏，以及把源视频用作背景、纹理序列或合成层。
3. **保留物理类别**：软体、布料、流体、关节机构等不得为了跑通而替换成刚体盒或视觉动画。仓库尚未实现的真实机制应明确报 `unsupported`，不要伪造结果。
4. **资产与碰撞分离**：可见几何负责外观，物理几何负责接触；两者可不同，但必须具有同一语义所有权、局部坐标和一次性世界变换，不能重复实例化碰撞参考副本。
5. **随机化必须进入执行器**：每个声明范围都要映射到 Blender 构建、Genesis batch 构建或逐环境 setter。不能只改配置文件，也不能把 Genesis 共享参数伪称为逐环境变化。
6. **证据优先于状态声明**：静帧证明可渲染，短 smoke 证明代码路径，完整预览证明候选行为，只有产物、日志、验证指标和人工图像审查共同通过后，才可声称阶段完成。
7. **状态分级**：`passed`、`physics_validated`、`training_ready` 含义不同。单目先验、未收敛网格或未完成批量验收时保持 `training_ready=false`。
8. **保持仓库约定**：先读 `.agent/rules.md`；CLI 使用 dataclass + `tyro`，日志使用 `loguru`，Python 命令使用 `uv run python`，修改后运行 Ruff 与相关测试；不保留已被替代的 legacy/fallback 路径。

## 开始任务

1. 读取 `.agent/rules.md`、`README.md`、目标 `scenes/<scene-id>.json`、相关测试和即将修改的模块。
2. 运行当前契约检查器，避免依据旧对话猜测接口：

```bash
uv run python .agent/video2scene/scripts/inspect_repo.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id>
```

3. 运行自进化审计，发现代码接口漂移、已有失败、损坏 batch marker、skill 断链和场景硬编码风险：

```bash
uv run python .agent/video2scene/scripts/evolution_audit.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id> \
  --target-phase <phase1|phase2|phase3> \
  --report /tmp/<scene-id>-evolution-start.json
```

4. 明确用户要求停在哪一阶段：
   - Phase 1：视频证据 → Blender 初态与 handoff；
   - Phase 2：Blender handoff → Genesis 单场景纯仿真；
   - Phase 3：参数范围 → 并行、可续跑数据集；
   - 修复/重构：定位到对应阶段，不无关地重跑全部流水线。
5. 需要详细步骤时读取 [references/phase-playbook.md](references/phase-playbook.md)；修改代码前读取 [references/repository-contract.md](references/repository-contract.md)；验收前读取 [references/validation.md](references/validation.md)；发现漏洞或契约漂移时读取 [references/self-evolution.md](references/self-evolution.md)。`docs/s*.md` 是原始长对话，只在追查具体决策、错误或实验依据时按关键词局部检索。
6. 检查工作树并保护已有结果。除非用户明确要求，不覆盖已审查的正式输出；实验写入 `/tmp` 或 `outputs/debugs/`，可续跑生产使用 `--resume`。

## 核心迭代循环

按“证据 → 契约 → 构建 → 预览 → 验证 → 归纳”的短闭环推进，不一次性猜完整系统。

### A. 证据与物理命题

- 使用 `ffprobe` 和 `video2scene.evidence` 获取元数据、密集早期帧及全片采样；实际查看首帧、接触前后和末帧。
- 用一句可判定的话写出物理命题，再记录实体、关系、事件窗口、尺度锚点、相机、初始状态和不确定性。
- 区分：直接观测、由多帧估计、工程先验、不可辨识量。后续帧只用于验证事件与结果，不能成为运行时控制输入。

### B. Phase 1：可编辑 Blender 初态

- 在 `scenes/<scene-id>.json` 建立完整场景契约；同步扩展 schema、几何/材质生成器、资产导出和验证，不在构建器里散落匿名常量。
- 优先使用真实轮廓、倒角、厚度、缝线、粗糙度、尺度一致的资产。非刚体交付闭合静止表面、必要的顶点集合和材料先验；复杂机构需要真实关节/执行器表达。
- 首次场景必须完整渲染并查看 `comparison.jpg`、`preview.png`、`overview.png`。`--handoff-only` 仅用于已审查场景的批量变体或快速集成测试。
- 保存 `.blend` 后用独立进程重新打开并验证；OBJ 只代表几何，不得声称已携带 Blender 程序材质。

### C. Phase 2：Genesis 迁移与纯仿真

- 由 `scene_manifest.json` 编译显式 solver plan。每种物理类型都必须有真实实现；不支持时让 `MigrationError` 失败，而不是降级。
- 先做短时无渲染 smoke，检查构建、预松弛、初态恢复、状态有限性和接触；再做单环境完整渲染。
- 调参必须对应求解器物理量，并通过对照实验选择。不得用放大碰撞体、抬高物体、轨迹控制或末态改写掩盖穿透和动力学错误。
- 光照/色彩迁移与物理解耦。可以用 Genesis 支持的灯光近似 Blender，但要用静帧或 ROI 对比确认，不得做源视频像素合成。

### D. Phase 3：并行域随机化

- 先运行 `parameter_execution()` 审计每个参数的执行 lane。刚体几何、质量、初态可在实现允许时逐环境变化；FEM 网格/材料、共享表面、相机和灯光通常按 batch 变化。
- `sampled` 模式必须真正构建对应 Blender handoff；`reuse` 只适合吞吐或链路测试，不能声称覆盖所有几何/材质范围。
- 先生成小规模可视化预览并检查逐样本验证，再基准测试合理 `num_envs`，最后启动 `--resume` 生产。保留无效样本及原因，不以文件数量代替物理有效率。
- `_SUCCESS` 必须与 run identity、sample IDs 和 manifest 哈希一致；已有完整 batch 不重复生成，未完成 batch 可原子重建。

## 代码与 skill 共同演进

当任务暴露出可复用的新能力或约束时，先在仓库中形成可执行实现与回归测试，再更新本 skill 的对应契约：

- 新几何或实体类型：更新 schema、`geometry.py`/`build.py`、资产导出、Genesis morph、验证和测试。
- 新物理类型或机构：更新 `genesis_plan.py` 的显式路由、`simulation_runtime.py` 的真实 solver/material/coupler 实现、状态记录和物理检查；不增加静默 fallback。
- 新语义事件：将事件/关系写入 spec；名义场景通过 `physics_validation.checks` 选择显式验证器（当前为 `soft_impact`、`rolling_transition`），让检查消费实体/支撑/事件引用而不是固定名称。扩展新类型时同步实现 batch 路径；不要继续增加 `ball`、`block` 等实体名硬编码。
- 新随机参数：同时更新 spec/schema、采样约束、`parameter_execution()`、运行时应用、参数落盘与覆盖测试。
- 新目录或产物：只通过 `layout.py` 和 artifact 模块修改，更新全部生产者/消费者、测试、README 与本 skill 的 repository contract。
- 仅对单场景有效的数值留在 scene spec 或场景专属 validator；只有被实现、测试并证明可复用的经验才写入 skill。

如果代码接口已变化，本 skill 与 references 必须在同一任务中更新；如果只是场景参数调优，不要把一次性阈值堆进 skill。

## 自进化闭环

每次实际使用都把可复现失败当作改进信号，但禁止无证据自我改写：

1. 保存失败命令、seed、日志、trace、验证项和可视产物；
2. 用最小真实路径复现并定位责任模块；
3. 先修代码并添加行为回归测试，再验证 Blender/Genesis 实际路径；
4. 只有可复用、已验证的规则才写回本 skill 或 references；单场景参数留在 spec/validator；
5. 接口或 schema 有意变化后，审查 skill 的全部依赖并刷新结构基线；
6. 交付前重新运行审计，阻断项未清零时不得声称完成。

结构基线只能在代码测试、相关真实路径和 skill 审查完成后更新：

```bash
uv run python .agent/video2scene/scripts/evolution_audit.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id> \
  --refresh-baseline --reviewed
```

审计器只发现问题、生成报告和更新经确认的结构基线，不自动修改生产代码、场景、输出或成功标记。具体闭环、升级标准和安全边界见 [references/self-evolution.md](references/self-evolution.md)。

## 完成与汇报

- 报告实际执行的命令、修改文件、产物路径、检查数量/失败项、视频编码信息、物理指标、随机化有效率和仍未验证的内容。
- 明确区分“代码通过”“任务已启动”“预览可接受”“物理已验证”“训练可用”。
- 用户说“继续”时，先检查现有日志、markers、manifest 和产物哈希，从最近未完成的门槛继续；不要无理由从头重建。
- 最小代码验收通常包括：

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
uv run python /home/yongce/.codex/skills/.system/skill-creator/scripts/quick_validate.py \
  .agent/video2scene
uv run python .agent/video2scene/scripts/evolution_audit.py \
  --spec scenes/<scene-id>.json \
  --output-root outputs/<scene-id> \
  --fail-on-findings
```

真实场景仍需按请求阶段完成相应渲染/仿真验收；单元测试和 skill validator 不能代替它。
