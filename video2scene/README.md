# Video → Blender → Genesis（Phase 1 + Phase 2 + Phase 3）

从视频证据出发，由 Agent 进行视觉与三维推理，生成可编辑、可重复构建、保留物理语义的 Blender 初始场景。目前完成 `inputs/physics_iq_videos/ball-and-block-fall.mp4` 的 Blender 重建、Genesis 物理迁移，以及基于 Blender 参数先验的并行域随机化数据生产。Phase 2/3 只继承 `t=0` 场景、材质先验和初速度，之后的运动完全由 Genesis 求解，不读取源视频、不回放轨迹。

## 新场景：ball-rolls-on-glass

`scenes/ball-rolls-on-glass.json` 重建蓝色小球从木质台面滚上透明玻璃板的事件。首帧中心投影误差约 2.57 px；`t=0` 之后不再读取源视频，Genesis 只从规格中的初始线速度与滚动角速度继续求解。

```bash
uv run python -m video2scene.build \
  --spec scenes/ball-rolls-on-glass.json \
  --output-root outputs/ball-rolls-on-glass

UV_CACHE_DIR=/tmp/skill2scene-uv-cache \
MPLCONFIGDIR=/tmp/skill2scene-mpl \
XDG_CACHE_HOME=/tmp/skill2scene-xdg-cache \
uv run python -m video2scene.simulate \
  --output-root outputs/ball-rolls-on-glass \
  --duration-s 0.5 --simulation-hz 1920 \
  --ambient-light 0.62 --key-light-scale 0.22 --fill-light-scale 0.28 \
  --render-shadows --surface-color-space linear --overwrite
```

该场景使用声明式 `physics_validation.checks` 中的 `rolling_transition`：检查木板到玻璃的进入时序、连续支撑、正向位移、横向漂移、竖直波动、滚动滑移率及速度界限。Blender 使用程序玻璃材质；Genesis Rasterizer 使用中性半透明灰近似外观，但碰撞仍是独立、共面的刚体玻璃板。

Phase 3 已完成 4 条可视化预览和 64 条正式数据：

- 预览：`outputs/ball-rolls-on-glass-phase3-preview/batches/preview-grid.jpg`；
- 正式数据：`outputs/ball-rolls-on-glass/batches/`，8 批 × 8 环境；
- 总结：`outputs/ball-rolls-on-glass/phase3-report.json`；
- 64/64 样本通过逐环境物理检查，且 `--resume` 已验证会跳过全部完整批次；
- `physics_validated=true`，但由于随机范围仍来自单目工程先验，保持 `training_ready=false`。

## Phase 1：Video → Blender

```bash
uv run python -m video2scene.build \
  --output-root outputs/ball-and-block-fall
```

依赖当前 uv 环境的 Blender `bpy`、NumPy、SciPy、Pillow、Trimesh、JSON Schema，以及系统 `ffmpeg` / `ffprobe`。构建在无 GUI 环境实际执行 Cycles CPU 渲染，默认 64 samples、12 线程；保存后启动独立 Python 进程重新打开 `.blend` 校验。生成过程可重复，重跑会更新指定输出目录。

## 交付内容

本次已审查的名义构建目录：`outputs/ball-and-block-fall/blender/`。

| 产物 | 内容 |
| --- | --- |
| `scene.blend` | 独立可打开的完整场景，默认参考相机；包含程序材质与嵌入的 JSON |
| `preview.png` | 640 × 360 的首帧重建渲染 |
| `comparison.jpg` | 源视频首帧与实际 Blender 渲染对照 |
| `overview.png` | 960 × 540 的斜侧结构视图 |
| `scene_spec.json` | 本次构建的具体配置，随机变体同时记录 seed 和采样值 |
| `scene_manifest.json` | 场景语义、证据、全部参数范围、物理类型、质量状态、相机内外参、资产与变换映射 |
| `validation.json` | 重新打开保存文件后的结构校验、网格体积、初始间隙和投影一致性 |
| `render_review.json` | Agent 的实际图像审查、已知差异与被审查图片的 SHA256；重建会重置为待审查 |
| `scene.schema.json` | 随产物携带的 JSON Schema，可离线解析本目录的场景 JSON |
| `../frames/` | 原始时间戳、视频 SHA256、密集早期帧、全片采样、证据联系表 |
| `assets/visual/*.obj` | 每个语义实体的可见几何；子对象已合并到所属实体的局部坐标 |
| `assets/physics/*.obj` | 刚体/静态物体的几何及软垫闭合静止表面，不含装饰缝线 |
| `assets/physics/*_rest.npz` | 软垫三角化表面、局部顶点、底部候选点、上表面和缝边分组 |

`.blend` 集合为 `DynamicObjects`、`DeformableObjects`、`Environment`、`Fixtures`、`VisualDetails`、`CollisionGeometry`、`CamerasLights`。碰撞集合中的对象是隐藏的参考副本，标注 `is_duplicate_reference_only`，转换时不能重复实例化。软垫参考副本保留原曲面，不是静态盒体。

OBJ 只交付几何。完整程序 BSDF 在 `.blend` 内；JSON 保留材质参数，但不能声称 OBJ 已包含 Blender 的织物、皮革和光照效果。后续如需跨渲染器复用外观，应烘焙 PBR 通道并重新检查色彩响应。

## 场景理解与真实性边界

源视频为 640 × 360、30 fps、150 帧、5 秒。首帧已经释放：棕色缝线球落向左垫，橙色细长方块落向右垫；软垫下陷，方块侧倒。主要事件发生在前约 0.3 秒。

- 球、方块、软垫、上方夹具和横杆是独立的三维资产；没有使用源视频背景贴图或合成。
- 软垫为有正体积的闭合网格，具有鼓起轮廓、局部褶皱、缝边与织物程序材质；物理类型为 `soft`。
- 球体带独立缝线与细微表面纹理，方块有真实倒角，夹具由曲柄、夹爪与金属连接件组成。
- 初始状态对应源帧 0；只有静态初始场景，未写入未来运动关键帧。初速度为低置信估计，供后续仿真初始化参考。
- 球半径假设为 0.040 m，软垫约 0.556 × 0.769 × 0.176 m。单目绝对尺度、隐藏支撑、真实材料和初始应力均未测量。
- 参考中心 `(180, 100)` 与 `(510, 148)` 为人工标注；投影误差只检查参数执行一致性，不能作为独立三维精度指标。
- 静态渲染没有复现源视频的运动模糊；夹具细节、布面褶皱和背景阴影仍是近似。

当前 Agent 已读取关键帧并把判断固化在 `scenes/ball-and-block-fall.json`。CLI 重建这份分析结果；没有配置远程视觉模型，也不是仅替换视频路径就能自动处理任意新视频的通用模型。新视频流程为抽帧 → Agent 分析并写规格 → 必要时增加几何生成器 → 渲染审查 → 保存迁移资产。

## JSON 与参数化

`scenes/ball-and-block-fall.json` 是可编辑的输入规格，`schemas/scene.schema.json` 提供版本化结构校验。输出 `scene_manifest.json` 继承完整规格，并增加实际导出的实体、质量检查和环境版本信息。

| 字段 | 内容 |
| --- | --- |
| `source` / `evidence` | 视频哈希、帧号、标注、观察与不确定性 |
| `semantics` | 场景在做什么、事件区间、物体关系、非刚体清单 |
| `coordinates` | 米/千克/秒、右手 Z-up、相机约定、尺度假设 |
| `entities` | 几何、位姿、外观、物理类型、参数、初始速度与推断来源 |
| `physics.deformable` | 体积/表面表示、布面绑定、支持边界、离散化要求、Genesis 候选映射 |
| `domain_randomization` | 48 个 JSON-pointer 参数、分布、范围、单位、确定性 seed 与约束 |
| `resolved_entities` | 实际局部资产、世界变换、体积、密度推导的质量与顶点分组 |
| `resolved_camera` | 内参 K、Blender 相机到世界矩阵、Blender/CV 坐标转换 |
| `quality` / `genesis_handoff` | 已验证内容、未标定内容及后续阶段必需的验证 |

参数区间是探索先验，不是从视频得到的统计置信区间。48 个参数覆盖球/方块的尺寸、密度、摩擦、回弹、初始高度和速度，软垫的尺寸、密度、弹性模量、泊松比、阻尼、褶皱和颜色，以及相机与照明。所有实体的其他几何、位置和材质也可直接编辑；本构建器尚未实现任意外部资产类别替换。

只生成确定性的配置，不渲染、不仿真：

```bash
uv run python -m video2scene.spec --output outputs/config_samples --seed 0 --count 100
```

实际构建并渲染一个变体：

```bash
uv run python -m video2scene.build --seed 7 --samples 32
```

默认写入 `outputs/ball-and-block-fall/blender/variants/seed_000007/`。采样器检查正尺寸、有效材料范围、初始离地间隙、目标软垫范围和软垫不重叠；厚度变化后重算软垫中心高度，导出时按密度 × 实际体积重算质量。随机种子与具体取值落盘。随机相机及位置变体不要求匹配源图像坐标，但仍必须通过结构校验。

重建、仿真和数据集命令都以同一个场景根目录为边界，禁止再写入 `reference/`、`main/`、`dataset/` 或批次内的 `genesis/` 兼容目录：

```text
outputs/<scene-id>/
├── frames/                         # Source evidence
├── blender/
│   ├── assets/{physics,visual}/
│   ├── build.log
│   └── variants/seed_XXXXXX/
├── sim/                            # Nominal Genesis run, including run.log
└── batches/
    ├── generate.log
    ├── dataset_manifest.json
    └── batch_XXXXXX/sim/           # Parallel batch artifacts
```

数据依赖固定为 `inputs/*.mp4 → frames/ → blender/ → sim/`；批量路径为
`blender/variants/seed_XXXXXX → batches/batch_XXXXXX/sim/`。各阶段只接收
`--output-root`，内部路径由 `SceneLayout` 统一解析，避免调用方自行拼接目录。

## Genesis 迁移准备

本机安装版本为 Genesis 1.4.0。已检查本机源码的 `MPM.Elastic` / `PBD.Elastic` 字段；本阶段没有初始化求解器或执行 `scene.step()`。

- **刚体**：球保留解析球形碰撞；方块保留盒形尺寸与带倒角的可见网格。变换只应用一次，质量来自假设密度与实际几何体积。初速度来自 JSON，不得默认清零。
- **软垫**：导入闭合局部表面，采用体积软体材料。`MPM.Elastic` 候选映射为 `E ← young_modulus_pa`、`nu ← poisson_ratio`、`rho ← density_kg_m3`。该选择仍需接触和材料校准，不能把软垫冻结为刚体后宣称迁移成功。
- **PBD 备选**：记录密度、摩擦和待标定的柔度参数。不能把 `1/E` 直接当作与分辨率无关的 PBD 柔度。
- **离散化与支撑**：提供静止表面及候选支持点，默认与下方支撑接触，不自动固定底部。若改用固定底层，必须作为边界假设记录。FEM 需要另行生成有质量保证的四面体网格；本阶段没有伪造四面体文件。
- **布套**：当前布面是体积软垫的视觉表皮，缝边需绑定到模拟表面。若需要独立布套的拉伸/滑移，应新增薄壳模型与耦合，不能让缝边作为独立刚体掉落。
- **初始应力**：视频中的垫子已经受重力支撑，重建形状不是测量得到的无应力形状；后续需先进行重力松弛，再恢复刚体至源帧 0 状态。
- **流体扩展**：本视频没有流体。契约预留 `cloth` / `fluid` 和薄壳/流体体积碰撞语义；新场景仍需具体生成器、容器/发射器/边界等配置及验证，当前不会假装已经支持流体重建。

材料与网格依据：[Genesis soft solvers](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/soft_solvers.html)、[mesh processing](https://genesis-world.readthedocs.io/en/latest/user_guide/assets/mesh_processing.html)。版本变化后需重新核对实际接口。

Phase-1 的 `blender/validation.json` 保持 `physics_validated=false`、`training_ready=false`；它只证明几何与交付契约成立。Phase-2 的动力学结论单独记录在 `sim/validation.json`。

## 检查与模块

```bash
uv run python -m video2scene.evidence inputs/physics_iq_videos/ball-and-block-fall.mp4 \
  --output-root outputs/ball-and-block-fall
uv run python -m video2scene.validate outputs/ball-and-block-fall/blender
uv run python -m unittest discover -s tests -v
```

`layout.py` 统一场景根目录和各阶段路径，`filesystem.py`、`hashing.py`、`logging_utils.py` 提供共享基础设施；`evidence.py` 负责证据；`spec.py` 负责契约与单配置采样；`domain_randomization.py` 负责批级/环境级参数分层；`geometry.py` / `materials.py` 负责几何及外观；`assets.py` 导出局部资产；`build.py` 构建与渲染或生成轻量 handoff；`simulation_config.py`、`simulation_runtime.py`、`simulation_artifacts.py` 提供单场景和批量仿真共享能力；`simulate.py` 负责 Phase 2 编排与验证；`dataset_simulation.py` 负责单批并行执行；`generate_dataset.py` 负责可续跑批次编排；`validate.py` 重新打开保存文件核验。

验证覆盖闭合网格、正体积、法线一致性、有限变换、质量/密度一致性、软垫顶点分组与 OBJ/NPZ 对应、初始间隙、相机投影、无源视频贴图及无运动关键帧。全局三角形自交、真实尺度和材料动力学并未通过这些结构检查得到证明。


## Phase 2：Blender → Genesis

运行已经审查过的 Phase-1 输出：

```bash
UV_CACHE_DIR=/tmp/skill2scene-uv-cache \
MPLCONFIGDIR=/tmp/skill2scene-mpl \
XDG_CACHE_HOME=/tmp/skill2scene-xdg-cache \
uv run python -m video2scene.simulate --overwrite
```

默认读取 `outputs/ball-and-block-fall/blender/scene_manifest.json`，写入 `outputs/ball-and-block-fall/sim/`。主要产物：

| 产物 | 内容 |
| --- | --- |
| `migration_plan.json` | 每个实体的 solver/material/collision 选择、源资产哈希和不可降级规则 |
| `preview.png` | 重力短预松弛后、刚体恢复到源帧 0 的 Genesis 初始帧 |
| `comparison.jpg` | 源帧 0 / Blender / Genesis 三联对照 |
| `simulation.mp4` | `t=0` 后纯 Genesis 求解的视频 |
| `contact_sheet.jpg` | 接触与稳定阶段的抽帧 |
| `trace.json` / `trace.npz` | 刚体位姿速度、软体边界、局部压陷与接触间隙轨迹 |
| `validation.json` | 初态、事件时间、双向软硬接触、穿透、侧倒、留存和稳定性检查 |
| `render_review.json` | 本次渲染使用的光照近似、自动验证状态与待用户审查的产物哈希 |
| `run_manifest.json` | Genesis 版本、运行参数、耗时、离散化规模、视频信息和产物哈希 |

### 求解器选择

- `ball` / `block`：Genesis `RigidSolver`，碰撞分别使用 Phase-1 声明的球体和盒体；Blender 导出的高细节 OBJ 作为无碰撞视觉跟随体。
- `cushion_left` / `cushion_right`：Genesis `FEMSolver + FEM.Elastic(model="linear")`。直接使用 Phase-1 的闭合软体 OBJ，保留全部 11,362 个表面顶点，不用刚体盒替代；TetGen 仅生成内部四面体。
- `support` / `wall`：按 Phase-1 碰撞尺寸建立隐藏静态碰撞体，同时使用 Blender OBJ 渲染。
- `rail` / `clamp_*`：Phase 1 已标记为脱离接触路径，仅作为视觉实体。
- 默认采用 Genesis 1.4.0 的 `LegacyCoupler` 做双向 rigid–FEM 耦合。实测 SAP 模式可运行，但该场景中会抑制细长方块的接触角响应；可用 `--coupler sap` 做诊断对照。
- Genesis Rasterizer 不支持 Blender Area Light 的一一等价迁移，因此默认保留 Phase-1 的灯位和颜色，并使用实测标定后的 Point Light 近似：环境光 `0.85`、主光功率倍率 `0.35`、补光功率倍率 `0.45`。这些参数只改变渲染，不参与物理求解。

软垫先进行 0.0125 秒重力 seating，球和方块暂时停放在接触区外；随后软垫速度清零，球和方块严格恢复 manifest 中的 `t=0` 位姿和速度。默认内部步频为 1920 Hz，输出 30 fps。Legacy 刚体–FEM 接触作用距离默认为 15 mm；该值由无渲染参数对照选定，用于减少细长方块停止后嵌入顶层 FEM 表面的现象。球的接触带仍按半径设置。这里调整的是求解器接触离散化参数，不是碰撞体放大、轨迹控制或末帧位姿修正。

### 当前名义场景验证

当前 `outputs/ball-and-block-fall/sim/validation.json` 的物理检查通过，包括：

- 两个刚体的初始位置和姿态误差为零；
- 两块软垫预松弛后的表面 P95 偏差小于 1 mm；
- 方块接触约 0.067 秒、球接触约 0.100 秒，均落入源视频事件窗口；
- 两块软垫均产生真实 FEM 局部压陷，球和方块均有双向动力学响应；
- 方块最终侧倒，球和方块均留在各自软垫上且未穿透支撑；方块末态还需通过基于接触带宽度的过度嵌入检查；
- 全部状态有限，1 秒视频为 640 × 360、30 fps、31 帧。

`physics_validated=true` 表示当前预览在初态、事件、接触和结果层面通过。少量局部四面体翻转仅作为诊断记录，不否定用户已接受的成片；`training_ready=false` 仍然保留，因为材料参数来自单目工程先验，且尚未完成严格网格收敛与批量验收。

可运行无渲染批处理烟测：

```bash
uv run python -m video2scene.build \
  --output-root /tmp/skill2scene-batch-smoke \
  --handoff-only --samples 1 --threads 1

UV_CACHE_DIR=/tmp/skill2scene-uv-cache \
MPLCONFIGDIR=/tmp/skill2scene-mpl \
XDG_CACHE_HOME=/tmp/skill2scene-xdg-cache \
uv run python -m video2scene.simulate \
  --num-envs 2 --duration-s 0.1 --no-render \
  --output-root /tmp/skill2scene-batch-smoke --overwrite
```


## Phase 3：域随机化与并行数据生产

Phase 3 使用 Phase 1 声明的 48 个参数区间。Genesis 单个 batched scene 内逐环境随机刚体几何、质量、初始位姿和速度；FEM 网格/材料、耦合材料、外观、相机和灯光按 batch 随机。这样不会把 Genesis 实际共享的参数伪装成逐环境变化，同时仍能利用 GPU 并行推进大量环境。

先做小规模可视化预览：

```bash
UV_CACHE_DIR=/tmp/skill2scene-uv-cache \
MPLCONFIGDIR=/tmp/skill2scene-mpl \
XDG_CACHE_HOME=/tmp/skill2scene-xdg-cache \
uv run python -m video2scene.generate_dataset \
  --samples 4 --num-envs 4 --duration-s 1.0 \
  --render-preview \
  --output-root outputs/ball-and-block-fall-preview \
  --overwrite
```

确认预览和有效率后，再进行可续跑生产：

```bash
UV_CACHE_DIR=/tmp/skill2scene-uv-cache \
MPLCONFIGDIR=/tmp/skill2scene-mpl \
XDG_CACHE_HOME=/tmp/skill2scene-xdg-cache \
uv run python -m video2scene.generate_dataset \
  --asset-mode sampled \
  --samples 1024 --num-envs 16 --seed 20260908 \
  --duration-s 1.0 \
  --output-root outputs/ball-and-block-fall \
  --resume
```

默认 `sampled` 模式将每批 Blender handoff 写入 `blender/variants/seed_XXXXXX/`，并将对应 Genesis 结果写入 `batches/batch_XXXXXX/sim/`，确保软垫尺寸、褶皱及材质来自该批实际采样，而不是只改 JSON。`reuse` 模式只复用 `blender/` 名义场景，适合快速测试 Genesis 并行吞吐，不代表完整的 48 参数覆盖。批次输出包含逐环境参数、时间序列、物理验证、吞吐和哈希；详见 [`docs/s2.md`](docs/s2.md)。
