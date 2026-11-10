# Quadruped Morphology Workbench

轻量的四足机器人构型配置器：基于 MuJoCo 生成参数化模型，使用逆运动学探索姿态，并显示载荷、质量、重心及其地面投影。

[English README](README.md) · [英文详细操作](README_EN.md) · [代码来源与参考项目](SOURCES.md)

![四足与后肢直立姿态](docs/images/pose-preview.png)

## 开发基础与来源

当前代码是在本项目中独立实现的，使用 **MuJoCo** 作为模型编译、运动学和渲染基础，使用 NumPy、Pillow 和 Python 自带的 Tkinter。生成器、参数窗口、解析 IK、质量惯量校核和测试在项目中逐步开发，编码过程使用了 OpenAI Codex 辅助。

最初的研究参考是 [Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design](https://github.com/real-stanford/transformer-transformer)。**当前版本没有复用该项目的源码、机器人资产、数据集、模型或训练流程，也没有集成它的生成器。** 因此应将它标为研究参考，而不是声称本仓库由其代码改造而来。详细来源见 [SOURCES.md](SOURCES.md)。

## 安装和启动

当前已在 Apple Silicon Mac 上验证；使用 Python 3.12。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，在终端运行：

```sh
git clone https://github.com/peiyuwang1018/quadruped-morphology-workbench.git
cd quadruped-morphology-workbench
uv sync --locked
.venv/bin/python -m quadruped_codesign.cli panel --config configs/fusion_reference.toml
```

环境创建后，也可在 Finder 双击 `Open Workbench.command`。窗口需要支持 Tk 的 Python 和桌面图形会话。项目内部模块名仍是 `quadruped_codesign`，与 GitHub 仓库名不同不会影响使用。

## 一起做一个小实验

1. 在 **Stance** 选择 **Hind-leg upright**，观察后肢直立姿态。
2. 金色点是重心，粉色点是地面投影，青色线是后足支撑线段。点击 **Align CoM projection**，比较对齐前后的线段偏差。
3. 切回 **Four-foot stance**，此前的四足参数会保留；点击 **Save config & model** 可保存独立候选。

**Geometry** 调整前后腿长和安装间距；**Stance** 调整姿态与足端目标；**Mass** 调整声明的质量估计；**Payload** 调整固定载荷。Fusion 界面使用 mm、deg、kg，配置文件使用 SI 单位，重心读数为世界坐标。无效参数会保留上次有效预览并禁止保存。

## 输出与验证

每个候选保存到 `outputs/panel/时间戳/`，包括完整配置、MJCF、检查报告、当前渲染图和范围说明。输出与虚拟环境不进入 Git。

```sh
.venv/bin/python -m quadruped_codesign.cli generate --config configs/fusion_reference.toml --output outputs/robot.xml
.venv/bin/python -m quadruped_codesign.cli check outputs/robot.xml
.venv/bin/python -m unittest discover -s tests -v
# 在有桌面的环境运行完整窗口/渲染检查：
QUADRUPED_UI_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

macOS 独立 MuJoCo 查看器需要用 `.venv/bin/mjpython -m quadruped_codesign.cli view outputs/robot.xml`；参数窗口使用普通 Python。当前完整桌面测试共 41 项。早期八关节教学配置和 Fusion v1 快照仍可重载。

## 当前边界

这是构型与冻结姿态探索工具。十二关节模型尚无 actuator；传动、转子惯量、力矩/速度限制、主动站立控制、步态与 RL 尚未实现。Isaac Lab 迁移未经验证。

部分安装距离、框架质量、连杆密度和关节范围仍为明确声明的估计。示例总质量约 30.53 kg，不能作为实机辨识值。直立使用单独暂定的髋范围约 −77° 至 +160°，未按硬件校准。

四足投影位于支撑多边形内是几何筛查；两只理想点足的线段没有面积，即使投影对齐也不代表能稳定站立或电机能承载。详细说明见 [构型假设](docs/fusion-reference.md) 与 [站姿和支撑几何](docs/support-preview.md)。

尚未选定项目开源许可证；依赖与参考项目保留各自许可。
