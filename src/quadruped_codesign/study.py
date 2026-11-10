"""A reproducible four-candidate experiment, not an optimization or standing task."""

import json
from pathlib import Path

import mujoco
import numpy as np

from .generator import Payload, write_robot
from .mass_properties import inspect_mass
from .validation import evaluate_robot
from .visualization import preview_data


def run_study(directory: Path, *, render: bool = False) -> Path:
    variants = {
        "baseline": {},
        "front_longer": {"front_thigh_length": 0.26},
        "payload_center": {"payload": Payload(mass=2)},
        "payload_front": {"payload": Payload(mass=2, position=(0.12, 0, 0.12))},
    }
    paths = {name: write_robot(directory / f"{name}.xml", **parameters) for name, parameters in variants.items()}
    models = {name: mujoco.MjModel.from_xml_path(str(path.resolve())) for name, path in paths.items()}
    # Equal base position, base orientation and all eight joint angles in every comparison.
    # The shared height clears the longest leg; no support forces are inferred from this pose.
    reference_qpos = preview_data(models["front_longer"]).qpos.copy()
    reports = {}
    for name, model in models.items():
        report = evaluate_robot(model)
        data = mujoco.MjData(model)
        data.qpos[:] = reference_qpos
        mujoco.mj_forward(model, data)
        report["comparison_pose_mass"] = inspect_mass(model, data)
        report["comparison_pose_feet_torso_frame_m"] = {
            leg: (data.site(f"{leg}_foot").xpos - data.body("torso").xpos).tolist()
            for leg in ("FL", "FR", "RL", "RR")
        }
        reports[name] = report

    def mass(name):
        return reports[name]["comparison_pose_mass"]

    base, centre, front = mass("baseline"), mass("payload_center"), mass("payload_front")
    observed_shift = np.asarray(front["system_com_torso_frame_m"]) - centre["system_com_torso_frame_m"]
    expected_shift = 2 / front["total_mass_kg"] * np.array([0.12, 0, 0])
    front_leg_report = reports["front_longer"]
    rear_names = [f"{leg}_{segment}" for leg in ("RL", "RR") for segment in ("thigh", "shank")]
    checks = {
        "all_models_pass": all(r["passed"] for r in reports.values()),
        "payload_adds_exactly_2kg": bool(np.isclose(centre["total_mass_kg"] - base["total_mass_kg"], 2, rtol=0, atol=1e-9)),
        "moving_payload_keeps_total_mass": bool(np.isclose(centre["total_mass_kg"], front["total_mass_kg"], rtol=0, atol=1e-9)),
        "payload_com_shift_matches_weighted_average": bool(np.allclose(observed_shift, expected_shift, rtol=0, atol=1e-10)),
        "changing_front_keeps_rear_inertial_properties": all(mass("front_longer")["bodies"][name] == base["bodies"][name] for name in rear_names),
        "changing_front_keeps_rear_relative_feet": all(np.allclose(front_leg_report["comparison_pose_feet_torso_frame_m"][leg],
            reports["baseline"]["comparison_pose_feet_torso_frame_m"][leg], rtol=0, atol=1e-10) for leg in ("RL", "RR")),
    }
    result = {"passed": all(checks.values()), "checks": checks,
              "comparison_qpos": reference_qpos.tolist(),
              "comparison_scope": "Same base and joint pose; coordinates relative to torso. Frozen inspection, not a supported stance.",
              "payload_com_shift_m": {"expected": expected_shift.tolist(), "observed": observed_shift.tolist()},
              "models": reports}
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = ["# 前后肢与固定载荷：最小对照实验", "",
             "这是助手执行的模型检查，不代表用户已完成学习。所有候选使用相同机身位置、朝向与关节角；质心坐标在机身坐标系中，x 前、y 左、z 上。", "",
             "| 候选 | 前肢上/下段 m | 后肢上/下段 m | 裸机 kg | 载荷 kg | 总质量 kg | 整机质心 x / y / z（mm） |",
             "|---|---|---|---:|---:|---:|---|"]
    for name, report in reports.items():
        shape = report["morphology_m"]
        properties = mass(name)
        com = " / ".join(f"{v * 1000:.3f}" for v in properties["system_com_torso_frame_m"])
        lines.append(f"| {name} | {shape['front_thigh_length']:.2f} / {shape['front_shank_length']:.2f} | {shape['thigh_length']:.2f} / {shape['shank_length']:.2f} | {properties['bare_robot_mass_kg']:.6f} | {properties['payload_mass_kg']:.1f} | {properties['total_mass_kg']:.6f} | {com} |")
    lines += ["", "比较关系：baseline → front_longer 只改变前肢上段 0.22 → 0.26 m；baseline → payload_center 加入背部 2 kg 均匀方块；payload_center → payload_front 只把载荷向前移动 0.12 m。", "",
              f"移动载荷后的整机质心向前变化：{observed_shift[0] * 1000:.6f} mm；加权平均解析预测：{expected_shift[0] * 1000:.6f} mm。总质量保持不变。", "",
              "## 验证", ""]
    lines += [f"- {'通过' if passed else '失败'}：`{name}`" for name, passed in checks.items()]
    lines += ["", "每个模型还校核：每个刚体质量、局部质心、完整质心惯量张量与均匀胶囊/球/箱体解析模型一致；前后腿运动学、限位、两个预览姿态的穿透与 1 秒无控制下落检查。完整数值见 report.json。", "",
              "## 解释与边界", "",
              "腿长改变时截面和密度固定，所以质量也变化。小腿刚体的质量与惯量包含足端球体，电机质量尚未单独建模。载荷固定在机身，未建模支架质量与装卸。",
              "相同关节角下，不等长前后肢的足端高度可以不同；显示姿态不是四足支撑解。尚未实现后肢直立、平衡、静态支撑力或力矩评价，也未验证 Isaac Lab 导入。", "",
              "## 复现", "", "在项目根目录运行：", "", "```sh", ".venv/bin/python -m quadruped_codesign.cli study --render", "```", ""]
    destination = directory / "report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"Study failed; see {directory / 'report.json'}")
    if render:
        from .visualization import render_comparison
        render_comparison(list(paths.values()), directory / "comparison.png", shared_qpos=reference_qpos)
    return destination
