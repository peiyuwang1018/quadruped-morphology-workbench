"""Run from project root: .venv/bin/python examples/generate_robot.py"""

from quadruped_codesign import generate_robot, evaluate_robot, write_robot

model = generate_robot(thigh_length=0.22, shank_length=0.20)
results = evaluate_robot(model)
print(f"Checks passed: {results['passed']}; mass: {results['robot_mass_kg']:.3f} kg")
write_robot("outputs/my_robot.xml", thigh_length=0.22, shank_length=0.20)
