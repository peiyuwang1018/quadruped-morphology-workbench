# Sources and development provenance

## Implementation origin

This repository was developed as a standalone, AI-assisted Python project through iterative design and coding with OpenAI Codex. The procedural MJCF generators, analytic inverse-kinematics solver, primitive mass/inertia checks, Tk parameter panel, support geometry, alignment search and tests were implemented within this project.

The initial `assets/base_quadruped.xml` is this project's own primitive-geometry model, retained as a regression fixture. It is not a model imported from ANYmal, Unitree, MuJoCo Menagerie or Transformer Transformer. The later Fusion reference profiles use manually entered design parameters and mounting descriptions; they do not contain exported CAD assemblies or meshes. The early eight-joint generator and the current twelve-joint profile are stages of this same local project.

## Research reference: Transformer Transformer

The initial research motivation referenced **Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design**:

- [Official repository](https://github.com/real-stanford/transformer-transformer)
- [Project page](https://transformer-transformer.github.io/)

It is a research reference for procedural morphology generation and future morphology–control co-design. **This repository is not a fork or a reproduction of that framework. No source code, robot assets, trained models, datasets or training pipeline from it have been incorporated into this version.** Its generator has not been integrated. Any future reuse should identify the exact upstream files/version, preserve applicable attribution and document modifications at that time.

## Software foundation

| Component | Role | Upstream |
|---|---|---|
| MuJoCo | MJCF compilation, kinematics, recursive Newton–Euler and Jacobians for static loads, contact-distance queries and rendering | [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), [Python documentation](https://mujoco.readthedocs.io/en/stable/python.html) |
| NumPy | Numerical geometry, transforms and validation | [numpy/numpy](https://github.com/numpy/numpy) |
| Pillow | Rendered image display and export | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| Python standard library | Tkinter UI, TOML loading, XML generation, tests and CLI | [Python documentation](https://docs.python.org/3.12/) |
| uv / Hatchling | Environment locking and package build tooling | [uv](https://github.com/astral-sh/uv), [Hatch](https://github.com/pypa/hatch) |

Dependencies are installed separately rather than vendored. Exact resolved versions and package hashes are in `uv.lock`; their licenses remain those of their respective upstream projects. No upstream project's license is automatically assigned to this project's original code.

## Model data and analytical methods

- Example link lengths, body spacings, motor arrangement and clearance envelopes were supplied during the project's design discussions. Several mounting distances remain declared estimates. See [model assumptions](docs/fusion-reference.md).
- The nominal AKE90-8 KV35 module dimensions and mass are referenced to the [manufacturer's product page](https://www.cubemars.com/product/ake90-8-kv35-quasi-direct-drive-actuator.html). The 118 × 60 mm clearance envelope is an assembly assumption, separate from nominal module dimensions. No manufacturer CAD model is bundled.
- Inverse kinematics uses an analytic roll joint plus a planar two-link construction with a lateral offset. Mass/inertia checks use standard box, cylinder, sphere and capsule formulas and the parallel-axis theorem. They are implemented here, rather than copied from a robot-specific generator.
- Support-region visualization is geometric screening. [Feasible Region: an Actuation-Aware Extension of the Support Region](https://arxiv.org/abs/1903.07999) is background reading on the additional role of actuator constraints, not an algorithm implemented by this workbench.

The body/subtree inspector and static pair-load evaluator were also implemented in this project. The latter uses MuJoCo's dynamics primitives under explicit equal-left/right vertical loading assumptions; it does not integrate Pinocchio or a research project's optimization method. The implementation is documented in [English](documents/kinematics-dynamics-en.md) and [Chinese](documents/kinematics-dynamics-zh.md).

## Project licensing

A license for this project's original code has not yet been selected. No project-level `LICENSE` is included in this initial version. Dependency and reference-project licenses remain separate.
