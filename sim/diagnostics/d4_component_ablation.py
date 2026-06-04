"""D4: component ablation — fluidshape="none" 으로 비대칭 source 분리.

각 mesh geom (base_link / tail_link / fin_1) 의 fluidshape 를 임시로 끄고
ctrl_mirror sim 으로 y·yaw 비대칭 잔존도 측정. baseline 대비 감소량 = 그 component
fluid 의 비대칭 기여도. case 2 (all none) 잔존 = mass/inertia 비대칭 기여도.

xml 디스크 원본 보존: 같은 디렉토리에 임시 .xml 파일 생성 후 from_xml_path 로 로드
(meshdir 상대 경로 유지). 측정 후 삭제.

ctrl: 0.5·sin(2π·4·t) + 0.3 vs -(...)  — D3 와 동일.
sim_seconds=20 (D2 에서 transient 제거 충분 확인).
"""
import os
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML


def load_from_string(xml_str):
    """xml string 을 디스크 원본 디렉토리의 임시 파일로 저장 → from_xml_path 로 로드."""
    parent = Path(DEFAULT_XML).parent
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=parent, delete=False, encoding="utf-8"
    ) as f:
        f.write(xml_str)
        tmp_path = f.name
    try:
        return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)


def make_xml(disable_meshes):
    """disable_meshes: list of mesh names. 해당 geom 의 mesh="..." 뒤에 fluidshape="none" inject."""
    with open(DEFAULT_XML, "r", encoding="utf-8") as f:
        xml = f.read()
    for name in disable_meshes:
        needle = f'mesh="{name}"'
        repl = f'mesh="{name}" fluidshape="none"'
        # geom 의 attribute (asset 의 `mesh name="..."` 와 패턴 다름)
        if needle not in xml:
            raise RuntimeError(f"geom attribute not found: {needle}")
        xml = xml.replace(needle, repl, 1)
    return xml


def run_ctrl_mirror(xml_str, sim_seconds=20.0):
    """ctrl(t) vs -ctrl(t) 두 sim 비교. 좌우 대칭이면 yA(t)=-yB(t)."""
    model = load_from_string(xml_str)

    def run(ctrl_fn):
        data = mujoco.MjData(model)
        n_steps = int(sim_seconds / model.opt.timestep)
        ys, yaws = np.empty(n_steps), np.empty(n_steps)
        for i in range(n_steps):
            t = i * model.opt.timestep
            data.ctrl[0] = float(ctrl_fn(t))
            mujoco.mj_step(model, data)
            ys[i] = data.qpos[1]
            yaws[i] = data.qpos[2]
        return ys, yaws

    ctrl_A = lambda t: 0.5 * np.sin(2 * np.pi * 4 * t) + 0.3
    ctrl_B = lambda t: -(0.5 * np.sin(2 * np.pi * 4 * t) + 0.3)
    yA, yawA = run(ctrl_A)
    yB, yawB = run(ctrl_B)

    y_err = yA + yB
    yaw_err = yawA + yawB
    rel_y = abs(y_err).max() / (abs(yA).max() + 1e-9) * 100
    rel_yaw = abs(yaw_err).max() / (abs(yawA).max() + 1e-9) * 100
    return rel_y, rel_yaw, abs(yA).max(), np.degrees(abs(yawA).max())


def main():
    cases = [
        ("1 baseline",  []),
        ("2 all none",  ["base_link", "tail_link", "fin_1"]),
        ("3 fin none",  ["fin_1"]),
        ("4 base none", ["base_link"]),
        ("5 tail none", ["tail_link"]),
    ]

    print(f"D4 component ablation — sim_seconds=20, ctrl: 0.5·sin(2π·4t)+0.3 vs −(...)\n")
    header = f"{'case':<14} {'|y_err|':>8} {'|yaw_err|':>10} {'|y|max(m)':>10} {'|yaw|max(°)':>12}"
    print(header)
    print("-" * len(header))

    results = {}
    for label, disable in cases:
        xml = make_xml(disable)
        rel_y, rel_yaw, y_max, yaw_max = run_ctrl_mirror(xml)
        results[label] = (rel_y, rel_yaw)
        print(f"{label:<14} {rel_y:>7.2f}% {rel_yaw:>9.2f}% {y_max:>10.4f} {yaw_max:>12.2f}")

    # Δ vs baseline
    base_y, base_yaw = results["1 baseline"]
    print()
    print(f"{'case':<14} {'y Δ vs base':>13} {'yaw Δ vs base':>15}")
    print("-" * 44)
    for label, _ in cases:
        if label == "1 baseline":
            continue
        ry, ryaw = results[label]
        print(f"{label:<14} {base_y - ry:>+12.2f}% {base_yaw - ryaw:>+14.2f}%")


if __name__ == "__main__":
    main()
