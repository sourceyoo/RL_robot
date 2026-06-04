"""D8: D7 의 -62% 감소가 진짜 numerical 원인인지 vs deterministic bias 좁히기.

현 xml: timestep=0.002, integrator=implicitfast.

검증:
Step A. dt sweep (0.002, 0.001, 0.0005, 0.0001) → 비대칭이 0 으로 수렴(numerical) vs plateau(inherent)
Step B. integrator 변경 (implicitfast, Euler, RK4) → 같은 dt 에서 정확도 비교
Step C. component ablation 재실행: base 만 / fin 만 / tail 만 fluidshape="none"
Step D. mesh STL vertex y 분포 직접 측정 (y mean, +y vs -y count)
"""
import os
import struct
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML

STL_DIR = Path(__file__).resolve().parent.parent.parent / "fish_urdf" / "RL_SIM_MODE_3_description" / "meshes"


def parse_stl_binary(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        verts = np.empty((n * 3, 3), dtype=np.float64)
        for i in range(n):
            f.read(12)
            for j in range(3):
                verts[i * 3 + j] = struct.unpack("<fff", f.read(12))
            f.read(2)
    return verts


def ctrl_mirror_sim(model, sim_seconds=20.0, freq=4.0, amp=0.5, offset=0.3):
    dt = model.opt.timestep
    n_steps = int(sim_seconds / dt)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tail")

    def run(sign):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        ys, yaws = [], []
        for k in range(n_steps):
            t = k * dt
            data.ctrl[actuator_id] = sign * (amp * np.sin(2 * np.pi * freq * t) + offset)
            mujoco.mj_step(model, data)
            ys.append(data.qpos[1])
            yaws.append(data.qpos[2])
        return np.array(ys), np.array(yaws)

    y_pos, yaw_pos = run(+1)
    y_neg, yaw_neg = run(-1)
    y_sum = y_pos + y_neg
    yaw_sum = yaw_pos + yaw_neg
    return {
        "y_pos_end": y_pos[-1], "y_neg_end": y_neg[-1],
        "y_sum_max": np.abs(y_sum).max(),
        "y_sum_end": y_sum[-1],     # signed (+ 면 +y 쪽 bias)
        "yaw_sum_max": np.abs(yaw_sum).max(),
        "yaw_sum_end": yaw_sum[-1],
    }


def load_with_mods(mods, label):
    with open(DEFAULT_XML, "r", encoding="utf-8") as f:
        xml = f.read()
    for find, replace in mods:
        if find not in xml:
            raise RuntimeError(f"[{label}] '{find[:60]}...' not in xml")
        xml = xml.replace(find, replace)
    parent = Path(DEFAULT_XML).parent
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{label}.xml", dir=parent, delete=False, encoding="utf-8"
    ) as f:
        f.write(xml)
        tmp = f.name
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)


# ============================================================
# Step A: dt sweep (integrator 고정 implicitfast)
# ============================================================
print("=" * 64)
print("Step A: dt sweep (integrator=implicitfast 고정)")
print("=" * 64)

dt_results = {}
for dt in [0.002, 0.001, 0.0005, 0.0001]:
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    model.opt.timestep = dt
    r = ctrl_mirror_sim(model)
    dt_results[dt] = r
    print(f"  dt={dt:.5f}  y_sum_end={r['y_sum_end']:+.5f}  y_sum_max={r['y_sum_max']:.5f}  "
          f"yaw_sum_end={r['yaw_sum_end']:+.5f}  yaw_sum_max={r['yaw_sum_max']:.5f}")

# convergence check
ys = [dt_results[dt]['y_sum_max'] for dt in [0.002, 0.001, 0.0005, 0.0001]]
print(f"\n  y_sum_max series (dt 0.002→0.0001): {[f'{v:.5f}' for v in ys]}")
print(f"  ratio (smaller dt / 0.002): {[f'{v/ys[0]*100:5.1f}%' for v in ys]}")
print(f"  plateau? (0.0001 / 0.0005): {ys[3]/ys[2]*100:.1f}%  (1에 가까우면 plateau = inherent bias)")

# ============================================================
# Step B: integrator 비교 (dt 0.002 고정)
# ============================================================
print("\n" + "=" * 64)
print("Step B: integrator 비교 (dt=0.002 고정)")
print("=" * 64)

# mujoco integrator: 0=Euler, 1=RK4, 2=implicit, 3=implicitfast
INTEGRATORS = {
    "Euler":        mujoco.mjtIntegrator.mjINT_EULER,
    "RK4":          mujoco.mjtIntegrator.mjINT_RK4,
    "implicit":     mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
integrator_results = {}
for name, code in INTEGRATORS.items():
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    model.opt.timestep = 0.002
    model.opt.integrator = code
    r = ctrl_mirror_sim(model)
    integrator_results[name] = r
    print(f"  {name:>12}  y_sum_end={r['y_sum_end']:+.5f}  y_sum_max={r['y_sum_max']:.5f}  "
          f"yaw_sum_max={r['yaw_sum_max']:.5f}")

# ============================================================
# Step C: component ablation 재실행 (base/tail/fin 단독)
# ============================================================
print("\n" + "=" * 64)
print("Step C: component ablation (fluidshape='none' 단독)")
print("=" * 64)

BASE_GEOM_FIND = '<geom type="mesh" mesh="base_link"/>'
BASE_GEOM_NONE = '<geom type="mesh" mesh="base_link" fluidshape="none"/>'
TAIL_GEOM_FIND = '<geom pos="-0.144817 0 0.03565" type="mesh" mesh="tail_link"/>'
TAIL_GEOM_NONE = '<geom pos="-0.144817 0 0.03565" type="mesh" mesh="tail_link" fluidshape="none"/>'
FIN_GEOM_FIND = '<geom pos="-0.201817 0 0.047025"\n                type="mesh" mesh="fin_1"\n                mass="0.024"\n                material="fiberglass"/>'
FIN_GEOM_NONE = '<geom pos="-0.201817 0 0.047025"\n                type="mesh" mesh="fin_1"\n                mass="0.024"\n                material="fiberglass" fluidshape="none"/>'

ablation_cases = {
    "baseline":     [],
    "base_off":     [(BASE_GEOM_FIND, BASE_GEOM_NONE)],
    "tail_off":     [(TAIL_GEOM_FIND, TAIL_GEOM_NONE)],
    "fin_off":      [(FIN_GEOM_FIND, FIN_GEOM_NONE)],
    "all_off":      [(BASE_GEOM_FIND, BASE_GEOM_NONE), (TAIL_GEOM_FIND, TAIL_GEOM_NONE), (FIN_GEOM_FIND, FIN_GEOM_NONE)],
}

abl = {}
for name, mods in ablation_cases.items():
    model = load_with_mods(mods, name)
    r = ctrl_mirror_sim(model)
    abl[name] = r
    print(f"  {name:>12}  y_sum_end={r['y_sum_end']:+.5f}  y_sum_max={r['y_sum_max']:.5f}  "
          f"yaw_sum_max={r['yaw_sum_max']:.5f}")

b = abl["baseline"]
print(f"\n  baseline y_sum_max={b['y_sum_max']:.5f}")
for name in ["base_off", "tail_off", "fin_off", "all_off"]:
    r = abl[name]
    dy = (r['y_sum_max'] - b['y_sum_max']) / max(b['y_sum_max'], 1e-9) * 100
    print(f"    {name:>10}  y_sum_max={r['y_sum_max']:.5f} ({dy:+6.1f}%)")

# ============================================================
# Step D: mesh STL vertex y 분포 직접 측정
# ============================================================
print("\n" + "=" * 64)
print("Step D: mesh STL vertex y 분포 (raw STL frame, m 단위)")
print("=" * 64)

for name, scale in [("base_link", np.array([0.001, 0.001, 0.001])),
                    ("tail_link_1_1", np.array([0.001, 0.001, 0.001])),
                    ("fin_1", np.array([0.001, 0.0011, 0.001]))]:
    v = parse_stl_binary(STL_DIR / f"{name}.stl") * scale
    print(f"\n  [{name}]  n={len(v)}")
    for axis_name, ai in [("x", 0), ("y", 1), ("z", 2)]:
        vals = v[:, ai]
        n_pos = (vals > 0).sum()
        n_neg = (vals < 0).sum()
        mean = vals.mean()
        med = np.median(vals)
        diff_pct = (n_pos - n_neg) / len(v) * 100
        print(f"    {axis_name}: mean={mean:+.6f}  median={med:+.6f}  "
              f"#(>0)={n_pos}  #(<0)={n_neg}  Δ%={diff_pct:+.2f}")
