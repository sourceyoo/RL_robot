"""D7: D4 96% 비대칭 source 의 진짜 메커니즘 좁히기.

D6 결과:
- 8.2° 는 PCA degenerate noise (frame cancel 시 비대칭 +78% 증가)
- fin_1 ipos.y 비대칭 무관
- D4 의 tail_link fluid 96% 감소는 frame 회전이 아닌 다른 메커니즘

후보:
(1) offset ctrl 이 비대칭 trigger: offset=0 에서는 좌우 swing 대칭 → 비대칭 사라질 수도
(2) tail ellipsoid b≠c (13.45 vs 17.92mm) 의 yaw·pitch drag coupling
(3) numerical integrator 정밀도

case 별 ctrl_mirror sim:
A. amp=0.5 offset=0.0  → offset 제거. 비대칭 사라지면 offset 가 trigger
B. amp=0.5 offset=0.3  → baseline (D6 와 동일)
C. amp=0.0 offset=0.3  → 정적 자세 (sin 제거). fluid drag 자체의 asym
D. amp=0.0 offset=0.0  → null ctrl. 잔존 비대칭 = numerical
E. amp=0.5 offset=0.3 + dt=0.001 (10x 작은 dt) → integrator 정밀도
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML


def ctrl_mirror_sim(model, sim_seconds=20.0, freq=4.0, amp=0.5, offset=0.3, dt_override=None):
    if dt_override is not None:
        model.opt.timestep = dt_override
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
    y_asym = np.abs(y_pos + y_neg)
    yaw_asym = np.abs(yaw_pos + yaw_neg)
    return {
        "y_end_pos": y_pos[-1], "y_end_neg": y_neg[-1],
        "yaw_end_pos": yaw_pos[-1], "yaw_end_neg": yaw_neg[-1],
        "y_asym_max": y_asym.max(), "yaw_asym_max": yaw_asym.max(),
        "y_asym_end": y_asym[-1], "yaw_asym_end": yaw_asym[-1],
        "y_pos_traj": y_pos, "y_neg_traj": y_neg,
    }


print("=" * 64)
print("D7: offset / amp / dt 영향 분해")
print("=" * 64)

cases = [
    ("A_amp_offset_0",    {"amp": 0.5, "offset": 0.0}),
    ("B_baseline",        {"amp": 0.5, "offset": 0.3}),
    ("C_static_offset",   {"amp": 0.0, "offset": 0.3}),
    ("D_null_ctrl",       {"amp": 0.0, "offset": 0.0}),
    ("E_baseline_dt_0p001", {"amp": 0.5, "offset": 0.3, "dt_override": 0.001}),
    ("F_offset_neg",      {"amp": 0.5, "offset": -0.3}),
    ("G_offset_0p1",      {"amp": 0.5, "offset": 0.1}),
    ("H_offset_0p6",      {"amp": 0.5, "offset": 0.6}),
]

results = {}
for name, kwargs in cases:
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    r = ctrl_mirror_sim(model, **kwargs)
    results[name] = r
    print(f"\n[{name}]  amp={kwargs.get('amp')} offset={kwargs.get('offset')}"
          f"{'  dt=' + str(kwargs['dt_override']) if 'dt_override' in kwargs else ''}")
    print(f"  y(+ctrl) end={r['y_end_pos']:+.5f}  y(-ctrl) end={r['y_end_neg']:+.5f}")
    print(f"  yaw(+) end ={r['yaw_end_pos']:+.5f}  yaw(-) end ={r['yaw_end_neg']:+.5f}")
    print(f"  |y+y_neg| max={r['y_asym_max']:.5f}  yaw_asym_max={r['yaw_asym_max']:.5f}")

print("\n" + "=" * 64)
print("Summary: B(baseline) 대비 변화")
print("=" * 64)
b = results["B_baseline"]
print(f"  B baseline                y_asym={b['y_asym_max']:.5f}  yaw_asym={b['yaw_asym_max']:.5f}")
for name in ["A_amp_offset_0", "C_static_offset", "D_null_ctrl",
             "E_baseline_dt_0p001", "F_offset_neg", "G_offset_0p1", "H_offset_0p6"]:
    r = results[name]
    dy = (r["y_asym_max"] - b["y_asym_max"]) / max(b["y_asym_max"], 1e-9) * 100
    dyaw = (r["yaw_asym_max"] - b["yaw_asym_max"]) / max(b["yaw_asym_max"], 1e-9) * 100
    print(f"  {name:>26}  y_asym={r['y_asym_max']:.5f} ({dy:+6.1f}%)  yaw_asym={r['yaw_asym_max']:.5f} ({dyaw:+6.1f}%)")

# 핵심 결론 가이드
print("\n" + "=" * 64)
print("판정")
print("=" * 64)
a, c, d, e = results["A_amp_offset_0"], results["C_static_offset"], results["D_null_ctrl"], results["E_baseline_dt_0p001"]
print(f"  offset=0 에서 비대칭 = {a['y_asym_max']:.5f} (B 의 {a['y_asym_max']/b['y_asym_max']*100:.1f}%)")
print(f"  amp=0 offset=0.3 정적 = {c['y_asym_max']:.5f}  → fluid drag 정적 asym")
print(f"  null ctrl = {d['y_asym_max']:.5f}  → numerical floor")
print(f"  dt=0.001 (10x 작음) = {e['y_asym_max']:.5f} ({e['y_asym_max']/b['y_asym_max']*100:.1f}%)  → integrator 영향")
