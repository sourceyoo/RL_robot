"""CPG offset 좌우 대칭 검증 — fluid asymmetry 진단 (D 카드).

amp=1.0, freq=4Hz fixed. offset ∈ {-0.8, -0.4, 0, +0.4, +0.8} 각 5s simulate.
mean yaw_rate · final yaw · final pos 측정 → |+X| vs |-X| 비교.

대칭 = |+X|·|-X| 크기 ≈ 동일, 부호 반대. 비대칭이면 sim 인공물 (xml fluidshape /
geom orientation).

사용: cd /home/yoo/RL_robot/sim && python3 diagnostics/cpg_offset_symmetry.py
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML


def run_cpg(offset: float, amp: float = 1.0, freq: float = 4.0, sim_seconds: float = 5.0):
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    data = mujoco.MjData(model)
    phase = 0.0
    yaw_rates = []
    n_steps = int(sim_seconds / model.opt.timestep)
    for _ in range(n_steps):
        ctrl = float(np.clip(amp * np.sin(2.0 * np.pi * phase) + offset, -1.0, 1.0))
        data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)
        phase = (phase + freq * model.opt.timestep) % 1.0
        yaw_rates.append(float(data.qvel[2]))
    return {
        "final_yaw_deg": float(np.degrees(data.qpos[2])),
        "mean_yaw_rate_dps": float(np.degrees(np.mean(yaw_rates))),
        "final_x": float(data.qpos[0]),
        "final_y": float(data.qpos[1]),
    }


def main():
    offsets = [-0.8, -0.4, 0.0, +0.4, +0.8]
    print(f"\namp=1.0, freq=4Hz, 5s simulate")
    print(f"{'offset':>7}  {'final_yaw°':>11}  {'mean_ω°/s':>11}  {'x':>7}  {'y':>7}")
    print("-" * 60)
    results = {}
    for o in offsets:
        r = run_cpg(o)
        results[o] = r
        print(f"{o:+5.1f}    {r['final_yaw_deg']:+9.2f}    {r['mean_yaw_rate_dps']:+9.2f}    "
              f"{r['final_x']:+.3f}  {r['final_y']:+.3f}")

    print("\n좌우 대칭 검증 (|+X| vs |-X|):")
    for x in [0.4, 0.8]:
        plus = results[+x]['mean_yaw_rate_dps']
        minus = results[-x]['mean_yaw_rate_dps']
        # asymmetry = (|plus| - |minus|) / mean(|plus|, |minus|), 부호 일치 가정
        denom = (abs(plus) + abs(minus)) / 2 + 1e-9
        diff = abs(plus) - abs(minus)
        rel = diff / denom * 100
        sign_match = "✓" if plus * minus < 0 else "✗ 부호 동일!"
        print(f"  offset ±{x}: +={plus:+.2f}°/s, -={minus:+.2f}°/s, "
              f"|+|−|−|={diff:+.2f}°/s ({rel:+.1f}%), 부호반대 {sign_match}")

    print("\noffset=0 drift (완벽 대칭이면 ≈ 0):")
    z = results[0.0]
    print(f"  yaw_rate = {z['mean_yaw_rate_dps']:+.3f}°/s, y_drift = {z['final_y']:+.4f} m")


if __name__ == "__main__":
    main()
