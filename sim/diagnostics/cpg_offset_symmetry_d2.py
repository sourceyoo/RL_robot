"""D2: CPG 좌우 비대칭 — sim 시간 + phase 초기값 의존성 검증.

D1 (5s, phase=0) 에서 ±0.8 비대칭 -153% 발견 — transient 인지 steady-state 인지 분리.

Test 1: sim_seconds {5, 20} × offset ±0.8 — transient 제거 후 비대칭 잔존 여부.
Test 2: phase_init {0, 0.25, 0.5, 0.75} × offset ±0.4 — phase 초기값 의존성.
        의존성 ↑ → transient (phase 평균하면 대칭 회복).
        의존성 X → steady-state 비대칭 (xml orientation/fluidshape 원인).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import DEFAULT_XML


def run_cpg(offset, amp=1.0, freq=4.0, sim_seconds=5.0, phase_init=0.0):
    model = mujoco.MjModel.from_xml_path(DEFAULT_XML)
    data = mujoco.MjData(model)
    phase = phase_init
    yaw_rates = []
    n_steps = int(sim_seconds / model.opt.timestep)
    for _ in range(n_steps):
        ctrl = float(np.clip(amp * np.sin(2 * np.pi * phase) + offset, -1.0, 1.0))
        data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)
        phase = (phase + freq * model.opt.timestep) % 1.0
        yaw_rates.append(float(data.qvel[2]))
    # transient 제거 — 후반 70% 만 평균
    skip = int(len(yaw_rates) * 0.3)
    return {
        "mean_w_dps": float(np.degrees(np.mean(yaw_rates[skip:]))),
        "final_yaw_deg": float(np.degrees(data.qpos[2])),
    }


def main():
    # Test 1: sim_seconds 비교
    print("Test 1: sim_seconds 비교 (offset ±0.8, transient 제외 70%, phase_init=0)")
    print(f"{'sim_s':>5}  {'offset':>7}  {'mean_ω°/s':>11}  {'final_yaw°':>11}")
    for ss in [5.0, 20.0]:
        for o in [-0.8, +0.8]:
            r = run_cpg(o, sim_seconds=ss)
            print(f"{ss:>5.1f}  {o:+5.1f}     {r['mean_w_dps']:+9.3f}    {r['final_yaw_deg']:+9.2f}")

    # Test 2: phase_init sweep
    print("\nTest 2: phase_init sweep (sim 20s, offset ±0.4)")
    print(f"{'phase':>5}  {'offset':>7}  {'mean_ω°/s':>11}  {'final_yaw°':>11}")
    for pi in [0.0, 0.25, 0.5, 0.75]:
        for o in [-0.4, +0.4]:
            r = run_cpg(o, sim_seconds=20.0, phase_init=pi)
            print(f"{pi:>5.2f}  {o:+5.1f}     {r['mean_w_dps']:+9.3f}    {r['final_yaw_deg']:+9.2f}")

    # 대칭 비교 — phase_init 평균
    print("\n비대칭 정량 (sim 20s, phase_init 4점 평균, transient 제외):")
    for x in [0.4, 0.8]:
        plus_total, minus_total = [], []
        for pi in [0.0, 0.25, 0.5, 0.75]:
            rp = run_cpg(+x, sim_seconds=20.0, phase_init=pi)
            rm = run_cpg(-x, sim_seconds=20.0, phase_init=pi)
            plus_total.append(rp['mean_w_dps'])
            minus_total.append(rm['mean_w_dps'])
        pm = np.mean(plus_total)
        mm = np.mean(minus_total)
        asym = (abs(pm) - abs(mm)) / ((abs(pm) + abs(mm)) / 2 + 1e-9) * 100
        pstd = np.std(plus_total)
        mstd = np.std(minus_total)
        print(f"  offset ±{x}: +={pm:+.3f}±{pstd:.3f}°/s, -={mm:+.3f}±{mstd:.3f}°/s, "
              f"asym={asym:+.1f}%  (phase σ: +={pstd:.3f}, -={mstd:.3f})")


if __name__ == "__main__":
    main()
