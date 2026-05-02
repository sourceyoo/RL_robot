"""주파수 스윕 진단 (MODE_3 + Ecoflex fin + BL4260 + 3DOF planar).

qpos layout: [root_x, root_y, root_yaw, tail_joint, fin_joint] (5)

각 주파수에서 sine wave (ctrl = sin(2π f t))를 인가하고
- 꼬리 관절 도달 각도 (mean/max/min) — 추종성/대칭
- passive fin 도달 각도 (mean/max/min) — 변형
- body 변위 (x, y) — 추진 방향 (x_disp < 0 = head-forward)
- body yaw — 회전 발산 여부
를 측정.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).parent / "rl_fish.xml"


def sweep(freqs, duration=15.0, settle=3.0):
    model = mujoco.MjModel.from_xml_path(str(XML))
    # qpos: root_x(0), root_y(1), root_yaw(2), tail(3), fin(4)
    X, Y, YAW, TAIL, FIN = 0, 1, 2, 3, 4

    print(f"명령 ±20° (BL4260, kp=100, kv=5, forcerange=±3Nm). 3DOF planar.")
    print(f"{duration}s 시뮬, settle={settle}s\n")
    print(f"{'f[Hz]':>5} | {'tail mean/max/min[°]':>22} | {'fin mean/max/min[°]':>22} | "
          f"{'x_disp':>9} {'y_disp':>9} {'yaw[°]':>8} 방향")
    print("-" * 130)

    for f in freqs:
        data = mujoco.MjData(model)
        tail_a, fin_a = [], []
        x_pos, y_pos = [], []
        yaws = []

        n_steps = int(duration / model.opt.timestep)
        for _ in range(n_steps):
            t = data.time
            data.ctrl[0] = np.sin(2 * np.pi * f * t)
            mujoco.mj_step(model, data)
            if t >= settle:
                tail_a.append(data.qpos[TAIL])
                fin_a.append(data.qpos[FIN])
                x_pos.append(data.qpos[X])
                y_pos.append(data.qpos[Y])
                yaws.append(data.qpos[YAW])

        t_mean = np.degrees(np.mean(tail_a)); t_max = np.degrees(max(tail_a)); t_min = np.degrees(min(tail_a))
        f_mean = np.degrees(np.mean(fin_a)); f_max = np.degrees(max(fin_a)); f_min = np.degrees(min(fin_a))
        x_d = x_pos[-1] - x_pos[0]
        y_d = y_pos[-1] - y_pos[0]
        yaw_d = np.degrees(yaws[-1] - yaws[0])

        if x_d < -0.01:
            verdict = "← 머리쪽(전진) ✓"
        elif x_d > 0.01:
            verdict = "→ 꼬리쪽(후진)"
        else:
            verdict = "  거의 정지"

        print(f"{f:5.2f} | {t_mean:+6.2f}/{t_max:+6.2f}/{t_min:+6.2f}      | "
              f"{f_mean:+6.2f}/{f_max:+6.2f}/{f_min:+6.2f}      | "
              f"{x_d:+9.4f} {y_d:+9.4f} {yaw_d:+8.2f} {verdict}")


if __name__ == "__main__":
    sweep([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
