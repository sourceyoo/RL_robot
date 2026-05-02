"""body의 roll/pitch만 막고 fin 대칭성 검증.

freejoint qvel layout: [vx, vy, vz, ωx, ωy, ωz]
qvel[3:5] = 0 매 step → body는 yaw + 평행이동만 가능.
이때 fin이 대칭이면, 비대칭 원인 = body의 roll/pitch (특히 pitch).
"""

from pathlib import Path
import mujoco
import numpy as np

XML = Path(__file__).parent / "rl_fish.xml"


def sweep(freqs, duration=15.0, settle=3.0):
    model = mujoco.MjModel.from_xml_path(str(XML))
    tail_idx = 7
    fin_idx = 8

    print("body roll/pitch 잠금 (qvel[3:5]=0 매 step). yaw + 평행이동 자유.\n")
    print(f"{'f[Hz]':>5} | {'tail mean/max/min[°]':>22} | {'fin mean/max/min[°]':>22} | "
          f"{'x_disp':>8} {'y_disp':>8}")
    print("-" * 110)

    for f in freqs:
        data = mujoco.MjData(model)
        tail_a, fin_a = [], []
        x_pos, y_pos = [], []

        n_steps = int(duration / model.opt.timestep)
        for _ in range(n_steps):
            t = data.time
            data.ctrl[0] = np.sin(2 * np.pi * f * t)
            # roll, pitch rate 0
            data.qvel[3] = 0
            data.qvel[4] = 0
            mujoco.mj_step(model, data)
            if t >= settle:
                tail_a.append(data.qpos[tail_idx])
                fin_a.append(data.qpos[fin_idx])
                x_pos.append(data.qpos[0])
                y_pos.append(data.qpos[1])

        t_mean = np.degrees(np.mean(tail_a)); t_max = np.degrees(max(tail_a)); t_min = np.degrees(min(tail_a))
        f_mean = np.degrees(np.mean(fin_a)); f_max = np.degrees(max(fin_a)); f_min = np.degrees(min(fin_a))
        x_d = x_pos[-1] - x_pos[0]
        y_d = y_pos[-1] - y_pos[0]

        print(f"{f:5.2f} | {t_mean:+6.2f}/{t_max:+6.2f}/{t_min:+6.2f}      | "
              f"{f_mean:+6.2f}/{f_max:+6.2f}/{f_min:+6.2f}      | "
              f"{x_d:+8.4f} {y_d:+8.4f}")


if __name__ == "__main__":
    sweep([0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
