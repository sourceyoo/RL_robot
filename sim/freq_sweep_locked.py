"""body를 강제로 고정해서 fin 대칭성 검증.

base_link의 freejoint qvel을 매 step 0으로 리셋 → body 안 움직임.
순수하게 active tail의 회전만 fin에 전달.
fin이 이때 대칭이라면, 비대칭 원인 = body의 자유 운동.
"""

from pathlib import Path
import mujoco
import numpy as np

XML = Path(__file__).parent / "rl_fish.xml"


def sweep(freqs, duration=15.0, settle=3.0):
    model = mujoco.MjModel.from_xml_path(str(XML))
    tail_idx = 7
    fin_idx = 8

    print("body 강제 고정 (qvel[0:6]=0 매 step). 순수 tail→fin 응답만 측정.\n")
    print(f"{'f[Hz]':>5} | {'tail mean/max/min[°]':>22} | {'fin mean/max/min[°]':>22}")
    print("-" * 80)

    for f in freqs:
        data = mujoco.MjData(model)
        tail_a, fin_a = [], []

        n_steps = int(duration / model.opt.timestep)
        for _ in range(n_steps):
            t = data.time
            data.ctrl[0] = np.sin(2 * np.pi * f * t)
            # body 고정
            data.qvel[0:6] = 0
            mujoco.mj_step(model, data)
            # 위치도 리셋 (속도가 0이어도 가속이 누적되면 표류)
            data.qpos[0:3] = 0
            if t >= settle:
                tail_a.append(data.qpos[tail_idx])
                fin_a.append(data.qpos[fin_idx])

        t_mean = np.degrees(np.mean(tail_a)); t_max = np.degrees(max(tail_a)); t_min = np.degrees(min(tail_a))
        f_mean = np.degrees(np.mean(fin_a)); f_max = np.degrees(max(fin_a)); f_min = np.degrees(min(fin_a))

        print(f"{f:5.2f} | {t_mean:+6.2f}/{t_max:+6.2f}/{t_min:+6.2f}      | "
              f"{f_mean:+6.2f}/{f_max:+6.2f}/{f_min:+6.2f}")


if __name__ == "__main__":
    sweep([0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
