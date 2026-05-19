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

XML = Path(__file__).parent.parent / "rl_fish.xml"


def sweep(freqs, duration=15.0, settle=3.0):
    model = mujoco.MjModel.from_xml_path(str(XML))
    X, Y, YAW, TAIL, FIN = 0, 1, 2, 3, 4

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    fin_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fin_1")
    fin_geom_id = -1
    for i in range(model.ngeom):
        if model.geom_dataid[i] >= 0:
            mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[i])
            if mname == "fin_1":
                fin_geom_id = i
                break

    # V-cut left/right tips in mesh PCA-aligned frame (m, from mesh_vert analysis)
    fork_left = np.array([0.00044, -0.08135, +0.08275])
    fork_right = np.array([-0.00044, -0.08135, -0.08275])
    fork_mid = 0.5 * (fork_left + fork_right)

    print(f"명령 ±20° (BL4260, kp=100, kv=5, forcerange=±3Nm). 3DOF planar.")
    print(f"{duration}s 시뮬, settle={settle}s")
    print(f"fin_tip = V-cut left tip world 위치. ang_tail = tail end → tip 방향 vs tail axis (deg).")
    print(f"ang_base = base 중심 → tip 방향 vs base head_dir (deg).\n")
    print(f"{'f[Hz]':>5} | {'tail joint[°]':>22} | {'fin joint[°]':>22} | "
          f"{'ang_tail[°]':>22} | {'ang_base[°]':>22} | "
          f"{'x_disp':>9} {'y_disp':>9} {'yaw[°]':>8} 방향")
    print("-" * 180)

    for f in freqs:
        data = mujoco.MjData(model)
        tail_a, fin_a = [], []
        ang_tail_list, ang_base_list = [], []
        x_pos, y_pos = [], []
        yaws = []

        n_steps = int(duration / model.opt.timestep)
        for _ in range(n_steps):
            t = data.time
            data.ctrl[0] = np.sin(2 * np.pi * f * t)
            mujoco.mj_step(model, data)
            if t >= settle:
                tj = float(data.qpos[TAIL])
                fj = float(data.qpos[FIN])
                by = float(data.qpos[YAW])
                tail_a.append(tj)
                fin_a.append(fj)

                # fin tip world (V-cut left)
                geom_pos = data.geom_xpos[fin_geom_id]
                geom_mat = data.geom_xmat[fin_geom_id].reshape(3, 3)
                tip_w = geom_pos + geom_mat @ fork_left

                # base center, tail end (= fin body origin)
                base_w = data.xpos[base_id]
                tail_end_w = data.xpos[fin_body_id]

                # tail axis: tail의 length 방향 (tail_end - base 방향, world xy)
                tail_axis = tail_end_w[:2] - base_w[:2]
                tail_axis_n = tail_axis / (np.linalg.norm(tail_axis) + 1e-9)

                # ang_tail: tail axis 대비 (tail_end → tip) 방향 angle
                v_tail = tip_w[:2] - tail_end_w[:2]
                ang_tail = np.arctan2(
                    tail_axis_n[0] * v_tail[1] - tail_axis_n[1] * v_tail[0],
                    tail_axis_n[0] * v_tail[0] + tail_axis_n[1] * v_tail[1],
                )

                # head_dir: base의 forward (CLAUDE.md axis 보정)
                head_dir = np.array([-np.cos(by), np.sin(by)])
                # ang_base: head_dir 반대 방향 (= tail 방향) 대비 (base → tip) angle
                tail_dir_from_base = -head_dir
                v_base = tip_w[:2] - base_w[:2]
                ang_base = np.arctan2(
                    tail_dir_from_base[0] * v_base[1] - tail_dir_from_base[1] * v_base[0],
                    tail_dir_from_base[0] * v_base[0] + tail_dir_from_base[1] * v_base[1],
                )

                ang_tail_list.append(ang_tail)
                ang_base_list.append(ang_base)
                x_pos.append(data.qpos[X])
                y_pos.append(data.qpos[Y])
                yaws.append(by)

        t_mean = np.degrees(np.mean(tail_a)); t_max = np.degrees(max(tail_a)); t_min = np.degrees(min(tail_a))
        f_mean = np.degrees(np.mean(fin_a)); f_max = np.degrees(max(fin_a)); f_min = np.degrees(min(fin_a))
        at_mean = np.degrees(np.mean(ang_tail_list)); at_max = np.degrees(max(ang_tail_list)); at_min = np.degrees(min(ang_tail_list))
        ab_mean = np.degrees(np.mean(ang_base_list)); ab_max = np.degrees(max(ang_base_list)); ab_min = np.degrees(min(ang_base_list))
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
              f"{at_mean:+6.2f}/{at_max:+6.2f}/{at_min:+6.2f}      | "
              f"{ab_mean:+6.2f}/{ab_max:+6.2f}/{ab_min:+6.2f}      | "
              f"{x_d:+9.4f} {y_d:+9.4f} {yaw_d:+8.2f} {verdict}")


if __name__ == "__main__":
    sweep([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
