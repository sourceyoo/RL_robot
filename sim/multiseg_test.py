"""다단 passive fin (3-segment 실리콘 이산화) 진단.

목표:
  1. 각 fin segment가 head→tail traveling wave로 위상 lag 누적하는지
  2. body가 +x(후진) 대신 -x(전진)로 가는지

부호 규약: x_disp < 0 → 머리 방향(전진) ✓
"""

from pathlib import Path
import mujoco
import numpy as np

XML = Path(__file__).parent / "rl_fish.xml"
DURATION = 10.0
SETTLE = 2.0


def run(f):
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    # qpos layout: freejoint(7) + tail(1) + fin_seg1(1) + fin_seg2(1) + fin_seg3(1)
    tail_idx = 7
    seg_idx = [8, 9, 10]

    tail_a = []
    seg_a = [[], [], []]
    x_pos = []

    n = int(DURATION / model.opt.timestep)
    for _ in range(n):
        t = data.time
        data.ctrl[0] = np.sin(2 * np.pi * f * t)
        mujoco.mj_step(model, data)
        if t >= SETTLE:
            tail_a.append(data.qpos[tail_idx])
            for i, idx in enumerate(seg_idx):
                seg_a[i].append(data.qpos[idx])
            x_pos.append(data.qpos[0])

    tail_amp = (max(tail_a) - min(tail_a)) / 2
    seg_amps = [(max(s) - min(s)) / 2 for s in seg_a]
    tail_mean = np.mean(tail_a)
    seg_means = [np.mean(s) for s in seg_a]
    x_disp = x_pos[-1] - x_pos[0]

    if x_disp < -0.005:
        verdict = "← 전진 ✓"
    elif x_disp > 0.005:
        verdict = "→ 후진"
    else:
        verdict = "  정지"

    print(f"f={f:>4.1f}Hz | tail amp={np.degrees(tail_amp):5.1f}° mean={np.degrees(tail_mean):+5.1f}° | "
          f"seg1 amp={np.degrees(seg_amps[0]):5.1f}° mean={np.degrees(seg_means[0]):+5.1f}° | "
          f"seg2 amp={np.degrees(seg_amps[1]):5.1f}° mean={np.degrees(seg_means[1]):+5.1f}° | "
          f"seg3 amp={np.degrees(seg_amps[2]):5.1f}° mean={np.degrees(seg_means[2]):+5.1f}° | "
          f"x_disp={x_disp:+7.4f}m {verdict}")


def main():
    print("3-segment passive fin (k=2e-3, d=1e-5 each segment).\n")
    print("Wave 잘 나면: tail < seg1 < seg2 < seg3 진폭으로 점점 커지거나, mean이 segment 따라 변함\n")
    for f in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        run(f)


if __name__ == "__main__":
    main()
