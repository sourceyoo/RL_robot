"""offset 부호별 추진·회전 비대칭 (정책 무관, read-only).

질문: "우회전(-offset)은 parallel 가능, 좌회전(+offset)은 불가"가 왜인가.
직진 자세에서 고정 offset 주입 → 추진(head방향 전진)·회전(yaw rate) 측정. +/- 비대칭 확인.
사용: python3 diagnostics/offset_sign_sweep.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_YAW

DT = 0.084
FREQ_N, AMP_N = 0.0, 0.0   # freq 5Hz, amp 0.5 (c1=0 → amp 0.5)
N_STEP = 60                # 초기 회전율 (arc로 여러바퀴 돌기 전 — yaw 단조 구간)
OFFSETS = [round(o, 2) for o in np.linspace(-0.9, 0.9, 13)]


def run(offset):
    env = FishSwimEnv(episode_seconds=600.0, success_radius=0.001, action_history_n=20)
    env.reset(seed=0)
    yaw0 = float(env.data.qpos[IDX_YAW])
    prev = env._torso_pos()[:2].copy()
    fwd = 0.0
    yaw_rates = []
    a = np.array([FREQ_N, AMP_N, offset], dtype=np.float32)
    for _ in range(N_STEP):
        env.step(a)
        xy = env._torso_pos()[:2]
        yaw = float(env.data.qpos[IDX_YAW])
        hd = np.array([-np.cos(yaw), np.sin(yaw)])
        fwd += float(np.dot(xy - prev, hd))   # head방향 누적 전진 (추진)
        yaw_rates.append(float(env.data.qvel[IDX_YAW]))
        prev = xy.copy()
    # net yaw_rate (tail wag ±진동은 평균으로 상쇄, net rotation 만 남음)
    yaw_rate_mean = np.degrees(np.mean(yaw_rates))   # deg/s
    yaw_net = np.degrees(float(env.data.qpos[IDX_YAW]) - yaw0)
    env.close()
    return fwd, yaw_rate_mean, yaw_net


print(f"[offset sign sweep] amp=0.5, freq=5Hz, {N_STEP}step(초기) 고정 offset. (+offset=좌회전 키)")
print(f"  {'offset':>7}  {'추진(head전진 m)':>16}  {'회전율(deg/s)':>14}  {'net yaw(deg)':>13}")
print(f"  {'-'*52}")
rows = []
for o in OFFSETS:
    fwd, yr, yn = run(o)
    rows.append((o, fwd, yr, yn))
    note = "  ← 직진" if abs(o) < 1e-6 else ""
    print(f"  {o:>+7.2f}  {fwd:>16.4f}  {yr:>+14.2f}  {yn:>+13.1f}{note}")
# +/- 대칭쌍 회전율 비교
print(f"\n  === |offset| 같은 좌(+)/우(-) 회전율 비대칭 ===")
d = {o: yr for o, fwd, yr, yn in rows}
for mag in [0.15, 0.30, 0.45, 0.60]:
    p = d.get(round(mag, 2)); m = d.get(round(-mag, 2))
    if p is not None and m is not None:
        print(f"    |off|={mag}: 우(-) {m:+.2f}°/s  vs  좌(+) {p:+.2f}°/s   "
              f"(좌/우 회전 효율비 {abs(p)/(abs(m)+1e-9):.2f})")
