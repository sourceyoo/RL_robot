"""논의1: 단일 모터 CPG로 '소각 선회 + 직진(parallel)'이 물리적으로 가능한가 (read-only).

정책 없이 고정 action 직접 주입. action=[freq_norm, amp_norm, offset], freq[4,6] amp[0,1] offset[-1,1].
질문: offset(선회 bias)을 주면 머리가 도는데, 진행방향도 머리방향과 정렬(sideslip 작음)되나?
 A 직진 baseline(offset=0): sideslip 작고 직진? (sanity)
 B 정상 선회(offset 고정): 머리 회전율·sideslip·선회반경. sideslip 작으면 '선회=parallel arc' 물리OK.
 C 펄스(offset 잠깐→0): 틀고 나서 머리방향=진행방향 직진되나. parallel 본질.
출력: /home/yoo/RL_robot/images/cpg_offset_physics.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

DT = 0.084
AMP_N = 0.4   # amp=0.7
FREQ_N = 0.0  # 5Hz


def run(action_seq):
    """action_seq: (offset 시퀀스). freq/amp 고정. pos(x,y)·yaw 시계열 반환."""
    env = FishSwimEnv(episode_seconds=600.0, success_radius=0.001, action_history_n=20)
    env.reset(seed=0)
    xs, ys, yaws = [], [], []
    p0 = env._torso_pos()[:2].copy()
    for off in action_seq:
        env.step(np.array([FREQ_N, AMP_N, off], dtype=np.float32))
        xy = env._torso_pos()[:2]
        xs.append(xy[0] - p0[0]); ys.append(xy[1] - p0[1])
        yaws.append(float(env.data.qpos[2]))
    env.close()
    return np.array(xs), np.array(ys), np.array(yaws)


def fwd_ang(vx, vy):  # 전진(-x)=0°, 좌회전 +
    return np.degrees(np.arctan2(vy, -vx))


def metrics(xs, ys, yaws, seg=None):
    """seg=(i0,i1) 구간. 머리회전, net 진행방향, sideslip(머리-진행), 전진거리."""
    i0, i1 = seg if seg else (0, len(xs) - 1)
    dyaw = np.degrees(yaws[i1] - yaws[i0])
    dx, dy = xs[i1] - xs[i0], ys[i1] - ys[i0]
    disp = np.hypot(dx, dy)
    vel_ang = fwd_ang(dx, dy) if disp > 1e-4 else float("nan")
    head_ang = np.degrees(yaws[i1] - yaws[0])  # 시작 대비 머리각
    # step별 instantaneous sideslip (꼬리질 진동 포함) 평균 + net
    vxx = np.diff(xs[i0:i1 + 1]); vyy = np.diff(ys[i0:i1 + 1])
    yy = yaws[i0:i1 + 1][1:]
    hd = np.stack([-np.cos(yy), np.sin(yy)], 1)
    v = np.stack([vxx, vyy], 1); vn = np.hypot(vxx, vyy)
    mv = vn > 1e-5
    inst_slip = np.nan
    if mv.any():
        cos = (v[mv] * hd[mv]).sum(1) / vn[mv]
        inst_slip = float(np.degrees(np.mean(np.arccos(np.clip(cos, -1, 1)))))
    # net sideslip: net 진행방향 vs 구간 끝 머리방향
    head_dir_ang = fwd_ang(-np.cos(yaws[i1]), np.sin(yaws[i1]))
    net_slip = abs(vel_ang - head_dir_ang) if disp > 1e-4 else float("nan")
    return dict(dyaw=dyaw, disp=disp, vel_ang=vel_ang, head_ang=head_ang,
                inst_slip=inst_slip, net_slip=net_slip)


N = 120  # ~10s
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# A 직진 baseline
xa, ya, ywa = run([0.0] * N)
mA = metrics(xa, ya, ywa)
print(f"[A 직진 offset=0]  머리회전 {mA['dyaw']:+.1f}°  전진 {mA['disp']:.3f}m  "
      f"진행방향 {mA['vel_ang']:+.1f}°  inst_sideslip {mA['inst_slip']:.1f}°  net_sideslip {mA['net_slip']:.1f}°")

# B 정상 선회 (offset sweep)
print("[B 정상 선회 — offset 고정]")
for off in (0.2, 0.4, 0.6, 0.8, 1.0):
    xb, yb, ywb = run([off] * N)
    mB = metrics(xb, yb, ywb)
    # 선회반경: net disp 와 dyaw 로 호 근사 R = disp / (2 sin(dyaw/2))
    dyr = np.radians(abs(mB['dyaw']))
    R = mB['disp'] / (2 * np.sin(dyr / 2)) if dyr > 1e-3 else float("inf")
    print(f"  offset={off:.1f}: 머리회전 {mB['dyaw']:+6.1f}°  전진 {mB['disp']:.3f}m  "
          f"inst_slip {mB['inst_slip']:5.1f}°  net_slip {mB['net_slip']:5.1f}°  선회반경≈{R:.3f}m")
    axes[1].plot(xb, yb, label=f"off={off:.1f} ({mB['dyaw']:+.0f}°)")

# C 펄스 (offset 0.6 잠깐 → 0 직진)
PULSE = 20  # ~1.7s
seq = [0.6] * PULSE + [0.0] * (N - PULSE)
xc, yc, ywc = run(seq)
mC_turn = metrics(xc, yc, ywc, seg=(0, PULSE))
mC_str = metrics(xc, yc, ywc, seg=(PULSE, N - 1))
print(f"[C 펄스 offset=0.6×{PULSE} → 0]")
print(f"  펄스구간(틀기):  머리회전 {mC_turn['dyaw']:+.1f}°  net_slip {mC_turn['net_slip']:.1f}°")
print(f"  직진구간(틀고나서): 머리회전 {mC_str['dyaw']:+.1f}°(추가)  진행방향 {mC_str['vel_ang']:+.1f}°  "
      f"머리방향 {fwd_ang(-np.cos(ywc[-1]), np.sin(ywc[-1])):+.1f}°  inst_slip {mC_str['inst_slip']:.1f}°  net_slip {mC_str['net_slip']:.1f}°")
print("  → 직진구간 net_slip 작으면(머리방향≈진행방향): 틀고 직진 가능 = parallel 물리 OK = 보상문제.")
print("    net_slip 크면: 단일모터로 '틀고 직진' 불가 = 물리 한계.")

# A·C 궤적
for xs, ys, yws, ax, ttl in [(xa, ya, ywa, axes[0], "A 직진 offset=0"),
                              (xc, yc, ywc, axes[2], f"C 펄스 off0.6×{PULSE}→0")]:
    ax.plot(xs, ys, "-", color="tab:green", lw=1.4)
    st = max(1, len(xs) // 18)
    for i in range(0, len(xs), st):
        hd = np.array([-np.cos(yws[i]), np.sin(yws[i])]) * 0.02
        ax.arrow(xs[i], ys[i], hd[0], hd[1], head_width=0.006, color="navy", alpha=0.7)
    ax.plot(0, 0, "k*", ms=13); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(ttl)
axes[1].set_aspect("equal"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
axes[1].set_title("B 정상 선회 (offset 고정)")
fig.suptitle("논의1: 단일모터 CPG로 소각 선회+직진(parallel) 물리 가능성. 화살표=머리방향", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/home/yoo/RL_robot/images/cpg_offset_physics.png"
fig.savefig(out, dpi=110)
print("저장:", out)
