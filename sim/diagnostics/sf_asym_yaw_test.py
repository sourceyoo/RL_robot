"""m4_v17 sf_asym ↔ yaw 회전 방향 매핑 검증.

env.step()으로 sf_asym sustained 주입 12초 → x, y, yaw 시계열·궤적 측정.
sf_asym ∈ {-0.5, 0, +0.5} 3 cases.

질문: sf_asym>0이 head를 어느 방향으로 회전시키나? (yaw_sign reward 부호 검증용)
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fish_env import FishSwimEnv  # noqa: E402

# 한글 폰트
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
    try:
        prop = font_manager.FontProperties(fname=fp)
        mpl.rcParams['font.family'] = prop.get_name()
        break
    except Exception:
        continue
mpl.rcParams['axes.unicode_minus'] = False

REPO = Path(__file__).resolve().parents[2]


def run(sf_asym: float, duration: float = 12.0):
    env = FishSwimEnv(episode_seconds=duration + 0.01,
                       target_theta_range=(np.pi, np.pi),  # 직진 (target 무관, no truncate)
                       success_radius=0.0)  # 도달 안 함
    env.reset(seed=0)

    n_steps = int(duration / env.dt)
    t_arr = np.zeros(n_steps)
    x_arr = np.zeros(n_steps)
    y_arr = np.zeros(n_steps)
    yaw_arr = np.zeros(n_steps)

    action = np.array([sf_asym], dtype=np.float32)
    for i in range(n_steps):
        env.step(action)
        t_arr[i] = env.data.time
        x_arr[i] = float(env.data.qpos[0])
        y_arr[i] = float(env.data.qpos[1])
        yaw_arr[i] = float(env.data.qpos[2])

    env.close()
    return {"t": t_arr, "x": x_arr, "y": y_arr, "yaw": yaw_arr}


def main():
    cases = [(-0.5, "sf_asym = -0.5 (−쪽 slow)"),
             (0.0,  "sf_asym = 0 (대칭, 직진 기대)"),
             (+0.5, "sf_asym = +0.5 (+쪽 slow)")]
    results = {}
    print(f"{'sf_asym':>8} | {'x_disp':>9} {'y_disp':>9} {'yaw_disp[°]':>13} | 회전 방향")
    print("-" * 75)
    for sf, label in cases:
        r = run(sf, duration=12.0)
        x_d = r["x"][-1] - r["x"][0]
        y_d = r["y"][-1] - r["y"][0]
        yaw_d = np.degrees(r["yaw"][-1] - r["yaw"][0])
        if abs(yaw_d) < 5:
            verdict = "직진 (회전 거의 없음)"
        elif yaw_d > 0:
            verdict = f"yaw +방향 (head +y쪽)"
        else:
            verdict = f"yaw −방향 (head −y쪽)"
        print(f"{sf:+8.2f} | {x_d:+9.4f} {y_d:+9.4f} {yaw_d:+13.2f} | {verdict}")
        results[sf] = (r, label)

    # 플롯
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    colors = {-0.5: 'tab:red', 0.0: 'tab:gray', 0.5: 'tab:blue'}
    # 궤적 (xy)
    ax = axes[0]
    for sf, (r, label) in results.items():
        ax.plot(r["x"], r["y"], color=colors[sf], lw=2, label=label)
        ax.plot(r["x"][0], r["y"][0], 'o', color=colors[sf], ms=8)
        ax.plot(r["x"][-1], r["y"][-1], 's', color=colors[sf], ms=8)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("궤적 (xy)  ● 시작  ■ 끝   world −x = 머리(전진)")
    ax.axhline(0, color='k', alpha=0.2, lw=0.5); ax.axvline(0, color='k', alpha=0.2, lw=0.5)
    ax.set_aspect('equal'); ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)
    # yaw 시계열
    ax = axes[1]
    for sf, (r, label) in results.items():
        ax.plot(r["t"], np.degrees(r["yaw"]), color=colors[sf], lw=2, label=label)
    ax.set_xlabel("t [s]"); ax.set_ylabel("yaw [°]")
    ax.set_title("yaw 시계열  (+ = head +y, − = head −y)")
    ax.axhline(0, color='k', alpha=0.2, lw=0.5)
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=9)

    fig.suptitle("m4_v17: sf_asym → yaw 회전 방향 매핑 (12s sustained)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPO / "images" / "sf_asym_yaw_mapping.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=120)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
