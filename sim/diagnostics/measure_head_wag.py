"""Sine ctrl 주입 시 head_yaw / v_world / slip_angle 시계열 측정.

질문: tail wagging이 sim에서 실제로 head를 진동시키는가?
- 0.5Hz, 3Hz, 6Hz sine ctrl 12s 주입
- yaw, vx, vy, head_dir vs v_world 각도 기록 후 PNG 출력
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib import font_manager

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
XML = REPO / "sim" / "rl_fish.xml"


def run_sine(freq_hz: float, duration_s: float = 12.0, amplitude: float = 1.0):
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    dt = model.opt.timestep  # 0.002s
    n_steps = int(duration_s / dt)

    t_arr = np.zeros(n_steps)
    yaw_arr = np.zeros(n_steps)
    vx_arr = np.zeros(n_steps)
    vy_arr = np.zeros(n_steps)
    slip_arr = np.zeros(n_steps)
    ctrl_arr = np.zeros(n_steps)

    for i in range(n_steps):
        t = i * dt
        ctrl = amplitude * np.sin(2 * np.pi * freq_hz * t)
        data.ctrl[0] = ctrl
        mujoco.mj_step(model, data)

        yaw = float(data.qpos[2])
        vx = float(data.qvel[0])
        vy = float(data.qvel[1])
        head_dir = np.array([-np.cos(yaw), np.sin(yaw)])
        v = np.array([vx, vy])
        v_norm = float(np.linalg.norm(v))
        if v_norm > 0.02:
            cos_s = np.clip(float(np.dot(head_dir, v / v_norm)), -1.0, 1.0)
            slip = float(np.arccos(cos_s))
        else:
            slip = np.nan
        t_arr[i] = t
        yaw_arr[i] = yaw
        vx_arr[i] = vx
        vy_arr[i] = vy
        slip_arr[i] = slip
        ctrl_arr[i] = ctrl

    return {"t": t_arr, "yaw": yaw_arr, "vx": vx_arr, "vy": vy_arr,
            "slip": slip_arr, "ctrl": ctrl_arr}


def analyze_steady(res, t_min=4.0, t_max=10.0):
    """초기 전이 제외 정상상태 통계."""
    mask = (res["t"] >= t_min) & (res["t"] <= t_max)
    yaw = res["yaw"][mask]
    slip = res["slip"][mask]
    return {
        "yaw_mean": float(np.mean(yaw)),
        "yaw_std": float(np.std(yaw)),
        "yaw_peak2peak_deg": float(np.degrees(yaw.max() - yaw.min())),
        "slip_mean_deg": float(np.degrees(np.nanmean(slip))),
        "slip_max_deg": float(np.degrees(np.nanmax(slip))),
        "v_mean": float(np.mean(np.sqrt(res["vx"][mask] ** 2 + res["vy"][mask] ** 2))),
    }


def main():
    freqs = [0.5, 3.0, 6.0]
    results = {}
    for f in freqs:
        print(f"=== sine {f} Hz ===")
        r = run_sine(f, duration_s=12.0)
        s = analyze_steady(r, t_min=4.0, t_max=10.0)
        print(f"  yaw peak-to-peak: {s['yaw_peak2peak_deg']:.2f}°")
        print(f"  yaw std:          {np.degrees(s['yaw_std']):.2f}°")
        print(f"  slip mean:        {s['slip_mean_deg']:.2f}°")
        print(f"  slip max:         {s['slip_max_deg']:.2f}°")
        print(f"  v mean:           {s['v_mean']:.4f} m/s")
        results[f] = (r, s)

    # 플롯
    fig, axes = plt.subplots(len(freqs), 3, figsize=(16, 4 * len(freqs)), dpi=120)
    for row, f in enumerate(freqs):
        r, s = results[f]
        # ctrl + yaw
        ax = axes[row, 0]
        ax.plot(r["t"], r["ctrl"], 'k-', alpha=0.4, label='ctrl')
        ax.set_ylabel("ctrl", color='k')
        ax2 = ax.twinx()
        ax2.plot(r["t"], np.degrees(r["yaw"]), 'r-', label='yaw [°]')
        ax2.set_ylabel("yaw [°]", color='r')
        ax.set_title(f"{f} Hz sine — ctrl & yaw")
        ax.set_xlabel("t [s]")
        # v_world
        ax = axes[row, 1]
        ax.plot(r["t"], r["vx"], label='vx')
        ax.plot(r["t"], r["vy"], label='vy')
        ax.set_title(f"{f} Hz — v_world")
        ax.set_xlabel("t [s]"); ax.set_ylabel("[m/s]")
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        # slip
        ax = axes[row, 2]
        ax.plot(r["t"], np.degrees(r["slip"]), 'm-')
        ax.axhline(15, color='g', ls='--', alpha=0.5, label='15° (자연 wag 가설)')
        ax.axhline(30, color='orange', ls='--', alpha=0.5, label='30°')
        ax.set_title(f"{f} Hz — slip_angle (head vs v)")
        ax.set_xlabel("t [s]"); ax.set_ylabel("[°]")
        ax.set_ylim(0, 180)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        # 통계 텍스트
        txt = (f"yaw pk-pk: {s['yaw_peak2peak_deg']:.1f}°\n"
               f"slip mean: {s['slip_mean_deg']:.1f}°\n"
               f"slip max:  {s['slip_max_deg']:.1f}°\n"
               f"|v| mean:  {s['v_mean']:.3f} m/s")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray'))

    fig.suptitle("Tail wagging이 head를 흔드는가? (정상상태 4~10s 분석)", fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPO / "images" / "head_wag_diagnosis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight', dpi=120)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
