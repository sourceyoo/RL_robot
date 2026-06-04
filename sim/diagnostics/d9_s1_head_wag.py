"""s1_forward 정책에서 머리 yaw·slip_deg 시간 trace 측정.

목적: 직진 s1 에서 slip_μ=24° 가 CPG sine wag 인지, 다른 패턴인지 확인.
출력: /home/yoo/RL_robot/images/d9_s1_head_wag.png
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stable_baselines3 import SAC

from fish_env import FishSwimEnv, IDX_X, IDX_Y, IDX_YAW

PI = math.pi


def rollout_trace(model_path: Path, seed: int = 0, n_ep: int = 3):
    env = FishSwimEnv(
        episode_seconds=20.0,
        target_theta_range=(PI, PI),
        success_radius=0.08,
        action_history_n=20,
    )
    model = SAC.load(str(model_path), device="cpu")

    eps = []
    for ep in range(n_ep):
        obs, _ = env.reset(seed=seed + ep)
        ts, yaws, slips, vlats, vlons, head_yerrs = [], [], [], [], [], []
        done = False
        t = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            yaw = float(env.data.qpos[IDX_YAW])
            head_dir = np.array([-math.cos(yaw), math.sin(yaw)])
            v = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
            vnorm = float(np.linalg.norm(v))
            if vnorm > 1e-6:
                v_unit = v / vnorm
                v_lon = float(np.dot(v_unit, head_dir)) * vnorm     # 전후
                # lateral: head_dir 의 수직축 (head 좌우)
                perp = np.array([-head_dir[1], head_dir[0]])
                v_lat = float(np.dot(v_unit, perp)) * vnorm
            else:
                v_lon = v_lat = 0.0
            ts.append(t)
            yaws.append(yaw)
            slips.append(float(info.get("slip_deg", 0.0)))
            vlats.append(v_lat)
            vlons.append(v_lon)
            # head_dir 의 world x축 (target=PI 방향=-x) 과 각도 — 진행 방향에서 머리 흔들림
            head_yerrs.append(math.degrees(math.atan2(head_dir[1], -head_dir[0])))
            t += env.dt
        eps.append({
            "t": np.array(ts), "yaw": np.array(yaws), "slip": np.array(slips),
            "vlat": np.array(vlats), "vlon": np.array(vlons),
            "head_yerr": np.array(head_yerrs),
        })
    return eps


def main():
    model_path = ROOT / "runs/m4_cpg_v4/seed0/s1_forward/model.zip"
    if not model_path.exists():
        print(f"missing: {model_path}", file=sys.stderr)
        sys.exit(1)
    eps = rollout_trace(model_path, seed=0, n_ep=3)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    colors = ["C0", "C1", "C2"]
    for i, ep in enumerate(eps):
        c = colors[i]
        axes[0].plot(ep["t"], ep["head_yerr"], color=c, alpha=0.8, label=f"ep{i}")
        axes[1].plot(ep["t"], ep["slip"], color=c, alpha=0.8)
        axes[2].plot(ep["t"], ep["vlat"], color=c, alpha=0.8)
        axes[3].plot(ep["t"], ep["vlon"], color=c, alpha=0.8)

    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("head yaw err (deg)\n[target=-x 방향]")
    axes[0].set_title("s1_forward 직진 시 머리·slip 시간 trace (seed0 v4)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel("slip_deg\n= angle(head, v)")
    axes[1].axhline(24, color="r", lw=0.5, ls="--", label="평균 24°")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel("v_lateral (m/s)\n[head 좌우 속도]")
    axes[2].grid(alpha=0.3)

    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_ylabel("v_longitudinal (m/s)\n[head 전후 속도]")
    axes[3].set_xlabel("time (s)")
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    out = Path("/home/yoo/RL_robot/images/d9_s1_head_wag.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"saved {out}")

    # FFT 한 ep 의 yaw 진동 frequency
    ep = eps[0]
    yaw_centered = ep["yaw"] - np.mean(ep["yaw"])
    dt = ep["t"][1] - ep["t"][0] if len(ep["t"]) > 1 else 0.084
    fft = np.fft.rfft(yaw_centered)
    freqs = np.fft.rfftfreq(len(yaw_centered), d=dt)
    peak_idx = np.argmax(np.abs(fft[1:])) + 1
    peak_freq = freqs[peak_idx]
    print(f"yaw FFT peak: {peak_freq:.2f} Hz (CPG carrier 와 비교: 4~6 Hz)")
    print(f"yaw range: [{np.degrees(np.min(ep['yaw'])):.1f}, {np.degrees(np.max(ep['yaw'])):.1f}] deg")
    print(f"head_yerr stats: mean={np.mean(ep['head_yerr']):+.2f}, std={np.std(ep['head_yerr']):.2f}, "
          f"range=[{np.min(ep['head_yerr']):+.1f}, {np.max(ep['head_yerr']):+.1f}] deg")
    print(f"slip_deg stats: mean={np.mean(ep['slip']):.2f}, std={np.std(ep['slip']):.2f}, "
          f"max={np.max(ep['slip']):.1f} deg")
    print(f"v_lat stats: mean={np.mean(ep['vlat']):+.4f}, std={np.std(ep['vlat']):.4f} m/s")
    print(f"v_lon stats: mean={np.mean(ep['vlon']):+.4f}, std={np.std(ep['vlon']):.4f} m/s")


if __name__ == "__main__":
    main()
