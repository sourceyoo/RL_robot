"""m4_cpg_v13 시계열 (read-only). s3a·s3b 대표 ep 의 yaw(t)·v_fwd(t)·v_lat(t).
과회전(yaw 목표 초과) + 계단식(회전구간 v_fwd≤0) 시각 확인.
출력: /home/yoo/RL_robot/images/v13_timeseries.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv

PI = np.pi
DT = 0.084
N_EP = 30
STAGES = [("s3a_arc15", (PI / 12, PI / 12), 15), ("s3b_arc30", (PI / 6, PI / 6), 30)]


def rollout(model, off, seed):
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08,
                      action_history_n=20, target_theta_offset_range=off)
    obs, _ = env.reset(seed=seed)
    yaw0 = float(env.data.qpos[2])
    tgt = env._target_pos()[:2].copy()
    xs, ys, yaws, vfwd, vlat = [], [], [], [], []
    prev = env._torso_pos()[:2].copy()
    term = trunc = reached = False
    while not (term or trunc):
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        xy = env._torso_pos()[:2]; yaw = float(env.data.qpos[2])
        hd = np.array([-np.cos(yaw), np.sin(yaw)]); perp = np.array([-hd[1], hd[0]])
        d = (xy - prev) / DT
        xs.append(xy[0]); ys.append(xy[1]); yaws.append(yaw)
        vfwd.append(float(np.dot(d, hd))); vlat.append(float(np.dot(d, perp)))
        prev = xy.copy(); reached = bool(info.get("reached", False))
        if reached:
            break
    env.close()
    return dict(xs=np.array(xs), ys=np.array(ys), yaws=np.array(yaws),
                vfwd=np.array(vfwd), vlat=np.array(vlat), yaw0=yaw0, tgt=tgt, reached=reached)


def pick(model, off):
    cand = []
    for ep in range(N_EP):
        r = rollout(model, off, 6000 + ep)
        if r["reached"]:
            cand.append((6000 + ep, abs(r["yaws"][-1] - r["yaw0"])))
    if not cand:
        return None
    med = np.median([c[1] for c in cand])
    return min(cand, key=lambda c: abs(c[1] - med))[0]


CARD = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v13"
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
base = Path(__file__).parent.parent / "runs" / CARD / "seed0"
for row, (stage, off, tgt_deg) in enumerate(STAGES):
    mp = base / stage / "model_best.zip"
    model = SAC.load(str(mp), device="cpu")
    ep = pick(model, off)
    r = rollout(model, off, ep)
    t = np.arange(len(r["xs"])) * DT
    dyaw = np.degrees(r["yaws"] - r["yaw0"])
    tgt_ang = np.degrees(np.arctan2(r["tgt"][1], -r["tgt"][0]))

    ax = axes[row, 0]
    ax.plot(r["xs"], r["ys"], "-", color="tab:green", lw=1.2)
    step = max(1, len(r["xs"]) // 18)
    for i in range(0, len(r["xs"]), step):
        hd = np.array([-np.cos(r["yaws"][i]), np.sin(r["yaws"][i])]) * 0.03
        ax.arrow(r["xs"][i], r["ys"][i], hd[0], hd[1], head_width=0.008, color="navy", alpha=0.7)
    ax.plot(0, 0, "k*", ms=14); ax.plot(r["tgt"][0], r["tgt"][1], "rx", ms=12, mew=3)
    ax.set_aspect("equal"); ax.set_title(f"{stage} path (arrow=head dir)")
    ax.grid(alpha=0.3)

    ax = axes[row, 1]
    ax.plot(t, dyaw, "-", color="tab:blue", lw=1.8)
    ax.axhline(tgt_ang, ls="--", color="gray", label=f"target {tgt_ang:+.0f}deg")
    ax.set_title(f"{stage} head dyaw(t)  final {dyaw[-1]:+.1f}deg (overshoot)")
    ax.set_xlabel("t (s)"); ax.set_ylabel("dyaw (deg)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[row, 2]
    ax.plot(t, r["vfwd"], "-", color="tab:green", lw=1.5, label="v_fwd")
    ax.plot(t, r["vlat"], "-", color="tab:red", lw=1.2, label="v_lat")
    ax.axhline(0, ls=":", color="k", lw=0.8)
    ax.set_title(f"{stage} speed(t)  v_fwd<0 = stop/back")
    ax.set_xlabel("t (s)"); ax.set_ylabel("speed (m/s)"); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("v13 seed0 — head overshoot + stop-and-go check (parallel = smooth yaw + v_fwd>0 always)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path(f"/home/yoo/RL_robot/images/{CARD}_timeseries.png")
fig.savefig(out, dpi=110)
print(f"저장: {out}")
