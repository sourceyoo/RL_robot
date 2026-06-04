"""m4_cpg_v12 계단식/게걸음 시계열 시각화 (read-only, 학습 trigger 아님).

사용자 요청('먼저 더 진단'): seed별 yaw(t)·v_fwd(t)·v_lat(t) 시계열로
parallel(매끄러운 yaw 증가 + v_fwd 항상 양수) vs 계단식(yaw 평탄→급증, 회전구간 v_fwd 0/음수)
vs 게걸음(yaw 거의 0, v_lat 큼) 을 정밀 분해.

3행(seed0/1/2) × 3열(경로+머리화살표 / yaw(t)·turn목표 / v_fwd·v_lat(t)).
대표 ep = 그 seed에서 머리회전 |Δyaw| 가 중앙값에 가장 가까운 도달 ep.
출력: /home/yoo/RL_robot/images/v12_staircase_timeseries.png
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
ACTION_HISTORY_N = 20
DT = 0.084
N_EP = 30
S3B_KW = dict(episode_seconds=60.0, success_radius=0.08, action_history_n=ACTION_HISTORY_N,
              target_theta_offset_range=(PI / 6, PI / 6))


def rollout(model, seed):
    env = FishSwimEnv(**S3B_KW)
    obs, _ = env.reset(seed=seed)
    yaw0 = float(env.data.qpos[2])
    tgt = env._target_pos()[:2].copy()
    xs, ys, yaws, vfwd, vlat = [], [], [], [], []
    prev_xy = env._torso_pos()[:2].copy()
    term = trunc = False
    reached = False
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        xy = env._torso_pos()[:2]
        yaw = float(env.data.qpos[2])
        hd = np.array([-np.cos(yaw), np.sin(yaw)])
        perp = np.array([-hd[1], hd[0]])
        d = (xy - prev_xy) / DT
        xs.append(xy[0]); ys.append(xy[1]); yaws.append(yaw)
        vfwd.append(float(np.dot(d, hd))); vlat.append(float(np.dot(d, perp)))
        prev_xy = xy.copy()
        reached = bool(info.get("reached", False))
        if reached:
            break
    env.close()
    return dict(xs=np.array(xs), ys=np.array(ys), yaws=np.array(yaws),
                vfwd=np.array(vfwd), vlat=np.array(vlat), yaw0=yaw0, tgt=tgt, reached=reached)


def pick_representative(model):
    """도달 ep 중 |Δyaw| 가 중앙값에 가장 가까운 ep 의 seed 반환."""
    cand = []
    for ep in range(N_EP):
        r = rollout(model, 6000 + ep)
        if r["reached"]:
            cand.append((6000 + ep, abs(r["yaws"][-1] - r["yaw0"])))
    if not cand:
        return None
    med = np.median([c[1] for c in cand])
    return min(cand, key=lambda c: abs(c[1] - med))[0]


fig, axes = plt.subplots(3, 3, figsize=(15, 12))
base = Path(__file__).parent.parent / "runs" / "m4_cpg_v12"
for row, s in enumerate((0, 1, 2)):
    mp = base / f"seed{s}" / "s3b_arc30" / "model_best.zip"
    if not mp.exists():
        for c in range(3):
            axes[row, c].set_title(f"seed{s}: 모델 없음")
        continue
    model = SAC.load(str(mp), device="cpu")
    ep = pick_representative(model)
    if ep is None:
        for c in range(3):
            axes[row, c].set_title(f"seed{s}: 도달 ep 없음")
        continue
    r = rollout(model, ep)
    t = np.arange(len(r["xs"])) * DT
    dyaw_deg = np.degrees(r["yaws"] - r["yaw0"])
    tgt_ang = np.degrees(np.arctan2(r["tgt"][1], -r["tgt"][0]))  # 머리가 돌아야 할 목표각

    # 1열: 경로 + 머리방향 화살표
    ax = axes[row, 0]
    ax.plot(r["xs"], r["ys"], "-", color="tab:green", lw=1.2)
    step = max(1, len(r["xs"]) // 18)
    for i in range(0, len(r["xs"]), step):
        hd = np.array([-np.cos(r["yaws"][i]), np.sin(r["yaws"][i])]) * 0.03
        ax.arrow(r["xs"][i], r["ys"][i], hd[0], hd[1], head_width=0.008, color="navy", alpha=0.7)
    ax.plot(0, 0, "k*", ms=14); ax.plot(r["tgt"][0], r["tgt"][1], "rx", ms=12, mew=3)
    ax.set_aspect("equal"); ax.set_title(f"seed{s} 경로 (화살표=머리방향)")
    ax.set_xlim(-0.55, 0.1); ax.grid(alpha=0.3)

    # 2열: yaw(t) — 매끄러운 단조 vs 계단
    ax = axes[row, 1]
    ax.plot(t, dyaw_deg, "-", color="tab:blue", lw=1.8)
    ax.axhline(tgt_ang, ls="--", color="gray", label=f"목표 {tgt_ang:+.0f}°")
    ax.set_title(f"seed{s} 머리각 Δyaw(t)  최종 {dyaw_deg[-1]:+.1f}°")
    ax.set_xlabel("t (s)"); ax.set_ylabel("Δyaw (도)"); ax.legend(); ax.grid(alpha=0.3)

    # 3열: v_fwd, v_lat(t) — 회전 중 멈춤(계단식) 여부
    ax = axes[row, 2]
    ax.plot(t, r["vfwd"], "-", color="tab:green", lw=1.5, label="v_fwd (전진)")
    ax.plot(t, r["vlat"], "-", color="tab:red", lw=1.2, label="v_lat (측면)")
    ax.axhline(0, ls=":", color="k", lw=0.8)
    ax.set_title(f"seed{s} 속도(t)  v_fwd<0 구간=멈춤/후진")
    ax.set_xlabel("t (s)"); ax.set_ylabel("속도 (m/s)"); ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("v12 s3b(±30°) 계단식/게걸음 분해 — 대표 ep (|Δyaw| 중앙값)\n"
             "parallel = yaw 매끄러운 증가 + v_fwd 항상>0  /  계단식 = yaw 평탄→급증 + 회전구간 v_fwd≤0  /  게걸음 = yaw≈0 + v_lat 큼",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("/home/yoo/RL_robot/images/v12_staircase_timeseries.png")
fig.savefig(out, dpi=110)
print(f"저장: {out}")
