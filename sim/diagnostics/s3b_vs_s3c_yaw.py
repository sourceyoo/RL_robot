"""s3b vs s3c — 머리 실제 회전(dyaw) vs sideslip(head_angle) 정량 비교 (read-only).

질문: s3b strict 통과 = 진짜 머리 틀기(yaw 회전)인가, 허용범위(40°) 안의 sideslip인가?
      s3c는 머리를 못 트나(yaw saturate), 아니면 s3b와 같은 게걸음인데 각도만 큰가?

측정: deterministic rollout. 좌(+y target)·우(-y target) 분리.
  - dyaw_final = 머리가 target쪽으로 실제 회전한 양 (deg, +면 target쪽)
  - slip_mean  = head_dir 와 velocity 각도 평균 (게걸음 정도)
  - course_mean= velocity·target 정렬 평균
모델: runs/m4_cpg_v24/seed0/{s3b_arc30,s3c_arc60}/model.zip
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from fish_env import FishSwimEnv

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT.parent / "images"


def rollout(model, offset_range, seed, ep_sec, target_radius):
    env = FishSwimEnv(episode_seconds=ep_sec, success_radius=0.08,
                      action_history_n=20, target_theta_offset_range=offset_range,
                      target_radius=target_radius)
    obs, _ = env.reset(seed=seed)
    yaw0 = float(env.data.qpos[2])
    tgt = env._target_pos()[:2].copy()
    xs, ys, dyaws, slips, courses = [], [], [], [], []
    reached = False
    done = False
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
        xy = env._torso_pos()[:2]
        yaw = float(env.data.qpos[2])
        hd = np.array([-np.cos(yaw), np.sin(yaw)])
        v = np.array([float(env.data.qvel[0]), float(env.data.qvel[1])])
        vn = np.linalg.norm(v)
        # 시계열 plot용 (env deadzone 0.02와 동일 조건으로 맞춤)
        if vn > 0.02:
            slip = np.degrees(np.arccos(np.clip(np.dot(hd, v / vn), -1, 1)))
        else:
            slip = np.nan
        xs.append(xy[0]); ys.append(xy[1])
        dyaws.append(yaw - yaw0); slips.append(slip)
        if env._distance_to_target() < 0.08:
            reached = True
    # 요약은 env 공식 strict 판정(info) 사용 — 정의 100% 일치
    course_avg = info.get("course_avg", 0.0)
    head_angle_avg = info.get("head_angle_avg", 90.0)
    success_strict = info.get("success_strict", False)
    # target쪽으로 튼 양: +y target이면 dyaw>0가 target쪽. sign(tgt_y)로 정렬.
    side = 1.0 if tgt[1] >= 0 else -1.0
    dyaw_toward = np.degrees(np.array(dyaws)) * side
    return dict(xs=np.array(xs), ys=np.array(ys),
                dyaw_toward=dyaw_toward, slips=np.array(slips),
                course_avg=course_avg, head_angle_avg=head_angle_avg,
                success_strict=success_strict, tgt=tgt, reached=reached,
                side="L" if side > 0 else "R")


def collect(tag, offset_range, ep_sec, target_radius, n_seed=12):
    mp = ROOT / "runs" / "m4_cpg_v24" / "seed0" / tag / "model.zip"
    model = SAC.load(str(mp), device="cpu")
    L, R = [], []
    for s in range(n_seed):
        r = rollout(model, offset_range, seed=1000 + s, ep_sec=ep_sec,
                    target_radius=target_radius)
        (L if r["side"] == "L" else R).append(r)
    return L, R


PI = np.pi
# s3b: |offset| 15~30° @0.5m, s3c: 30~60° @0.7m (학습 거리와 일치).
CONFIGS = [
    ("s3b_arc30", (PI / 12, PI / 6), 60.0, 0.5),   # 15~30°
    ("s3c_arc60", (PI / 6, PI / 3), 60.0, 0.7),    # 30~60°
]

print(f"{'model':10s} {'side':4s} {'n':>2s} {'tgt_off°':>8s} "
      f"{'dyaw_fin°':>9s} {'headang_μ°':>10s} {'course_μ':>8s} "
      f"{'reach':>6s} {'STRICT':>7s}")
print("-" * 78)

reps = {}  # (tag, side) -> 대표 rollout
for tag, off, ep, tr in CONFIGS:
    L, R = collect(tag, off, ep, tr)
    for side_name, group in [("L", L), ("R", R)]:
        if not group:
            print(f"{tag:10s} {side_name:4s}  0   (없음)")
            continue
        tgt_off = np.degrees([np.arctan2(abs(g["tgt"][1]), abs(g["tgt"][0])) for g in group])
        dyaw_fin = np.array([g["dyaw_toward"][-1] for g in group])
        head_m = np.array([g["head_angle_avg"] for g in group])  # env 공식 head_angle_avg
        course_m = np.array([g["course_avg"] for g in group])    # env 공식 course_avg
        reach = np.array([g["reached"] for g in group])
        strict = np.array([g["success_strict"] for g in group])
        print(f"{tag:10s} {side_name:4s} {len(group):2d} "
              f"{tgt_off.mean():8.1f} {dyaw_fin.mean():9.1f} "
              f"{head_m.mean():10.1f} {course_m.mean():8.3f} "
              f"{reach.mean():6.0%} {strict.mean():7.0%}")
        # 대표 = head_angle 중앙값 ep
        idx = int(np.argsort(head_m)[len(head_m) // 2])
        reps[(tag, side_name)] = group[idx]

# ---- plot: 각 (model, side) 대표의 trajectory + dyaw(t) + slip(t) ----
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
cols = [("s3b_arc30", "L"), ("s3b_arc30", "R"), ("s3c_arc60", "L"), ("s3c_arc60", "R")]
for c, key in enumerate(cols):
    r = reps.get(key)
    if r is None:
        continue
    tag, side = key
    t = np.arange(len(r["xs"])) * 0.084
    # row0: trajectory + 머리방향 화살표
    ax = axes[0][c]
    ax.plot(r["xs"], r["ys"], "-", color="tab:green" if r["reached"] else "tab:red", lw=1.5)
    ax.plot(0, 0, "k*", ms=12)
    ax.plot(r["tgt"][0], r["tgt"][1], "kx", ms=10)
    ax.set_title(f"{tag} {side} traj reach={r['reached']}")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(-0.9, 0.2); ax.set_ylim(-0.9, 0.9)
    # row1: dyaw(t) — 머리 실제 회전
    ax = axes[1][c]
    ax.plot(t, r["dyaw_toward"], "-", color="tab:blue", lw=1.8)
    tgt_off = np.degrees(np.arctan2(abs(r["tgt"][1]), abs(r["tgt"][0])))
    ax.axhline(tgt_off, ls="--", color="gray", label=f"target {tgt_off:.0f}°")
    ax.set_title(f"dyaw(t) head turn  final {r['dyaw_toward'][-1]:+.1f}°")
    ax.set_ylabel("dyaw toward target (°)"); ax.set_ylim(-20, 70)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # row2: slip(t) — sideslip
    ax = axes[2][c]
    ax.plot(t, r["slips"], "-", color="tab:orange", lw=1.5)
    ax.axhline(40, ls="--", color="red", label="strict 40°")
    ax.set_title(f"slip(t) head_angle  env avg {r['head_angle_avg']:.1f}° "
                 f"strict={r['success_strict']}")
    ax.set_xlabel("t (s)"); ax.set_ylabel("slip (°)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("s3b vs s3c — 머리 회전(dyaw) vs sideslip(slip). 머리 틀기 천장이면 s3c dyaw saturate.",
             fontsize=13)
fig.tight_layout()
out = IMG / "s3b_vs_s3c_yaw.png"
fig.savefig(out, dpi=110)
print(f"\n[saved] {out}")
