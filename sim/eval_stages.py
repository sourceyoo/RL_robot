"""모델 한 개를 모든 stage 분포에서 평가 — catastrophic forgetting 측정.

수치 평가 외에 처음 --plot-episodes (default 20) trajectory를 plot으로 저장.
plot 디렉토리: sim/plots/<카드명>/<학습stage>/eval_<YYYYMMDD_HHMMSS>/trajectories_<stage>.png
  (카드명·학습stage = runs/<카드명>/<학습stage>/model.zip 구조에서 추출)

분포 (m4_v17, s2 제거 후): s1은 π 고정. s3a~d는 annular disjoint sampling (|θ-π| ∈ [inner, outer]):
  s3a [9.2°, 15°], s3b [15°, 30°], s3c [30°, 60°], s3d [60°, 90°].
  inner 9.2° = arcsin(0.08/0.5) = 직진 자연 한계. 우연 도달 제거 + 인접 disjoint.

사용:
    python3 sim/eval_stages.py <model.zip> [--episodes 100] [--plot-episodes 20]
"""
import argparse
import datetime
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless (cv2 Qt 플러그인 충돌 회피)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent))
from fish_env import FishSwimEnv

matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

PI = math.pi
ACTION_HISTORY_N = 20

STAGES = [
    # ep_sec은 curriculum.py STAGES와 sync (m4 환경: s1 20s, s3a~d 60s; m4_v17에서 s2 제거).
    # s3a~d는 annular: |θ-π| ∈ (offset_min, offset_max). inner = 9.2° (자연 한계 arcsin(0.08/0.5)).
    # 직진 우연 도달률 제거 + 인접 stage disjoint로 각 stage 고유 회전만 검증.
    {"tag": "s1_forward", "theta": (PI, PI),                                       "radius": 0.08, "ep_sec": 20.0},
    # m4_v17: s2_anchor 제거 (amp 고정으로 정밀 정지 학습 불가능, 의미 없는 stage).
    {"tag": "s3a_arc15",  "offset": (PI * 9.2 / 180.0, PI/12),                     "radius": 0.08, "ep_sec": 60.0},
    {"tag": "s3b_arc30",  "offset": (PI/12,            PI/6),                      "radius": 0.08, "ep_sec": 60.0},
    {"tag": "s3c_arc60",  "offset": (PI/6,             PI/3),                      "radius": 0.08, "ep_sec": 60.0},
    {"tag": "s3d_arc90",  "offset": (PI/3,             PI/2),                      "radius": 0.08, "ep_sec": 60.0},
]

def plot_stage(ax, stage, episodes):
    """plot_trajectories.py와 동일한 trajectory plot logic (circular import 회피 위해 복제)."""
    n = len(episodes)
    n_reach = sum(1 for e in episodes if e["reached"])
    ax.set_title(f"{stage['tag']} — reach {n_reach}/{n} "
                 f"({100*n_reach/max(1,n):.0f}%)")
    seen = set()
    for e in episodes:
        key = (round(e["target"][0], 4), round(e["target"][1], 4))
        if key in seen:
            continue
        seen.add(key)
        ax.add_patch(Circle(e["target"], stage["radius"],
                            facecolor="none", edgecolor="0.7", lw=0.8, ls="--"))
        ax.plot(e["target"][0], e["target"][1], marker="x",
                color="0.4", ms=6, mew=1.2)
    # 점은 매 SAMPLE_EVERY step (~1초, m4_v17 frame_skip=125·dt=0.25), 헤딩 화살표 동반.
    # world −x = 머리 규약 → head_dir = (-cos yaw, sin yaw).
    SAMPLE_EVERY = 4
    for e in episodes:
        color = "tab:green" if e["reached"] else "tab:red"
        alpha = 0.55 if e["reached"] else 0.85
        idx = slice(0, None, SAMPLE_EVERY)
        xi = e["xs"][idx]
        yi = e["ys"][idx]
        yawi = e["yaws"][idx]
        hx = -np.cos(yawi)
        hy = np.sin(yawi)
        ax.scatter(xi, yi, color=color, s=8, alpha=alpha, edgecolors="none")
        ax.quiver(xi, yi, hx, hy, color=color, alpha=alpha,
                  scale=35, width=0.003, headwidth=4, headlength=5, pivot="mid")
        ax.plot(e["xs"][-1], e["ys"][-1], marker="o",
                color=color, ms=5, alpha=0.95, mec="white", mew=0.5)
    ax.plot(0, 0, marker="*", color="black", ms=12, mec="white", mew=0.6, zorder=5)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color="0.85", lw=0.5, zorder=0)


def eval_stage(model, stage, n_episodes, seed, n_plot):
    env_kwargs = dict(
        episode_seconds=stage["ep_sec"],
        success_radius=stage["radius"],
        action_history_n=ACTION_HISTORY_N,
    )
    if "offset" in stage:
        env_kwargs["target_theta_offset_range"] = stage["offset"]
    else:
        env_kwargs["target_theta_range"] = stage["theta"]
    env = FishSwimEnv(**env_kwargs)
    reaches = 0
    aligns = []
    final_dists = []
    ep_seconds = []
    slip_means = []
    slip_maxes = []
    plot_episodes = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        ep_align_sum = 0.0
        ep_slip_sum = 0.0
        ep_slip_max = 0.0
        ep_step = 0
        last_dist = None
        record = ep < n_plot
        if record:
            target_xy = env._target_pos()[:2].copy()
            xs = [float(env._torso_pos()[0])]
            ys = [float(env._torso_pos()[1])]
            yaws = [float(env.data.qpos[2])]  # IDX_YAW=2
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            ep_align_sum += float(info.get("align", 0.0))
            slip = float(info.get("slip_deg", 0.0))
            ep_slip_sum += slip
            if slip > ep_slip_max:
                ep_slip_max = slip
            ep_step += 1
            last_dist = float(info.get("distance", 0.0))
            if record:
                pos = env._torso_pos()
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
                yaws.append(float(env.data.qpos[2]))
        reached = bool(info.get("reached", False))
        if reached:
            reaches += 1
        aligns.append(ep_align_sum / max(1, ep_step))
        slip_means.append(ep_slip_sum / max(1, ep_step))
        slip_maxes.append(ep_slip_max)
        final_dists.append(last_dist if last_dist is not None else 0.0)
        ep_seconds.append(ep_step * env.dt)
        if record:
            plot_episodes.append({
                "xs": np.array(xs),
                "ys": np.array(ys),
                "yaws": np.array(yaws),
                "target": target_xy,
                "reached": reached,
            })
    env.close()
    return {
        "reach_rate":     reaches / n_episodes,
        "avg_align":      float(np.mean(aligns)),
        "final_dist":     float(np.mean(final_dists)),
        "ep_seconds":     float(np.mean(ep_seconds)),
        "slip_mean_deg":  float(np.mean(slip_means)),
        "slip_max_deg":   float(np.max(slip_maxes)),
        "slip_std_deg":   float(np.std(slip_means)),
    }, plot_episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=str, help="model.zip path")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-episodes", type=int, default=20,
                   help="처음 N ep trajectory plot 저장 (0 = plot 안 함)")
    p.add_argument("--only-stage", type=str, default=None,
                   help="특정 stage tag만 평가 (e.g. s1_forward)")
    p.add_argument("--include-future", action="store_true",
                   help="학습 안 한 stage까지 전체 평가 (generalization). default는 학습 stage까지만")
    args = p.parse_args()

    model_path = Path(args.model).resolve()
    # runs/<카드명>/<학습stage>/model.zip 또는 runs/<카드명>/<seedN>/<학습stage>/model.zip 모두 지원.
    # card_name = runs/ 이후 모델 부모 path 전체 (예: "m4_cpg_v2/seed0").
    sim_root = Path(__file__).resolve().parent
    runs_root = sim_root / "runs"
    try:
        card_name = str(model_path.parent.parent.relative_to(runs_root))
    except ValueError:
        # runs/ 밖에 있는 모델 호출 시 fallback
        card_name = model_path.parent.parent.name
    train_stage = model_path.parent.name

    # stages 결정: --only-stage > --include-future > default (train_stage까지)
    if args.only_stage is not None:
        stages = [s for s in STAGES if s["tag"] == args.only_stage]
        if not stages:
            raise SystemExit(f"unknown stage: {args.only_stage}. options: {[s['tag'] for s in STAGES]}")
    elif args.include_future:
        stages = STAGES
    else:
        idx = next((i for i, s in enumerate(STAGES) if s["tag"] == train_stage), None)
        if idx is None:
            print(f"[warn] train_stage '{train_stage}'를 STAGES에서 못 찾음 — 전체 평가 fallback")
            stages = STAGES
        else:
            stages = STAGES[:idx + 1]
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")     # 파일명
    timestamp_human = now.strftime("%Y-%m-%d %H:%M:%S")  # plot 표시용
    # plot 저장 구조: plots/<card>/<eval_stage>/{ts}_model_{train_stage}.png
    # 폴더 = 평가 대상 stage. 같은 stage의 여러 시점·모델 비교 누적 가능.
    plots_root = sim_root / "plots" / card_name
    if args.plot_episodes > 0:
        plots_root.mkdir(parents=True, exist_ok=True)

    model = SAC.load(str(model_path), device="cpu")
    print(f"\n[eval] model={args.model}, n_ep={args.episodes}, seed_base={args.seed}")
    if args.plot_episodes > 0:
        print(f"[plot] out_dir={plots_root}/<eval_stage>/{timestamp}_model_{train_stage}.png "
              f"(first {args.plot_episodes} ep)")
    print(f"\n{'stage':<12} {'reach':>7} {'align':>7} {'fdist':>7} {'ep_s':>6} "
          f"{'slip_μ':>7} {'slip_max':>9} {'slip_σ':>7}")
    print("-" * 72)
    for stage in stages:
        r, plot_eps = eval_stage(model, stage, args.episodes, args.seed, args.plot_episodes)
        print(f"{stage['tag']:<12} {r['reach_rate']*100:>6.1f}% "
              f"{r['avg_align']:>+7.3f} {r['final_dist']:>7.3f} "
              f"{r['ep_seconds']:>6.2f} "
              f"{r['slip_mean_deg']:>6.1f}° {r['slip_max_deg']:>8.1f}° "
              f"{r['slip_std_deg']:>6.2f}")
        if plot_eps:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            plot_stage(ax, {"tag": stage["tag"], "radius": stage["radius"]}, plot_eps)
            subtitle = ("world −x = 머리(전진). ★ 시작 (0,0), × target, "
                        "원 = success_radius. 초록=성공, 빨강=실패.")
            fig.suptitle(f"Trajectories — {model_path.name} · {stage['tag']} "
                         f"({len(plot_eps)} ep) · {timestamp_human}\n{subtitle}",
                         fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stage_dir = plots_root / stage["tag"]
            stage_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(stage_dir / f"{timestamp}_model_{train_stage}.png", dpi=120)
            plt.close(fig)


if __name__ == "__main__":
    main()
