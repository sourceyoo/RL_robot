"""약방향에서 정책이 action(freq/amp/offset)을 어떻게 쓰는가 (read-only, 학습 trigger 아님).

질문: 약방향 도달 0% — 정책이 약방향에서 "머리만 돌리고 추진 amp를 죽이는지", offset(회전 bias)을
약방향으로 충분히 주는지. 강/약(turn_sign ±) 그룹별 action·운동 step당 평균 + 대표 ep 시계열.
사용: python3 diagnostics/action_usage_asymmetry.py [card] [seedN]
출력: images/{card}_{seed}_action_asym.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_X, IDX_Y, IDX_YAW, FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX

PI = np.pi
N_EP = 100
STAGES = {"s3a_arc15": (PI * 9.2 / 180.0, PI / 12), "s3b_arc30": (PI / 12, PI / 6)}


def decode(a):
    c = np.clip(a, -1.0, 1.0)
    freq = FREQ_MIN + (float(c[0]) + 1.0) * 0.5 * (FREQ_MAX - FREQ_MIN)
    amp = AMP_MIN + (float(c[1]) + 1.0) * 0.5 * (AMP_MAX - AMP_MIN)
    return freq, amp, float(c[2])


def step_metrics(env, a):
    yaw = float(env.data.qpos[IDX_YAW])
    hd = np.array([-np.cos(yaw), np.sin(yaw)])
    v = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
    vfwd = float(np.dot(v, hd))
    yaw_rate = float(env.data.qvel[IDX_YAW])
    return vfwd, yaw_rate


def analyze(model, off):
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=off)
    g = {+1: dict(n=0, reach=0, freq=[], amp=[], offset=[], vfwd=[], yawrate=[]),
         -1: dict(n=0, reach=0, freq=[], amp=[], offset=[], vfwd=[], yawrate=[])}
    series = {+1: None, -1: None}   # 대표 ep 시계열 (마지막 ep)
    for ep in range(N_EP):
        obs, _ = env.reset(seed=4000 + ep)
        ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
        fr, am, of, vf, yr = [], [], [], [], []
        reached = False
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            f, m, o = decode(a)
            obs, _, term, trunc, info = env.step(a)
            vfwd, yawrate = step_metrics(env, a)
            fr.append(f); am.append(m); of.append(o); vf.append(vfwd); yr.append(yawrate)
            reached = bool(info.get("reached", False))
            if reached:
                break
        d = g[ts]
        d["n"] += 1; d["reach"] += int(reached)
        d["freq"].append(np.mean(fr)); d["amp"].append(np.mean(am)); d["offset"].append(np.mean(of))
        d["vfwd"].append(np.mean(vf)); d["yawrate"].append(np.mean(yr))
        series[ts] = dict(amp=np.array(am), offset=np.array(of), vfwd=np.array(vf),
                          yaw=None, reached=reached)
    env.close()
    return g, series


def report(tag, g):
    print(f"\n  {tag}:")
    for ts, name in [(+1, "강방향(+1)"), (-1, "약방향(-1)")]:
        d = g[ts]
        if d["n"] == 0:
            print(f"    {name}: ep 없음"); continue
        rr = 100.0 * d["reach"] / d["n"]
        print(f"    {name}: {d['n']}ep 도달 {rr:4.0f}%  |  "
              f"freq {np.mean(d['freq']):.2f}Hz  amp {np.mean(d['amp']):.3f}  "
              f"offset {np.mean(d['offset']):+.3f}  |  v_fwd {np.mean(d['vfwd']):+.4f}  "
              f"yaw_rate {np.mean(d['yawrate']):+.4f}")


def main():
    card = sys.argv[1] if len(sys.argv) > 1 else "m4_cpg_v17"
    seedn = sys.argv[2] if len(sys.argv) > 2 else "seed0"
    base = Path(__file__).parent.parent / "runs" / card / seedn
    print(f"[action usage asymmetry] N_EP={N_EP}, deterministic. card={card} {seedn}")
    print("  amp=추진진폭[0,1], offset=회전bias[-1,1], v_fwd=머리방향 전진속도")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (stage, off) in enumerate(STAGES.items()):
        mp = base / stage / "model_best.zip"
        if not mp.exists():
            mp = base / stage / "model.zip"
        if not mp.exists():
            print(f"  {stage}: 모델 없음"); continue
        model = SAC.load(str(mp), device="cpu")
        g, series = analyze(model, off)
        report(stage, g)
        for col, (key, ylab) in enumerate([("amp", "amp (추진)"), ("offset", "offset (회전bias)"), ("vfwd", "v_fwd (전진)")]):
            ax = axes[row, col]
            for ts, c, lab in [(+1, "tab:blue", "강(+1)"), (-1, "tab:red", "약(-1)")]:
                s = series.get(ts)
                if s is not None:
                    ax.plot(np.arange(len(s[key])) * 0.084, s[key], color=c, lw=1.2,
                            label=f"{lab}{'✓' if s['reached'] else '✗'}")
            if key == "vfwd":
                ax.axhline(0, ls=":", color="k", lw=0.8)
            ax.set_title(f"{stage} {ylab}")
            ax.set_xlabel("t(s)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"{card} {seedn} — 강/약방향 action 사용 (대표 ep 시계열)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(f"/home/yoo/RL_robot/images/{card}_{seedn}_action_asym.png")
    fig.savefig(out, dpi=110)
    print(f"\n  플롯: {out}")


if __name__ == "__main__":
    main()
