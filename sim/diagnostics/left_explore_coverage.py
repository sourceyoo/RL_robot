"""D-H3a: 좌회전 탐색이 parallel 영역을 닿는가 (탐색 커버리지, read-only).

H1·H2 기각: 좌 parallel 물리 가능 + 보상도 parallel 선호. 그런데 정책이 s3b 큰 각에서 게걸음.
질문: 정책의 좌회전 stochastic 탐색이 D-H1 의 parallel 영역(큰 amp + 큰 offset)을 방문하는가?
좌가 그 영역을 거의 안 닿으면 = 탐색이 parallel 을 못 찾음(H3 탐색 실패). 우와 비교.

parallel 영역(D-H1): amp ≥ 0.6 & |offset| ∈ [0.6, 0.9] (|offset|≈amp 라 tail 왕복추진+큰 선회).
사용: python3 diagnostics/left_explore_coverage.py
"""
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, AMP_MIN, AMP_MAX

PI = np.pi
SEED = 4000
N_EP = 40           # 좌/우 각 방향 ep 목표
STAGES = {"s3a_arc15": (PI * 9.2 / 180.0, PI / 12), "s3b_arc30": (PI / 12, PI / 6)}
AMP_LO, OFF_LO, OFF_HI = 0.6, 0.6, 0.9   # parallel 영역 정의


def decode_amp_off(a):
    c = np.clip(a, -1.0, 1.0)
    amp = AMP_MIN + (float(c[1]) + 1.0) * 0.5 * (AMP_MAX - AMP_MIN)
    return amp, float(c[2])


def in_parallel(amp, off):
    return (amp >= AMP_LO) and (OFF_LO <= abs(off) <= OFF_HI)


def collect(model, off_range):
    """좌/우 ep 진행하며 stochastic + deterministic action(amp,offset) 수집."""
    env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                      target_theta_offset_range=off_range)
    g = {+1: dict(n=0, sto=[], det=[]), -1: dict(n=0, sto=[], det=[])}
    ep = 0
    while (g[+1]["n"] < N_EP or g[-1]["n"] < N_EP) and ep < 600:
        obs, _ = env.reset(seed=SEED + ep); ep += 1
        ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
        if g[ts]["n"] >= N_EP:
            continue
        g[ts]["n"] += 1
        term = trunc = False
        while not (term or trunc):
            a_det, _ = model.predict(obs, deterministic=True)
            a_sto, _ = model.predict(obs, deterministic=False)
            g[ts]["det"].append(decode_amp_off(a_det))
            g[ts]["sto"].append(decode_amp_off(a_sto))
            obs, _, term, trunc, info = env.step(a_det)
            if bool(info.get("reached", False)):
                break
    env.close()
    return g


def summ(tag, arr):
    if not arr:
        return f"{tag}: 데이터 없음"
    amps = np.array([x[0] for x in arr]); offs = np.array([x[1] for x in arr])
    cover = 100.0 * np.mean([in_parallel(a, o) for a, o in arr])
    return (f"{tag}: amp {amps.mean():.3f}±{amps.std():.3f}  offset {offs.mean():+.3f}±{offs.std():.3f}  "
            f"|off| {np.abs(offs).mean():.3f}  parallel영역 커버 {cover:4.1f}%")


def main():
    print("[D-H3a 좌회전 탐색 커버리지] stochastic action 이 parallel 영역 닿는가")
    print(f"  parallel 영역: amp≥{AMP_LO} & |offset|∈[{OFF_LO},{OFF_HI}]")
    base = Path(__file__).parent.parent / "runs" / "m4_cpg_v19" / "seed0"
    for stage, off_range in STAGES.items():
        mp = base / stage / "model.zip"
        if not mp.exists():
            print(f"\n  {stage}: 모델 없음"); continue
        model = SAC.load(str(mp), device="cpu")
        g = collect(model, off_range)
        print(f"\n  ### {stage}")
        for ts, name in [(+1, "우(+1)"), (-1, "좌(-1)")]:
            print(f"    {name} ({g[ts]['n']}ep)")
            print(f"      det   {summ('det', g[ts]['det'])}")
            print(f"      stoch {summ('stoch', g[ts]['sto'])}")
        # 판정 (stochastic 커버율 좌 vs 우)
        cl = 100.0 * np.mean([in_parallel(a, o) for a, o in g[-1]["sto"]]) if g[-1]["sto"] else 0.0
        cr = 100.0 * np.mean([in_parallel(a, o) for a, o in g[+1]["sto"]]) if g[+1]["sto"] else 0.0
        print(f"    → 좌 parallel영역 stochastic 커버 {cl:.1f}% vs 우 {cr:.1f}%  "
              f"({'좌 탐색이 parallel 못 닿음 = H3 탐색실패' if cl < 0.5 * cr + 1e-9 else '좌도 parallel 영역 탐색함'})")


if __name__ == "__main__":
    main()
