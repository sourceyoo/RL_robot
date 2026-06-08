"""D-H1: 좌 parallel 실재성 + 좌/우 선회 효율비 (open-loop, 정책 무관, read-only).

좌회전이 정책에서 sideslip 게걸음인 원인이 물리(H1)인지 가른다.
질문: 좌 target 에서 '진짜 parallel'(머리 돌며 추진: turn_ratio↑·슬립비↓·yaw_net target쪽+)로
도달하는 constant action 이 존재하는가? weak_dir_reachable 의 '좌 도달'이 실은 sideslip 이었는지
turn_ratio/슬립비 측정으로 재검증. + 좌/우 mirror 선회 효율비(정상상태 구간, fin transient 제외).

판정:
 좌에 turn_ratio≥0.4 & 슬립비≤0.6 & yaw_net target쪽(+) 도달 조합 존재 → 좌 parallel 물리 가능 → H1 기각.
 좌 도달이 전부 turn_ratio<0.2 또는 슬립비>0.8 (sideslip 도달뿐) → 좌 parallel 물리 불가 → H1 지지.
 효율비 |net_yaw_좌|/|net_yaw_우| < 0.5 → H1 부분 지지(가능하나 심한 비효율).
사용: python3 diagnostics/left_parallel_reachable.py
"""
import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).parent.parent))
from fish_env import FishSwimEnv, IDX_X, IDX_Y, IDX_YAW, FREQ_MIN, FREQ_MAX, AMP_MIN, AMP_MAX

PI = np.pi
SEED = 4000
DT = 0.084
OFFS = {"s3a(12°)": PI * 12.0 / 180.0, "s3b(30°)": PI * 30.0 / 180.0}
# parallel 후보: 큰 amp + 작은 |offset| (|offset|<amp → tail 이 0 가로질러 왕복추진).
# 좌(+offset)/우(-offset) mirror 쌍. 효율비도 이 쌍으로.
PAR_OFFSETS = [0.15, 0.30, 0.45]
PAR_FREQ, PAR_AMP = 5.0, 1.0
# 기존 weak_dir 그리드 (좌 '도달'이 sideslip 이었는지 재검증용)
GRID = [(f, a, o) for f in [4.0, 5.0, 6.0] for a in [0.5, 0.75, 1.0]
        for o in np.round(np.linspace(-1.0, 1.0, 9), 3)]
EFF_STEPS = 80  # 효율비용 고정 step (도달 무시). 후반 절반만 rate 계산 → transient 제외


def amp_c(a):
    return 2.0 * (a - AMP_MIN) / (AMP_MAX - AMP_MIN) - 1.0


def freq_c(f):
    return 2.0 * (f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN) - 1.0


def set_target(env, theta):
    """weak_dir_reachable.set_target 복제: target 위치 변경 + turn state 재계산."""
    r = float(np.linalg.norm(env._target_pos()[:2]))
    gid = env._target_geom_id
    env.model.geom_pos[gid] = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
    mujoco.mj_forward(env.model, env.data)
    yaw0 = float(env.data.qpos[IDX_YAW])
    hd0 = np.array([-np.cos(yaw0), np.sin(yaw0)])
    rel0 = env._target_pos()[:2] - env._torso_pos()[:2]
    n = float(np.linalg.norm(rel0))
    env._yaw0 = yaw0
    env._turn_sign = -float(np.sign(hd0[0] * rel0[1] - hd0[1] * rel0[0])) if n > 1e-6 else 0.0
    env._target_offset = float(np.arccos(np.clip(float(np.dot(hd0, rel0 / n)), -1.0, 1.0))) if n > 1e-6 else 0.0
    env._prev_distance = env._distance_to_target()
    env._prev_pos = env._torso_pos()[:2].copy()


def run_measured(env, theta, freq, amp, offset, fixed_steps=None):
    """constant action open-loop + per-step parallel 지표 측정.
    fixed_steps=None: 도달/ep끝까지. 정수: 그 step 만 (효율비용)."""
    env.reset(seed=SEED)
    set_target(env, theta)
    ts = int(np.sign(env._turn_sign)) if env._turn_sign != 0 else 1
    yaw0, toff, tsign = env._yaw0, env._target_offset, env._turn_sign
    a = np.array([freq_c(freq), amp_c(amp), offset], dtype=np.float32)
    vf, vl, al, tr, yaws = [], [], [], [], []
    best = env._distance_to_target()
    reached = False
    t = 0
    term = trunc = False
    while not (term or trunc):
        _, _, term, trunc, info = env.step(a)
        yaw = float(env.data.qpos[IDX_YAW])
        hd = np.array([-np.cos(yaw), np.sin(yaw)])
        perp = np.array([-hd[1], hd[0]])
        v = np.array([float(env.data.qvel[IDX_X]), float(env.data.qvel[IDX_Y])])
        rel = env._target_pos()[:2] - env._torso_pos()[:2]
        rn = float(np.linalg.norm(rel))
        vf.append(float(np.dot(v, hd)))
        vl.append(abs(float(np.dot(v, perp))))
        al.append(float(np.dot(hd, rel / rn)) if rn > 1e-6 else 0.0)
        tr.append(float(np.clip((yaw - yaw0) * tsign / toff, 0.0, 1.0)) if toff > 1e-3 else 1.0)
        yaws.append(yaw)
        best = min(best, env._distance_to_target())
        t += 1
        if fixed_steps is not None and t >= fixed_steps:
            break
        if bool(info.get("reached", False)):
            reached = True
            break
    vfm = float(np.mean(vf))
    slip = float(np.mean(vl)) / (abs(vfm) + 1e-9)
    yaw_net = float(np.degrees((float(env.data.qpos[IDX_YAW]) - yaw0) * tsign))
    return dict(ts=ts, reached=reached, best=best, vfwd=vfm, slip=slip,
                turn_ratio=float(np.mean(tr)), align=float(np.mean(al)),
                yaw_net=yaw_net, nstep=t, yaws=np.array(yaws), yaw0=yaw0, tsign=tsign)


def is_parallel(r):
    return (r["turn_ratio"] >= 0.4) and (r["slip"] <= 0.6) and (r["yaw_net"] > 0)


def sweep_reach(env, theta, label, cands):
    """도달 조합 수집 + parallel 지표. parallel 도달 존재 판정."""
    ts = None
    par_hits, ss_hits = [], []  # parallel 도달 / sideslip 도달
    for (f, a, o) in cands:
        r = run_measured(env, theta, f, a, o)
        ts = r["ts"]
        if r["reached"]:
            (par_hits if is_parallel(r) else ss_hits).append((f, a, o, r))
    print(f"\n  [{label}] θ={np.degrees(theta):.1f}°, turn_sign={ts:+d}  ({len(cands)} 조합)")
    print(f"    도달 {len(par_hits)+len(ss_hits)}개  (parallel {len(par_hits)} / sideslip {len(ss_hits)})")
    print(f"    {'유형':>9} {'freq':>4} {'amp':>4} {'offset':>7} {'turn_ratio':>10} {'슬립비':>7} {'yaw_net':>8} {'근접':>6}")
    show = [("PAR", h) for h in par_hits[:6]] + [("ss", h) for h in ss_hits[:6]]
    for tag, (f, a, o, r) in show:
        print(f"    {tag:>9} {f:>4.1f} {a:>4.2f} {o:>+7.2f} {r['turn_ratio']:>10.3f} "
              f"{r['slip']:>7.2f} {r['yaw_net']:>+8.1f}° {r['best']:>6.3f}")
    return ts, len(par_hits), len(ss_hits)


def efficiency(env, off):
    """좌/우 mirror 쌍 net_yaw 효율비 (정상상태: 후반 절반 rate)."""
    print(f"\n  [효율비] mirror 쌍 (amp={PAR_AMP}, freq={PAR_FREQ}Hz, {EFF_STEPS}step 중 후반절반 rate)")
    print(f"    {'|offset|':>8} {'우 net_yaw_rate':>16} {'좌 net_yaw_rate':>16} {'효율비(좌/우)':>14}")
    ratios = []
    for mag in PAR_OFFSETS:
        rR = run_measured(env, PI - off, PAR_FREQ, PAR_AMP, -mag, fixed_steps=EFF_STEPS)
        rL = run_measured(env, PI + off, PAR_FREQ, PAR_AMP, +mag, fixed_steps=EFF_STEPS)
        h = EFF_STEPS // 2
        # 후반 절반 net yaw (target쪽 +), rate deg/s
        rateR = float(np.degrees((rR["yaws"][-1] - rR["yaws"][h - 1]) * rR["tsign"]) / ((EFF_STEPS - h) * DT))
        rateL = float(np.degrees((rL["yaws"][-1] - rL["yaws"][h - 1]) * rL["tsign"]) / ((EFF_STEPS - h) * DT))
        ratio = abs(rateL) / (abs(rateR) + 1e-9)
        ratios.append(ratio)
        print(f"    {mag:>8.2f} {rateR:>+15.2f}° {rateL:>+15.2f}° {ratio:>14.2f}")
    return float(np.mean(ratios))


def main():
    print("[D-H1 좌 parallel 실재성 + 효율비] open-loop, 정책 무관")
    print("  parallel 판정: turn_ratio≥0.4 & 슬립비≤0.6 & yaw_net target쪽(+)")
    for name, off in OFFS.items():
        print(f"\n{'='*78}\n### {name}")
        env = FishSwimEnv(episode_seconds=60.0, success_radius=0.08, action_history_n=20,
                          target_theta_offset_range=(off, off))
        cands = [(PAR_FREQ, PAR_AMP, o) for o in PAR_OFFSETS] + \
                [(PAR_FREQ, PAR_AMP, -o) for o in PAR_OFFSETS] + GRID
        _, parL, ssL = sweep_reach(env, PI + off, "좌 θ=π+off", cands)
        _, parR, ssR = sweep_reach(env, PI - off, "우 θ=π-off", cands)
        eff = efficiency(env, off)
        env.close()
        print(f"\n  --- {name} 판정 ---")
        print(f"    좌: parallel 도달 {parL} / sideslip 도달 {ssL}   우: parallel {parR} / sideslip {ssR}")
        print(f"    좌/우 선회 효율비(평균) {eff:.2f}")
        if parL > 0:
            verdict = "좌 parallel 물리 가능 → H1 기각 (D-H2 보상 진단으로)"
        elif ssL > 0:
            verdict = "좌 도달이 sideslip 뿐 → H1 지지 (좌 parallel 물리 불가, sim 한정)"
        else:
            verdict = "좌 도달 0 → constant 부족(시변 필요) or 물리 강차단"
        if 0 < eff < 0.5:
            verdict += f" / 효율비 {eff:.2f}<0.5 → 좌선회 심한 비효율(H1 부분 지지)"
        print(f"    → {verdict}")


if __name__ == "__main__":
    main()
