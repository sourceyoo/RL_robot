"""학습된 SAC 정책의 ctrl pattern 분석 — F (asymmetric swing) vs D (step) vs B (DC offset) mode 식별.

deterministic rollout → ctrl·tail·yaw 시계열 → slow_frac + DC offset + step rate + dominant freq + outcome.

사용:
    python3 analyze_policy.py runs/m4_v2_seed0/s3a_arc15/model.zip --theta 3.403 --episode-seconds 60
    (theta = π + 0.262 rad = π + 15°)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from fish_env import FishSwimEnv, IDX_YAW, IDX_X, IDX_Y, IDX_TAIL

ROOT = Path(__file__).parent


def infer_action_history_n(sac_model) -> int:
    return sac_model.observation_space.shape[0] - 13


def analyze_ctrl(ctrl_series: list[float], dt: float) -> dict:
    """ctrl 시계열에서 mode 식별 metric 계산."""
    ctrl = np.asarray(ctrl_series, dtype=np.float64)
    n = len(ctrl)
    if n < 4:
        return {}

    dc_offset = float(np.mean(ctrl))
    step_rate = float((np.abs(np.diff(ctrl)) > 0.5).mean())  # D mode 신호

    # FFT dominant freq (DC 제거 후)
    spec = np.abs(np.fft.rfft(ctrl - dc_offset))
    freqs = np.fft.rfftfreq(n, d=dt)
    dom_freq = float(freqs[int(np.argmax(spec[1:]) + 1)]) if len(spec) > 1 else 0.0

    # slow_frac: AC 성분(ctrl - DC offset) zero-crossing 기반 +half / cycle 시간 비율
    # DC offset된 정책도 AC swing의 시간 비대칭 측정 가능
    ctrl_ac = ctrl - dc_offset
    sign = np.sign(ctrl_ac)
    sign[sign == 0] = 1
    sign_changes = np.where(np.diff(sign) != 0)[0]

    if len(sign_changes) < 2:
        return dict(slow_frac=None, slow_frac_std=None, n_cycles=0,
                    dc_offset=dc_offset, step_rate=step_rate, dom_freq=dom_freq)

    # halves: (sign, duration_in_samples)
    halves = []
    prev = 0
    for idx in sign_changes:
        dur = idx - prev
        if dur > 0:
            halves.append((sign[prev], dur))
        prev = idx + 1

    # cycles = 연속 부호 다른 두 half
    slow_fracs = []
    i = 0
    while i + 1 < len(halves):
        s1, d1 = halves[i]
        s2, d2 = halves[i + 1]
        if s1 == s2:
            i += 1
            continue
        total = d1 + d2
        if total > 0:
            pos = d1 if s1 > 0 else d2
            slow_fracs.append(pos / total)
        i += 2

    if not slow_fracs:
        return dict(slow_frac=None, slow_frac_std=None, n_cycles=0,
                    dc_offset=dc_offset, step_rate=step_rate, dom_freq=dom_freq)

    return dict(
        slow_frac=float(np.mean(slow_fracs)),
        slow_frac_std=float(np.std(slow_fracs)),
        n_cycles=len(slow_fracs),
        dc_offset=dc_offset,
        step_rate=step_rate,
        dom_freq=dom_freq,
    )


def classify_mode(stats: dict, target_theta_offset_deg: float) -> str:
    """target 회전 offset 대비 mode 판정."""
    sf = stats.get("slow_frac")
    dc = stats.get("dc_offset", 0.0)
    sr = stats.get("step_rate", 0.0)
    if sf is None:
        return "low-cycle (n_cycles<1) — 정책이 swing 학습 X 또는 비주기"

    sf_dev = abs(sf - 0.5)
    dc_dev = abs(dc)
    rotation_needed = abs(target_theta_offset_deg) > 1.0

    if not rotation_needed:
        if sf_dev < 0.05 and dc_dev < 0.05:
            return "Direct (대칭 swing, 직진 mode) ✓"
        else:
            return f"비대칭이지만 target=직진 — 정렬 mode 학습 X 신호 (slow_frac={sf:.3f}, dc={dc:+.3f})"

    # 회전 필요
    if sr > 0.2:
        return f"D (step ctrl, step_rate={sr:.2f}) — ACTION_DIFF_W 강화 효과 약함"
    if sf_dev > 0.05:
        return f"F (asymmetric swing, slow_frac={sf:.3f} dev {sf_dev:+.3f}) ✓"
    if dc_dev > 0.1:
        return f"B (DC offset sine, dc={dc:+.3f}) — sweep 시간 대칭이지만 평균 편향"
    return f"불분명 (slow_frac={sf:.3f}, dc={dc:+.3f}, sr={sr:.2f}) — 회전 학습 약함"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("--theta", type=float, required=True, help="target 각도 (rad). π+0.262 = +15°.")
    p.add_argument("--episode-seconds", type=float, default=60.0)
    p.add_argument("--success-radius", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from stable_baselines3 import SAC
    if not Path(args.model_path).exists():
        print(f"모델 파일 없음: {args.model_path}")
        return

    sac = SAC.load(args.model_path, device="cpu")
    action_history_n = infer_action_history_n(sac)

    env = FishSwimEnv(
        target_theta_range=(args.theta, args.theta),
        success_radius=args.success_radius,
        episode_seconds=args.episode_seconds,
        action_history_n=action_history_n,
    )
    obs, _ = env.reset(seed=args.seed)
    data = env.data
    dt = env.dt

    ctrls: list[float] = []
    yaws: list[float] = []
    tails: list[float] = []
    slip_angles: list[float] = []
    head_target_align_series: list[float] = []

    target_offset_deg = math.degrees(args.theta - math.pi)
    initial_yaw = float(data.qpos[IDX_YAW])
    initial_x = float(data.qpos[IDX_X])
    initial_y = float(data.qpos[IDX_Y])

    info = {}
    while True:
        action, _ = sac.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        ctrls.append(float(action[0]))
        yaws.append(float(data.qpos[IDX_YAW]))
        tails.append(float(data.qpos[IDX_TAIL]))
        vx = float(data.qvel[IDX_X]); vy = float(data.qvel[IDX_Y])
        yaw_now = float(data.qpos[IDX_YAW])
        head = np.array([-math.cos(yaw_now), math.sin(yaw_now)])
        speed = math.hypot(vx, vy)
        if speed > 0.01:
            vel = np.array([vx, vy]) / speed
            cos_s = float(np.clip(np.dot(head, vel), -1.0, 1.0))
            slip_angles.append(math.degrees(math.acos(cos_s)))
        rel = env._target_pos()[:2] - env._torso_pos()[:2]
        rel_n = float(np.linalg.norm(rel))
        if rel_n > 1e-3:
            head_target_align_series.append(float(np.dot(head, rel / rel_n)))
        if terminated or truncated:
            break

    yaw_total_deg = math.degrees(float(data.qpos[IDX_YAW]) - initial_yaw)
    yaw_max_deg = math.degrees(max(abs(y - initial_yaw) for y in yaws))
    x_disp = float(data.qpos[IDX_X]) - initial_x
    y_disp = float(data.qpos[IDX_Y]) - initial_y

    stats = analyze_ctrl(ctrls, dt)
    mode = classify_mode(stats, target_offset_deg)

    print(f"[analyze] model={args.model_path}")
    print(f"          target θ = π + {target_offset_deg:+.1f}°,  ep={args.episode_seconds}s,  dt={dt:.4f}s,  n_steps={len(ctrls)}")
    print()
    print(f"ctrl pattern:")
    if stats.get("slow_frac") is not None:
        print(f"  slow_frac    = {stats['slow_frac']:.3f} ± {stats['slow_frac_std']:.3f}  (n_cycles={stats['n_cycles']})")
    else:
        print(f"  slow_frac    = N/A (n_cycles=0)")
    print(f"  DC offset    = {stats['dc_offset']:+.4f}  (mean ctrl)")
    print(f"  step_rate    = {stats['step_rate']:.3f}  (|Δctrl|>0.5 빈도)")
    print(f"  dominant freq= {stats['dom_freq']:.2f} Hz")
    print()
    print(f"outcome:")
    print(f"  yaw_total    = {yaw_total_deg:+.2f}°  (target offset {target_offset_deg:+.1f}°)")
    print(f"  yaw_max_abs  = {yaw_max_deg:.2f}°")
    print(f"  reached      = {info.get('reached', False)}  (final_dist {info.get('distance', 0):.3f})")
    print(f"  x_disp       = {x_disp:+.3f}m,  y_disp = {y_disp:+.3f}m")
    print()
    slip_mean = float(np.mean(slip_angles)) if slip_angles else float("nan")
    slip_max = float(np.max(slip_angles)) if slip_angles else float("nan")
    align_final = float(np.mean(head_target_align_series[-20:])) if head_target_align_series else float("nan")
    yaw_range_deg = (max(yaws) - min(yaws)) * 180 / math.pi
    yaw_osc = yaw_range_deg / max(1e-3, abs(yaw_total_deg))
    print(f"side-slip 검출:")
    print(f"  slip_angle   = {slip_mean:.1f}° (max {slip_max:.1f}°)")
    print(f"  head_target_align_final = {align_final:.3f}")
    print(f"  yaw_oscillation = {yaw_osc:.2f}× (range {yaw_range_deg:.1f}° / total {yaw_total_deg:.1f}°)")
    if slip_mean > 30:
        slip_verdict = "side-slip ✗"
    elif slip_mean < 15 and align_final > 0.7:
        slip_verdict = "F/D 회전 ✓"
    else:
        slip_verdict = "mixed"
    print(f"  진행 mode: {slip_verdict}")
    print()
    print(f"mode 판정: {mode}")


if __name__ == "__main__":
    main()
