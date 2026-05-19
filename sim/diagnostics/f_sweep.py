"""F duty sweep — sweep 시간 비대칭 ctrl pattern으로 ±60°/±90° 회전 capability 측정.

한 cycle 안에서 +sweep 시간 / -sweep 시간 비대칭 (예: 75/25 = +sweep이 1.5π, -sweep이 0.5π).
빠른 stroke 반작용으로 그 방향으로 회전 (carangiform power/recovery stroke 비대칭).

9 patterns: 50/50 (대칭 baseline) + 60/40·70/30·75/25·80/20 × 양방향 (+slow / -slow).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).parent.parent / "rl_fish.xml"
X, Y, YAW, TAIL, FIN = 0, 1, 2, 3, 4


def f_pattern(t: float, f: float, slow_frac: float, sign: int) -> float:
    """Asymmetric duty sine.

    한 cycle (1/f) 안에서:
      - 첫 slow_frac × cycle 시간 동안: sign × half-sine 0→1→0 (slow sweep)
      - 나머지 (1-slow_frac) × cycle 시간 동안: -sign × half-sine 0→-1→0 (fast sweep)

    Args:
        slow_frac: 느린 sweep 시간 비율 ∈ (0.5, 1.0). 0.5는 대칭 (slow=fast).
        sign: +1 (slow sweep이 +쪽) / -1 (slow sweep이 -쪽).
    """
    period = 1.0 / f
    phase = (t % period) / period   # 0 ~ 1

    if phase < slow_frac:
        # slow half-sine: 0 → +1 → 0 (slow_frac 시간 동안)
        return sign * np.sin(np.pi * phase / slow_frac)
    else:
        # fast half-sine: 0 → -1 → 0 ((1-slow_frac) 시간 동안)
        return -sign * np.sin(np.pi * (phase - slow_frac) / (1.0 - slow_frac))


def run(name: str, ctrl_fn, duration: float = 60.0, settle: float = 2.0) -> dict | None:
    """한 pattern 실행 → yaw/x/y 측정."""
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    n_steps = int(duration / model.opt.timestep)
    yaws, x_pos, y_pos = [], [], []

    for _ in range(n_steps):
        t = data.time
        data.ctrl[0] = float(np.clip(ctrl_fn(t), -1.0, 1.0))
        mujoco.mj_step(model, data)
        if t >= settle:
            yaws.append(data.qpos[YAW])
            x_pos.append(data.qpos[X])
            y_pos.append(data.qpos[Y])

    if not yaws:
        return None

    elapsed = duration - settle
    yaw_total = yaws[-1] - yaws[0]
    yaw_rate = yaw_total / elapsed
    yaw_max_abs = max(abs(y - yaws[0]) for y in yaws)

    return {
        "name": name,
        "yaw_total_deg": np.degrees(yaw_total),
        "yaw_rate_dps": np.degrees(yaw_rate),
        "yaw_max_abs_deg": np.degrees(yaw_max_abs),
        "x_disp": x_pos[-1] - x_pos[0],
        "y_disp": y_pos[-1] - y_pos[0],
    }


def main():
    f = 3.0  # Hz (전진 검증 영역)
    duration = 60.0  # s (학습 ep_sec 정합)

    patterns = []

    # 50/50 대칭 baseline
    patterns.append(("F 50/50 (대칭)", lambda t: f_pattern(t, f, 0.5, +1)))

    # 4 ratio × 양방향
    for ratio in [0.6, 0.7, 0.75, 0.8]:
        a = int(ratio * 100)
        b = 100 - a
        patterns.append((f"F {a}/{b} (+slow)", lambda t, r=ratio: f_pattern(t, f, r, +1)))
        patterns.append((f"F {a}/{b} (-slow)", lambda t, r=ratio: f_pattern(t, f, r, -1)))

    print(f"F duty sweep: f={f} Hz, duration={duration}s (settle 2s).")
    print(f"각 pattern당 ctrl ∈ [-1, 1]. 한 cycle 안 sweep 시간 비대칭.\n")
    print(f"{'pattern':<22} {'yaw_total[°]':>14} {'yaw_rate[°/s]':>14} {'max|yaw|[°]':>14} {'x_disp':>9} {'y_disp':>9}")
    print("-" * 92)

    results = []
    for name, fn in patterns:
        try:
            r = run(name, fn, duration=duration)
            if r is None:
                continue
            results.append(r)
            print(f"{r['name']:<22} {r['yaw_total_deg']:>+14.2f} {r['yaw_rate_dps']:>+14.3f} "
                  f"{r['yaw_max_abs_deg']:>14.2f} {r['x_disp']:>+9.3f} {r['y_disp']:>+9.3f}")
        except Exception as e:
            print(f"{name}: ERROR {e}")

    # 판정
    print()
    print("판정 (yaw_max_abs_deg 기준):")
    reached_60 = [r for r in results if r["yaw_max_abs_deg"] >= 60.0]
    reached_90 = [r for r in results if r["yaw_max_abs_deg"] >= 90.0]
    print(f"  ±60° 도달 pattern: {len(reached_60)}개 — {[r['name'] for r in reached_60]}")
    print(f"  ±90° 도달 pattern: {len(reached_90)}개 — {[r['name'] for r in reached_90]}")

    baseline = next((r for r in results if "50/50" in r["name"]), None)
    if baseline:
        sym_ok = abs(baseline["yaw_rate_dps"]) < 1.0
        print(f"  sim 좌우 대칭 (50/50 |yaw_rate|<1): {'OK' if sym_ok else 'FAIL'} ({baseline['yaw_rate_dps']:+.3f}°/s)")


if __name__ == "__main__":
    main()
