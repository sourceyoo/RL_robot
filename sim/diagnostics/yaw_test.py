"""Yaw 회전 진단 — 단일 모터 + passive fin 물리적 한계 측정.

여러 비대칭 ctrl 패턴을 30초 인가하고 yaw 변화량 → yaw rate (°/s).
SAC 학습 한계인지 물리 한계인지 판별.

패턴:
  A. 대칭 sine (baseline): yaw rate ~0 기대
  B. DC offset sine: 평균 tail 각도가 ±쪽으로 편향
  C. 비대칭 sine (강한 +스트로크 + 약한 -스트로크)
  D. Asymmetric square (한쪽 편향 펄스)
  E. 한쪽 hold (constant ctrl): 정적 자세 + drift
  F. Asymmetric duty cycle: 한쪽으로 시간 길게
  G. Burst-and-coast (한쪽으로 burst + 정지)

3 Hz 기준 (전진 추진 검증된 영역). 진폭 ctrl=±1 (명령 ±20°).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

XML = Path(__file__).parent.parent / "rl_fish.xml"

# qpos layout
X, Y, YAW, TAIL, FIN = 0, 1, 2, 3, 4


def run(name: str, ctrl_fn, duration=30.0, settle=2.0):
    """한 패턴 실행. ctrl_fn(t) → ctrl ∈ [-1, 1]."""
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
    yaw_total = yaws[-1] - yaws[0]            # 누적 yaw (rad)
    yaw_rate = yaw_total / elapsed             # rad/s
    yaw_max_abs = max(abs(y - yaws[0]) for y in yaws)  # 최대 편차

    return {
        "name": name,
        "yaw_total_deg": np.degrees(yaw_total),
        "yaw_rate_dps": np.degrees(yaw_rate),
        "yaw_max_abs_deg": np.degrees(yaw_max_abs),
        "x_disp": x_pos[-1] - x_pos[0],
        "y_disp": y_pos[-1] - y_pos[0],
    }


def main():
    f = 3.0  # Hz, MODE_3 정착 영역 (전진 검증됨)
    omega = 2 * np.pi * f

    patterns = [
        # A. baseline 대칭
        ("A. 대칭 sine",                lambda t: np.sin(omega * t)),

        # B. DC offset (평균 tail 한쪽 편향)
        ("B1. sine + 0.3 (오른쪽 편향)", lambda t: 0.7 * np.sin(omega * t) + 0.3),
        ("B2. sine - 0.3 (왼쪽 편향)",   lambda t: 0.7 * np.sin(omega * t) - 0.3),
        ("B3. sine + 0.5",               lambda t: 0.5 * np.sin(omega * t) + 0.5),
        ("B4. sine - 0.5",               lambda t: 0.5 * np.sin(omega * t) - 0.5),

        # C. 비대칭 진폭
        ("C1. +쪽 1.0 / -쪽 0.3",        lambda t: 1.0 * max(0, np.sin(omega * t)) + 0.3 * min(0, np.sin(omega * t))),
        ("C2. +쪽 0.3 / -쪽 1.0",        lambda t: 0.3 * max(0, np.sin(omega * t)) + 1.0 * min(0, np.sin(omega * t))),

        # D. asymmetric square (편향 펄스)
        ("D1. sq +0.5 (75% +) ",         lambda t: 1.0 if (omega * t) % (2 * np.pi) < 1.5 * np.pi else -1.0),
        ("D2. sq -0.5 (75% -)",          lambda t: -1.0 if (omega * t) % (2 * np.pi) < 1.5 * np.pi else 1.0),

        # E. 한쪽 hold (constant)
        ("E1. ctrl = +1.0 (오른쪽 maxhold)", lambda t: 1.0),
        ("E2. ctrl = -1.0 (왼쪽 maxhold)",   lambda t: -1.0),
        ("E3. ctrl = +0.7",              lambda t: 0.7),
        ("E4. ctrl = -0.7",              lambda t: -0.7),

        # F. duty cycle 비대칭 (3 Hz 주기 안에서 +시간 75% / -시간 25%)
        ("F1. duty +75%/-25% asymsine",  lambda t: np.sin(np.pi * (omega * t) % (2 * np.pi) / 1.5) if (omega * t) % (2 * np.pi) < 1.5 * np.pi else -np.sin(np.pi * ((omega * t) % (2 * np.pi) - 1.5 * np.pi) / 0.5)),

        # G. burst-and-coast (1초 burst sine + 1초 정지)
        ("G1. burst+coast (한쪽 편향)",    lambda t: (0.7 * np.sin(omega * t) + 0.3) if int(t) % 2 == 0 else 0.0),
    ]

    print(f"각 패턴 30초 인가, 첫 2초 settle 후 측정. f={f} Hz, ctrl ∈ [-1, 1].\n")
    print(f"{'pattern':<40} {'yaw_total[°]':>14} {'yaw_rate[°/s]':>14} {'max|yaw|[°]':>14} {'x_disp':>9} {'y_disp':>9}")
    print("-" * 110)

    for name, fn in patterns:
        try:
            r = run(name, fn)
            if r is None:
                continue
            print(f"{r['name']:<40} {r['yaw_total_deg']:>+14.2f} {r['yaw_rate_dps']:>+14.3f} "
                  f"{r['yaw_max_abs_deg']:>14.2f} {r['x_disp']:>+9.3f} {r['y_disp']:>+9.3f}")
        except Exception as e:
            print(f"{name}: ERROR {e}")

    print()
    print("해석 가이드:")
    print("  - yaw_rate_dps |값| > 5°/s: SAC가 학습할 여지 있음 (충분한 회전 능력)")
    print("  - yaw_rate_dps |값| < 1°/s: 물리적 한계 — 단일 모터로 회전 불가능")
    print("  - max|yaw|이 yaw_total보다 훨씬 크면: 진동·복귀하는 yaw (steady-state 회전 X)")


if __name__ == "__main__":
    main()
