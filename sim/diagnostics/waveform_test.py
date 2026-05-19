"""다양한 ctrl waveform이 head 방향(-x) 추진을 만들어내는지 테스트.

부호 규약:
  x_disp < 0  → head 방향(전진) ✓
  x_disp > 0  → tail 방향(후진)

기준: f=3Hz (현 actuator로 ±20° 추종 잘 됨), 10초 시뮬, 2초 settle.
"""

from pathlib import Path
import mujoco
import numpy as np

XML = Path(__file__).parent.parent / "rl_fish.xml"
F = 3.0  # 기본 주파수
DURATION = 10.0
SETTLE = 2.0


def run(ctrl_fn, label):
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    x_pos, y_pos = [], []
    n = int(DURATION / model.opt.timestep)
    for _ in range(n):
        t = data.time
        c = float(np.clip(ctrl_fn(t), -1.0, 1.0))
        data.ctrl[0] = c
        mujoco.mj_step(model, data)
        if t >= SETTLE:
            x_pos.append(data.qpos[0])
            y_pos.append(data.qpos[1])
    x_d = x_pos[-1] - x_pos[0]
    y_d = y_pos[-1] - y_pos[0]
    if x_d < -0.01:
        verdict = "← 머리쪽 (전진!) ✓"
    elif x_d > 0.01:
        verdict = "→ 꼬리쪽 (후진)"
    else:
        verdict = "  거의 정지"
    print(f"{label:38s}  x_disp={x_d:+8.4f}  y_disp={y_d:+8.4f}  {verdict}")


def asymm_stroke(t, fast_positive=True):
    """비대칭 스트로크: 한쪽 빠르게, 다른쪽 느리게.
    fast_positive=True: +방향으로 빠르게 가고 -방향으로 천천히 복귀
    """
    period = 1.0 / F
    phase = (t / period) % 1.0
    if fast_positive:
        # 0~0.3 phase: 0 → +1 (빠른 +스트로크)
        # 0.3~1.0 phase: +1 → -1 → 0 (천천히 -로 갔다가 0으로)
        if phase < 0.3:
            return phase / 0.3
        elif phase < 0.65:
            return 1 - 2 * (phase - 0.3) / 0.35  # +1 → -1
        else:
            return -1 + (phase - 0.65) / 0.35  # -1 → 0
    else:
        return -asymm_stroke(t, fast_positive=True)


def main():
    print(f"기준 주파수 f={F}Hz, {DURATION}s 시뮬, settle={SETTLE}s\n")
    print(f"{'Waveform':38s}  {'x_disp':>10}  {'y_disp':>10}  방향")
    print("-" * 95)

    # 1. baseline
    run(lambda t: np.sin(2*np.pi*F*t), "1. 대칭 sine (baseline)")

    # 2-3. DC offset
    run(lambda t: 0.3 + 0.6*np.sin(2*np.pi*F*t), "2. DC offset +0.3 (꼬리 평균 +y)")
    run(lambda t: -0.3 + 0.6*np.sin(2*np.pi*F*t), "3. DC offset -0.3 (꼬리 평균 -y)")

    # 4-5. 2nd harmonic (Fourier asymmetric)
    run(lambda t: 0.7*np.sin(2*np.pi*F*t) + 0.3*np.sin(2*np.pi*2*F*t),
        "4. sin + 0.3*sin(2x) (2nd harm in-phase)")
    run(lambda t: 0.7*np.sin(2*np.pi*F*t) - 0.3*np.sin(2*np.pi*2*F*t),
        "5. sin - 0.3*sin(2x) (2nd harm anti-phase)")
    run(lambda t: 0.7*np.sin(2*np.pi*F*t) + 0.3*np.sin(2*np.pi*2*F*t + np.pi/2),
        "6. sin + 0.3*sin(2x+90°) (quadrature)")

    # 7. Burst-and-coast
    def burst(t):
        period = 1.0
        phase = t % period
        if phase < 0.5:
            return np.sin(2*np.pi*F*t)
        else:
            return 0.0
    run(burst, "7. Burst-and-coast (0.5s on / 0.5s off)")

    # 8-9. asymmetric stroke
    run(lambda t: asymm_stroke(t, fast_positive=True),
        "8. fast+/slow- stroke")
    run(lambda t: asymm_stroke(t, fast_positive=False),
        "9. fast-/slow+ stroke")

    # 10. high frequency pulse
    run(lambda t: np.sign(np.sin(2*np.pi*F*t)) * 0.9,
        "10. square wave (rapid transitions)")

    # 11-12. 더 큰 DC offset
    run(lambda t: 0.6 + 0.4*np.sin(2*np.pi*F*t), "11. DC offset +0.6 (강한 +쪽 평균)")
    run(lambda t: -0.6 + 0.4*np.sin(2*np.pi*F*t), "12. DC offset -0.6 (강한 -쪽 평균)")

    # 13-14. 더 낮은 주파수에서 비대칭
    F_low = 1.5
    run(lambda t: 0.7*np.sin(2*np.pi*F_low*t) + 0.3*np.sin(2*np.pi*2*F_low*t),
        f"13. sin+0.3*sin(2x) at {F_low}Hz")

    # 15. 정상 sine을 다른 주파수에서 (참조)
    run(lambda t: 0.0, "0. ctrl=0 (정지 reference)")


if __name__ == "__main__":
    main()
