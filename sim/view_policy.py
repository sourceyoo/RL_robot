"""학습된 SAC 정책을 MuJoCo viewer 안에서 자동 rollout.

사용:
    python3 view_policy.py                       # smoke test 모델로
    python3 view_policy.py runs/<태그>/model.zip  # 다른 모델 지정
    python3 view_policy.py --sine                # 정책 대신 2Hz 사인파
    python3 view_policy.py --zero                # ctrl=0 (정지 상태)

종료: viewer 창 닫기 또는 Ctrl+C
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from fish_env import FishSwimEnv

ROOT = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", nargs="?", default=str(ROOT / "runs/smoke/model.zip"))
    p.add_argument("--sine", action="store_true", help="정책 대신 2Hz 사인파")
    p.add_argument("--zero", action="store_true", help="ctrl=0 정지 상태")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = FishSwimEnv()
    obs, _ = env.reset(seed=args.seed)
    model, data = env.model, env.data

    sac = None
    if not (args.sine or args.zero):
        from stable_baselines3 import SAC
        if not Path(args.model_path).exists():
            print(f"모델 파일 없음: {args.model_path}\n--sine 또는 --zero 옵션 사용 가능.")
            return
        sac = SAC.load(args.model_path, device="cpu")
        print(f"[정책] {args.model_path}")
    elif args.sine:
        print("[수동] 2Hz 사인파")
    else:
        print("[정지] ctrl=0")

    print("viewer 창에서 마우스로 카메라 조작. 창 닫으면 종료.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        ep_reward = 0.0
        ep_count = 1
        last_print = time.time()

        while viewer.is_running():
            sim_start = time.time()

            # 행동 결정
            if sac is not None:
                action, _ = sac.predict(obs, deterministic=True)
            elif args.sine:
                action = np.array([0.95 * np.sin(2 * np.pi * 2.0 * data.time)])
            else:
                action = np.zeros(model.nu)

            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            step += 1

            viewer.sync()

            # 1초마다 상태 출력
            if time.time() - last_print > 1.0:
                print(f"  [ep{ep_count}] step={step}  dist={info['distance']:.3f}  "
                      f"reward_so_far={ep_reward:.2f}")
                last_print = time.time()

            if terminated or truncated:
                print(f"  [ep{ep_count} 끝] reward={ep_reward:.3f}  "
                      f"final_dist={info['distance']:.3f}  reached={info['reached']}")
                obs, _ = env.reset(seed=args.seed + ep_count)
                ep_reward = 0.0
                step = 0
                ep_count += 1

            # 실시간 페이스 (50Hz env step)
            elapsed = time.time() - sim_start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)


if __name__ == "__main__":
    main()
