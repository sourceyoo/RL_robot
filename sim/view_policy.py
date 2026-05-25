"""학습된 SAC 정책을 MuJoCo viewer 안에서 자동 rollout.

obs dim을 SAC 모델에서 자동 추론해 env action_history_n을 맞춤 (v4: 8, v11+: 16).
stage별 theta_range, success_radius, episode_seconds 인자로 학습 환경 그대로 재현 가능.

사용:
    python3 view_policy.py runs/s3d_arc90/model.zip                    # 자동 추론 + random target
    python3 view_policy.py runs/s3d_arc90/model.zip \\
        --theta-min 1.5708 --theta-max 4.7124 --episode-seconds 30      # s3d 학습 환경 그대로
    python3 view_policy.py --sine                                       # 정책 대신 6Hz 사인파
    python3 view_policy.py --zero                                       # ctrl=0 정지

종료: viewer 창 닫기 또는 Ctrl+C
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from fish_env import FishSwimEnv

ROOT = Path(__file__).parent


def infer_action_history_n(sac_model) -> int:
    """SAC 모델의 obs_space dim에서 action_history_n 추론. obs dim = 13 + N (v32-C+)."""
    obs_dim = sac_model.observation_space.shape[0]
    n = obs_dim - 13
    if n < 0:
        raise RuntimeError(f"비호환 모델: obs_dim={obs_dim} (13 미만)")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_path", nargs="?", default=str(ROOT / "runs/smoke/model.zip"))
    p.add_argument("--sine", action="store_true", help="정책 대신 6Hz 사인파")
    p.add_argument("--zero", action="store_true", help="ctrl=0 정지 상태")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--theta-min", type=float, default=-math.pi,
                   help="목표 각도 최소 (rad). 학습 stage 그대로면 stage 그 값.")
    p.add_argument("--theta-max", type=float, default=math.pi)
    p.add_argument("--success-radius", type=float, default=0.08)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--action-history-n", type=int, default=None,
                   help="None이면 SAC 모델에서 자동 추론")
    p.add_argument("--max-episodes", type=int, default=0,
                   help="이 ep 수만큼 끝나면 viewer 자동 종료 (0=무한).")
    args = p.parse_args()

    sac = None
    action_history_n = args.action_history_n if args.action_history_n is not None else 0

    if not (args.sine or args.zero):
        from stable_baselines3 import SAC
        if not Path(args.model_path).exists():
            print(f"모델 파일 없음: {args.model_path}\n--sine 또는 --zero 옵션 사용 가능.")
            return
        sac = SAC.load(args.model_path, device="cpu")
        if args.action_history_n is None:
            action_history_n = infer_action_history_n(sac)
        print(f"[정책] {args.model_path}  (obs_dim={sac.observation_space.shape[0]}, "
              f"action_history_n={action_history_n})")
    elif args.sine:
        print("[수동] 6Hz 사인파")
    else:
        print("[정지] ctrl=0")

    env = FishSwimEnv(
        target_theta_range=(args.theta_min, args.theta_max),
        success_radius=args.success_radius,
        episode_seconds=args.episode_seconds,
        action_history_n=action_history_n,
    )
    obs, _ = env.reset(seed=args.seed)
    model, data = env.model, env.data

    print(f"환경: theta=[{args.theta_min:.3f}, {args.theta_max:.3f}], "
          f"success={args.success_radius}m, ep={args.episode_seconds}s, "
          f"align_w={env.align_weight:.5f}")
    print("viewer 창에서 마우스로 카메라 조작. 창 닫으면 종료.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        ep_reward = 0.0
        ep_count = 1
        ep_reached_count = 0
        last_print = time.time()
        tail_buf: list[float] = []
        fin_buf: list[float] = []

        while viewer.is_running():
            sim_start = time.time()

            if sac is not None:
                action, _ = sac.predict(obs, deterministic=True)
            elif args.sine:
                action = np.array([1.0 * np.sin(2 * np.pi * 6.0 * data.time)])
            else:
                action = np.zeros(model.nu)

            obs, r, terminated, truncated, info = env.step(action)
            ep_reward += r
            step += 1
            tail_buf.append(float(data.qpos[3]))  # IDX_TAIL=3
            fin_buf.append(float(data.qpos[4]))   # IDX_FIN=4

            viewer.sync()

            if time.time() - last_print > 1.0:
                tail_amp = float(np.degrees(max(tail_buf) - min(tail_buf))) if tail_buf else 0.0
                fin_amp = float(np.degrees(max(fin_buf) - min(fin_buf))) if fin_buf else 0.0
                print(f"  [ep{ep_count}] step={step}  dist={info['distance']:.3f}  "
                      f"align={info['align']:+.2f}  tail_amp={tail_amp:.1f}°  fin_amp={fin_amp:.1f}°  "
                      f"reward={ep_reward:.2f}")
                tail_buf.clear()
                fin_buf.clear()
                last_print = time.time()

            if terminated or truncated:
                if info["reached"]:
                    ep_reached_count += 1
                print(f"  [ep{ep_count} 끝] reward={ep_reward:.3f}  "
                      f"final_dist={info['distance']:.3f}  reached={info['reached']}  "
                      f"누적 reach: {ep_reached_count}/{ep_count}")
                if args.max_episodes > 0 and ep_count >= args.max_episodes:
                    print(f"[done] {args.max_episodes} ep 완료. viewer 종료.")
                    break
                obs, _ = env.reset(seed=args.seed + ep_count)
                ep_reward = 0.0
                step = 0
                ep_count += 1
                tail_buf.clear()
                fin_buf.clear()

            elapsed = time.time() - sim_start
            if elapsed < env.dt:
                time.sleep(env.dt - elapsed)


if __name__ == "__main__":
    main()
