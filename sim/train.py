"""SAC로 FishSwimEnv 학습.

기본은 학습 중 별도 thread에서 MuJoCo viewer를 띄워 정책 진화를 실시간 관찰.
viewer가 필요 없으면 --no-viewer.

사용:
    python3 train.py                                # 기본 50만 step + viewer
    python3 train.py --steps 20000 --tag smoke      # 짧은 smoke run
    python3 train.py --no-viewer                    # viewer 없이 빠르게
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import copy
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from fish_env import FishSwimEnv


class PolicySnapshotCallback(BaseCallback):
    """매 sync_every step마다 model.policy를 deepcopy해서 viewer thread가 쓸
    snapshot으로 model_holder에 둔다. 학습 thread와 viewer thread 사이의
    PyTorch race condition을 피하기 위함."""

    def __init__(self, model_holder: dict, sync_every: int = 500):
        super().__init__()
        self.model_holder = model_holder
        self.sync_every = sync_every

    def _on_step(self) -> bool:
        if self.n_calls % self.sync_every == 0:
            snap = copy.deepcopy(self.model.policy).cpu().eval()
            self.model_holder["policy"] = snap
        return True


def make_env():
    return Monitor(FishSwimEnv())


def start_viewer_thread(model_holder: dict, stop_event: threading.Event) -> threading.Thread:
    """별도 thread에서 viewer를 띄우고 model_holder['policy'] (deepcopy 스냅샷)로 rollout.

    snapshot은 학습 thread의 callback이 매 N step마다 갱신.
    snapshot이 아직 없으면(=학습 초기 N step 미만) random action.
    """
    eval_env = FishSwimEnv()
    state = {"obs": eval_env.reset()[0], "ep": 1, "ep_reward": 0.0, "step": 0}

    def loop():
        with mujoco.viewer.launch_passive(eval_env.model, eval_env.data) as v:
            print("[viewer] 창 열림. 마우스로 카메라 조작. 창 닫으면 viewer만 꺼지고 학습은 계속.")
            last_print = time.time()
            while v.is_running() and not stop_event.is_set():
                t0 = time.time()
                snap = model_holder.get("policy")
                if snap is None:
                    action = eval_env.action_space.sample()
                else:
                    action, _ = snap.predict(state["obs"], deterministic=False)
                state["obs"], r, term, trunc, info = eval_env.step(action)
                state["ep_reward"] += r
                state["step"] += 1

                v.sync()

                if time.time() - last_print > 2.0:
                    print(f"[viewer] ep{state['ep']} step={state['step']} "
                          f"dist={info['distance']:.3f} reward={state['ep_reward']:.2f}")
                    last_print = time.time()

                if term or trunc:
                    print(f"[viewer] ep{state['ep']} 끝: reward={state['ep_reward']:.3f} "
                          f"final_dist={info['distance']:.3f} reached={info['reached']}")
                    state["obs"], _ = eval_env.reset(seed=state["ep"])
                    state["ep"] += 1
                    state["ep_reward"] = 0.0
                    state["step"] = 0

                # 실시간 페이스 (50Hz)
                rest = eval_env.dt - (time.time() - t0)
                if rest > 0:
                    time.sleep(rest)
            print("[viewer] 종료.")

    t = threading.Thread(target=loop, daemon=True, name="viewer")
    t.start()
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--tag", type=str, default="sac")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--no-viewer", action="store_true",
                   help="학습 중 viewer를 띄우지 않음 (빠른 학습)")
    p.add_argument("--final-viewer", action="store_true", default=True,
                   help="학습 완료 후 viewer로 최종 정책 무한 rollout (Ctrl+C로 종료)")
    args = p.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(make_env, n_envs=args.n_envs)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        tensorboard_log=str(run_dir),
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1_000,
    )

    # 학습 중 viewer thread (정책 snapshot 방식, race-free)
    viewer_thread = None
    stop_event = threading.Event()
    model_holder = {"policy": None}
    callbacks = None

    if not args.no_viewer:
        viewer_thread = start_viewer_thread(model_holder, stop_event)
        time.sleep(1.0)  # viewer가 뜰 시간 확보
        callbacks = PolicySnapshotCallback(model_holder, sync_every=500)
        print(f"[train] {args.steps} step 학습 시작. viewer 정책은 500 step마다 갱신.")

    try:
        model.learn(total_timesteps=args.steps, progress_bar=True, callback=callbacks)
    finally:
        if viewer_thread is not None:
            stop_event.set()

    model.save(run_dir / "model")
    print(f"saved -> {run_dir / 'model.zip'}")

    eval_env = make_env()
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"eval mean_reward = {mean:.3f} ± {std:.3f}")
    eval_env.close()

    # 최종 정책 viewer — 사용자가 창 닫을 때까지
    if args.final_viewer and not args.no_viewer:
        print("\n[final] 학습 완료. 최종 정책으로 viewer 무한 rollout. 창 닫거나 Ctrl+C로 종료.")
        try:
            from view_policy import main as view_main
            import sys
            sys.argv = ["view_policy.py", str(run_dir / "model.zip")]
            view_main()
        except KeyboardInterrupt:
            print("\n[final] 사용자 종료.")


if __name__ == "__main__":
    main()
