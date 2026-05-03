"""SAC로 FishSwimEnv 학습 (curriculum 지원).

기본은 학습 중 별도 thread에서 MuJoCo viewer를 띄워 정책 진화를 실시간 관찰.
viewer가 필요 없으면 --no-viewer.

curriculum 인자:
  --theta-min, --theta-max  목표 각도 범위 (라디안)
  --success-radius          도달 판정 거리
  --init-from <model.zip>   이전 단계 정책에서 fine-tuning 시작
  --success-threshold       reach_rate가 이 비율 이상이면 학습 조기 종료 (0=비활성)
  --eval-window             reach_rate 측정용 최근 에피소드 수
  --check-every             reach_rate 체크 주기 (step)

사용:
    python3 train.py                                # 기본 50만 step + viewer
    python3 train.py --steps 20000 --tag smoke      # smoke run
    python3 train.py --no-viewer                    # viewer 없이 빠르게
    python3 train.py --tag s2 --init-from runs/s1/model.zip --success-radius 0.04
"""

from __future__ import annotations

import argparse
import collections
import copy
import math
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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


class CurriculumStopCallback(BaseCallback):
    """에피소드 통계를 추적해 (1) 직관적 메트릭을 TB에 기록하고 (2) reach_rate가
    threshold 이상이면 학습을 조기 종료한다.

    TB에 기록하는 metric (`fish/` namespace):
      - fish/success_rate    최근 N ep 도달률 (0~1)
      - fish/final_distance  최근 N ep 평균 종료 거리 (m)
      - fish/episode_seconds 최근 N ep 평균 ep 길이 (초, ep_len * dt)
      - fish/avg_align       최근 N ep 평균 정렬도 (-1~+1, 종료 시점)

    threshold=0이면 조기 종료 비활성 (정해진 step까지 학습).
    """

    def __init__(self, threshold: float = 0.9, window: int = 100,
                 check_every: int = 5000, min_steps: int = 30_000,
                 dt: float = 0.02, verbose: int = 1):
        super().__init__(verbose)
        self.threshold = threshold
        self.window = window
        self.check_every = check_every
        self.min_steps = min_steps
        self.dt = dt
        self.recent_reached: collections.deque = collections.deque(maxlen=window)
        self.recent_distances: collections.deque = collections.deque(maxlen=window)
        self.recent_lengths: collections.deque = collections.deque(maxlen=window)
        self.recent_aligns: collections.deque = collections.deque(maxlen=window)
        self.last_check = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        ep_ended = False
        for info, done in zip(infos, dones):
            if done:
                ep_ended = True
                self.recent_reached.append(1.0 if info.get("reached", False) else 0.0)
                self.recent_distances.append(float(info.get("distance", 0.0)))
                self.recent_aligns.append(float(info.get("align", 0.0)))
                # SB3 Monitor가 ep 길이를 info["episode"]["l"]로 자동 채움
                ep_block = info.get("episode")
                if ep_block is not None:
                    self.recent_lengths.append(float(ep_block.get("l", 0.0)))

        # ep 종료 시점에 TB 메트릭 갱신 (학습 부담 최소)
        if ep_ended and self.recent_reached:
            n = len(self.recent_reached)
            self.logger.record("fish/success_rate",
                               sum(self.recent_reached) / n)
            self.logger.record("fish/final_distance",
                               sum(self.recent_distances) / len(self.recent_distances))
            self.logger.record("fish/avg_align",
                               sum(self.recent_aligns) / len(self.recent_aligns))
            if self.recent_lengths:
                self.logger.record(
                    "fish/episode_seconds",
                    (sum(self.recent_lengths) / len(self.recent_lengths)) * self.dt,
                )

        # 조기 종료 체크 (threshold > 0일 때만)
        if self.threshold <= 0:
            return True
        if self.num_timesteps < self.min_steps:
            return True
        if self.n_calls - self.last_check >= self.check_every:
            self.last_check = self.n_calls
            if len(self.recent_reached) >= max(20, self.window // 2):
                rate = sum(self.recent_reached) / len(self.recent_reached)
                print(f"[curriculum] step {self.num_timesteps}: reach_rate = {rate:.1%} "
                      f"(window={len(self.recent_reached)})")
                if rate >= self.threshold:
                    print(f"[curriculum] ✓ reach_rate {rate:.1%} >= {self.threshold:.0%} "
                          f"— 학습 조기 종료 (다음 단계로)")
                    return False
        return True


def save_training_plots(tb_log_root: Path, tag: str, plot_dir: Path) -> None:
    """tensorboard event를 읽어 학습 곡선 PNG 저장.

    tb_log_root 안에 tag로 시작하는 dir(예: s1_forward_1)을 찾아 가장 최근 것 사용.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import matplotlib
        matplotlib.use("Agg")  # 헤드리스 backend (Qt 충돌 회피)
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[plot] 의존성 없음 ({e}), skip")
        return

    # tag로 시작하는 dir 중 가장 최근 것 (예: s1_forward, s1_forward_1, s1_forward_2)
    candidates = sorted([d for d in tb_log_root.iterdir()
                         if d.is_dir() and d.name.startswith(tag)])
    if not candidates:
        print(f"[plot] {tb_log_root}에 '{tag}*' dir 없음, skip")
        return
    sac_dir = candidates[-1]

    ea = EventAccumulator(str(sac_dir))
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    wanted = [
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
    ]
    metrics = [m for m in wanted if m in available]
    if not metrics:
        print(f"[plot] 측정 가능 지표 없음 (available={available[:5]}), skip")
        return

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        events = ea.Scalars(m)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, linewidth=1.2)
        ax.set_title(f"{tag} — {m}")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"{tag}.png"
    plt.savefig(out, dpi=100)
    plt.close(fig)
    print(f"[plot] saved -> {out}")


def make_env_factory(target_theta_range, success_radius,
                     episode_seconds=10.0, align_weight=0.02):
    """env 인자(curriculum용)를 closure로 묶어 SB3가 부를 수 있는 0-arg make_env 반환."""
    def make_env():
        return Monitor(FishSwimEnv(
            target_theta_range=target_theta_range,
            success_radius=success_radius,
            episode_seconds=episode_seconds,
            align_weight=align_weight,
        ))
    return make_env


def start_viewer_thread(model_holder: dict, stop_event: threading.Event,
                        target_theta_range, success_radius,
                        episode_seconds=10.0, align_weight=0.02) -> threading.Thread:
    """별도 thread에서 viewer를 띄우고 model_holder['policy'] (deepcopy 스냅샷)로 rollout.

    snapshot은 학습 thread의 callback이 매 N step마다 갱신.
    snapshot이 아직 없으면(=학습 초기 N step 미만) random action.
    """
    eval_env = FishSwimEnv(
        target_theta_range=target_theta_range,
        success_radius=success_radius,
        episode_seconds=episode_seconds,
        align_weight=align_weight,
    )
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
    p.add_argument("--steps", type=int, default=500_000,
                   help="학습 최대 step (curriculum 조기 종료 가능)")
    p.add_argument("--tag", type=str, default="sac")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--no-final-viewer", action="store_true",
                   help="학습 후 최종 정책 viewer 자동 띄움 비활성")
    # curriculum
    p.add_argument("--theta-min", type=float, default=-math.pi,
                   help="목표 각도 최소값 (rad). 직진=π, 머리쪽=π/2")
    p.add_argument("--theta-max", type=float, default=math.pi)
    p.add_argument("--success-radius", type=float, default=0.08)
    p.add_argument("--init-from", type=str, default=None,
                   help="이전 단계 model.zip에서 로드해 fine-tuning")
    p.add_argument("--success-threshold", type=float, default=0.0,
                   help="reach_rate가 이 비율 이상 도달하면 학습 조기 종료. 0이면 비활성")
    p.add_argument("--eval-window", type=int, default=100,
                   help="reach_rate 측정용 최근 에피소드 수")
    p.add_argument("--check-every", type=int, default=5000,
                   help="reach_rate 체크 주기 step")
    args = p.parse_args()

    run_dir = Path(__file__).parent / "runs" / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    theta_range = (args.theta_min, args.theta_max)
    make_env = make_env_factory(theta_range, args.success_radius)
    env = make_vec_env(make_env, n_envs=args.n_envs)

    if args.init_from:
        print(f"[train] {args.init_from} 정책 로드 — fine-tuning")
        model = SAC.load(args.init_from, env=env, device=args.device)
        model.tensorboard_log = str(run_dir)
    else:
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
            ent_coef="auto_0.1",  # entropy 초기값 0.1 (collapse 늦춤)
        )

    # callbacks 구성
    callbacks = []
    viewer_thread = None
    stop_event = threading.Event()
    model_holder = {"policy": None}

    if not args.no_viewer:
        viewer_thread = start_viewer_thread(model_holder, stop_event,
                                            theta_range, args.success_radius)
        time.sleep(1.0)
        callbacks.append(PolicySnapshotCallback(model_holder, sync_every=500))
        print(f"[train] {args.steps} step max + viewer (정책 500 step마다 갱신)")

    if args.success_threshold > 0:
        callbacks.append(CurriculumStopCallback(
            threshold=args.success_threshold,
            window=args.eval_window,
            check_every=args.check_every,
        ))
        print(f"[train] curriculum: reach_rate ≥ {args.success_threshold:.0%}이면 조기 종료")

    cb = CallbackList(callbacks) if callbacks else None

    try:
        model.learn(total_timesteps=args.steps, progress_bar=True, callback=cb)
    finally:
        if viewer_thread is not None:
            stop_event.set()

    model.save(run_dir / "model")
    print(f"saved -> {run_dir / 'model.zip'}")

    # 학습 곡선 PNG로 저장 (sim/plots/<tag>.png).
    # train.py 단독 사용 시 tb_log_root는 run_dir 자체 (SAC_1 하위).
    plot_dir = Path(__file__).parent / "plots"
    save_training_plots(run_dir, "SAC", plot_dir)  # default tag 'SAC'로 SAC_N dir 찾음

    eval_env = make_env()
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"eval mean_reward = {mean:.3f} ± {std:.3f}")
    eval_env.close()

    # 최종 정책 viewer (curriculum 모드에선 보통 끔)
    if not args.no_final_viewer and not args.no_viewer:
        print("\n[final] 학습 완료. 최종 정책으로 viewer 무한 rollout. 창 닫으면 종료.")
        try:
            from view_policy import main as view_main
            import sys
            sys.argv = ["view_policy.py", str(run_dir / "model.zip")]
            view_main()
        except KeyboardInterrupt:
            print("\n[final] 사용자 종료.")


if __name__ == "__main__":
    main()
