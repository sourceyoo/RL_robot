"""Curriculum 학습 — 단일 프로세스 (viewer 한 번만 열림).

각 stage가 reach_rate ≥ threshold (기본 90%)에 도달하면 다음 stage로.
min_steps (기본 30k) 이전엔 조기 종료 안 함 — 충분히 학습 후 평가.
도달 못해도 max_steps에 걸리면 강제 진행 (안전장치).

이전 디자인 (subprocess로 train.py 호출)과 달리 단일 프로세스에서 SAC를
새로 만들거나 이전 모델 로드해 stage를 이어감. viewer thread는 한 번만 시작
되어 모든 stage 통과 — 깜빡임 없음.

사용:
    python3 curriculum.py                      # 전체 단계 + viewer
    python3 curriculum.py --no-viewer          # viewer 없이 빠르게
    python3 curriculum.py --threshold 0.85     # 도달률 임계 변경
    python3 curriculum.py --min-steps 50000    # 최소 step 강화
    python3 curriculum.py --start-stage 2      # Stage 2부터 시작
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env

from train import (
    PolicySnapshotCallback,
    CurriculumStopCallback,
    EntCoefFloorCallback,
    make_env_factory,
    start_viewer_thread,
    save_training_plots,
)


PI = math.pi

# v4: action history obs 추가 (시간적 비대칭 ctrl 패턴 학습 enable).
# yaw_test.py 진단으로 D2 패턴 등 비대칭이 yaw 회전 핵심임 확인.
# v11: 8 → 16. v8/v9/v10 모두 ent_floor 카드로 천장 못 뚫음 — 진짜 병목 = 정책
# 표현력. N=8은 1/2 wag cycle만 커버 (3Hz × 17 step/cycle / 2 ≈ 8).
# N=16은 1 wag cycle 커버 → D2 패턴(75/25 비대칭 stroke) 표현 가능.
# obs dim: 11 + 16 = 27.
ACTION_HISTORY_N = 16

# v10: Stage별 차등 ent_floor.
# v8 floor 0.005 / v9 floor 0.002 결과 종합 — 단일 floor로 전체 cover 불가 입증.
# 작은 회전(s3a/b)은 정확도 stage → 낮은 floor (0.002~0.003), 큰 회전(s3c/d)은
# 비대칭 ctrl 패턴 발견 위해 높은 floor (0.005~0.006). v7 floor 0.02는 재앙이라
# 상한 보수적으로.
STAGES = [
    {
        "tag": "s1_forward",
        "desc": "Stage 1 — 직진 (target θ=π fixed)",
        "theta_min": PI, "theta_max": PI,
        "success_radius": 0.08,
        "max_steps": 400_000,
        "ent_floor": 0.002,
    },
    {
        "tag": "s2_anchor",
        "desc": "Stage 2 — 안착 (success_radius 0.04)",
        "theta_min": PI, "theta_max": PI,
        "success_radius": 0.04,
        "max_steps": 400_000,
        "ent_floor": 0.002,
    },
    # v6: Stage 3 전체 ep_seconds 30s (10s/20s → 30s 통일).
    # yaw_test.py D2 best 4.6°/s × 30s = 138° 회전 가능 → ±90° 회전+추진(0.5m, 1.8s)에 9s 여유.
    {
        "tag": "s3a_arc15",
        "desc": "Stage 3a — 좌우 ±15° (θ ∈ π ± π/12)",
        "theta_min": PI - PI / 12, "theta_max": PI + PI / 12,
        "success_radius": 0.08,
        "max_steps": 200_000,
        "episode_seconds": 30.0,
        "ent_floor": 0.002,
    },
    {
        "tag": "s3b_arc30",
        "desc": "Stage 3b — 좌우 ±30° (θ ∈ π ± π/6)",
        "theta_min": PI - PI / 6, "theta_max": PI + PI / 6,
        "success_radius": 0.08,
        "max_steps": 250_000,
        "episode_seconds": 30.0,
        "ent_floor": 0.003,
    },
    {
        "tag": "s3c_arc60",
        "desc": "Stage 3c — 좌우 ±60° (θ ∈ π ± π/3)",
        "theta_min": PI - PI / 3, "theta_max": PI + PI / 3,
        "success_radius": 0.08,
        "max_steps": 350_000,
        "episode_seconds": 30.0,
        "ent_floor": 0.005,
    },
    {
        "tag": "s3d_arc90",
        "desc": "Stage 3d — 좌우 ±90° (θ ∈ [π/2, 3π/2], 전체 head arc)",
        "theta_min": PI / 2, "theta_max": 3 * PI / 2,
        "success_radius": 0.08,
        "max_steps": 500_000,
        "episode_seconds": 30.0,
        "ent_floor": 0.006,
    },
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.9,
                   help="다음 단계로 넘어갈 reach_rate 임계 (기본 0.9 = 90%%)")
    p.add_argument("--min-steps", type=int, default=30_000,
                   help="각 단계 최소 학습 step (이 이전엔 조기 종료 안 함)")
    p.add_argument("--no-viewer", action="store_true",
                   help="viewer 없이 빠르게")
    p.add_argument("--start-stage", type=int, default=1,
                   help="시작 단계 번호 (1-indexed). 이전 단계 model.zip이 있어야 함")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.start_stage < 1 or args.start_stage > len(STAGES):
        print(f"start-stage는 1~{len(STAGES)} 사이여야 합니다.")
        return

    sim_dir = Path(__file__).parent
    runs_dir = sim_dir / "runs"
    plot_dir = sim_dir / "plots"
    # 모든 단계의 tensorboard log를 model3_vN 폴더 안에 묶어 TB UI에서 v별 비교 가능.
    # 새 학습 시작할 때마다 v숫자를 올려도 되고, 같은 v 안에서 stage 진행도 가능.
    tb_dir = sim_dir / "tb_logs" / "model3_v11"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # Viewer thread 단 한 번만 — 첫 stage env로 시작, 모든 stage 통과
    model_holder = {"policy": None}
    stop_event = threading.Event()
    viewer_thread = None

    if not args.no_viewer:
        first = STAGES[args.start_stage - 1]
        viewer_thread = start_viewer_thread(
            model_holder, stop_event,
            (first["theta_min"], first["theta_max"]),
            first["success_radius"],
            episode_seconds=first.get("episode_seconds", 10.0),
            action_history_n=ACTION_HISTORY_N,
        )
        time.sleep(1.0)  # viewer가 뜰 시간

    # 시작점 모델 (Stage 2 이상부터 시작 시)
    prev_model_path = None
    if args.start_stage > 1:
        prev_tag = STAGES[args.start_stage - 2]["tag"]
        prev_model_path = runs_dir / prev_tag / "model.zip"
        if not prev_model_path.exists():
            print(f"이전 단계 모델 {prev_model_path} 없음. Stage 1부터 시작하세요.")
            if viewer_thread is not None:
                stop_event.set()
            return
        print(f"[curriculum] Stage {args.start_stage}부터. init = {prev_model_path}")

    try:
        for i in range(args.start_stage - 1, len(STAGES)):
            stage = STAGES[i]
            print(f"\n{'='*60}\n{stage['desc']}\n  threshold={args.threshold:.0%}, "
                  f"min_steps={args.min_steps:,}, max_steps={stage['max_steps']:,}\n"
                  f"{'='*60}")

            run_dir = runs_dir / stage["tag"]
            run_dir.mkdir(parents=True, exist_ok=True)

            theta_range = (stage["theta_min"], stage["theta_max"])
            ep_sec = stage.get("episode_seconds", 10.0)
            make_env = make_env_factory(theta_range, stage["success_radius"],
                                        episode_seconds=ep_sec,
                                        action_history_n=ACTION_HISTORY_N)
            env = make_vec_env(make_env, n_envs=1)

            if prev_model_path is not None:
                print(f"[curriculum] {prev_model_path} 정책 로드 (fine-tuning)")
                model = SAC.load(str(prev_model_path), env=env, device=args.device)
                model.tensorboard_log = str(tb_dir)
            else:
                model = SAC(
                    "MlpPolicy", env, verbose=1, device=args.device,
                    tensorboard_log=str(tb_dir),
                    learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                    tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                    learning_starts=1_000,
                    ent_coef="auto_0.1",  # entropy 초기값 0.1 (collapse 늦춤)
                )

            callbacks = []
            if not args.no_viewer:
                callbacks.append(PolicySnapshotCallback(model_holder, sync_every=500))
            callbacks.append(CurriculumStopCallback(
                threshold=args.threshold, window=100,
                check_every=5000, min_steps=args.min_steps,
            ))
            # v10: stage별 차등 floor. 작은 회전(s3a/b 0.002~0.003)은 정확도, 큰 회전(s3c 0.005,
            # s3d 0.006)은 비대칭 ctrl 패턴 발견 위해 entropy 유지. v8 0.005·v9 0.002 모두
            # 한쪽만 만족했던 결과의 종합.
            callbacks.append(EntCoefFloorCallback(floor=stage["ent_floor"]))
            print(f"[curriculum] ent_floor = {stage['ent_floor']}")
            cb = CallbackList(callbacks) if callbacks else None

            model.learn(
                total_timesteps=stage["max_steps"],
                progress_bar=True,
                callback=cb,
                reset_num_timesteps=True,           # stage별 step 0부터 카운트
                tb_log_name=stage["tag"],           # TB run 이름을 stage tag로 (s1_forward, s2_anchor, ...)
            )

            model.save(run_dir / "model")
            print(f"[curriculum] saved -> {run_dir / 'model.zip'}")

            save_training_plots(tb_dir, stage["tag"], plot_dir)

            prev_model_path = run_dir / "model.zip"
            env.close()

        print(f"\n{'='*60}\n[curriculum] 모든 단계 완료!\n"
              f"최종 모델: {prev_model_path}\n"
              f"학습 곡선: {plot_dir}/\n"
              f"확인: python3 view_policy.py {prev_model_path}\n"
              f"{'='*60}")
    finally:
        if viewer_thread is not None:
            stop_event.set()


if __name__ == "__main__":
    main()
