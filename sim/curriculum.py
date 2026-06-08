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
import subprocess
import sys
import threading
import time
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env

from fish_env import FishSwimEnv
from train import (
    PolicySnapshotCallback,
    CurriculumStopCallback,
    EntCoefFloorCallback,
    make_env_factory,
    start_viewer_thread,
    save_training_plots,
)


PI = math.pi

# action history obs: 비대칭 ctrl 패턴(yaw 회전 핵심) 학습용. N=20 = 1.25 wag cycle.
# 변경 시 SAC.load obs Box mismatch — s1부터 새 학습 필요 (함정 #9).
ACTION_HISTORY_N = 20

# Stage별 차등 ent_floor: 작은 회전(s3a/b)은 정확도 → 낮은 floor, 큰 회전(s3c/d)은
# 비대칭 ctrl 패턴 탐색 → 높은 floor. floor 0.02 이상은 학습 붕괴.
STAGES = [
    {
        "tag": "s1_forward",
        "desc": "Stage 1 — 직진 (target θ=π fixed)",
        "theta_min": PI, "theta_max": PI,
        "success_radius": 0.08,
        "max_steps": 1_000_000,
        "episode_seconds": 20.0,
        "ent_floor": 0.002,
    },
    # m4_v17: s2_anchor 제거 — amp 고정 1.0으로 정밀 정지 학습 불가능. 본질이 s1 sr 0.04 버전이라 의미 없음.
    # Stage 3 전체 ep_seconds 60s: m4 fluidcoef 변경 후 추진 속도 ~50%↓·yaw rate 1/3.6↓
    # 보정 (이전 30s에서 2배). yaw rate 1.27°/s × 60s = 76° 회전 여유.
    # m4_v13~: annular offset_range로 sampling (theta = π ± U(min, max), sign random).
    # inner = 9.2° (= arcsin(0.08/0.5), 직진 자연 한계). 직진 우연 도달률을 평가/학습에서 제거.
    # 인접 stage disjoint: s3a/b/c/d outer = 이전 stage outer로부터 시작.
    {
        "tag": "s3a_arc15",
        "desc": "Stage 3a — annular |θ-π| ∈ [9.2°, 15°]",
        "theta_min": PI - PI / 12, "theta_max": PI + PI / 12,  # legacy (offset_range 있으면 미사용)
        "theta_offset_min": PI * 9.2 / 180.0, "theta_offset_max": PI / 12,
        "success_radius": 0.08,
        "max_steps": 2_000_000,   # m4_cpg_v21: 1M→2M. lat6로 좌 수렴 더 빡빡 → 여유.
        "episode_seconds": 60.0,
        "ent_floor": 0.008,       # m4_cpg_v21: 0.002→0.008. v19에서 좌 dead 탈출시킨 검증값(탐색 강화).
        "ent_floor_end": 0.004,   # 0.008→0.004 linear decay: 초반 강탐색(dead 탈출)+후반 곧장 정밀 수렴.
    },
    {
        "tag": "s3b_arc30",
        "desc": "Stage 3b — annular |θ-π| ∈ [15°, 30°]",
        "theta_min": PI - PI / 6, "theta_max": PI + PI / 6,
        "theta_offset_min": PI / 12, "theta_offset_max": PI / 6,
        "success_radius": 0.08,
        "max_steps": 1_000_000,
        "episode_seconds": 60.0,
        "ent_floor": 0.003,
        # ent_floor linear decay 0.003 → 0: 학습 후반 deterministic policy 수렴.
        # stochastic-deterministic gap 좁히기 (함정 #13 대응).
        "ent_floor_end": 0.0,
    },
    {
        "tag": "s3c_arc60",
        "desc": "Stage 3c — annular |θ-π| ∈ [30°, 60°]",
        "theta_min": PI - PI / 3, "theta_max": PI + PI / 3,
        "theta_offset_min": PI / 6, "theta_offset_max": PI / 3,
        "success_radius": 0.08,
        "max_steps": 1_000_000,
        "episode_seconds": 60.0,
        # 큰 회전: floor ↑로 narrow mode 깨고 비대칭 ctrl 탐색.
        "ent_floor": 0.008,
    },
    {
        "tag": "s3d_arc90",
        "desc": "Stage 3d — annular |θ-π| ∈ [60°, 90°]",
        "theta_min": PI / 2, "theta_max": 3 * PI / 2,
        "theta_offset_min": PI / 3, "theta_offset_max": PI / 2,
        "success_radius": 0.08,
        "max_steps": 1_000_000,
        "episode_seconds": 60.0,
        "ent_floor": 0.010,
    },
    # m4_v20: forgetting 회복 stage. 5 sub 균등 mixed sampling + sub trigger
    # (모든 sub의 reach_rate ≥ threshold일 때만 trigger). 마지막 stage에 chain.
    {
        "tag": "s_all_mix",
        "desc": "Stage 6 — 5 sub mixed (s1+s3a+s3b+s3c+s3d 균등, sub trigger)",
        "theta_min": -PI, "theta_max": PI,  # legacy fallback (sub_distributions 우선)
        "success_radius": 0.08,
        "max_steps": 2_000_000,
        "episode_seconds": 60.0,
        "ent_floor": 0.005,
        "sub_distributions": [
            {"sub_id": "s1",  "kind": "theta",  "range": (PI, PI),                  "weight": 1.0},
            {"sub_id": "s3a", "kind": "offset", "range": (PI * 9.2 / 180.0, PI / 12), "weight": 1.0},
            {"sub_id": "s3b", "kind": "offset", "range": (PI / 12, PI / 6),         "weight": 1.0},
            {"sub_id": "s3c", "kind": "offset", "range": (PI / 6,  PI / 3),         "weight": 1.0},
            {"sub_id": "s3d", "kind": "offset", "range": (PI / 3,  PI / 2),         "weight": 1.0},
        ],
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
    p.add_argument("--end-stage", type=int, default=None,
                   help="마지막 진행 단계 (inclusive, 1-indexed). 미지정 시 끝(s3d)까지 자동 진행. "
                        "현 미달 stage만 돌리려면 --start-stage X --end-stage X.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=None,
                   help="SAC seed (multi-seed 평가용). 미지정시 SB3 default.")
    p.add_argument("--tb-tag", type=str, default="m4_v1",
                   help="tb_logs sub-dir 이름 (multi-seed면 m4_v1_seed{N}). m4 환경 default.")
    p.add_argument("--runs-subdir", type=str, default=None,
                   help="runs/ 안에서 stage tag prefix (예: m4_v1_seed0 → runs/m4_v1_seed0/s3b_arc30)")
    p.add_argument("--init-from", type=str, default=None,
                   help="start-stage init 모델 path. None이면 이전 stage model.zip 자동.")
    p.add_argument("--det-episodes", type=int, default=100,
                   help="stage 졸업 deterministic eval ep 수 (기본 100, eval_stages.py와 통일). "
                        "TB stochastic 90% trigger 후 이 ep 수로 진짜 90% 확인. 비용 ~8분/회.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="모든 stage total_timesteps override. None이면 STAGES default (1M).")
    p.add_argument("--det-cooldown-steps", type=int, default=100_000,
                   help="det eval 미달 시 다음 평가까지 학습 step (기본 100k).")
    args = p.parse_args()

    if args.start_stage < 1 or args.start_stage > len(STAGES):
        print(f"start-stage는 1~{len(STAGES)} 사이여야 합니다.")
        return
    end_stage = args.end_stage if args.end_stage is not None else len(STAGES)
    if end_stage < args.start_stage or end_stage > len(STAGES):
        print(f"end-stage는 start-stage({args.start_stage})~{len(STAGES)} 사이여야 합니다.")
        return

    sim_dir = Path(__file__).parent
    runs_dir = sim_dir / "runs"
    # multi-seed 시 seed별 plot 충돌 방지 (tb-tag 별 sub-dir).
    plot_dir = sim_dir / "plots" / args.tb_tag
    # 카드 버전·seed별 tensorboard log 분리: --tb-tag로 디렉토리 명시.
    tb_dir = sim_dir / "tb_logs" / args.tb_tag
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

    # 시작점 모델 (Stage 2 이상부터 시작 시).
    # --runs-subdir 지정 시 init은 공용 runs_dir/{prev_tag}, 결과는 {subdir}/{tag}/로 저장.
    prev_model_path = None
    if args.start_stage > 1:
        if args.init_from is not None:
            prev_model_path = Path(args.init_from)
        else:
            prev_tag = STAGES[args.start_stage - 2]["tag"]
            prev_model_path = runs_dir / prev_tag / "model.zip"
        if not prev_model_path.exists():
            print(f"이전 단계 모델 {prev_model_path} 없음. Stage 1부터 시작하세요.")
            if viewer_thread is not None:
                stop_event.set()
            return
        print(f"[curriculum] Stage {args.start_stage}부터. init = {prev_model_path}")

    if args.runs_subdir:
        out_runs_dir = runs_dir / args.runs_subdir
        out_runs_dir.mkdir(parents=True, exist_ok=True)
        print(f"[curriculum] 학습 결과 저장 경로 = {out_runs_dir}")
    else:
        out_runs_dir = runs_dir

    try:
        for i in range(args.start_stage - 1, end_stage):
            stage = STAGES[i]
            print(f"\n{'='*60}\n{stage['desc']}\n  threshold={args.threshold:.0%}, "
                  f"min_steps={args.min_steps:,}, max_steps={stage['max_steps']:,}\n"
                  f"{'='*60}")

            run_dir = out_runs_dir / stage["tag"]
            run_dir.mkdir(parents=True, exist_ok=True)

            theta_range = (stage["theta_min"], stage["theta_max"])
            offset_range = None
            if "theta_offset_min" in stage and "theta_offset_max" in stage:
                offset_range = (stage["theta_offset_min"], stage["theta_offset_max"])
            sub_distributions = stage.get("sub_distributions")
            sub_ids = [s["sub_id"] for s in sub_distributions] if sub_distributions else None
            ep_sec = stage.get("episode_seconds", 10.0)
            make_env = make_env_factory(theta_range, stage["success_radius"],
                                        episode_seconds=ep_sec,
                                        action_history_n=ACTION_HISTORY_N,
                                        target_theta_offset_range=offset_range,
                                        sub_distributions=sub_distributions)
            env = make_vec_env(make_env, n_envs=1)

            if prev_model_path is not None:
                print(f"[curriculum] {prev_model_path} 정책 로드 (fine-tuning)")
                model = SAC.load(str(prev_model_path), env=env, device=args.device)
                model.tensorboard_log = str(tb_dir)
                # m4_v20: replay buffer rehearsal (literature: CLEAR 2019 + Continual World 2021).
                # SAC.load는 buffer 안 가져옴 — chain 학습 시 이전 stage 경험 0% → forgetting.
                # 이전 stage save_replay_buffer.pkl 있으면 load → forgetting 완화.
                prev_buffer_path = prev_model_path.parent / "replay_buffer.pkl"
                if prev_buffer_path.exists():
                    model.load_replay_buffer(str(prev_buffer_path))
                    print(f"[curriculum] replay buffer 로드: {prev_buffer_path} "
                          f"(size={model.replay_buffer.size():,})")
                else:
                    print(f"[curriculum] (replay buffer 파일 없음 — 첫 chain stage)")
                # SAC.load 후 seed property는 init seed 그대로 — multi-seed fine-tune엔 재설정 필요.
                if args.seed is not None:
                    model.set_random_seed(args.seed)
                    print(f"[curriculum] seed = {args.seed} (post-load reset)")
            else:
                # policy_kwargs 생략 = SB3 default NN [256, 256].
                model = SAC(
                    "MlpPolicy", env, verbose=1, device=args.device,
                    tensorboard_log=str(tb_dir),
                    learning_rate=3e-4, buffer_size=3_000_000, batch_size=256,
                    tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                    learning_starts=1_000,
                    ent_coef="auto_0.1",  # entropy 초기값 0.1 (collapse 늦춤)
                    seed=args.seed,
                )

            callbacks = []
            if not args.no_viewer:
                callbacks.append(PolicySnapshotCallback(model_holder, sync_every=500))
            # peak 시점 model_best.zip 별도 저장 (후반 후퇴 대비).
            # deterministic check: stochastic 90% trigger 시 det eval로 진짜 90% 확인 (함정 #13).
            def _det_env_builder(_theta_range=theta_range, _offset_range=offset_range,
                                 _sub=sub_distributions,
                                 _sr=stage["success_radius"],
                                 _ep=ep_sec, _N=ACTION_HISTORY_N):
                return FishSwimEnv(
                    target_theta_range=_theta_range,
                    target_theta_offset_range=_offset_range,
                    sub_distributions=_sub,
                    success_radius=_sr,
                    episode_seconds=_ep,
                    action_history_n=_N,
                )
            callbacks.append(CurriculumStopCallback(
                threshold=args.threshold, window=100,
                check_every=5000, min_steps=args.min_steps,
                best_save_path=run_dir / "model_best",
                det_env_fn=_det_env_builder,
                det_episodes=args.det_episodes,
                det_cooldown_steps=args.det_cooldown_steps,
                sub_ids=sub_ids,
            ))
            # ent_floor_end 있으면 linear decay (0 → max_steps).
            ent_floor_end = stage.get("ent_floor_end")
            decay_end = stage["max_steps"] if ent_floor_end is not None else 0
            callbacks.append(EntCoefFloorCallback(
                floor=stage["ent_floor"],
                floor_end=ent_floor_end,
                decay_end_step=decay_end,
            ))
            if ent_floor_end is not None:
                print(f"[curriculum] ent_floor schedule: {stage['ent_floor']} → "
                      f"{ent_floor_end} (linear, 0~{decay_end:,} step)")
            else:
                print(f"[curriculum] ent_floor = {stage['ent_floor']}")
            cb = CallbackList(callbacks) if callbacks else None

            model.learn(
                total_timesteps=args.max_steps if args.max_steps is not None else stage["max_steps"],
                progress_bar=True,
                callback=cb,
                reset_num_timesteps=True,           # stage별 step 0부터 카운트
                tb_log_name=stage["tag"],           # TB run 이름을 stage tag로 (s1_forward, s2_anchor, ...)
            )

            model.save(run_dir / "model")
            print(f"[curriculum] saved -> {run_dir / 'model.zip'}")
            # m4_v20: replay buffer 보존 (다음 stage가 load해 forgetting 완화).
            buffer_path = run_dir / "replay_buffer.pkl"
            model.save_replay_buffer(str(buffer_path))
            print(f"[curriculum] replay buffer saved -> {buffer_path} "
                  f"(size={model.replay_buffer.size():,})")

            save_training_plots(tb_dir, stage["tag"], plot_dir)

            # m4_cpg_v2: stage 종료 시 trajectory plot 자동 생성 (eval_stages.py).
            # 학습 stage 까지만 deterministic eval (default 100 ep × stage 수).
            eval_script = Path(__file__).parent / "eval_stages.py"
            print(f"[curriculum] auto-eval (trajectory plot 생성) → {eval_script.name}")
            subprocess.run(
                [sys.executable, str(eval_script), str(run_dir / "model.zip")],
                check=False,
            )

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
