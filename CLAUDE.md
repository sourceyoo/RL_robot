# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 목적

MuJoCo 기반 물고기 로봇의 강화학습. 현재 저장소에는 시뮬레이션 에셋만 있고 Python 학습 코드, 빌드 시스템, 테스트는 아직 **없습니다**. 앞으로 추가될 작업은 대체로 `robotic_fish.xml`을 로드해 Gym/Gymnasium 환경으로 감싸고 정책을 학습시키는 Python 패키지가 될 것입니다.

## 저장소 구성

서로 **독립적인 두 개의 모델링 스택**이 공존합니다. 한쪽을 수정해도 다른 쪽에 반영되지 않습니다.

### 1. MuJoCo 모델 — `robotic_fish.xml` (RL 학습 대상)

RL이 실제로 사용할 모델입니다. 한눈에 보이지 않는 중요한 사실들:

- **수중 물리 환경, 공기 중이 아님.** `<option density="5000">`이 물 같은 매질을 시뮬레이션하고 `<flag gravity="disable" .../>`로 중력이 꺼져 있습니다. 부력/항력은 명시적인 힘이 아니라 이 고밀도 매질에서 비롯됩니다. 꺼진 중력을 "버그"로 보고 켜지 마세요.
- **include 경로는 상대 경로.** `./common/visual.xml`과 `./common/materials.xml`을 끌어옵니다. skybox는 인라인으로 선언되어 있고, 별도의 `common/skybox.xml`은 사용되지 않습니다. `robotic_fish.xml`을 옮기면 이 경로들도 같이 수정해야 합니다.
- **관절 구조:** torso(free joint) → 2단 꼬리 (`tail1` hinge + `tail_twist` hinge → `tail2` hinge, 끝단은 passive stiffness), 양쪽 가슴지느러미는 각각 `roll` + `pitch`. 두 지느러미의 roll은 `<tendon>` 두 개 (`fins_flap` 반대칭, `fins_sym` 대칭+stiffness)로 묶여 있습니다.
- **액션 공간 (5개 position 액추에이터, 모두 ctrlrange `-1..1`):** `tail`, `tail_twist`, `fins_flap` (tendon, 반대칭 펄럭임), `finleft_pitch`, `finright_pitch`. 좌/우 지느러미의 **roll**은 직접 구동되지 않고 **오직 `fins_flap` tendon을 통해서만** 움직입니다.
- **센서:** `torso` site에 `velocimeter`와 `gyro`. `target` 구는 `(0, 0.4, 0.1)`에 위치하며, 목표 지점 도달형 보상의 기준점입니다.
- **카메라:** `tracking_top`, `tracking_x`, `tracking_y`, `fixed_top`, 그리고 1인칭 `eye`. 학습 영상 로깅에 유용합니다.

이 모델은 DeepMind Control Suite의 "fish" 태스크와 거의 동일한 구조입니다. 관측/보상을 설계할 때 그쪽 코드가 좋은 레퍼런스입니다.

### 2. URDF / Gazebo / ROS 패키지 — `fish_urdf/RL_SIM_MODE_2_description/`

Fusion 360에서 `fusion2urdf`로 내보낸 별도의 단순 모델입니다. 움직이는 관절은 **딱 하나**: `tail_joint` (revolute, ±0.349 rad). ROS launch 파일들 (`display.launch`, `gazebo.launch`, `controller.launch`)과 catkin/CMake 패키지 골격을 포함합니다.

**깨진 파일:** `launch/controller.yaml`과 `launch/controller.launch`에는 한국어 관절명 "회전 3"이 CP949로 저장되어 mojibake (`ȸ�� 3`)로 보입니다. 게다가 URDF의 실제 관절명인 `tail_joint`와도 일치하지 않습니다. 그대로는 ros_control이 동작하지 않습니다 — 인코딩 수정 *그리고* `tail_joint`로 이름 변경이 모두 필요합니다.

이 스택은 **RL 학습 대상이 아닙니다**. RViz/Gazebo 데모용으로 보존된 CAD export로 보세요. RL을 얹어달라는 요청이 오면 ROS를 거치지 말고 형상 정보만 MuJoCo로 옮기세요.

## 작업 진행 규칙 — 한 단계씩 분리

다단계 작업은 **한 단계씩 분리해서** 진행한다. 한 단계 결과를 보고하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- "한 단계"의 단위는 **사용자가 결과를 보고 다음 결정을 내릴 수 있는 지점**.
- 예: 의존성 설치 → 멈춤. URDF 로드 검증 → 멈춤. MJCF 작성 → 멈춤. .gitignore 작성 → 멈춤. commit → 멈춤. push → 멈춤.
- TaskCreate로 전체 단계를 미리 나열해 두는 건 OK (전체 그림 표시). 하지만 status는 한 번에 한 task만 in_progress.
- 자명하게 묶이는 미세 작업(같은 파일 수정 + 같은 줄 검증)은 굳이 쪼개지 않아도 된다. 핵심은 **사용자가 검수·중단할 기회를 주는 것**.
- 결정·규칙은 md 파일(이 CLAUDE.md 또는 PROGRESS.md)에 명시해 미래 세션도 같은 방식으로 동작하게 한다.

## 알아둘 컨벤션

- `common/materials.xml`의 `self_highlight`, `target_highlight` 같은 오버라이드 머티리얼은 보상 이벤트 발생 시 색을 바꿔 시각화하는 용도입니다. 학습 중 시각화를 추가할 때 이름 규칙을 유지하세요.
- `robotic_fish.xml`은 `<default class="fish">`로 관절 기본값(damping, range, solver 파라미터)을 잡아둡니다. 새 body를 추가할 때 `childclass="fish"`를 지정해야 상속됩니다. 안 그러면 같은 모양이라도 물리 동작이 조용히 달라집니다.

## RL 코드를 새로 추가할 때

`requirements.txt`, `pyproject.toml`, 학습 진입점이 아직 없습니다. 첫 학습 스크립트를 만들 때 같이 정립하세요. 모델은 작아서 별도 전처리 없이 `mujoco.MjModel.from_xml_path("robotic_fish.xml")`로 바로 로드 가능합니다.
