"""m4_v15 reward 표를 PNG로 렌더링 → images/m4_v15_reward_table.png"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 강제 등록 (Noto Sans CJK KR 우선)
KR_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
kr_prop = None
for fp in KR_FONT_CANDIDATES:
    try:
        kr_prop = font_manager.FontProperties(fname=fp)
        mpl.rcParams['font.family'] = kr_prop.get_name()
        break
    except Exception:
        continue
mpl.rcParams['axes.unicode_minus'] = False

rows = [
    ["progress",    "(prev_dist − dist) · 5.0",                       "+",    "±0.005~0.02", "목표 접근 시 +, 멀어지면 −"],
    ["reach bonus", "+10.0 (도달 시 1회)",                            "+",    "sparse +10",  "도달 incentive"],
    ["align",       "0.05 · cos(head, target)",                       "±",    "±0.05",       "머리 정렬"],
    ["yaw_sign",    "0.005 · yaw_rate · sign(yaw_err)",               "±",    "±0.005~0.01", "회전 방향 정렬"],
    ["action_diff", "−0.01 · (Δctrl)²",                               "−",    "−0~−0.04",    "smooth ctrl"],
    ["DC_pen",      "−0.05 · |window_mean|",                          "−",    "−0~−0.05",    "B mode(편향) 차단"],
    ["asym_bonus",  "+0.02 · max(0, target_sign·sf_signed − 0.05)",   "+",    "0~+0.009",    "비대칭 sweep 정합"],
    ["amp_bonus",   "+0.01 · max(0, target_sign·amp_signed − 0.1)",   "+",    "0~+0.009",    "비대칭 진폭 정합"],
    ["time_pen",    "−1e-5 · distance · time",                        "−",    "0~−0.002",    "떠도는 mode 억제"],
    ["slip_pen",    "−0.2 · slip_angle  (|v|>0.02)",                  "−",    "0~−0.63",     "sideslip 직접 차단"],
    ["still_pen",   "−0.05  (|v|≤0.02)",                              "−",    "−0.05",       "정지/제자리 회전 차단"],
    ["back clip",   "head_vel < −0.01  →  reward = −1.0",             "clip", "−1.0 덮어쓰기", "후진 절대 금지"],
]
cols = ["항", "식", "부호", "크기 (step당 typical)", "의도"]

fig, ax = plt.subplots(figsize=(14, 6.8), dpi=150)
ax.axis('off')
ax.set_title("m4_v15 보상 항목별 step당 영향", fontsize=15, fontweight='bold', pad=14)

table = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='left', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.55)

widths = [0.10, 0.34, 0.06, 0.18, 0.32]
for i, w in enumerate(widths):
    for j in range(len(rows) + 1):
        table[(j, i)].set_width(w)

sign_colors = {"+": "#1b7a1b", "−": "#b03030", "±": "#555555", "clip": "#a04000"}
for r, row in enumerate(rows, start=1):
    cell = table[(r, 2)]
    cell.set_text_props(color=sign_colors.get(row[2], "black"), fontweight='bold', ha='center')

# 헤더
for c in range(len(cols)):
    h = table[(0, c)]
    h.set_facecolor("#2c3e50")
    h.set_text_props(color="white", fontweight='bold')

# zebra
for r in range(1, len(rows) + 1):
    if r % 2 == 0:
        for c in range(len(cols)):
            table[(r, c)].set_facecolor("#f4f4f4")

# 크기 큰 페널티 강조 (slip_pen, back clip)
for r_idx in [10, 12]:
    for c in range(len(cols)):
        table[(r_idx, c)].set_facecolor("#fff0e0")

out = Path(__file__).resolve().parents[2] / "images" / "m4_v15_reward_table.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f"saved: {out}")
