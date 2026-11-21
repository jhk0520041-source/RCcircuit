import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

st.set_page_config(page_title="직류 RC회로 시뮬레이션", layout="wide")


@st.cache_resource(show_spinner=False)
def load_korean_font_name() -> str | None:
    """앱 실행 중 한 번만 시스템 폰트를 탐색·등록하고 사용할 한글 폰트 이름을 반환합니다."""
    try:
        font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    candidate_fonts = [
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "UnDotum",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            return font_name

    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            return prop.get_name()

    return None


def configure_korean_font():
    """캐시된 폰트 이름을 matplotlib 전역 설정에 반영합니다."""
    font_name = load_korean_font_name()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False


configure_korean_font()

# 1. 제목 및 안내 문구
st.title("⚡직류 RC회로 시뮬레이션")
st.caption("변수를 설정한 뒤, 아래 적용하기 버튼을 눌러 시뮬레이션을 확인하세요!")


def parse_float_field(raw_val, field_name: str, min_val: float, strict_min: bool = False):
    """입력값을 float로 변환하고 범위를 검증합니다."""
    if raw_val is None or str(raw_val).strip() == "":
        return False, None, f"[{field_name}] 값을 입력해주세요."
    try:
        val = float(str(raw_val).strip())
    except (ValueError, TypeError):
        return False, None, f"[{field_name}]에는 올바른 숫자(float)를 입력해주세요. (입력값: '{raw_val}')"

    if np.isnan(val) or np.isinf(val):
        return False, None, f"[{field_name}]에는 유한한 숫자를 입력해주세요."

    if strict_min:
        if val <= min_val:
            return False, None, f"[{field_name}]은(는) {min_val:g}보다 큰 값이어야 합니다. (입력값: {val:g})"
    else:
        if val < min_val:
            return False, None, f"[{field_name}]은(는) {min_val:g} 이상의 값이어야 합니다. (입력값: {val:g})"

    return True, val, None


def calc_equiv_resistance_kohm(topology: str, r_list: list[float]) -> float:
    """R1..Rn (단위: kΩ)의 직렬/병렬 합성저항(kΩ)을 계산합니다."""
    if topology == "직렬":
        return float(np.sum(r_list))
    inv_sum = sum(1.0 / r for r in r_list)
    return float(1.0 / inv_sum)


def calc_equiv_capacitance_uf(topology: str, c_list: list[float]) -> float:
    """C1..Cn (단위: μF)의 직렬/병렬 합성전기용량(μF)을 계산합니다."""
    if topology == "병렬":
        return float(np.sum(c_list))
    inv_sum = sum(1.0 / c for c in c_list)
    return float(1.0 / inv_sum)


@st.cache_data(show_spinner=False)
def compute_rc_simulation(
    r_topo: str,
    r_vals_kohm: tuple[float, ...],
    c_topo: str,
    c_vals_uf: tuple[float, ...],
    V_supply: float,
    use_r_int: bool,
    R_int_ohm: float,
):
    """
    R_int_ohm을 '기전력 장치의 내부저항'으로 취급해 충·방전 시계열 및 마일스톤을 계산합니다.
    - 충전 시: 기전력 장치(V_supply)와 내부저항(R_int_ohm)이 회로에 연결되므로
      전체 직렬 저항은 R_eq_ohm + R_int_ohm, 충전 시정수는 (R_eq_ohm + R_int_ohm) * C_eq_f 입니다.
    - 방전 시: 기전력 장치 및 내부저항이 분리된 RC 폐회로 방전을 기준으로
      방전 저항은 R_eq_ohm, 방전 시정수는 R_eq_ohm * C_eq_f 입니다.
    """
    R_eq_kohm = calc_equiv_resistance_kohm(r_topo, list(r_vals_kohm))
    R_eq_ohm = R_eq_kohm * 1e3

    C_eq_uf = calc_equiv_capacitance_uf(c_topo, list(c_vals_uf))
    C_eq_f = C_eq_uf * 1e-6

    r_int_effective_ohm = R_int_ohm if use_r_int else 0.0

    # 충전 경로: 기전력 장치 내부저항 포함
    R_charge_total_ohm = R_eq_ohm + r_int_effective_ohm
    tau_charge_s = R_charge_total_ohm * C_eq_f
    tau_charge_ms = tau_charge_s * 1e3

    # 방전 경로: 외부 RC 루프 기준
    tau_dis_s = R_eq_ohm * C_eq_f
    tau_dis_ms = tau_dis_s * 1e3

    V_c0 = 0.0
    # 충전 초기전류 I0: t=0에서 커패시터 전압이 0V일 때 (V_supply - 0) / (R_eq + R_int)
    I0_charge_a = (V_supply - V_c0) / R_charge_total_ohm
    I0_charge_ma = I0_charge_a * 1e3

    # 방전 시작 전압(V_supply) 및 초기전류 I0
    V_dis_start = V_supply
    I0_discharge_a = -(V_dis_start / R_eq_ohm)
    I0_discharge_ma = I0_discharge_a * 1e3

    t_charge_max = max(5.0 * tau_charge_s, 1e-6)
    t_dis_max = max(5.0 * tau_dis_s, 1e-6)

    t_charge_curve = np.linspace(0.0, t_charge_max, 600)
    t_charge_csv = np.linspace(0.0, t_charge_max, 100)

    t_dis_curve = np.linspace(0.0, t_dis_max, 600)
    t_dis_csv = np.linspace(0.0, t_dis_max, 100)

    # 충전 시계열 (기전력 장치 내부저항 반영)
    vc_charge_curve = V_supply * (1.0 - np.exp(-t_charge_curve / tau_charge_s))
    ic_charge_curve_ma = (V_supply / R_charge_total_ohm) * np.exp(-t_charge_curve / tau_charge_s) * 1e3

    vc_charge_csv = V_supply * (1.0 - np.exp(-t_charge_csv / tau_charge_s))
    ic_charge_csv_a = (V_supply / R_charge_total_ohm) * np.exp(-t_charge_csv / tau_charge_s)

    # 방전 시계열
    vc_dis_curve = V_dis_start * np.exp(-t_dis_curve / tau_dis_s)
    ic_dis_curve_ma = -(V_dis_start / R_eq_ohm) * np.exp(-t_dis_curve / tau_dis_s) * 1e3

    vc_dis_csv = V_dis_start * np.exp(-t_dis_csv / tau_dis_s)
    ic_dis_csv_a = -(V_dis_start / R_eq_ohm) * np.exp(-t_dis_csv / tau_dis_s)

    # t = tau 시점의 값 및 비율(%)
    vc_charge_tau = float(V_supply * (1.0 - np.exp(-1.0)))
    ic_charge_tau_ma = float(I0_charge_ma * np.exp(-1.0))
    vc_charge_tau_pct = (vc_charge_tau / V_supply * 100.0) if V_supply != 0 else 0.0
    ic_charge_tau_pct = (ic_charge_tau_ma / I0_charge_ma * 100.0) if I0_charge_ma != 0 else 0.0

    vc_dis_tau = float(V_dis_start * np.exp(-1.0))
    ic_dis_tau_ma = float(I0_discharge_ma * np.exp(-1.0))
    vc_dis_tau_pct = (vc_dis_tau / V_dis_start * 100.0) if V_dis_start != 0 else 0.0
    ic_dis_tau_pct = (ic_dis_tau_ma / I0_discharge_ma * 100.0) if I0_discharge_ma != 0 else 0.0

    # 1τ ~ 5τ 마일스톤 표 (충전)
    k_arr = np.arange(1, 6)
    t_ch_ms_arr = k_arr * tau_charge_s
    vc_charge_ms = V_supply * (1.0 - np.exp(-t_ch_ms_arr / tau_charge_s))
    ic_charge_ms_ma = (V_supply / R_charge_total_ohm) * np.exp(-t_ch_ms_arr / tau_charge_s) * 1e3
    charge_rate_pct = (vc_charge_ms / V_supply * 100.0) if V_supply != 0 else np.zeros_like(vc_charge_ms)

    df_charge_ms = pd.DataFrame({
        "시정수 구간": [f"{k}τ ({k * tau_charge_ms:.2f} ms)" for k in k_arr],
        "전압 값 [V]": np.round(vc_charge_ms, 4),
        "충전율 [%]": np.round(charge_rate_pct, 2),
        "전류 값 [mA]": np.round(ic_charge_ms_ma, 3),
    }).set_index("시정수 구간")

    # 1τ ~ 5τ 마일스톤 표 (방전)
    t_dis_ms_arr = k_arr * tau_dis_s
    vc_dis_ms = V_dis_start * np.exp(-t_dis_ms_arr / tau_dis_s)
    ic_dis_ms_ma = -(V_dis_start / R_eq_ohm) * np.exp(-t_dis_ms_arr / tau_dis_s) * 1e3
    discharge_rate_pct = (
        ((V_dis_start - vc_dis_ms) / V_dis_start * 100.0) if V_dis_start != 0 else np.zeros_like(vc_dis_ms)
    )

    df_discharge_ms = pd.DataFrame({
        "시정수 구간": [f"{k}τ ({k * tau_dis_ms:.2f} ms)" for k in k_arr],
        "전압 값 [V]": np.round(vc_dis_ms, 4),
        "방전율 [%]": np.round(discharge_rate_pct, 2),
        "전류 값 [mA]": np.round(ic_dis_ms_ma, 3),
    }).set_index("시정수 구간")

    return {
        "tau_charge_ms": tau_charge_ms,
        "tau_dis_ms": tau_dis_ms,
        "V_c0": V_c0,
        "I0_charge_ma": I0_charge_ma,
        "I0_discharge_ma": I0_discharge_ma,
        "t_charge_curve": t_charge_curve,
        "t_charge_csv": t_charge_csv,
        "t_dis_curve": t_dis_curve,
        "t_dis_csv": t_dis_csv,
        "vc_charge_curve": vc_charge_curve,
        "ic_charge_curve_ma": ic_charge_curve_ma,
        "vc_charge_csv": vc_charge_csv,
        "ic_charge_csv_a": ic_charge_csv_a,
        "vc_dis_curve": vc_dis_curve,
        "ic_dis_curve_ma": ic_dis_curve_ma,
        "vc_dis_csv": vc_dis_csv,
        "ic_dis_csv_a": ic_dis_csv_a,
        "vc_charge_tau": vc_charge_tau,
        "ic_charge_tau_ma": ic_charge_tau_ma,
        "vc_charge_tau_pct": vc_charge_tau_pct,
        "ic_charge_tau_pct": ic_charge_tau_pct,
        "vc_dis_tau": vc_dis_tau,
        "ic_dis_tau_ma": ic_dis_tau_ma,
        "vc_dis_tau_pct": vc_dis_tau_pct,
        "ic_dis_tau_pct": ic_dis_tau_pct,
        "df_charge_ms": df_charge_ms,
        "df_discharge_ms": df_discharge_ms,
    }


def draw_circuit_schematic(
    r_topo: str,
    r_vals_kohm: list[float],
    c_topo: str,
    c_vals_uf: list[float],
    V_supply: float,
    use_r_int: bool,
    R_int_ohm: float,
):
    """소자 겹침과 배선 뭉침 없이 직렬/병렬 연결 및 기전력 내부저항을 표시하는 회로도를 그립니다."""
    n_r = len(r_vals_kohm)
    n_c = len(c_vals_uf)

    FIXED_FONT_SIZE = 8.5
    FIXED_FONT_WEIGHT = "normal"

    # 병렬 저항일 경우 가지 개수에 맞춰 가로 폭 확보
    r_span = max(5.6, n_r * 2.3) if r_topo == "병렬" else max(5.4, n_r * 2.4)
    c_span = max(3.6, n_c * 2.3) if c_topo == "병렬" else 2.6

    x_src = 1.0
    x_r_start = x_src + 1.5
    x_r_end = x_r_start + r_span
    x_c_center = x_r_end + (c_span / 2.0 + 1.1 if c_topo == "병렬" else 1.6)
    x_max = x_c_center + (c_span / 2.0 + 1.5 if c_topo == "병렬" else 2.8)

    y_bottom = 2.4
    y_top = 6.2
    y_mid = (y_bottom + y_top) / 2.0
    y_max = 7.6

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.axis("off")

    label_bbox = dict(
        boxstyle="round,pad=0.2",
        facecolor="white",
        edgecolor="none",
        alpha=0.95,
        zorder=5,
    )

    # 기본 외곽 배선
    ax.plot([x_src, x_src], [y_bottom, y_top], color="#222222", lw=2.0, zorder=1)
    ax.plot([x_src, x_r_start], [y_top, y_top], color="#222222", lw=2.0, zorder=1)
    ax.plot([x_src, x_c_center], [y_bottom, y_bottom], color="#222222", lw=2.0, zorder=1)

    # 좌측 기전력 장치
    src_circle = patches.Circle(
        (x_src, y_mid), 0.72, facecolor="#FFF8E1", edgecolor="#D97706", lw=2.0, zorder=4
    )
    ax.add_patch(src_circle)
    ax.text(
        x_src, y_mid + 0.18, "기전력",
        ha="center", va="center", fontsize=FIXED_FONT_SIZE, fontweight=FIXED_FONT_WEIGHT, color="#92400E", zorder=5
    )
    ax.text(
        x_src, y_mid - 0.22, f"{V_supply:g} V",
        ha="center", va="center", fontsize=FIXED_FONT_SIZE, fontweight=FIXED_FONT_WEIGHT, color="#92400E", zorder=5
    )

    # 상단 저항 블록: 직렬 vs 병렬
    if r_topo == "직렬":
        ax.plot([x_r_start, x_r_end], [y_top, y_top], color="#222222", lw=2.0, zorder=1)
        ax.plot([x_r_end, x_c_center], [y_top, y_top], color="#222222", lw=2.0, zorder=1)

        seg_w = r_span / n_r
        box_w = min(1.35, seg_w * 0.55)
        for idx, r_val in enumerate(r_vals_kohm):
            cx = x_r_start + (idx + 0.5) * seg_w
            rect = patches.Rectangle(
                (cx - box_w / 2, y_top - 0.34), box_w, 0.68,
                facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.8, zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                cx,
                y_top + 0.65,
                f"R{idx+1} = {r_val:g}kΩ",
                ha="center",
                va="center",
                fontsize=FIXED_FONT_SIZE,
                fontweight=FIXED_FONT_WEIGHT,
                color="#1E40AF",
                bbox=label_bbox,
            )
    else:
        # 병렬 저항: 상단 주 배선 아래에 각 저항을 수직 가지로 분리 배치
        y_r_bus_top = y_top
        y_r_bus_bot = y_top - 1.85
        y_r_mid = (y_r_bus_top + y_r_bus_bot) / 2.0

        x_r_branches = (
            np.linspace(x_r_start + 0.85, x_r_end - 0.85, n_r)
            if n_r > 1
            else [(x_r_start + x_r_end) / 2.0]
        )

        ax.plot([x_r_start, x_r_end], [y_r_bus_top, y_r_bus_top], color="#222222", lw=2.0, zorder=1)
        ax.plot([x_r_end, x_c_center], [y_r_bus_top, y_r_bus_top], color="#222222", lw=2.0, zorder=1)
        ax.plot([min(x_r_branches), max(x_r_branches)], [y_r_bus_bot, y_r_bus_bot], color="#222222", lw=1.8, zorder=1)
        ax.plot([max(x_r_branches), max(x_r_branches)], [y_r_bus_bot, y_r_bus_top], color="#222222", lw=1.8, zorder=1)

        box_w = 1.45
        box_h = 0.54
        for idx, (r_val, rx) in enumerate(zip(r_vals_kohm, x_r_branches)):
            ax.plot([rx, rx], [y_r_bus_bot, y_r_bus_top], color="#222222", lw=1.6, zorder=1)
            rect = patches.Rectangle(
                (rx - box_w / 2, y_r_mid - box_h / 2), box_w, box_h,
                facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.8, zorder=4
            )
            ax.add_patch(rect)
            ax.text(
                rx,
                y_r_mid,
                f"R{idx+1}={r_val:g}kΩ",
                ha="center",
                va="center",
                fontsize=FIXED_FONT_SIZE,
                fontweight=FIXED_FONT_WEIGHT,
                color="#1E40AF",
                zorder=5,
            )

    # 우측 축전기 블록: 직렬 vs 병렬
    if c_topo == "직렬":
        ax.plot([x_c_center, x_c_center], [y_bottom, y_top], color="#222222", lw=2.0, zorder=1)
        y_centers = np.linspace(y_top - 1.0, y_bottom + 1.0, n_c) if n_c > 1 else [y_mid]
        for idx, (c_val, yc) in enumerate(zip(c_vals_uf, y_centers)):
            ax.plot([x_c_center - 0.34, x_c_center + 0.34], [yc + 0.16, yc + 0.16], color="#059669", lw=2.4, zorder=4)
            ax.plot([x_c_center - 0.34, x_c_center + 0.34], [yc - 0.16, yc - 0.16], color="#059669", lw=2.4, zorder=4)
            ax.text(
                x_c_center + 0.55,
                yc,
                f"C{idx+1} = {c_val:g}μF",
                ha="left",
                va="center",
                fontsize=FIXED_FONT_SIZE,
                fontweight=FIXED_FONT_WEIGHT,
                color="#065F46",
                bbox=label_bbox,
            )
    else:
        x_branches = np.linspace(x_c_center - c_span * 0.4, x_c_center + c_span * 0.4, n_c) if n_c > 1 else [x_c_center]
        ax.plot([min(x_branches), max(x_branches)], [y_top, y_top], color="#222222", lw=2.0, zorder=1)
        ax.plot([min(x_branches), max(x_branches)], [y_bottom, y_bottom], color="#222222", lw=2.0, zorder=1)
        ax.plot([x_r_end, min(x_branches)], [y_top, y_top], color="#222222", lw=2.0, zorder=1)
        ax.plot([x_src, min(x_branches)], [y_bottom, y_bottom], color="#222222", lw=2.0, zorder=1)

        for idx, (c_val, xp) in enumerate(zip(c_vals_uf, x_branches)):
            ax.plot([xp, xp], [y_bottom, y_mid - 0.16], color="#222222", lw=1.6, zorder=1)
            ax.plot([xp, xp], [y_mid + 0.16, y_top], color="#222222", lw=1.6, zorder=1)
            ax.plot([xp - 0.28, xp + 0.28], [y_mid + 0.16, y_mid + 0.16], color="#059669", lw=2.4, zorder=4)
            ax.plot([xp - 0.28, xp + 0.28], [y_mid - 0.16, y_mid - 0.16], color="#059669", lw=2.4, zorder=4)
            ax.text(
                xp,
                y_bottom - 0.35,
                f"C{idx+1}\n{c_val:g}μF",
                ha="center",
                va="top",
                fontsize=FIXED_FONT_SIZE,
                fontweight=FIXED_FONT_WEIGHT,
                color="#065F46",
                bbox=label_bbox,
            )

    # 하단 상태 표시 (축전기 라벨 아래로 충분히 분리)
    r_int_display = f"{R_int_ohm:g} Ω" if use_r_int else "0 Ω"
    status_txt = f"기전력 내부저항 r_int = {r_int_display}"
    ax.text(
        x_max / 2.0,
        0.55,
        status_txt,
        ha="center",
        va="center",
        fontsize=FIXED_FONT_SIZE,
        fontweight=FIXED_FONT_WEIGHT,
        color="#374151",
        bbox=label_bbox,
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.05)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_vc_ic_plots(
    t_arr: np.ndarray,
    vc_arr: np.ndarray,
    ic_arr_ma: np.ndarray,
    tau_ms: float,
    vc_tau_val: float,
    vc_tau_pct: float,
    ic_tau_val_ma: float,
    ic_tau_pct: float,
    mode_label: str,
):
    """동일 크기·여백으로 Vc-t(왼쪽), Ic-t(오른쪽) 그래프를 그리고 시정수 τ 시점의 노란색 가로 점선 및 퍼센트를 표기합니다."""
    tau_s = tau_ms / 1e3
    FIXED_FIG_SIZE = (6.4, 4.2)
    c_left, c_right = st.columns(2)

    with c_left:
        fig_v, ax_v = plt.subplots(figsize=FIXED_FIG_SIZE)
        ax_v.plot(t_arr, vc_arr, color="#2563EB", lw=2, label="Vc(t)")
        ax_v.axvline(x=tau_s, color="#DC2626", ls="--", lw=1.3, label=f"τ = {tau_ms:.2f} ms")
        ax_v.axhline(
            y=vc_tau_val,
            color="#EAB308",
            ls="--",
            lw=1.6,
            label=f"τ 시점 전압 ({vc_tau_pct:.1f}%)",
        )
        ax_v.scatter([tau_s], [vc_tau_val], color="#CA8A04", zorder=5, s=30)
        ax_v.annotate(
            f"{vc_tau_pct:.1f}% ({vc_tau_val:.2f}V)",
            xy=(tau_s, vc_tau_val),
            xytext=(tau_s * 1.12, vc_tau_val),
            fontsize=8.5,
            fontweight="normal",
            color="#854D0E",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9C3", edgecolor="#EAB308", lw=0.8),
        )
        v_min, v_max = float(np.min(vc_arr)), float(np.max(vc_arr))
        v_span = max(v_max - v_min, 1e-3)
        ax_v.set_ylim(v_min - 0.12 * v_span, v_max + 0.12 * v_span)
        ax_v.set_title(f"{mode_label} 커패시터 전압 Vc - t", fontsize=11, fontweight="normal", pad=10)
        ax_v.set_xlabel("시간 t [s]", fontsize=9.5, fontweight="normal", labelpad=6)
        ax_v.set_ylabel("전압 Vc [V]", fontsize=9.5, fontweight="normal", labelpad=6)
        ax_v.tick_params(axis="both", labelsize=8.5)
        ax_v.grid(True, alpha=0.4)
        ax_v.legend(loc="best", fontsize=8)
        fig_v.subplots_adjust(left=0.17, right=0.96, top=0.88, bottom=0.18)
        st.pyplot(fig_v, use_container_width=True)
        plt.close(fig_v)

    with c_right:
        fig_i, ax_i = plt.subplots(figsize=FIXED_FIG_SIZE)
        ax_i.plot(t_arr, ic_arr_ma, color="#EA580C", lw=2, label="Ic(t)")
        ax_i.axvline(x=tau_s, color="#DC2626", ls="--", lw=1.3, label=f"τ = {tau_ms:.2f} ms")
        ax_i.axhline(
            y=ic_tau_val_ma,
            color="#EAB308",
            ls="--",
            lw=1.6,
            label=f"τ 시점 전류 ({ic_tau_pct:.1f}%)",
        )
        ax_i.scatter([tau_s], [ic_tau_val_ma], color="#CA8A04", zorder=5, s=30)
        ax_i.annotate(
            f"{ic_tau_pct:.1f}% ({ic_tau_val_ma:.3f}mA)",
            xy=(tau_s, ic_tau_val_ma),
            xytext=(tau_s * 1.12, ic_tau_val_ma),
            fontsize=8.5,
            fontweight="normal",
            color="#854D0E",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9C3", edgecolor="#EAB308", lw=0.8),
        )
        i_min, i_max = float(np.min(ic_arr_ma)), float(np.max(ic_arr_ma))
        i_span = max(i_max - i_min, 1e-6)
        ax_i.set_ylim(i_min - 0.12 * i_span, i_max + 0.12 * i_span)
        ax_i.set_title(f"{mode_label} 전류 Ic - t", fontsize=11, fontweight="normal", pad=10)
        ax_i.set_xlabel("시간 t [s]", fontsize=9.5, fontweight="normal", labelpad=6)
        ax_i.set_ylabel("전류 Ic [mA]", fontsize=9.5, fontweight="normal", labelpad=6)
        ax_i.tick_params(axis="both", labelsize=8.5)
        ax_i.grid(True, alpha=0.4)
        ax_i.legend(loc="best", fontsize=8)
        fig_i.subplots_adjust(left=0.17, right=0.96, top=0.88, bottom=0.18)
        st.pyplot(fig_i, use_container_width=True)
        plt.close(fig_i)


# 적용된 파라미터 초기값 설정 ('적용하기' 버튼을 누를 때만 갱신됨)
if "applied_state" not in st.session_state:
    st.session_state.applied_state = {
        "r_topo": "직렬",
        "r_vals_kohm": (1.0,),
        "c_topo": "직렬",
        "c_vals_uf": (100.0,),
        "V_supply": 5.0,
        "use_r_int": False,
        "R_int_ohm": 0.0,
    }

# 2. 메인 영역: 1:1 비율 컬럼 분할
col_circuit, col_inputs = st.columns([1, 1])

# 오른쪽(1 비율) 영역을 가로로 3등분
with col_inputs:
    sub_r, sub_c, sub_v = st.columns(3)

    # 3.1 저항 R 값 입력
    with sub_r:
        st.subheader("저항 R 값 입력")
        r_topo_in = st.selectbox("R 연결 방식", options=["직렬", "병렬"], index=0, key="r_topo_sel")
        r_count_in = st.selectbox("저항 개수", options=[1, 2, 3], index=0, key="r_cnt_sel")

        r_options_kohm = [0.5, 1.0, 2.0, 4.7, 10.0]
        raw_r_list = []
        for idx in range(r_count_in):
            raw_r = st.selectbox(
                f"R{idx + 1} [㏀]",
                options=r_options_kohm,
                index=1,
                accept_new_options=True,
                key=f"r_val_{idx}",
            )
            raw_r_list.append(raw_r)

    # 3.2 축전기 C 값 입력
    with sub_c:
        st.subheader("축전기 C 값 입력")
        c_topo_in = st.selectbox("C 연결 방식", options=["직렬", "병렬"], index=0, key="c_topo_sel")
        c_count_in = st.selectbox("축전기 개수", options=[1, 2, 3], index=0, key="c_cnt_sel")

        c_options_uf = [10.0, 47.0, 100.0, 470.0, 1000.0]
        raw_c_list = []
        for idx in range(c_count_in):
            raw_c = st.selectbox(
                f"C{idx + 1} [㎌]",
                options=c_options_uf,
                index=2,
                accept_new_options=True,
                key=f"c_val_{idx}",
            )
            raw_c_list.append(raw_c)

    # 3.3 인가전압 V 및 기전력 장치 내부저항 입력
    with sub_v:
        st.subheader("인가전압 V 값 입력")
        v_options = [float(i) for i in range(1, 13)]
        raw_V_supply = st.selectbox(
            "인가전압 V [V]",
            options=v_options,
            index=4,
            accept_new_options=True,
            key="v_supply_sel",
        )

        use_r_int_chk = st.checkbox("기전력 내부저항 적용", value=False, key="r_int_chk")

        raw_R_int = "0.0"
        if use_r_int_chk:
            raw_R_int = st.selectbox(
                "내부저항 [Ω]",
                options=[10.0, 50.0, 100.0, 200.0, 500.0],
                index=0,
                accept_new_options=True,
                key="r_int_sel",
            )

# 4. columns 하단 '적용하기' 버튼 및 유효성 검사
error_box = st.container()
apply_clicked = st.button("적용하기", type="primary", use_container_width=True)

if apply_clicked:
    errors = []
    parsed_r_vals = []
    for idx, rv in enumerate(raw_r_list):
        ok, val, msg = parse_float_field(rv, f"저항 R{idx + 1} [㏀]", min_val=0.0, strict_min=True)
        if ok:
            parsed_r_vals.append(val)
        else:
            errors.append(msg)

    parsed_c_vals = []
    for idx, cv in enumerate(raw_c_list):
        ok, val, msg = parse_float_field(cv, f"축전기 C{idx + 1} [㎌]", min_val=0.0, strict_min=True)
        if ok:
            parsed_c_vals.append(val)
        else:
            errors.append(msg)

    ok_v, val_v, msg_v = parse_float_field(raw_V_supply, "인가전압 V [V]", min_val=0.0, strict_min=False)
    if not ok_v:
        errors.append(msg_v)

    val_r_int = 0.0
    if use_r_int_chk:
        ok_ri, val_ri, msg_ri = parse_float_field(raw_R_int, "내부저항 [Ω]", min_val=0.0, strict_min=False)
        if not ok_ri:
            errors.append(msg_ri)
        else:
            val_r_int = val_ri

    if errors:
        with error_box:
            for err in errors:
                st.markdown(f"<span style='color:#DC2626; font-weight:600;'>⚠ {err}</span>", unsafe_allow_html=True)
    else:
        st.session_state.applied_state = {
            "r_topo": r_topo_in,
            "r_vals_kohm": tuple(parsed_r_vals),
            "c_topo": c_topo_in,
            "c_vals_uf": tuple(parsed_c_vals),
            "V_supply": val_v,
            "use_r_int": use_r_int_chk,
            "R_int_ohm": val_r_int,
        }

app_state = st.session_state.applied_state
sim = compute_rc_simulation(
    r_topo=app_state["r_topo"],
    r_vals_kohm=app_state["r_vals_kohm"],
    c_topo=app_state["c_topo"],
    c_vals_uf=app_state["c_vals_uf"],
    V_supply=app_state["V_supply"],
    use_r_int=app_state["use_r_int"],
    R_int_ohm=app_state["R_int_ohm"],
)

# 왼쪽(1 비율) 영역: '적용하기'를 눌렀을 때만 반영된 applied_state 기준으로 회로도 그림
with col_circuit:
    draw_circuit_schematic(
        r_topo=app_state["r_topo"],
        r_vals_kohm=list(app_state["r_vals_kohm"]),
        c_topo=app_state["c_topo"],
        c_vals_uf=list(app_state["c_vals_uf"]),
        V_supply=app_state["V_supply"],
        use_r_int=app_state["use_r_int"],
        R_int_ohm=app_state["R_int_ohm"],
    )

# 5. 적용하기 버튼 아래 3개 탭 (충전, 방전, CSV 파일 다운로드)
st.divider()
tab_charge, tab_discharge, tab_csv = st.tabs(["충전", "방전", "CSV 파일 다운로드"])

with tab_charge:
    render_vc_ic_plots(
        t_arr=sim["t_charge_curve"],
        vc_arr=sim["vc_charge_curve"],
        ic_arr_ma=sim["ic_charge_curve_ma"],
        tau_ms=sim["tau_charge_ms"],
        vc_tau_val=sim["vc_charge_tau"],
        vc_tau_pct=sim["vc_charge_tau_pct"],
        ic_tau_val_ma=sim["ic_charge_tau_ma"],
        ic_tau_pct=sim["ic_charge_tau_pct"],
        mode_label="충전",
    )
    m_left, m_right = st.columns(2)
    with m_left:
        st.metric("시정수 τ (충전)", f"{sim['tau_charge_ms']:.2f} ms")
    with m_right:
        st.metric("초기전류 I0 (충전)", f"{sim['I0_charge_ma']:.3f} mA")

    st.subheader("주요 시정수 마일스톤 (1τ ~ 5τ)")
    st.dataframe(sim["df_charge_ms"], use_container_width=True)

with tab_discharge:
    render_vc_ic_plots(
        t_arr=sim["t_dis_curve"],
        vc_arr=sim["vc_dis_curve"],
        ic_arr_ma=sim["ic_dis_curve_ma"],
        tau_ms=sim["tau_dis_ms"],
        vc_tau_val=sim["vc_dis_tau"],
        vc_tau_pct=sim["vc_dis_tau_pct"],
        ic_tau_val_ma=sim["ic_dis_tau_ma"],
        ic_tau_pct=sim["ic_dis_tau_pct"],
        mode_label="방전",
    )
    m_left, m_right = st.columns(2)
    with m_left:
        st.metric("시정수 τ (방전)", f"{sim['tau_dis_ms']:.2f} ms")
    with m_right:
        st.metric("초기전류 I0 (방전)", f"{sim['I0_discharge_ma']:.3f} mA")

    st.subheader("주요 시정수 마일스톤 (1τ ~ 5τ)")
    st.dataframe(sim["df_discharge_ms"], use_container_width=True)

with tab_csv:
    st.write("현재 적용된 설정을 기준으로 CSV 파일을 내려 받을 수 있습니다.")

    df_ch_v = pd.DataFrame({"시간 [s]": np.round(sim["t_charge_csv"], 5), "전압 [V]": np.round(sim["vc_charge_csv"], 5)})
    df_ch_i = pd.DataFrame({"시간 [s]": np.round(sim["t_charge_csv"], 5), "전류 [A]": np.round(sim["ic_charge_csv_a"], 7)})
    df_dis_v = pd.DataFrame({"시간 [s]": np.round(sim["t_dis_csv"], 5), "전압 [V]": np.round(sim["vc_dis_csv"], 5)})
    df_dis_i = pd.DataFrame({"시간 [s]": np.round(sim["t_dis_csv"], 5), "전류 [A]": np.round(sim["ic_dis_csv_a"], 7)})

    st.markdown("**충전**")
    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        st.download_button(
            label="충전 시간-전압 CSV 다운로드",
            data=df_ch_v.to_csv(index=False).encode("utf-8-sig"),
            file_name="rc_charging_voltage.csv",
            mime="text/csv",
            key="dl_ch_v",
            use_container_width=True,
        )
    with c_dl2:
        st.download_button(
            label="충전 시간-전류 CSV 다운로드",
            data=df_ch_i.to_csv(index=False).encode("utf-8-sig"),
            file_name="rc_charging_current.csv",
            mime="text/csv",
            key="dl_ch_i",
            use_container_width=True,
        )

    st.markdown("**방전**")
    d_dl1, d_dl2 = st.columns(2)
    with d_dl1:
        st.download_button(
            label="방전 시간-전압 CSV 다운로드",
            data=df_dis_v.to_csv(index=False).encode("utf-8-sig"),
            file_name="rc_discharging_voltage.csv",
            mime="text/csv",
            key="dl_dis_v",
            use_container_width=True,
        )
    with d_dl2:
        st.download_button(
            label="방전 시간-전류 CSV 다운로드",
            data=df_dis_i.to_csv(index=False).encode("utf-8-sig"),
            file_name="rc_discharging_current.csv",
            mime="text/csv",
            key="dl_dis_i",
            use_container_width=True,
        )
