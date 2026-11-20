import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager

st.set_page_config(page_title="직류 RC회로 시뮬레이션", layout="wide")


def configure_korean_font():
    """matplotlib에서 한글이 깨지지 않도록 사용 가능한 한글 폰트를 설정합니다."""
    candidate_fonts = [
        "Malgun Gothic",      # Windows
        "AppleGothic",       # macOS
        "NanumGothic",       # Linux / Nanum
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "UnDotum",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            break
    else:
        # 시스템 경로에서 ttf/otf 직접 탐색
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_manager.fontManager.addfont(path)
                prop = font_manager.FontProperties(fname=path)
                plt.rcParams["font.family"] = prop.get_name()
                break

    plt.rcParams["axes.unicode_minus"] = False


configure_korean_font()

# 1. 제목 및 안내 문구
st.title("직류 RC회로 시뮬레이션")
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
    use_v0_nonzero: bool,
    V_init_input: float,
    R_int_ohm: float,
):
    """적용된 변수 조합에 대해 합성소자, 시정수, 초기전류, 충·방전 시계열 및 마일스톤을 계산합니다."""
    R_eq_kohm = calc_equiv_resistance_kohm(r_topo, list(r_vals_kohm))
    R_eq_ohm = R_eq_kohm * 1e3

    C_eq_uf = calc_equiv_capacitance_uf(c_topo, list(c_vals_uf))
    C_eq_f = C_eq_uf * 1e-6

    tau_s = R_eq_ohm * C_eq_f
    tau_ms = tau_s * 1e3

    if use_v0_nonzero:
        if R_int_ohm > 0:
            V_c0 = V_init_input * (R_eq_ohm / (R_eq_ohm + R_int_ohm))
        else:
            V_c0 = V_init_input
    else:
        V_c0 = 0.0

    R_charge_total = R_eq_ohm + (R_int_ohm if use_v0_nonzero else 0.0)

    # 초기전류 I0 (mA)
    I0_charge_a = (V_supply - V_c0) / R_charge_total
    I0_charge_ma = I0_charge_a * 1e3

    V_dis_start = V_c0 if use_v0_nonzero else V_supply
    I0_discharge_a = -(V_dis_start / R_eq_ohm)
    I0_discharge_ma = I0_discharge_a * 1e3

    t_max = max(5.0 * tau_s, 1e-6)
    t_curve = np.linspace(0.0, t_max, 600)
    t_csv = np.linspace(0.0, t_max, 100)

    # 충전 시계열
    vc_charge_curve = V_supply - (V_supply - V_c0) * np.exp(-t_curve / tau_s)
    ic_charge_curve_ma = ((V_supply - V_c0) / R_charge_total) * np.exp(-t_curve / tau_s) * 1e3

    vc_charge_csv = V_supply - (V_supply - V_c0) * np.exp(-t_csv / tau_s)
    ic_charge_csv_a = ((V_supply - V_c0) / R_charge_total) * np.exp(-t_csv / tau_s)

    # 방전 시계열
    vc_dis_curve = V_dis_start * np.exp(-t_curve / tau_s)
    ic_dis_curve_ma = -(V_dis_start / R_eq_ohm) * np.exp(-t_curve / tau_s) * 1e3

    vc_dis_csv = V_dis_start * np.exp(-t_csv / tau_s)
    ic_dis_csv_a = -(V_dis_start / R_eq_ohm) * np.exp(-t_csv / tau_s)

    # t = tau 시점의 값 및 비율(%)
    vc_charge_tau = float(V_supply - (V_supply - V_c0) * np.exp(-1.0))
    ic_charge_tau_ma = float(I0_charge_ma * np.exp(-1.0))
    vc_charge_tau_pct = (vc_charge_tau / V_supply * 100.0) if V_supply != 0 else 0.0
    ic_charge_tau_pct = (ic_charge_tau_ma / I0_charge_ma * 100.0) if I0_charge_ma != 0 else 0.0

    vc_dis_tau = float(V_dis_start * np.exp(-1.0))
    ic_dis_tau_ma = float(I0_discharge_ma * np.exp(-1.0))
    vc_dis_tau_pct = (vc_dis_tau / V_dis_start * 100.0) if V_dis_start != 0 else 0.0
    ic_dis_tau_pct = (ic_dis_tau_ma / I0_discharge_ma * 100.0) if I0_discharge_ma != 0 else 0.0

    # 1τ ~ 5τ 마일스톤 표
    k_arr = np.arange(1, 6)
    t_ms_arr = k_arr * tau_s

    vc_charge_ms = V_supply - (V_supply - V_c0) * np.exp(-t_ms_arr / tau_s)
    ic_charge_ms_ma = ((V_supply - V_c0) / R_charge_total) * np.exp(-t_ms_arr / tau_s) * 1e3
    charge_rate_pct = (vc_charge_ms / V_supply * 100.0) if V_supply != 0 else np.zeros_like(vc_charge_ms)

    df_charge_ms = pd.DataFrame({
        "시정수 구간": [f"{k}τ ({k * tau_ms:.2f} ms)" for k in k_arr],
        "전압 값 [V]": np.round(vc_charge_ms, 4),
        "충전율 [%]": np.round(charge_rate_pct, 2),
        "전류 값 [mA]": np.round(ic_charge_ms_ma, 3),
    }).set_index("시정수 구간")

    vc_dis_ms = V_dis_start * np.exp(-t_ms_arr / tau_s)
    ic_dis_ms_ma = -(V_dis_start / R_eq_ohm) * np.exp(-t_ms_arr / tau_s) * 1e3
    discharge_rate_pct = (
        ((V_dis_start - vc_dis_ms) / V_dis_start * 100.0) if V_dis_start != 0 else np.zeros_like(vc_dis_ms)
    )

    df_discharge_ms = pd.DataFrame({
        "시정수 구간": [f"{k}τ ({k * tau_ms:.2f} ms)" for k in k_arr],
        "전압 값 [V]": np.round(vc_dis_ms, 4),
        "방전율 [%]": np.round(discharge_rate_pct, 2),
        "전류 값 [mA]": np.round(ic_dis_ms_ma, 3),
    }).set_index("시정수 구간")

    return {
        "R_eq_kohm": R_eq_kohm,
        "R_eq_ohm": R_eq_ohm,
        "C_eq_uf": C_eq_uf,
        "tau_ms": tau_ms,
        "V_c0": V_c0,
        "I0_charge_ma": I0_charge_ma,
        "I0_discharge_ma": I0_discharge_ma,
        "t_curve": t_curve,
        "t_csv": t_csv,
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
    R_eq_kohm: float,
    c_topo: str,
    c_vals_uf: list[float],
    C_eq_uf: float,
    V_supply: float,
    use_v0_nonzero: bool,
    V_c0: float,
    R_int_ohm: float,
):
    """직렬/병렬 연결 상태에 따라 기전력 장치(V), 저항(R), 축전기(C) 회로도를 그립니다."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 왼쪽 기전력 장치 및 좌측 수직선
    ax.plot([1.4, 1.4], [1.6, 6.2], color="#222222", lw=2)
    ax.plot([1.4, 10.6], [6.2, 6.2], color="#222222", lw=2)
    ax.plot([1.4, 10.6], [1.6, 1.6], color="#222222", lw=2)
    ax.plot([10.6, 10.6], [1.6, 6.2], color="#222222", lw=2)

    src_circle = patches.Circle((1.4, 3.9), 0.72, facecolor="#FFF8E1", edgecolor="#D97706", lw=2, zorder=4)
    ax.add_patch(src_circle)
    ax.text(1.4, 4.08, "기전력", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#92400E")
    ax.text(1.4, 3.68, f"{V_supply:g} V", ha="center", va="center", fontsize=8.5, color="#92400E")

    # 상단 저항 블록: 직렬(가로 나열) vs 병렬(세로 분기)
    n_r = len(r_vals_kohm)
    if r_topo == "직렬":
        # 상단선 위에 R1..Rn을 가로로 일렬 배치
        ax.plot([2.5, 7.5], [6.2, 6.2], color="#222222", lw=2)
        seg_w = 4.4 / n_r
        box_w = min(1.25, seg_w * 0.75)
        for idx, r_val in enumerate(r_vals_kohm):
            cx = 2.8 + (idx + 0.5) * seg_w
            rect = patches.Rectangle(
                (cx - box_w / 2, 5.85), box_w, 0.7,
                facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.8, zorder=4
            )
            ax.add_patch(rect)
            ax.text(cx, 6.95, f"R{idx+1}={r_val:g}kΩ", ha="center", va="center", fontsize=8, color="#1E40AF", fontweight="bold")
        ax.text(5.0, 7.55, f"저항 직렬 (R_eq = {R_eq_kohm:.3g} kΩ)", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")
    else:
        # 병렬: 분기 노드(x=3.2, x=6.8) 사이 여러 세로 가지로 R1..Rn 배치
        y_positions = np.linspace(7.3, 5.2, n_r) if n_r > 1 else [6.2]
        ax.plot([2.5, 3.2], [6.2, 6.2], color="#222222", lw=2)
        ax.plot([6.8, 7.5], [6.2, 6.2], color="#222222", lw=2)
        ax.plot([3.2, 3.2], [min(y_positions), max(y_positions)], color="#222222", lw=2)
        ax.plot([6.8, 6.8], [min(y_positions), max(y_positions)], color="#222222", lw=2)
        for idx, (r_val, y_pos) in enumerate(zip(r_vals_kohm, y_positions)):
            ax.plot([3.2, 6.8], [y_pos, y_pos], color="#222222", lw=1.8)
            rect = patches.Rectangle(
                (4.35, y_pos - 0.25), 1.3, 0.5,
                facecolor="#EFF6FF", edgecolor="#2563EB", lw=1.8, zorder=4
            )
            ax.add_patch(rect)
            ax.text(5.0, y_pos, f"R{idx+1}={r_val:g}kΩ", ha="center", va="center", fontsize=7.8, color="#1E40AF", fontweight="bold")
        ax.text(5.0, 7.78, f"저항 병렬 (R_eq = {R_eq_kohm:.3g} kΩ)", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")

    # 우측 축전기 블록: 직렬(세로 나열) vs 병렬(가로 분기)
    n_c = len(c_vals_uf)
    if c_topo == "직렬":
        # 우측 수직선 위에 C1..Cn을 세로로 일렬 배치
        y_centers = np.linspace(5.3, 2.5, n_c) if n_c > 1 else [3.9]
        for idx, (c_val, yc) in enumerate(zip(c_vals_uf, y_centers)):
            ax.plot([10.3, 10.9], [yc + 0.14, yc + 0.14], color="#059669", lw=2.4, zorder=4)
            ax.plot([10.3, 10.9], [yc - 0.14, yc - 0.14], color="#059669", lw=2.4, zorder=4)
            ax.text(11.45, yc, f"C{idx+1}={c_val:g}μF", ha="left", va="center", fontsize=8, color="#065F46", fontweight="bold")
        ax.text(10.6, 6.65, f"축전기 직렬\nC_eq={C_eq_uf:.3g}μF", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#065F46")
    else:
        # 병렬: 상단/하단 버스선 사이에 C1..Cn을 가로로 나란히 병렬 배치
        x_positions = np.linspace(8.8, 11.2, n_c) if n_c > 1 else [10.6]
        ax.plot([8.5, 10.6], [6.2, 6.2], color="#222222", lw=2)
        ax.plot([8.5, 10.6], [1.6, 1.6], color="#222222", lw=2)
        for idx, (c_val, xp) in enumerate(zip(c_vals_uf, x_positions)):
            ax.plot([xp, xp], [1.6, 3.75], color="#222222", lw=1.6)
            ax.plot([xp, xp], [4.05, 6.2], color="#222222", lw=1.6)
            ax.plot([xp - 0.22, xp + 0.22], [3.9, 3.9], color="#059669", lw=2.5, zorder=4)
            ax.text(xp, 2.25, f"C{idx+1}\n{c_val:g}μF", ha="center", va="center", fontsize=7.5, color="#065F46", fontweight="bold")
        ax.text(10.0, 6.65, f"축전기 병렬 (C_eq={C_eq_uf:.3g}μF)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#065F46")

    # 하단 상태 표시
    status_txt = f"초기전압 V0≠0: {'ON' if use_v0_nonzero else 'OFF'} | Vc(0) = {V_c0:.3g} V"
    if use_v0_nonzero and R_int_ohm > 0:
        status_txt += f" | 내부저항 R_int = {R_int_ohm:g} Ω"
    ax.text(6.0, 0.55, status_txt, ha="center", va="center", fontsize=9, fontweight="bold", color="#374151")

    fig.tight_layout()
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
    """Vc-t(왼쪽), Ic-t(오른쪽) 그래프를 그리고 시정수 τ 시점의 노란색 가로 점선 및 퍼센트를 표기합니다."""
    tau_s = tau_ms / 1e3
    c_left, c_right = st.columns(2)

    with c_left:
        fig_v, ax_v = plt.subplots(figsize=(5.5, 3.8))
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
            xytext=(tau_s * 1.15, vc_tau_val),
            fontsize=8.5,
            fontweight="bold",
            color="#854D0E",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9C3", edgecolor="#EAB308", lw=0.8),
        )
        ax_v.set_title(f"{mode_label} 커패시터 전압 Vc - t")
        ax_v.set_xlabel("시간 t [s]")
        ax_v.set_ylabel("전압 Vc [V]")
        ax_v.set_aspect("auto")
        ax_v.grid(True, alpha=0.4)
        ax_v.legend(loc="best", fontsize=8)
        fig_v.tight_layout()
        st.pyplot(fig_v, use_container_width=True)
        plt.close(fig_v)

    with c_right:
        fig_i, ax_i = plt.subplots(figsize=(5.5, 3.8))
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
            f"{ic_tau_pct:.1f}% ({ic_tau_val_ma:.2f}mA)",
            xy=(tau_s, ic_tau_val_ma),
            xytext=(tau_s * 1.15, ic_tau_val_ma),
            fontsize=8.5,
            fontweight="bold",
            color="#854D0E",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9C3", edgecolor="#EAB308", lw=0.8),
        )
        ax_i.set_title(f"{mode_label} 전류 Ic - t")
        ax_i.set_xlabel("시간 t [s]")
        ax_i.set_ylabel("전류 Ic [mA]")
        ax_i.set_aspect("auto")
        ax_i.grid(True, alpha=0.4)
        ax_i.legend(loc="best", fontsize=8)
        fig_i.tight_layout()
        st.pyplot(fig_i, use_container_width=True)
        plt.close(fig_i)


# 적용된 파라미터 초기값 설정
if "applied_state" not in st.session_state:
    st.session_state.applied_state = {
        "r_topo": "직렬",
        "r_vals_kohm": (1.0,),
        "c_topo": "직렬",
        "c_vals_uf": (100.0,),
        "V_supply": 5.0,
        "use_v0_nonzero": False,
        "V_init_input": 0.0,
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

        preview_r = []
        valid_r_preview = True
        for rv in raw_r_list:
            ok, num, _ = parse_float_field(rv, "R", min_val=0.0, strict_min=True)
            if not ok:
                valid_r_preview = False
                break
            preview_r.append(num)
        if valid_r_preview:
            st.info(f"합성저항: **{calc_equiv_resistance_kohm(r_topo_in, preview_r):.4g} ㏀**")
        else:
            st.warning("R 값을 숫자로 입력하세요.")

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

        preview_c = []
        valid_c_preview = True
        for cv in raw_c_list:
            ok, num, _ = parse_float_field(cv, "C", min_val=0.0, strict_min=True)
            if not ok:
                valid_c_preview = False
                break
            preview_c.append(num)
        if valid_c_preview:
            st.info(f"합성전기용량: **{calc_equiv_capacitance_uf(c_topo_in, preview_c):.4g} ㎌**")
        else:
            st.warning("C 값을 숫자로 입력하세요.")

    # 3.3 인가전압 V 값 입력
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

        use_v0_chk = st.checkbox("초기 전압 V0≠0", value=False, key="v0_nonzero_chk")

        raw_V_init = "0.0"
        raw_R_int = "0.0"
        if use_v0_chk:
            raw_V_init = st.selectbox(
                "초기전압 V0 [V]",
                options=[0.0, 1.0, 2.0, 3.0, 5.0],
                index=1,
                accept_new_options=True,
                key="v0_init_sel",
            )
            raw_R_int = st.text_input(
                "내부저항 [Ω]",
                value="10",
                key="r_int_input",
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

    val_v_init = 0.0
    val_r_int = 0.0
    if use_v0_chk:
        ok_vi, val_vi, msg_vi = parse_float_field(raw_V_init, "초기전압 V0 [V]", min_val=0.0, strict_min=False)
        if not ok_vi:
            errors.append(msg_vi)
        else:
            val_v_init = val_vi

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
            "use_v0_nonzero": use_v0_chk,
            "V_init_input": val_v_init,
            "R_int_ohm": val_r_int,
        }

app_state = st.session_state.applied_state
sim = compute_rc_simulation(
    r_topo=app_state["r_topo"],
    r_vals_kohm=app_state["r_vals_kohm"],
    c_topo=app_state["c_topo"],
    c_vals_uf=app_state["c_vals_uf"],
    V_supply=app_state["V_supply"],
    use_v0_nonzero=app_state["use_v0_nonzero"],
    V_init_input=app_state["V_init_input"],
    R_int_ohm=app_state["R_int_ohm"],
)

# 왼쪽(1 비율) 영역: 적용된 변수 및 직렬/병렬 연결 형태 반영 회로도
with col_circuit:
    draw_circuit_schematic(
        r_topo=app_state["r_topo"],
        r_vals_kohm=list(app_state["r_vals_kohm"]),
        R_eq_kohm=sim["R_eq_kohm"],
        c_topo=app_state["c_topo"],
        c_vals_uf=list(app_state["c_vals_uf"]),
        C_eq_uf=sim["C_eq_uf"],
        V_supply=app_state["V_supply"],
        use_v0_nonzero=app_state["use_v0_nonzero"],
        V_c0=sim["V_c0"],
        R_int_ohm=app_state["R_int_ohm"],
    )

# 5. 적용하기 버튼 아래 3개 탭 (충전, 방전, CSV 파일 다운로드)
st.divider()
tab_charge, tab_discharge, tab_csv = st.tabs(["충전", "방전", "CSV 파일 다운로드"])

with tab_charge:
    render_vc_ic_plots(
        t_arr=sim["t_curve"],
        vc_arr=sim["vc_charge_curve"],
        ic_arr_ma=sim["ic_charge_curve_ma"],
        tau_ms=sim["tau_ms"],
        vc_tau_val=sim["vc_charge_tau"],
        vc_tau_pct=sim["vc_charge_tau_pct"],
        ic_tau_val_ma=sim["ic_charge_tau_ma"],
        ic_tau_pct=sim["ic_charge_tau_pct"],
        mode_label="충전",
    )
    m_left, m_right = st.columns(2)
    with m_left:
        st.metric("시정수 τ = RC", f"{sim['tau_ms']:.2f} ms")
    with m_right:
        st.metric("초기전류 I0 (충전)", f"{sim['I0_charge_ma']:.3f} mA")

    st.subheader("주요 시정수 마일스톤 (1τ ~ 5τ)")
    st.dataframe(sim["df_charge_ms"], use_container_width=True)

with tab_discharge:
    render_vc_ic_plots(
        t_arr=sim["t_curve"],
        vc_arr=sim["vc_dis_curve"],
        ic_arr_ma=sim["ic_dis_curve_ma"],
        tau_ms=sim["tau_ms"],
        vc_tau_val=sim["vc_dis_tau"],
        vc_tau_pct=sim["vc_dis_tau_pct"],
        ic_tau_val_ma=sim["ic_dis_tau_ma"],
        ic_tau_pct=sim["ic_dis_tau_pct"],
        mode_label="방전",
    )
    m_left, m_right = st.columns(2)
    with m_left:
        st.metric("시정수 τ = RC", f"{sim['tau_ms']:.2f} ms")
    with m_right:
        st.metric("초기전류 I0 (방전)", f"{sim['I0_discharge_ma']:.3f} mA")

    st.subheader("주요 시정수 마일스톤 (1τ ~ 5τ)")
    st.dataframe(sim["df_discharge_ms"], use_container_width=True)

with tab_csv:
    st.write("현재 적용된 설정을 기준으로 CSV 파일을 내려 받을 수 있습니다.")

    df_ch_v = pd.DataFrame({"시간 [s]": np.round(sim["t_csv"], 5), "전압 [V]": np.round(sim["vc_charge_csv"], 5)})
    df_ch_i = pd.DataFrame({"시간 [s]": np.round(sim["t_csv"], 5), "전류 [A]": np.round(sim["ic_charge_csv_a"], 7)})
    df_dis_v = pd.DataFrame({"시간 [s]": np.round(sim["t_csv"], 5), "전압 [V]": np.round(sim["vc_dis_csv"], 5)})
    df_dis_i = pd.DataFrame({"시간 [s]": np.round(sim["t_csv"], 5), "전류 [A]": np.round(sim["ic_dis_csv_a"], 7)})

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