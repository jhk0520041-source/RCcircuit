import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

st.set_page_config(page_title="직류 RC회로 시뮬레이션", layout="wide")

# 1. 제목 및 안내 문구
st.title("직류 RC회로 시뮬레이션")
st.caption("변수를 설정한 뒤, 아래 적용하기 버튼을 눌러 시뮬레이션을 확인하세요!")


def parse_float_field(raw_val, field_name: str, min_val: float, strict_min: bool = False):
    """입력값을 float로 변환하고 범위를 검증합니다. (유효성 통과 시 (True, 값, None), 실패 시 (False, None, 에러메시지))"""
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

    # 초기 커패시터 전압 V_c(0) 결정:
    # '초기 전압 V0≠0' 체크 시 입력된 초기전압(V)을 우선 사용하고,
    # 내부저항 R_int가 입력된 경우 R_eq와 R_int의 분압 효과도 함께 반영하여 계산
    if use_v0_nonzero:
        if R_int_ohm > 0:
            # 내부저항이 연결된 상태에서 초기 분압(또는 지정 초기전압) 반영
            V_c0 = V_init_input * (R_eq_ohm / (R_eq_ohm + R_int_ohm))
        else:
            V_c0 = V_init_input
    else:
        V_c0 = 0.0

    # 총 직렬 저항(충전 시 내부저항 포함 여부 반영)
    R_charge_total = R_eq_ohm + (R_int_ohm if use_v0_nonzero else 0.0)

    # 초기전류 I0 (mA): 충전과 방전을 구분하여 계산
    # 충전: t=0+ 에서 (V_supply - V_c0) / R_charge_total (방전 방향은 음수로 표현할지 크기 기준인지 구분)
    I0_charge_a = (V_supply - V_c0) / R_charge_total
    I0_charge_ma = I0_charge_a * 1e3

    # 방전: 초기 축전기 전압(체크 시 V_c0, 미체크 시 V_supply)에서 방전 시작
    V_dis_start = V_c0 if use_v0_nonzero else V_supply
    I0_discharge_a = -(V_dis_start / R_eq_ohm)
    I0_discharge_ma = I0_discharge_a * 1e3

    t_max = max(5.0 * tau_s, 1e-6)
    t_curve = np.linspace(0.0, t_max, 600)
    t_csv = np.linspace(0.0, t_max, 100)

    # 충전 시계열: V_c(t) = V_supply - (V_supply - V_c0) * exp(-t/tau)
    vc_charge_curve = V_supply - (V_supply - V_c0) * np.exp(-t_curve / tau_s)
    ic_charge_curve_ma = ((V_supply - V_c0) / R_charge_total) * np.exp(-t_curve / tau_s) * 1e3

    vc_charge_csv = V_supply - (V_supply - V_c0) * np.exp(-t_csv / tau_s)
    ic_charge_csv_a = ((V_supply - V_c0) / R_charge_total) * np.exp(-t_csv / tau_s)

    # 방전 시계열: V_c(t) = V_dis_start * exp(-t/tau), I_c(t) = -(V_dis_start / R_eq_ohm) * exp(-t/tau)
    vc_dis_curve = V_dis_start * np.exp(-t_curve / tau_s)
    ic_dis_curve_ma = -(V_dis_start / R_eq_ohm) * np.exp(-t_curve / tau_s) * 1e3

    vc_dis_csv = V_dis_start * np.exp(-t_csv / tau_s)
    ic_dis_csv_a = -(V_dis_start / R_eq_ohm) * np.exp(-t_csv / tau_s)

    # 1τ ~ 5τ 마일스톤 표 생성
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
        "V_dis_start": V_dis_start,
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
    """기전력 장치(V), 저항(R), 축전기(C)로 이루어진 간략 회로도를 그립니다."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 외곽 배선 사각형
    ax.plot([1.5, 10.5], [5.5, 5.5], color="#222222", lw=2)
    ax.plot([1.5, 10.5], [1.5, 1.5], color="#222222", lw=2)
    ax.plot([1.5, 1.5], [1.5, 5.5], color="#222222", lw=2)
    ax.plot([10.5, 10.5], [1.5, 5.5], color="#222222", lw=2)

    # 왼쪽: 기전력 장치 (DC Voltage Source)
    src_circle = patches.Circle((1.5, 3.5), 0.75, facecolor="#FFF8E1", edgecolor="#D97706", lw=2, zorder=3)
    ax.add_patch(src_circle)
    ax.text(1.5, 3.65, "EMF", ha="center", va="center", fontsize=9, fontweight="bold", color="#92400E")
    ax.text(1.5, 3.25, f"{V_supply:g} V", ha="center", va="center", fontsize=9, color="#92400E")
    ax.text(1.5, 4.5, "+", ha="center", va="center", fontsize=11, fontweight="bold", color="#B45309")
    ax.text(1.5, 2.5, "-", ha="center", va="center", fontsize=11, fontweight="bold", color="#B45309")

    # 상단 중앙: 저항 블록 (R_eq 및 R1..Rn 표시)
    r_box = patches.FancyBboxPatch(
        (3.6, 4.75), 3.2, 1.5,
        boxstyle="round,pad=0.1", facecolor="#EFF6FF", edgecolor="#2563EB", lw=2, zorder=3
    )
    ax.add_patch(r_box)
    r_detail = ", ".join([f"R{i+1}={v:g}kΩ" for i, v in enumerate(r_vals_kohm)])
    ax.text(5.2, 5.75, f"저항 ({r_topo}): {R_eq_kohm:.3g} kΩ", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1E40AF")
    ax.text(5.2, 5.15, r_detail, ha="center", va="center", fontsize=8, color="#1D4ED8")

    # 오른쪽: 축전기 블록 (C_eq 및 C1..Cn 표시)
    c_box = patches.FancyBboxPatch(
        (8.7, 2.35), 3.0, 2.3,
        boxstyle="round,pad=0.1", facecolor="#ECFDF5", edgecolor="#059669", lw=2, zorder=3
    )
    ax.add_patch(c_box)
    c_detail = ", ".join([f"C{i+1}={v:g}μF" for i, v in enumerate(c_vals_uf)])
    ax.text(10.2, 4.1, f"축전기 ({c_topo})", ha="center", va="center", fontsize=9, fontweight="bold", color="#065F46")
    ax.text(10.2, 3.55, f"C_eq = {C_eq_uf:.3g} μF", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#047857")
    ax.text(10.2, 2.9, c_detail, ha="center", va="center", fontsize=8, color="#065F46")

    # 하단 상태 요약 배지
    status_txt = f"초기전압 V0≠0: {'ON' if use_v0_nonzero else 'OFF'} | Vc(0) = {V_c0:.3g} V"
    if use_v0_nonzero and R_int_ohm > 0:
        status_txt += f" | 내부저항 R_int = {R_int_ohm:g} Ω"
    ax.text(6.0, 0.55, status_txt, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#374151")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_vc_ic_plots(
    t_arr: np.ndarray,
    vc_arr: np.ndarray,
    ic_arr_ma: np.ndarray,
    tau_ms: float,
    mode_label: str,
):
    """Vc-t(왼쪽), Ic-t(오른쪽) 그래프를 나란히 그립니다."""
    c_left, c_right = st.columns(2)

    with c_left:
        fig_v, ax_v = plt.subplots(figsize=(5.5, 3.6))
        ax_v.plot(t_arr, vc_arr, color="#2563EB", lw=2, label="Vc(t)")
        ax_v.axvline(x=tau_ms / 1e3, color="#DC2626", ls="--", lw=1.2, label=f"τ = {tau_ms:.2f} ms")
        ax_v.set_title(f"{mode_label} 커패시터 전압 Vc - t")
        ax_v.set_xlabel("시간 t [s]")
        ax_v.set_ylabel("전압 Vc [V]")
        ax_v.set_aspect("auto")
        ax_v.grid(True, alpha=0.4)
        ax_v.legend(loc="best")
        fig_v.tight_layout()
        st.pyplot(fig_v, use_container_width=True)
        plt.close(fig_v)

    with c_right:
        fig_i, ax_i = plt.subplots(figsize=(5.5, 3.6))
        ax_i.plot(t_arr, ic_arr_ma, color="#EA580C", lw=2, label="Ic(t)")
        ax_i.axvline(x=tau_ms / 1e3, color="#DC2626", ls="--", lw=1.2, label=f"τ = {tau_ms:.2f} ms")
        ax_i.set_title(f"{mode_label} 전류 Ic - t")
        ax_i.set_xlabel("시간 t [s]")
        ax_i.set_ylabel("전류 Ic [mA]")
        ax_i.set_aspect("auto")
        ax_i.grid(True, alpha=0.4)
        ax_i.legend(loc="best")
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

# 2. 메인 영역: 2:1 비율 컬럼 분할
col_circuit, col_inputs = st.columns([2, 1])

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

        # 실시간 미리보기용 합성저항 계산 (유효 숫자일 때만 표시)
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
            index=4,  # 5.0 V
            accept_new_options=True,
            key="v_supply_sel",
        )

        use_v0_chk = st.checkbox("초기 전압 V0≠0", value=False, key="v0_nonzero_chk")

        raw_V_init = "0.0"
        raw_R_int = "0.0"
        if use_v0_chk:
            # 드롭다운(직접 입력 가능) 2번째: 초기전압 [V] 입력 및 내부저항 [Ω] 입력
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

# 왼쪽(2 비율) 영역: 적용된 변수 기준 회로도 갱신
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