import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 웹페이지 제목 작성
st.title("⚡ 파이썬 RC 회로 시뮬레이터")
st.write("사이드바에서 변수를 설정한 뒤, 맨 아래 **'적용하기'** 버튼을 눌러 시뮬레이션을 업데이트하세요!")


def parse_positive_float(raw_value, fallback, label, allow_zero=False):
    """직접 입력 또는 선택된 값을 float로 변환하고 유효성을 검사합니다."""
    try:
        val = float(raw_value)
        if np.isnan(val) or np.isinf(val):
            st.sidebar.error(f"{label}은(는) 유효한 숫자여야 합니다. 기본값({fallback:g})을 사용합니다.")
            return float(fallback)
        if allow_zero:
            if val < 0:
                st.sidebar.error(f"{label}은(는) 0 이상이어야 합니다. 기본값({fallback:g})을 사용합니다.")
                return float(fallback)
        else:
            if val <= 0:
                st.sidebar.error(f"{label}은(는) 0보다 큰 값이어야 합니다. 기본값({fallback:g})을 사용합니다.")
                return float(fallback)
        return val
    except (ValueError, TypeError):
        st.sidebar.error(f"{label}에 올바른 숫자를 입력해주세요. 기본값({fallback:g})을 사용합니다.")
        return float(fallback)


def get_rc_series(
    mode: str,
    t_arr: np.ndarray,
    R: float,
    tau: float,
    V_supply: float,
    V_init: float,
    use_v_init: bool,
):
    """충전 또는 방전 모드에 따라 시간 배열 t_arr에 대한 전압 V(t)와 전류 I(t)를 계산합니다."""
    if mode == "charging":
        # V_init에서 시작해 인가전압 V_supply로 수렴: V(t) = V_supply + (V_init - V_supply) * exp(-t/tau)
        V_arr = V_supply + (V_init - V_supply) * np.exp(-t_arr / tau)
        I_arr = ((V_supply - V_init) / R) * np.exp(-t_arr / tau)
    else:
        # 초기 전압(체크 시 V_init, 미체크 시 V_supply)에서 0V로 방전
        V_start = V_init if use_v_init else V_supply
        V_arr = V_start * np.exp(-t_arr / tau)
        I_arr = -(V_start / R) * np.exp(-t_arr / tau)
    return V_arr, I_arr


def render_rc_tab(
    mode: str,
    R: float,
    C_micro: float,
    V_supply: float,
    V_init: float,
    use_v_init: bool,
    fix_axis: bool,
):
    """'charging' 또는 'discharging' 모드에 맞춰 그래프, 핵심 지표, 1τ~5τ 마일스톤 표를 렌더링합니다."""
    is_charging = mode == "charging"
    mode_kr = "충전" if is_charging else "방전"

    # 물리 공식 계산
    C = C_micro * 1e-6
    tau = R * C              # 시정수 (s)
    tau_ms = tau * 1e3       # 시정수 (ms)

    if is_charging:
        I0_signed = (V_supply - V_init) / R
        I0_mA_signed = I0_signed * 1e3
        v_ref_y = V_init + (V_supply - V_init) * 0.632
        v_ref_label = f"1τ V ({v_ref_y:.2f}V)"
        i_ref_y = I0_signed * 0.368
        i_ref_label = "36.8% I0"
        v_title = "Voltage over Time (Charging)"
        i_title = "Current over Time (Charging)"
    else:
        V_start = V_init if use_v_init else V_supply
        I0_signed = -(V_start / R)
        I0_mA_signed = I0_signed * 1e3
        v_ref_y = V_start * 0.368
        v_ref_label = f"36.8% V_start ({v_ref_y:.2f}V)"
        i_ref_y = I0_signed * 0.368
        i_ref_label = "36.8% I0"
        v_title = "Voltage over Time (Discharging)"
        i_title = "Current over Time (Discharging)"

    t_max = 10 if fix_axis else 5 * tau
    t = np.linspace(0, t_max, 1000)
    V_t, I_t = get_rc_series(mode, t, R, tau, V_supply, V_init, use_v_init)

    # 1. 웹 화면에 그래프 띄우기 (그래프 내 시정수 표시는 ms 단위)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 전압 그래프 (왼쪽)
    ax1.plot(t, V_t, label="Voltage $V(t)$", color="blue", linewidth=2)
    ax1.axvline(x=tau, color="red", linestyle="--", label=f"Tau = {tau_ms:.2f}ms")
    ax1.axhline(y=v_ref_y, color="green", linestyle=":", label=v_ref_label)
    ax1.set_title(v_title)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.legend()
    ax1.grid(True)

    # 전류 그래프 (오른쪽)
    ax2.plot(t, I_t, label="Current $I(t)$", color="orange", linewidth=2)
    ax2.axvline(x=tau, color="red", linestyle="--", label=f"Tau = {tau_ms:.2f}ms")
    ax2.axhline(y=i_ref_y, color="green", linestyle=":", label=i_ref_label)
    ax2.set_title(i_title)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Current (A)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # 2. 그래프 하단 핵심 회로 값 요약 (시정수 ms, 초기 전류 mA)
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric(
            label="시정수 τ = RC",
            value=f"{tau_ms:.2f} ms",
            delta=f"{tau:.4f} s (5τ: {5 * tau_ms:.1f} ms)",
            delta_color="off"
        )
    with metric_col2:
        st.metric(
            label="초기 전류 I(0)",
            value=f"{I0_mA_signed:.2f} mA",
            delta=f"{I0_signed:.6f} A",
            delta_color="off"
        )

    # 3. metric_columns 하단: 1τ ~ 5τ 주요 마일스톤 비교 표
    st.subheader(f"⏱️ {mode_kr} 주요 시정수 마일스톤 (1τ ~ 5τ)")
    k_values = np.arange(1, 6)
    t_milestones = k_values * tau
    V_milestones, I_milestones = get_rc_series(mode, t_milestones, R, tau, V_supply, V_init, use_v_init)

    if is_charging:
        rate_col_name = "충전율 (%)"
        rate_values = (V_milestones / V_supply * 100.0) if V_supply != 0 else np.zeros_like(V_milestones)
    else:
        rate_col_name = "방전율 (%)"
        V_start = V_init if use_v_init else V_supply
        rate_values = ((1.0 - V_milestones / V_start) * 100.0) if V_start != 0 else np.zeros_like(V_milestones)

    df_milestones = pd.DataFrame({
        "시정수 구간": [f"{k}τ ({(k * tau_ms):.2f} ms)" for k in k_values],
        "전압 값 (V)": np.round(V_milestones, 4),
        "전류 값 (mA)": np.round(I_milestones * 1e3, 3),
        rate_col_name: np.round(rate_values, 2),
    }).set_index("시정수 구간")

    st.dataframe(df_milestones, use_container_width=True)


def render_csv_download_tab(
    R: float,
    C_micro: float,
    V_supply: float,
    V_init: float,
    use_v_init: bool,
    fix_axis: bool,
):
    """충전 및 방전 시간-전압 / 시간-전류 CSV 다운로드 탭을 렌더링합니다."""
    st.subheader("📥 CSV 파일 다운로드")
    base_desc = (
        f"현재 적용된 설정(**R = {R:g} Ω**, **C = {C_micro:g} μF**, **인가전압 = {V_supply:g} V**"
    )
    if use_v_init:
        base_desc += f", **초기전압 = {V_init:.4g} V**"
    base_desc += ")을 기준으로 CSV 파일을 내려받을 수 있습니다."
    st.write(base_desc)

    C = C_micro * 1e-6
    tau = R * C
    t_max = 10 if fix_axis else 5 * tau
    t_table = np.linspace(0, t_max, 100)

    for mode, title_kr, prefix in [
        ("charging", "충전 (Charging)", "charging"),
        ("discharging", "방전 (Discharging)", "discharging"),
    ]:
        V_table, I_table = get_rc_series(mode, t_table, R, tau, V_supply, V_init, use_v_init)

        df_v = pd.DataFrame({
            "시간 (s)": np.round(t_table, 4),
            "전압 (V)": np.round(V_table, 4)
        })
        df_i = pd.DataFrame({
            "시간 (s)": np.round(t_table, 4),
            "전류 (A)": np.round(I_table, 6)
        })

        st.markdown(f"### {title_kr}")
        c1, c2 = st.columns(2)

        with c1:
            csv_v = df_v.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=f"📥 {title_kr} 시간-전압 CSV 다운로드",
                data=csv_v,
                file_name=f"rc_{prefix}_voltage_R{R:g}_C{C_micro:g}.csv",
                mime="text/csv",
                key=f"tab_dl_v_{prefix}",
                use_container_width=True
            )

        with c2:
            csv_i = df_i.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label=f"📥 {title_kr} 시간-전류 CSV 다운로드",
                data=csv_i,
                file_name=f"rc_{prefix}_current_R{R:g}_C{C_micro:g}.csv",
                mime="text/csv",
                key=f"tab_dl_i_{prefix}",
                use_container_width=True
            )


# 적용된 파라미터 초기값 설정 (세션 상태)
if "applied_params" not in st.session_state:
    st.session_state.applied_params = {
        "R": 1000.0,
        "C_micro": 500.0,
        "V_supply": 5.0,
        "use_v_init": False,
        "R_internal": 0.0,
        "V_init": 0.0,
        "fix_axis": False,
    }

# 2. 사이드바(입력 창) 만들기 - '적용하기' 버튼을 누르기 전까지는 임시 입력값으로만 유지
st.sidebar.header("변수 설정")

# 저항 R: 500 ~ 5000 범위(500 단위) 선택 + 직접 입력 허용
r_options = list(range(500, 5005, 500))
raw_R = st.sidebar.selectbox(
    "저항 R (Ω) - 선택 또는 직접 입력",
    options=r_options,
    index=1,  # 기본값 1000
    accept_new_options=True,
    help="목록에서 고르거나 원하는 저항값(Ω)을 직접 입력한 뒤 Enter를 누르세요."
)

# 전기용량 C: 500 단위 선택 + 직접 입력 허용
c_options = [500, 1000]
raw_C = st.sidebar.selectbox(
    "전기용량 C (μF) - 선택 또는 직접 입력",
    options=c_options,
    index=0,  # 기본값 500
    accept_new_options=True,
    help="목록에서 고르거나 원하는 전기용량(μF)을 직접 입력한 뒤 Enter를 누르세요."
)

# 인가전압: 1 V 단위 선택 + 직접 입력 허용
v_options = [float(i) for i in range(1, 13)]
raw_V_supply = st.sidebar.selectbox(
    "인가전압 (V) - 선택 또는 직접 입력",
    options=v_options,
    index=4,  # 기본값 5.0
    accept_new_options=True,
    help="목록에서 고르거나 원하는 인가전압(V)을 직접 입력한 뒤 Enter를 누르세요."
)

# 초기 전압 V0 != 0 체크박스 및 조건부 내부저항 입력창
use_v_init_input = st.sidebar.checkbox(
    "초기 전압 V0!=0",
    value=st.session_state.applied_params["use_v_init"],
    help="체크하면 내부저항을 입력받아 초기 전압을 계산합니다."
)

raw_R_internal = None
if use_v_init_input:
    raw_R_internal = st.sidebar.text_input(
        "내부저항 R_int (Ω)",
        value=str(int(st.session_state.applied_params["R_internal"]))
        if st.session_state.applied_params["R_internal"] > 0
        else "100",
        help="내부저항 값을 Ω 단위로 입력하세요 (예: 100). 초기 전압 V_init = 인가전압 × R / (R + R_int) 로 계산됩니다."
    )

fix_axis = st.sidebar.checkbox("시간축 고정 (0~10초)", value=st.session_state.applied_params["fix_axis"])

st.sidebar.divider()

# 사이드바 가장 아래 '적용하기' 버튼
if st.sidebar.button("적용하기", type="primary", use_container_width=True):
    parsed_R = parse_positive_float(raw_R, fallback=1000.0, label="저항 R")
    parsed_C = parse_positive_float(raw_C, fallback=500.0, label="전기용량 C")
    parsed_V_supply = parse_positive_float(raw_V_supply, fallback=5.0, label="인가전압", allow_zero=True)

    if use_v_init_input:
        parsed_R_int = parse_positive_float(raw_R_internal, fallback=100.0, label="내부저항", allow_zero=False)
        calculated_V_init = parsed_V_supply * (parsed_R / (parsed_R + parsed_R_int))
    else:
        parsed_R_int = 0.0
        calculated_V_init = 0.0

    st.session_state.applied_params = {
        "R": parsed_R,
        "C_micro": parsed_C,
        "V_supply": parsed_V_supply,
        "use_v_init": use_v_init_input,
        "R_internal": parsed_R_int,
        "V_init": calculated_V_init,
        "fix_axis": fix_axis,
    }

# 현재 시뮬레이션에 실제 반영된 값 불러오기
R = st.session_state.applied_params["R"]
C_micro = st.session_state.applied_params["C_micro"]
V_supply = st.session_state.applied_params["V_supply"]
use_v_init = st.session_state.applied_params["use_v_init"]
R_internal = st.session_state.applied_params["R_internal"]
V_init = st.session_state.applied_params["V_init"]
fix_axis = st.session_state.applied_params["fix_axis"]

# 현재 적용된 변수 값 안내 (내부저항 입력 적용 시에만 인가전압과 초기전압을 함께 표기)
if use_v_init and R_internal > 0:
    voltage_caption = (
        f"인가전압: **{V_supply:g} V**, 초기전압: **{V_init:.4g} V** (내부저항 {R_internal:g} Ω)"
    )
else:
    voltage_caption = f"인가전압: **{V_supply:g} V**"

st.caption(
    f"현재 적용된 파라미터 — 저항 R: **{R:g} Ω**, 전기용량 C: **{C_micro:g} μF**, "
    f"{voltage_caption}, 시간축 고정: **{'ON (0~10초)' if fix_axis else 'OFF (0~5τ)'}**"
)

# 3. 충전 / 방전 / CSV 파일 다운로드 3개 탭 구성
tab_charge, tab_discharge, tab_csv = st.tabs([
    "⚡ 충전",
    "🔋 방전",
    "📥 CSV 파일 다운로드"
])

with tab_charge:
    render_rc_tab("charging", R, C_micro, V_supply, V_init, use_v_init, fix_axis)

with tab_discharge:
    render_rc_tab("discharging", R, C_micro, V_supply, V_init, use_v_init, fix_axis)

with tab_csv:
    render_csv_download_tab(R, C_micro, V_supply, V_init, use_v_init, fix_axis)