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


def get_rc_series(mode: str, t_arr: np.ndarray, R: float, tau: float, V0: float, I0: float):
    """충전 또는 방전 모드에 따라 시간 배열 t_arr에 대한 전압 V(t)와 전류 I(t)를 계산합니다."""
    if mode == "charging":
        V_arr = V0 * (1 - np.exp(-t_arr / tau))
        I_arr = I0 * np.exp(-t_arr / tau)
    else:
        V_arr = V0 * np.exp(-t_arr / tau)
        I_arr = -I0 * np.exp(-t_arr / tau)
    return V_arr, I_arr


def render_rc_tab(mode: str, R: float, C_micro: float, V0: float, fix_axis: bool, num_points: int):
    """'charging' 또는 'discharging' 모드에 맞춰 그래프, 핵심 지표, 1τ~5τ 마일스톤 표, 데이터 표를 렌더링합니다."""
    is_charging = mode == "charging"
    mode_kr = "충전" if is_charging else "방전"

    # 물리 공식 계산
    C = C_micro * 1e-6
    tau = R * C              # 시정수 (s)
    tau_ms = tau * 1e3       # 시정수 (ms)
    I0 = V0 / R              # 초기 전류 (A)
    I0_mA = I0 * 1e3         # 초기 전류 (mA)

    t_max = 10 if fix_axis else 5 * tau
    t = np.linspace(0, t_max, 1000)
    V_t, I_t = get_rc_series(mode, t, R, tau, V0, I0)

    if is_charging:
        v_ref_y = V0 * 0.632
        v_ref_label = "63.2% V0"
        i_ref_y = I0 * 0.368
        i_ref_label = "36.8% I0"
        v_title = "Voltage over Time (Charging)"
        i_title = "Current over Time (Charging)"
    else:
        v_ref_y = V0 * 0.368
        v_ref_label = "36.8% V0"
        i_ref_y = -I0 * 0.368
        i_ref_label = "-36.8% I0"
        v_title = "Voltage over Time (Discharging)"
        i_title = "Current over Time (Discharging)"

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
            label="초기 전류 I₀ = V₀ / R",
            value=f"{I0_mA:.2f} mA",
            delta=f"{I0:.6f} A",
            delta_color="off"
        )

    # 3. metric_columns 하단: 1τ ~ 5τ 주요 마일스톤 비교 표
    st.subheader(f"⏱️ {mode_kr} 주요 시정수 마일스톤 (1τ ~ 5τ)")
    k_values = np.arange(1, 6)
    t_milestones = k_values * tau
    V_milestones, I_milestones = get_rc_series(mode, t_milestones, R, tau, V0, I0)

    df_milestones = pd.DataFrame({
        "시정수 구간": [f"{k}τ ({(k * tau_ms):.2f} ms)" for k in k_values],
        "전압 값 (V)": np.round(V_milestones, 4),
        "전류 값 (A)": np.round(I_milestones, 6),
        "전류 값 (mA)": np.round(I_milestones * 1e3, 3),
    }).set_index("시정수 구간")

    st.dataframe(df_milestones, use_container_width=True)

    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**시간 - 전압 데이터 ({mode_kr})**")
        st.dataframe(df_voltage, use_container_width=True, height=250)

    with col2:
        st.markdown(f"**시간 - 전류 데이터 ({mode_kr})**")
        st.dataframe(df_current, use_container_width=True, height=250)


def render_csv_download_tab(R: float, C_micro: float, V0: float, fix_axis: bool, num_points: int):
    """충전 및 방전 시간-전압 / 시간-전류 CSV 다운로드 탭을 렌더링합니다."""
    st.subheader("📥 CSV 파일 다운로드")
    st.write(
        f"현재 적용된 설정(**R = {R:g} Ω**, **C = {C_micro:g} μF**, **V0 = {V0:g} V**, "
        f"**분할 개수 = {num_points}개**)을 기준으로 CSV 파일을 내려받을 수 있습니다."
    )

    C = C_micro * 1e-6
    tau = R * C
    t_max = 10 if fix_axis else 5 * tau
    t_table = np.linspace(0, t_max, num_points)

    for mode, title_kr, prefix in [
        ("charging", "충전 (Charging)", "charging"),
        ("discharging", "방전 (Discharging)", "discharging"),
    ]:
        V_table, I_table = get_rc_series(mode, t_table, R, tau, V0, V0 / R)

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
        "V0": 5.0,
        "fix_axis": False,
        "num_points": 20,
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

# 인가전압 V0: 1 V 단위 선택 + 직접 입력 허용
v_options = [float(i) for i in range(1, 13)]
raw_V0 = st.sidebar.selectbox(
    "인가전압 V0 (V) - 선택 또는 직접 입력",
    options=v_options,
    index=4,  # 기본값 5.0
    accept_new_options=True,
    help="목록에서 고르거나 원하는 인가전압(V)을 직접 입력한 뒤 Enter를 누르세요."
)

fix_axis = st.sidebar.checkbox("시간축 고정 (0~10초)", value=st.session_state.applied_params["fix_axis"])
num_points = st.sidebar.slider(
    "표 데이터 분할 개수",
    min_value=10,
    max_value=100,
    value=st.session_state.applied_params["num_points"],
    step=5
)

st.sidebar.divider()

# 사이드바 가장 아래 '적용하기' 버튼
if st.sidebar.button("적용하기", type="primary", use_container_width=True):
    parsed_R = parse_positive_float(raw_R, fallback=1000.0, label="저항 R")
    parsed_C = parse_positive_float(raw_C, fallback=500.0, label="전기용량 C")
    parsed_V0 = parse_positive_float(raw_V0, fallback=5.0, label="인가전압 V0", allow_zero=True)

    st.session_state.applied_params = {
        "R": parsed_R,
        "C_micro": parsed_C,
        "V0": parsed_V0,
        "fix_axis": fix_axis,
        "num_points": num_points,
    }

# 현재 시뮬레이션에 실제 반영된 값 불러오기
R = st.session_state.applied_params["R"]
C_micro = st.session_state.applied_params["C_micro"]
V0 = st.session_state.applied_params["V0"]
fix_axis = st.session_state.applied_params["fix_axis"]
num_points = st.session_state.applied_params["num_points"]

# 현재 적용된 변수 값 안내
st.caption(
    f"현재 적용된 파라미터 — 저항 R: **{R:g} Ω**, 전기용량 C: **{C_micro:g} μF**, "
    f"인가전압 V0: **{V0:g} V**, 시간축 고정: **{'ON (0~10초)' if fix_axis else 'OFF (0~5τ)'}**, "
    f"표 분할 개수: **{num_points}개**"
)

# 3. 충전 / 방전 / CSV 파일 다운로드 3개 탭 구성
tab_charge, tab_discharge, tab_csv = st.tabs([
    "⚡ 충전",
    "🔋 방전",
    "📥 CSV 파일 다운로드"
])

with tab_charge:
    render_rc_tab("charging", R, C_micro, V0, fix_axis, num_points)

with tab_discharge:
    render_rc_tab("discharging", R, C_micro, V0, fix_axis, num_points)

with tab_csv:
    render_csv_download_tab(R, C_micro, V0, fix_axis, num_points)