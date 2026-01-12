import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import time
import joblib
import gc
from datetime import datetime
from utils import CoordinateSmoother, get_heat_center, MultiScaleBuffer
import platform

# 시스템 환경 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="SMART WALL GUARD", layout="wide")

# 세션 상태 초기화
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = None

if "log_history" not in st.session_state:
    st.session_state.log_history = []

if "emergency_triggered" not in st.session_state:
    st.session_state.emergency_triggered = False

if "show_emergency_dialog" not in st.session_state:
    st.session_state.show_emergency_dialog = False

if "page" not in st.session_state:
    st.session_state.page = "main"

# 상태 락
if "locked_event" not in st.session_state:
    st.session_state.locked_event = None  # None / "fall" / "impact"

if "event_lock_until" not in st.session_state:
    st.session_state.event_lock_until = 0

def open_modal():
    st.session_state.show_emergency_dialog = True

def close_modal():
    st.session_state.show_emergency_dialog = False

# 긴급 상황 팝업    
def get_alert_overlay(status, detail):
    # 그리드 모니터 위에 겹쳐질 빨간색 경고창 HTML
    alert_html = f"""
    <div class="alert-overlay-container">
        <div class="alert-content">
            <h2 style="margin: 0; color: white; font-size: 1.5rem;"> {status}</h2>
            <p style="margin: 5px 0; font-size: 1rem; font-weight: bold;">{detail}</p>
            <p style="font-size: 0.8rem; opacity: 0.8; margin: 0;">서울시 마포구 (MAPO-A1)</p>
        </div>
    </div>
    """
    return alert_html

# 신고 확인 팝업
@st.dialog("🚑 긴급 신고 및 위치 공유")
def confirm_emergency_report():
    st.warning("⚠️ 확인 버튼을 누르면 119/112 상황실로 긴급 신고 메시지가 즉시 발송됩니다.")
    
    # 전송될 내용 미리보기
    current_time = datetime.now().strftime("%H:%M:%S")
    latest_event = st.session_state.log_history[0]['이벤트'] if st.session_state.log_history else "정상 상황 감지"
    
    report_content = f"""[SMART WALL GUARD 긴급신고]
- 주소: 서울시 마포구 새창로4가길 123
- 상황: {latest_event}
- 시각: {current_time}
- 비상연락처: 010-ABCD-EFGH"""
    
    st.markdown("**전송 내용 미리보기:**")
    st.code(report_content, language=None)
    
    st.write("정말 전송하시겠습니까?")
    
    # 확인/취소 버튼
    c1, c2 = st.columns(2)
    with c1:
        if st.button("신고하기", use_container_width=True, type="primary"):
            # 실제 SMS API 연동 시 이 부분에 코드가 들어갑니다.
            st.success("신고 메시지가 전송되었습니다.")
            st.toast("🚑 119/112 긴급 신고 완료")
            time.sleep(1)
            close_modal()
            st.rerun()
    with c2:
        if st.button("취소", use_container_width=True):
            st.session_state.show_emergency_dialog = False
            st.toast("취소되었습니다")
            st.rerun()

# 데이터 엔진 및 기능 함수
def get_simulated_data():
    # 시연 모드일 경우 강제로 위험 데이터 생성
    if st.session_state.demo_mode == "impact":
        raw_pixels = np.random.uniform(35, 38, (8, 8)) # 아주 뜨거운 열원
        impact = np.random.uniform(26000, 30000)
        st.session_state.demo_mode = None # 일회성 실행 후 해제
    elif st.session_state.demo_mode == "fall":
        raw_pixels = np.random.uniform(32, 34, (8, 8))
        impact = np.random.uniform(18000, 21000) 
        st.session_state.demo_mode = None
    else:
        # 기존 일반 데이터 생성 로직
        raw_pixels = np.random.uniform(22, 26, (8, 8))
        is_detected = np.random.random() < 0.7 
        pos = (np.random.randint(1, 6), np.random.randint(1, 6))
        if is_detected:
            raw_pixels[pos[0]:pos[0]+2, pos[1]:pos[1]+2] += np.random.uniform(10, 15)
        impact = np.random.normal(16384, 600)

    return {
        "pixels": raw_pixels,
        "is_detected": True if raw_pixels.max() > 30 else False,
        "impact": impact,
        "time": datetime.now().strftime("%H:%M:%S")
    }

def min_max_normalize(matrix, min_temp=20.0, max_temp=40.0):
    normalized = (matrix - min_temp) / (max_temp - min_temp)
    return np.clip(normalized, 0, 1) # 0.0 ~ 1.0 사이로 값 고정

def emergency_button(label, phone_number, color="#007BFF"):
    button_html = f"""
        <a href="tel:{phone_number}" style="text-decoration: none;">
            <div style="
                width: 100%; height: 2.3rem; background-color: {color}; color: #FFFFFF;
                border: none; font-size: 0.95rem; font-weight: 600; border-radius: 6px;
                display: flex; align-items: center; justify-content: center;
                margin-bottom: 10px; cursor: pointer;
            ">
                {label}
            </div>
        </a>
    """
    st.markdown(button_html, unsafe_allow_html=True)

# CSS 설정
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; color: #000000 !important; }

    div[data-testid="stPopover"] button svg {
        display: none !important;
    }

    div[data-testid="stPopover"] button:hover {
        transform: scale(1.1);
        background-color: rgba(0,0,0,0.05) !important;
    }

    [data-testid="column"] {
        display: flex;
        align-items: center;
        justify-content: flex-start;
    }

    [data-testid="column"]:nth-child(2), [data-testid="column"]:nth-child(3) {
        justify-content: flex-end;
    }

    /* [핵심] 그리드 모니터 컨테이너를 기준점으로 설정 */
    [data-testid="stVerticalBlock"] > div:has(> .grid-monitor-box) {
        position: relative !important;
    }

    /* 경고창 전체 레이어 */
    .alert-overlay-container {
        position: absolute;
        top: -450px;
        left: 10px;
        right: 10px;
        z-index: 1000;
        pointer-events: none; /* 클릭 방해 금지 */
    }

    /* 경고창 내부 박스 */
    .alert-content {
        background-color: rgba(220, 20, 60, 0.9); /* 강렬한 크림슨 레드 */
        color: white;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #ffffff;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        animation: alert-blink 0.8s infinite;
    }

    @keyframes alert-blink {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(0.98); opacity: 0.8; }
        100% { transform: scale(1); opacity: 1; }
    }     
            
    .section-title {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
        margin-top: 0px !important;
        margin-bottom: 20px !important;
        display: block !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid #E9ECEF !important;
        padding: 10px 15px !important;
        border-radius: 8px !important;
        border-left: 5px solid #007BFF !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stMetricLabel"] { color: #666666 !important; font-size: 0.9rem !important; }
    [data-testid="stMetricValue"] { color: #000000 !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { transform: translateY(5px) !important; }

    div.stButton > button {
        width: 100% !important;
        height: 2.3rem !important;
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        border: none !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        margin-bottom: 5px !important;
    }
    div.stButton > button:hover { background-color: #0056B3 !important; color: #FFFFFF !important; }
    
    .back-btn {
        margin-top: 10px;
    }
    
    .back-button-container button {
        all: unset !important;
        cursor: pointer !important;
        font-size: 1.8rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
    }

    .back-button-container button:hover {
        transform: scale(1.2);
        color: #007BFF !important;
    }
    div[data-testid="column"] div.stButton > button {
        border: none !important;
        background-color: transparent !important;
        font-size: 1.5rem !important;
        padding: 0 !important;
        color: #333 !important;
    }
    div[data-testid="column"] div.stButton > button:hover {
        color: #007BFF !important;
        transform: scale(1.2);
    }

    /* 알림 카드 스타일 */
    .log-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #007BFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .log-card.danger { border-left-color: #ff4b4b; }
    .log-card.caution { border-left-color: #ffa500; }
    hr { margin: 20px 0 !important; background-color: #EEEEEE !important; }
    </style>
    """, unsafe_allow_html=True)
    
# 상단 헤더
header_cols = st.columns([10, 0.6, 0.5])

with header_cols[0]:
    st.markdown("<h2 style='margin:0;'>🛡️ SMART WALL GUARD</h2>", unsafe_allow_html=True)

with header_cols[1]:
    # 팝업 버튼 생성
    notif_popover = st.popover("🔔")
    
    # 팝업 내부 구조 잡기
    with notif_popover:
        st.markdown("### 🔔 최근 긴급 알림")
        
        # [중요] 실시간 로그가 들어갈 '빈 공간'만 미리 만들어둡니다.
        live_log_container = st.empty()
        
        st.divider()
        # '상세보기' 버튼은 여기서 한 번만 만듭니다 (중복 ID 에러 해결)
        if st.button("➕ 상세보기", key="static_notif_more", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

# 설정(⚙️)은 정적인 요소이므로 루프 밖에서 한 번만 그립니다.
with header_cols[2]:
    with st.popover("⚙️"):
        st.markdown("### ⚙️ 시스템 설정")
        st.divider()
        # key 값을 주어 명확히 구분합니다.
        st.slider("AI 감지 민감도", 0, 100, 85, key="sensitivity_slider")
        st.checkbox("실시간 로그 자동 저장", value=True, key="autosave_check")
        st.checkbox("위험 감지 시 경고음", value=False, key="sound_check")
        st.selectbox("열화상 컬러맵", ["magma", "inferno", "viridis", "hot"], key="colormap_select")

st.divider()

# 페이지 전환: 전체 알림 내역
if st.session_state.page == "history":
    st.session_state.show_emergency_dialog = False
    st.empty() 
    h_col1, h_col2, h_col3 = st.columns([1, 22, 3])
    with h_col1:
        # 버튼을 컨테이너로 감싸 CSS 적용
        st.markdown('<div class="back-button-container">', unsafe_allow_html=True)
        if st.button("⬅️", key="back_to_main"):
            st.session_state.page = "main"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    with h_col2:
        # 제목의 마진을 0으로 만들어 버튼과 높이를 맞춤
        st.markdown('<h2 class="header-title">전체 알림 내역</h2>', unsafe_allow_html=True)

    with h_col3:
        if st.button("🗑️ 전체 삭제", use_container_width=True, key="history_clear_all"):
            st.session_state.log_history = []
            st.rerun()
    
    st.divider()
    
    if not st.session_state.log_history:
        st.info("기록된 로그가 없습니다.")
    else:
        # 카드 형태로 내역 출력
        for log in st.session_state.log_history:
            
            if log['위험도'] != 'DANGER':
                continue
            
            # 위험도에 따른 카드 클래스 설정
            card_status = "danger"
            if log['위험도'] == "DANGER": card_status = "danger"
            
            st.markdown(f"""
                <div class="log-card {card_status}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.2rem; font-weight: 800;">{log['이벤트']}</span>
                        <span style="color: #888; font-size: 0.85rem;">{log['시각']}</span>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.95rem; color: #444;">
                        <strong>상세 정보:</strong> {log['상세수치']} | <strong>위험수준:</strong> {log['위험도']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.stop() # 상세보기 페이지일 때는 아래 실시간 루프를 멈춤

# 메인 레이아웃
col_left, col_right = st.columns([1.8, 1], gap="large")

with col_left:
    t_col, s_col = st.columns([16, 1])
    with t_col:
        st.markdown("<span class='section-title'>📍 THERMAL GRID MONITORING</span>", unsafe_allow_html=True)
    with s_col:
        is_icon_mode = st.toggle("", value=False, key="grid_mode")
    monitor_container = st.container()
    with monitor_container:
        # 이 공간 안에 플롯과 경고창이 동시에 렌더링됨
        st.markdown('<div class="grid-monitor-box"></div>', unsafe_allow_html=True)
        plot_spot = st.empty()
        alert_spot = st.empty() # 경고창이 들어갈 자리

with col_right:
    st.markdown("<span class='section-title'>📊 현재 상태</span>", unsafe_allow_html=True)
    m1_spot, m2_spot, m3_spot = st.empty(), st.empty(), st.empty()

    st.divider()
    
    st.markdown("<span class='section-title'>🚨 긴급 대응 조치</span>", unsafe_allow_html=True)
    emergency_button("🚑 119 신고하기", "119")
    emergency_button("🚓 112 신고하기", "112")
    st.button("📍 현재 위치 정보 공유", use_container_width=True, on_click=open_modal)

# 위치 정보 공유 팝업
if st.session_state.show_emergency_dialog:
    confirm_emergency_report()
    st.stop()

st.divider()

# footer
footer_spot = st.empty()

# 시나리오 테스트
st.markdown("<p style='font-size:0.8rem; color:#EEE;'>Scenario Test</p>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    if st.button("🚨 Test: Impact", key="test_in"):
        st.session_state.demo_mode = "impact"
        st.session_state.locked_event = "impact"
        st.session_state.event_lock_until = time.time() + 3
with c2:
    if st.button("🆘 Test: Fall", key="test_fall"):
        st.session_state.demo_mode = "fall"
        st.session_state.locked_event = "fall"
        st.session_state.event_lock_until = time.time() + 3

# 모델 불러오기 및 변수 초기화
try:
    model = joblib.load('model_rf.pkl')
    status_labels = ['✅ 정상', '👤 배회 감지', '🚨 이상 충격 감지!', '🆘 낙상 사고 발생!', '🐈 동물 감지']
except:
    model = None

FALL_IMPACT_MIN = 17000
FALL_IMPACT_MAX = 22000
IMPACT_MIN = 24000

# 에러 방지를 위한 변수 초기화
smoother = CoordinateSmoother(window_size=5) # 좌표 평활화
ms_buffer = MultiScaleBuffer(short_term_size=10, long_term_size=60) # 멀티 스케일
last_logged_status = "✅ 정상"
loop_counter = 0

# 실시간 업데이트 루프
while True:
    loop_counter += 1
    now = time.time()
    
    # ---------------------------------------------------------
    # [0] 상단 실시간 알림창 (가장 먼저 렌더링)
    # ---------------------------------------------------------
    live_log_container.empty()

    with live_log_container.container():
        danger_logs = [log for log in st.session_state.log_history if log['위험도'] == "DANGER"]
        if danger_logs:
            st.caption(f"총 {len(danger_logs)}건의 위험 감지")
            for log in danger_logs[:5]: 
                st.error(f"{log['시각']} - {log['이벤트']}")
        else:
            st.write("새로운 알림이 없습니다.")
    
    # ---------------------------------------------------------
    # [1] 데이터 획득
    # ---------------------------------------------------------
    data = get_simulated_data()
    raw_data = data["pixels"]
    impact = data["impact"]
    avg_temp = raw_data.max()
    normalized_data = min_max_normalize(raw_data)

    # ---------------------------------------------------------
    # [2] AI 추론 & 상황 판단 (Logic Layer) - 여기서 모든 변수 확정
    # ---------------------------------------------------------
    prediction = 0
    confidence = 99.1
    
    # 2-1. 모델 예측
    if model:
        ms_buffer.update(impact, data["is_detected"])
        peak_impact, loitering_score = ms_buffer.get_features()
        stay_time_calc = loitering_score * 30 
        
        features = [[avg_temp, peak_impact, stay_time_calc]]
        prediction = model.predict(features)[0]

        # 잔상 제거 필터 (충격량이 낮으면 과거 버퍼 무시)
        if prediction in [2, 3] and impact < 17000:
            prediction = 0
    
    # 2-2. 시연용 강제 오버라이드 (Demo Override)
    if time.time() < st.session_state.event_lock_until:
        if st.session_state.locked_event == "impact":
            prediction = 2  # 이상 충격
            impact = 28000  # 화면 표시용 수치도 높게 고정
            confidence = 98.5
        elif st.session_state.locked_event == "fall":
            prediction = 3  # 낙상
            impact = 20000  # 화면 표시용 수치 고정
            confidence = 96.2
            
    # 타이머가 없더라도, 순간적인 충격량이 높으면 감지 (기존 로직 유지)
    elif impact > 24000:
        prediction = 2 
        confidence = 98.5
    elif 17500 < impact < 23000:
        prediction = 3 
        confidence = 96.2

    # 2-3. 최종 상태 라벨 및 위험도(Color) 결정
    status = status_labels[prediction]
    if prediction != 0 and confidence == 99.1: # 데모 모드가 아닐 때 랜덤 confidence
        confidence = 92.4 + np.random.uniform(-1, 5)

    # 위험 수준(status_delta) 및 UI 색상(d_color) 결정
    if prediction in [2, 3]:   # 🚨 DANGER (충격, 낙상)
        status_delta = "DANGER"
        d_color = "inverse"
    elif prediction in [1, 4]: # ⚠️ CAUTION (배회, 동물)
        status_delta = "CAUTION"
        d_color = "normal"
    else:                      # ✅ SAFE
        status_delta = "SAFE"
        d_color = "normal"

    # ---------------------------------------------------------
    # [3] 시각화 및 알림 (View Layer)
    # ---------------------------------------------------------
    
    # 3-1. 좌측 열화상 모니터링 플롯
    fig, ax = plt.subplots(figsize=(8, 6.5)) 
    fig.patch.set_facecolor('#000000') 
    
    if not is_icon_mode:
        processed = zoom(normalized_data, 8, order=3)
        ax.imshow(processed, cmap='magma', aspect='auto', vmin=0, vmax=1)
        ax.axis('off')
    else:
        ax.set_facecolor('#111111') 
        for x in range(9):
            ax.axhline(x-0.5, color='white', lw=0.5, alpha=0.1)
            ax.axvline(x-0.5, color='white', lw=0.5, alpha=0.1)
        
        if data["is_detected"]:
            display_char, main_color, label_text = "?", "#FFFFFF", "감지 중"
            if prediction in [1, 2, 3]: 
                display_char, main_color, label_text = "P", "#00F2FF", "PERSON"
            elif prediction == 4: 
                display_char, main_color, label_text = "A", "#FFAB40", "ANIMAL"

            raw_r, raw_c = get_heat_center(raw_data) 
            smooth_r, smooth_c = smoother.update(raw_r, raw_c) 
            
            ax.scatter(smooth_c, smooth_r, s=8000, c=main_color, alpha=0.1, marker='o')
            ax.scatter(smooth_c, smooth_r, s=4000, c=main_color, alpha=0.3, marker='o')
            ax.scatter(smooth_c, smooth_r, s=1200, c=main_color, marker='o', edgecolors='white', linewidth=3)
            ax.text(smooth_c, smooth_r, display_char, color='white', fontsize=28, ha='center', va='center', fontweight='black')
            ax.text(smooth_c, smooth_r + 1.2, f"[{label_text}]", color=main_color, fontsize=10, ha='center', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor=main_color, boxstyle='round,pad=0.3'))
        ax.set_xlim(-0.5, 7.5); ax.set_ylim(7.5, -0.5); ax.axis('off')

    plt.subplots_adjust(0, 0, 1, 1)
    plot_spot.pyplot(fig)
    plt.close(fig)

    # 3-2. 긴급 상황 팝업 (Overlay)
    if status_delta == "DANGER":
        alert_msg = f"T: {avg_temp:.1f}°C / Impact: {int(impact)}"
        alert_spot.markdown(get_alert_overlay(status, alert_msg), unsafe_allow_html=True)
        st.session_state.emergency_triggered = True
    else:
        alert_spot.empty()
        st.session_state.emergency_triggered = False

    # ---------------------------------------------------------
    # [4] 데이터 저장 (Data Layer)
    # ---------------------------------------------------------
    
    # 상태가 변했고, 정상이 아니라면 로그 저장
    if status != "✅ 정상" and status != last_logged_status:
        # 이미 [2] 단계에서 확정된 status_delta를 사용하므로 로직이 깔끔함
        risk_level = status_delta # DANGER or CAUTION
        
        st.session_state.log_history.insert(0, {
            "시각": datetime.now().strftime("%H:%M:%S"),
            "이벤트": status,
            "위험도": risk_level,
            "상세수치": f"T: {avg_temp:.1f}°C / I: {int(impact)}"
        })
        if len(st.session_state.log_history) > 50: 
            st.session_state.log_history.pop()
    
    last_logged_status = status

    # ---------------------------------------------------------
    # [5] 우측 메트릭 업데이트
    # ---------------------------------------------------------
    m1_spot.metric(label="활성 센서", value="02 / 02 Units", delta="Thermal & Vibration Sync")
    m2_spot.metric(label="감지된 이벤트", value=f"{len(st.session_state.log_history)} 건", delta=f"최근: {data['time']}")
    m3_spot.metric(label="현재 상황 (AI 분석)", value=status, delta=f"신뢰도 {confidence:.1f}%", delta_color=d_color)

    footer_spot.markdown(f"<p style='color:#AAA; font-size:0.8rem; text-align:center;'>System Node: MAPO-A1 | Protocol: MQTT-JSON | Last Sync: {data['time']}</p>", unsafe_allow_html=True)
    time.sleep(0.4)