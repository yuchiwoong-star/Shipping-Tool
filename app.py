import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item
import random

# ==========================================
# 1. 기본 설정 및 데이터 정의
# ==========================================

st.set_page_config(layout="wide", page_title="물류 적재 최적화 도구")

# 차량 제원 (너비, 높이, 길이, 최대무게)
# 높이는 이미지에 없어서 윙바디 표준인 2350mm로 가정
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}

# 시각화용 색상 팔레트
COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C', '#FF9F1C', '#2B2D42', '#EF233C', '#D90429']

# ==========================================
# 2. 핵심 로직 함수 (계산 & 시각화)
# ==========================================

def get_optimized_trucks(items_df):
    """
    짐 목록을 받아서 최적의 차량 조합을 찾아내는 함수
    """
    # 1. 엑셀 데이터를 py3dbp Item 객체로 변환
    all_items = []
    for _, row in items_df.iterrows():
        for i in range(int(row['수량'])):
            # Item 생성 (이름, 가로, 높이, 세로, 무게) -> py3dbp는 W, H, D 순서
            item = Item(f"{row['박스명']}-{i}", row['가로'], row['높이'], row['세로'], row['무게'])
            all_items.append(item)

    remaining_items = all_items[:] # 복사본 생성
    used_trucks = [] # 결과로 배차된 트럭 리스트

    # 트럭 종류를 무게 기준 오름차순(작은차 -> 큰차) 정렬
    sorted_truck_keys = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    # 2. 모든 짐이 처리될 때까지 반복
    while remaining_items:
        best_bin = None
        best_packed_items = []
        max_packed_count = -1
        
        # 시도: 작은 트럭부터 큰 트럭 순으로 '현재 남은 짐'을 넣어봄
        # 전략: 남은 짐을 몽땅 실을 수 있는 가장 작은 트럭을 찾음.
        #       만약 제일 큰 차에도 다 안 들어가면, 제일 큰 차를 꽉 채워서 보냄.
        
        found_perfect_fit = False

        for t_name in sorted_truck_keys:
            spec = TRUCK_DB[t_name]
            packer = Packer()
            # Bin 생성 (이름, 가로, 높이, 세로, 최대무게)
            packer.add_bin(Bin(t_name, spec['w'], spec['h'], spec['l'], spec['weight']))
            
            for item in remaining_items:
                packer.add_item(item)
            
            packer.pack(bigger_first=True, distribute_items=False, number_of_decimals=0)
            
            # 테스트한 트럭 가져오기
            temp_bin = packer.bins[0]
            
            # 만약 남은 짐이 이 트럭에 100% 다 들어갔다면? -> 이 트럭이 최적 (비용 절감)
            if len(temp_bin.items) == len(remaining_items):
                best_bin = temp_bin
                best_packed_items = temp_bin.items
                found_perfect_fit = True
                break # 더 큰 트럭 볼 필요 없음
            
            # 다 안 들어갔다면? -> 가장 많이 실린 트럭을 일단 기록해둠 (보통 제일 큰 차가 됨)
            if len(temp_bin.items) > max_packed_count:
                max_packed_count = len(temp_bin.items)
                best_bin = temp_bin
                best_packed_items = temp_bin.items

        # 결과 확정
        if best_bin and len(best_packed_items) > 0:
            # 트럭 이름을 유니크하게 변경 (예: 11톤 -> 11톤 No.1)
            best_bin.name = f"{best_bin.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_bin)
            
            # 적재된 짐은 남은 목록에서 제거
            packed_names = [item.name for item in best_packed_items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            # 짐이 너무 커서 아무 차에도 안 들어가는 경우 등
            return used_trucks, "ERROR"

    return used_trucks, "SUCCESS"

def create_3d_figure(bin_obj):
    """
    적재된 Bin 객체를 받아 Plotly 3D 그래프를 반환하는 함수
    """
    fig = go.Figure()
    W, H, D = bin_obj.width, bin_obj.height, bin_obj.depth

    # 1. 트럭 프레임 (외곽선)
    lines_x = [0, W, W, 0, 0, 0, W, W, 0, 0, W, W, 0, 0, W, W]
    lines_y = [0, 0, D, D, 0, 0, 0, D, D, 0, 0, 0, D, D, D, D]
    lines_z = [0, 0, 0, 0, 0, H, H, H, H, H, H, 0, 0, H, H, 0]
    
    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines', line=dict(color='black', width=3), hoverinfo='none', name='적재함'
    ))

    # 2. 박스 그리기
    for i, item in enumerate(bin_obj.items):
        # py3dbp 위치: [x, y, z] -> Plotly: x=Width, y=Depth(Length), z=Height
        # 주의: py3dbp의 dimensions는 w, h, d 순서
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        
        # 큐브(박스) 생성
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+d, y+d, y, y, y+d, y+d],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=COLORS[i % len(COLORS)], opacity=1, flatshading=True,
            name=item.name.split('-')[0], hovertext=f"{item.name}<br>{w}x{d}x{h}"
        ))
        
        # 박스 테두리 (가독성을 위해)
        box_wire_x = [x, x+w, x+w, x, x, x, x+w, x+w, x, x, x+w, x+w, x+w, x+w, x, x]
        box_wire_y = [y, y, y+d, y+d, y, y, y, y+d, y+d, y, y, y, y+d, y+d, y+d, y+d]
        box_wire_z = [z, z, z, z, z, z+h, z+h, z+h, z+h, z+h, z+h, z, z, z+h, z+h, z]
        
        fig.add_trace(go.Scatter3d(
            x=box_wire_x, y=box_wire_y, z=box_wire_z,
            mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='none'
        ))

    # 레이아웃 설정
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='폭 (Width)', range=[-100, 2500], showbackground=False),
            yaxis=dict(title='길이 (Length)', range=[-100, 11000], showbackground=False),
            zaxis=dict(title='높이 (Height)', range=[-100, 2500], showbackground=False),
            aspectratio=dict(x=1, y=3, z=1) # 트럭 모양 비율 조정
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    return fig

# ==========================================
# 3. 메인 화면 구성
# ==========================================

st.title("🚛 자동 차량 배차 및 적재 시뮬레이터")
st.markdown("""
파일을 업로드하면 **최적의 차량 조합(예: 11톤 1대 + 5톤 1대)**을 자동으로 계산하고,
각 차량의 적재 모습을 3D로 시각화합니다.
""")

col1, col2 = st.columns([1, 3])

# [왼쪽] 파일 업로드 사이드바
with col1:
    st.header("1. 데이터 입력")
    uploaded_file = st.file_uploader("박스 리스트 엑셀 파일 (.xlsx)", type=['xlsx'])
    
    st.info("💡 엑셀 필수 컬럼: 박스명, 가로, 세로, 높이, 무게, 수량")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("입력 데이터 확인:", df.head(3))
        
        if st.button("최적 적재 실행", type="primary"):
            st.session_state['run_check'] = True
            st.session_state['data'] = df

# [오른쪽] 결과 화면
with col2:
    if st.session_state.get('run_check'):
        st.header("2. 분석 결과")
        
        with st.spinner("최적의 차량 조합을 계산 중입니다..."):
            trucks, status = get_optimized_trucks(st.session_state['data'])
            
        if status == "ERROR":
            st.error("오류 발생: 적재할 수 없는 크기의 화물이 포함되어 있거나 계산에 실패했습니다.")
        elif not trucks:
            st.warning("적재할 화물이 없습니다.")
        else:
            # 1. 결과 요약 출력
            st.success(f"✅ 총 **{len(trucks)}대**의 차량이 필요합니다.")
            
            # 차량 조합 텍스트 생성
            truck_names = [t.name.split(' ')[0] for t in trucks]
            from collections import Counter
            summary = Counter(truck_names)
            summary_text = ", ".join([f"**{k} {v}대**" for k, v in summary.items()])
            st.markdown(f"### 📋 추천 배차: {summary_text}")
            
            st.divider()

            # 2. 탭을 생성하여 각 차량별 결과 표시
            tabs = st.tabs([t.name for t in trucks])
            
            for i, tab in enumerate(tabs):
                current_truck = trucks[i]
                with tab:
                    # 정보 표시
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        st.markdown(f"**{current_truck.name}**")
                        st.write(f"- 적재 박스: {len(current_truck.items)}개")
                        st.write(f"- 적재 중량: {current_truck.get_total_weight():,} kg")
                        # 여유 공간 등의 정보 추가 가능
                        
                    with c2:
                        # 3D 그래프 그리기
                        fig = create_3d_figure(current_truck)
                        st.plotly_chart(fig, use_container_width=True)
