import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# [cite_start]1. 차량 및 제약 조건 설정 [cite: 1]
TRUCK_SPECS = {
    "11톤": {"w": 2350, "l": 9000, "h": 2300, "cap": 13000},
    "5톤": {"w": 2350, "l": 6200, "h": 2100, "cap": 7000}
}
MAX_STACK_H = 1300  # 사용자 요청: 최대 적재 높이 1.3m
MAX_STACK_COUNT = 4 # 사용자 요청: 최대 4단 적재

# 3D 박스 그리기 함수
def add_box_3d(fig, x0, y0, z0, l, w, h, name, color):
    fig.add_trace(go.Mesh3d(
        x=[x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0],
        y=[y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w],
        z=[z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6, color=color, name=f"Box {name}"
    ))

# 적재 알고리즘 (Simplified Lane Packing)
def calculate_packing(box_df, fleet):
    pending = box_df.to_dict('records')
    # [cite_start]길이순 정렬 [cite: 2]
    pending = sorted(pending, key=lambda x: x['l'], reverse=True)
    results = []

    for t_name in fleet:
        spec = TRUCK_SPECS[t_name]
        truck_res = {"name": t_name, "boxes": [], "weight": 0}
        curr_x, curr_y = 0, 0
        rem_w = spec['w']
        
        while pending and rem_w > 0:
            lane_w = 0
            curr_y = 0
            while pending and curr_y < spec['l']:
                # 한 지점에 쌓기 (Stacking)
                stack_h = 0
                stack_count = 0
                while pending and stack_count < MAX_STACK_COUNT:
                    b = pending[0]
                    if b['w'] <= rem_w and curr_y + b['l'] <= spec['l'] and \
                       stack_h + b['h'] <= MAX_STACK_H and \
                       truck_res['weight'] + b['weight'] <= spec['cap']:
                        
                        b['pos'] = [curr_y, spec['w'] - rem_w, stack_h]
                        truck_res['boxes'].append(b)
                        truck_res['weight'] += b['weight']
                        stack_h += b['h']
                        stack_count += 1
                        lane_w = max(lane_w, b['w'])
                        pending.pop(0)
                    else: break
                
                if stack_count > 0:
                    curr_y += truck_res['boxes'][-1]['l']
                else: break
            
            if lane_w > 0:
                rem_w -= lane_w
            else: break
        results.append(truck_res)
    return results, pending

# --- 웹 화면 구성 ---
st.set_page_config(layout="wide")
st.title("📦 3D 차량 적재 최적화 시스템")

# [cite_start]파일 업로드 (xaic.docx 기반 데이터 입력 가정) [cite: 2]
uploaded_file = st.file_sidebar.file_uploader("박스 정보 엑셀 업로드", type=['xlsx', 'csv'])

# 샘플 데이터 생성 (파일 없을 시)
if not uploaded_file:
    st.info("파일을 업로드하면 실제 데이터를 계산합니다. 현재는 샘플 데이터로 시뮬레이션 중입니다.")
    # [cite_start]제공해주신 박스 정보 예시 [cite: 2]
    sample_data = {
        'id': ['01', '13', '07', '48'],
        'w': [350, 500, 340, 570],
        'h': [230, 370, 250, 530],
        'l': [7700, 8700, 6700, 7300],
        'weight': [227, 956, 259, 465]
    }
    df = pd.DataFrame(sample_data)
else:
    df = pd.read_excel(uploaded_file) # 실제 운영 시 전처리 필요

fleet = ["11톤", "5톤", "5톤"] # 사용자 요청 조합
if st.sidebar.button("최적 적재 실행"):
    packed_trucks, remaining = calculate_packing(df, fleet)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        for i, truck in enumerate(packed_trucks):
            st.subheader(f"{i+1}호차: {truck['name']} (적재량: {truck['weight']}kg)")
            fig = go.Figure()
            # 차량 바닥 그리기
            add_box_3d(fig, 0, 0, 0, TRUCK_SPECS[truck['name']]['l'], TRUCK_SPECS[truck['name']]['w'], 10, "Floor", "gray")
            
            for b in truck['boxes']:
                add_box_3d(fig, b['pos'][0], b['pos'][1], b['pos'][2], b['l'], b['w'], b['h'], b['id'], np.random.choice(['blue', 'green', 'orange', 'red']))
            
            fig.update_layout(scene=dict(aspectmode='data'))
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.subheader("⚠️ 미적재 박스")
        st.write(f"총 {len(remaining)}개 박스가 실리지 못했습니다.")
        st.write(pd.DataFrame(remaining))
