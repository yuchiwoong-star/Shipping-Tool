import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

# 1. 차량 제원 및 제약 조건 (트럭 외형 포함)
TRUCK_SPECS = {
    "11톤": {"w": 2350, "l": 9000, "h": 2300, "cap": 13000, "cab_l": 2000, "wheel_r": 500},
    "5톤": {"w": 2350, "l": 6200, "h": 2100, "cap": 7000, "cab_l": 1800, "wheel_r": 450}
}
MAX_STACK_H = 1300  
MAX_STACK_COUNT = 4 

# 색상 정의
COLOR_LONG_BOX = '#d62728' # 상위 10% 긴 박스
COLOR_NORMAL_BOX = '#ffbb78' # 일반 박스 (연주황)
COLOR_TRUCK_FRAME = 'rgba(128, 128, 128, 0.2)' # 트럭 외형 (반투명 회색)
COLOR_TRUCK_TIRE = '#333333' # 트럭 타이어

def add_box_3d(fig, x0, y0, z0, l, w, h, name, color):
    # 박스 본체 (불투명도 0.8)
    fig.add_trace(go.Mesh3d(
        x=[x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0],
        y=[y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w],
        z=[z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.8, color=color, name=f"Box {name}",
        hoverinfo="text",
        text=f"📦 박스번호: {name}<br>📏 규격: {int(l)}x{int(w)}x{int(h)}<br>📍 위치(Z): {int(z0)}mm",
        showlegend=False
    ))
    
    # 박스 외곽선 (검은색 테두리)
    lines_x, lines_y, lines_z = [], [], []
    for s in [[0,1,2,3,0], [4,5,6,7,4], [0,4], [1,5], [2,6], [3,7]]:
        for i in s:
            lines_x.append([x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0][i])
            lines_y.append([y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w][i])
            lines_z.append([z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h][i])
        lines_x.append(None); lines_y.append(None); lines_z.append(None)

    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z, mode='lines',
        line=dict(color='black', width=3), showlegend=False, hoverinfo='skip'
    ))

    # 박스 번호 (중앙 상단에만 표시)
    fig.add_trace(go.Scatter3d(
        x=[x0 + l/2], y=[y0 + w/2], z=[z0 + h + 10], # 박스 상단에서 10mm 위
        mode='text', text=[name],
        textfont=dict(size=14, color="black", family="Arial Black"),
        showlegend=False, hoverinfo='skip'
    ))

def draw_truck_frame(fig, spec):
    # 적재함 부분 (반투명 박스)
    truck_l, truck_w, truck_h = spec['l'], spec['w'], spec['h']
    
    fig.add_trace(go.Mesh3d(
        x=[0, truck_l, truck_l, 0, 0, truck_l, truck_l, 0],
        y=[0, 0, truck_w, truck_w, 0, 0, truck_w, truck_w],
        z=[0, 0, 0, 0, truck_h, truck_h, truck_h, truck_h],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.2, color=COLOR_TRUCK_FRAME, showlegend=False, hoverinfo='skip'
    ))

    # 트럭 외곽선 (회색)
    lines_x, lines_y, lines_z = [], [], []
    for s in [[0,1,2,3,0], [4,5,6,7,4], [0,4], [1,5], [2,6], [3,7]]:
        for i in s:
            lines_x.append([0, truck_l, truck_l, 0, 0, truck_l, truck_l, 0][i])
            lines_y.append([0, 0, truck_w, truck_w, 0, 0, truck_w, truck_w][i])
            lines_z.append([0, 0, 0, 0, truck_h, truck_h, truck_h, truck_h][i])
        lines_x.append(None); lines_y.append(None); lines_z.append(None)

    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z, mode='lines',
        line=dict(color='gray', width=3), showlegend=False, hoverinfo='skip'
    ))
    
    # [새로운 추가] 트럭 헤드 (단순화된 사각형) - 트럭 길이 시작 지점(L=0)에서 앞으로
    cab_l_offset = spec['cab_l']
    fig.add_trace(go.Mesh3d(
        x=[-cab_l_offset, 0, 0, -cab_l_offset, -cab_l_offset, 0, 0, -cab_l_offset],
        y=[0, 0, truck_w, truck_w, 0, 0, truck_w, truck_w],
        z=[0, 0, 0, 0, truck_h * 0.7, truck_h * 0.7, truck_h * 0.7, truck_h * 0.7],
        opacity=0.8, color='darkgray', showlegend=False, hoverinfo='skip'
    ))

    # [새로운 추가] 바퀴 (원통형 대신 간단한 박스)
    wheel_r = spec['wheel_r']
    wheel_w = spec['w'] * 0.1 # 바퀴 폭
    
    # 뒷바퀴 (두 개)
    fig.add_trace(go.Mesh3d(
        x=[truck_l - wheel_r*1.5, truck_l - wheel_r*0.5, truck_l - wheel_r*0.5, truck_l - wheel_r*1.5,
           truck_l - wheel_r*1.5, truck_l - wheel_r*0.5, truck_l - wheel_r*0.5, truck_l - wheel_r*1.5],
        y=[-wheel_w, 0, 0, -wheel_w, -wheel_w, 0, 0, -wheel_w],
        z=[wheel_r, wheel_r, wheel_r, wheel_r, 0, 0, 0, 0],
        opacity=1.0, color=COLOR_TRUCK_TIRE, showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Mesh3d(
        x=[truck_l - wheel_r*1.5, truck_l - wheel_r*0.5, truck_l - wheel_r*0.5, truck_l - wheel_r*1.5,
           truck_l - wheel_r*1.5, truck_l - wheel_r*0.5, truck_l - wheel_r*0.5, truck_l - wheel_r*1.5],
        y=[truck_w, truck_w + wheel_w, truck_w + wheel_w, truck_w, truck_w, truck_w + wheel_w, truck_w + wheel_w, truck_w],
        z=[wheel_r, wheel_r, wheel_r, wheel_r, 0, 0, 0, 0],
        opacity=1.0, color=COLOR_TRUCK_TIRE, showlegend=False, hoverinfo='skip'
    ))

    # 앞바퀴 (두 개)
    fig.add_trace(go.Mesh3d(
        x=[-wheel_r*1.5, -wheel_r*0.5, -wheel_r*0.5, -wheel_r*1.5,
           -wheel_r*1.5, -wheel_r*0.5, -wheel_r*0.5, -wheel_r*1.5],
        y=[-wheel_w, 0, 0, -wheel_w, -wheel_w, 0, 0, -wheel_w],
        z=[wheel_r, wheel_r, wheel_r, wheel_r, 0, 0, 0, 0],
        opacity=1.0, color=COLOR_TRUCK_TIRE, showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Mesh3d(
        x=[-wheel_r*1.5, -wheel_r*0.5, -wheel_r*0.5, -wheel_r*1.5,
           -wheel_r*1.5, -wheel_r*0.5, -wheel_r*0.5, -wheel_r*1.5],
        y=[truck_w, truck_w + wheel_w, truck_w + wheel_w, truck_w, truck_w, truck_w + wheel_w, truck_w + wheel_w, truck_w],
        z=[wheel_r, wheel_r, wheel_r, wheel_r, 0, 0, 0, 0],
        opacity=1.0, color=COLOR_TRUCK_TIRE, showlegend=False, hoverinfo='skip'
    ))


def calculate_packing(box_df, fleet):
    cols = [str(c).lower().strip() for c in box_df.columns]
    def find_col(keys, default_idx):
        for i, c in enumerate(cols):
            if any(k in c for k in keys): return box_df.columns[i]
        return box_df.columns[default_idx] if len(box_df.columns) > default_idx else box_df.columns[0]

    t_l, t_w, t_h = find_col(['길이', 'l'], 3), find_col(['폭', 'w'], 1), find_col(['높이', 'h'], 2)
    t_weight, t_id = find_col(['중량', '무게', 'weight'], 4), find_col(['번호', 'id', '박스'], 0)

    clean_boxes = []
    for _, r in box_df.iterrows():
        try:
            clean_boxes.append({
                'id': str(r[t_id]), 'w': float(r[t_w]), 'h': float(r[t_h]), 
                'l': float(r[t_l]), 'weight': float(r[t_weight])
            })
        except: continue
    
    all_lengths = sorted([b['l'] for b in clean_boxes], reverse=True)
    threshold_idx = max(0, int(len(all_lengths) * 0.1) - 1)
    len_threshold = all_lengths[threshold_idx] if all_lengths else 0
    
    pending = sorted(clean_boxes, key=lambda x: x['l'], reverse=True)
    results = []
    for idx, t_name in enumerate(fleet):
        spec = TRUCK_SPECS[t_name]
        truck_res = {"name": t_name, "boxes": [], "weight": 0, "id": f"truck_{idx}", "spec": spec}
        rem_w = spec['w']
        while pending and rem_w > 0:
            lane_w, curr_y = 0, 0
            while pending and curr_y < spec['l']:
                stack_h, stack_count, temp_stack = 0, 0, []
                while pending and stack_count < MAX_STACK_COUNT:
                    b = pending[0]
                    if b['w'] <= rem_w and curr_y + b['l'] <= spec['l'] and \
                       stack_h + b['h'] <= MAX_STACK_H and \
                       truck_res['weight'] + b['weight'] <= spec['cap']:
                        b['pos'] = [curr_y, spec['w'] - rem_w, stack_h]
                        b['color'] = COLOR_LONG_BOX if b['l'] >= len_threshold else COLOR_NORMAL_BOX
                        temp_stack.append(b); truck_res['weight'] += b['weight']
                        stack_h += b['h']; stack_count += 1; lane_w = max(lane_w, b['w'])
                        pending.pop(0)
                    else: break
                if temp_stack:
                    truck_res['boxes'].extend(temp_stack)
                    curr_y += max([bx['l'] for bx in temp_stack])
                else: break
            if lane_w > 0: rem_w -= lane_w
            else: break
        results.append(truck_res)
    return results, pending

st.set_page_config(layout="wide")
st.title("📦 3D 차량 적재 최적화 시스템")
uploaded_file = st.sidebar.file_uploader("박스 정보 엑셀 업로드 (xlsx)", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # 차량 선택 드롭다운 추가
    selected_truck_type = st.sidebar.selectbox("차량 종류 선택", list(TRUCK_SPECS.keys()))

    if st.sidebar.button("최적 적재 실행"):
        packed_trucks, remaining = calculate_packing(df, [selected_truck_type]) # 선택된 차량만 계산
        for truck in packed_trucks:
            st.subheader(f"🚚 {truck['name']} ({truck['weight']:.1f}kg 적재) - 최적 적재 레이아웃")
            fig = go.Figure()
            s = truck['spec']
            
            # 차량 외형 그리기
            draw_truck_frame(fig, s)
            
            # 박스 적재
            for b in truck['boxes']:
                add_box_3d(fig, b['pos'][0], b['pos'][1], b['pos'][2], b['l'], b['w'], b['h'], b['id'], b['color'])
            
            # 3D 뷰 레이아웃 설정
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='길이 (L)', range=[-s['cab_l'], s['l']], showgrid=True), # 헤드 길이까지 포함
                    yaxis=dict(title='폭 (W)', range=[min(-s['w']*0.1, 0), max(s['w']*1.1, s['w'])], showgrid=True), # 바퀴 공간 포함
                    zaxis=dict(title='높이 (H)', range=[0, max(s['h']*1.2, s['wheel_r']*2)], showgrid=True), # 바퀴 높이 포함
                    aspectmode='manual',
                    aspectratio=dict(x=(s['l']+s['cab_l'])/2500, y=s['w']/2500, z=s['h']/2500), # 실제 비율에 가깝게
                    camera=dict(eye=dict(x=1.8, y=1.8, z=0.8)) # 고정된 시점
                ),
                margin=dict(l=0, r=0, b=0, t=50), height=700,
                hoverlabel=dict(bgcolor="white", font_size=16, font_family="Malgun Gothic")
            )
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{truck['id']}")

        if remaining:
            st.subheader("⚠️ 적재되지 못한 박스")
            remaining_df = pd.DataFrame(remaining)
            st.dataframe(remaining_df, use_container_width=True)
