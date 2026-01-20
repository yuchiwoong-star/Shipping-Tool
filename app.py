import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 1. 차량 제원 설정
TRUCK_SPECS = {
    "11톤": {"w": 2350, "l": 9000, "h": 2300, "cap": 13000},
    "5톤": {"w": 2350, "l": 6200, "h": 2100, "cap": 7000}
}
MAX_STACK_H = 1300  
MAX_STACK_COUNT = 4 

def add_box_3d(fig, x0, y0, z0, l, w, h, name, color):
    # 박스 본체 (완전 불투명)
    fig.add_trace(go.Mesh3d(
        x=[x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0],
        y=[y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w],
        z=[z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=1.0, color=color, name=f"Box {name}",
        hoverinfo="text",
        text=f"📦 번호: {name}<br>📏 규격: {int(l)}x{int(w)}x{int(h)}",
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
        line=dict(color='black', width=4), showlegend=False, hoverinfo='skip'
    ))

    # [수정] 노란색 스티커 형태의 번호 표시 (측면 양 끝단 부착)
    # 스티커 배경 (노란색 사각형)과 검은색 텍스트 결합
    fig.add_trace(go.Scatter3d(
        x=[x0 + 5, x0 + l - 5], # 양 끝단 면에 바짝 붙임
        y=[y0 + w/2, y0 + w/2],
        z=[z0 + h/2, z0 + h/2],
        mode='text+markers',
        text=[name, name],
        marker=dict(symbol='square', size=24, color='yellow', line=dict(color='black', width=2)),
        textfont=dict(size=13, color="black", family="Arial Black"),
        showlegend=False, hoverinfo='skip'
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
    
    # 상위 10% 길이 계산
    all_lengths = sorted([b['l'] for b in clean_boxes], reverse=True)
    threshold_idx = max(0, int(len(all_lengths) * 0.1) - 1)
    length_threshold = all_lengths[threshold_idx] if all_lengths else 0
    
    pending = sorted(clean_boxes, key=lambda x: x['l'], reverse=True)
    results = []
    for idx, t_name in enumerate(fleet):
        spec = TRUCK_SPECS[t_name]
        truck_res = {"name": t_name, "boxes": [], "weight": 0, "id": f"truck_{idx}"}
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
                        # 상위 10% 빨간색, 나머지 주황색
                        b['color'] = '#d62728' if b['l'] >= length_threshold else '#ffbb78'
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
    if st.sidebar.button("최적 적재 실행"):
        packed_trucks, remaining = calculate_packing(df, ["11톤", "5톤", "5톤"])
        for truck in packed_trucks:
            st.subheader(f"🚚 {truck['name']} ({truck['weight']:.1f}kg 적재)")
            fig = go.Figure()
            spec = TRUCK_SPECS[truck['name']]
            
            # 가이드 라인
            fig.add_trace(go.Scatter3d(
                x=[0, spec['l'], spec['l'], 0, 0, 0, spec['l'], spec['l'], 0, 0, spec['l'], spec['l']],
                y=[0, 0, spec['w'], spec['w'], 0, 0, 0, spec['w'], spec['w'], 0, 0, spec['w']],
                z=[0, 0, 0, 0, 0, spec['h'], spec['h'], spec['h'], spec['h'], spec['h'], 0, spec['h']],
                mode='lines', line=dict(color='gray', width=1), showlegend=False, hoverinfo='skip'
            ))
            
            for b in truck['boxes']:
                add_box_3d(fig, b['pos'][0], b['pos'][1], b['pos'][2], b['l'], b['w'], b['h'], b['id'], b['color'])
            
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title='길이 (L)', range=[0, 9000]),
                    yaxis=dict(title='폭 (W)', range=[0, 2350]),
                    zaxis=dict(title='높이 (H)', range=[0, 2300]),
                    aspectmode='manual',
                    aspectratio=dict(x=3, y=1, z=1),
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))
                ),
                margin=dict(l=0, r=0, b=0, t=50), height=800
            )
            st.plotly_chart(fig, key=f"chart_{truck['id']}")
