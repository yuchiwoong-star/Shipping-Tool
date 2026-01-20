import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 1. 차량 제원 설정 (Source 1 데이터 반영)
TRUCK_SPECS = {
    "11톤": {"w": 2350, "l": 9000, "h": 2300, "cap": 13000},
    "5톤": {"w": 2350, "l": 6200, "h": 2100, "cap": 7000}
}
MAX_STACK_H = 1300  # 사용자 요청 높이 제한 1.3m
MAX_STACK_COUNT = 4 # 최대 4단 적재

def add_box_3d(fig, x0, y0, z0, l, w, h, name, color):
    fig.add_trace(go.Mesh3d(
        x=[x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0],
        y=[y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w],
        z=[z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.6, color=color, name=f"Box {name}",
        showlegend=False
    ))

def calculate_packing(box_df, fleet):
    # 엑셀 열 이름을 유연하게 인식하도록 보강
    cols = [str(c).lower().strip() for c in box_df.columns]
    def find_col(keys, default_idx):
        for i, c in enumerate(cols):
            if any(k in c for k in keys): return box_df.columns[i]
        return box_df.columns[default_idx] if len(box_df.columns) > default_idx else box_df.columns[0]

    t_l = find_col(['l', '길이', 'length'], 3)
    t_w = find_col(['w', '폭', 'width'], 1)
    t_h = find_col(['h', '높이', 'height'], 2)
    t_weight = find_col(['weight', '중량', '무게'], 2)
    t_id = find_col(['id', '번호', '박스'], 0)

    clean_boxes = []
    for _, r in box_df.iterrows():
        try:
            clean_boxes.append({
                'id': str(r[t_id]), 
                'w': float(r[t_w]), 'h': float(r[t_h]), 
                'l': float(r[t_l]), 'weight': float(r[t_weight])
            })
        except: continue
    
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

# --- 웹 페이지 화면 구현 ---
st.set_page_config(page_title="3D 적재 시뮬레이터", layout="wide")
st.title("📦 3D 차량 적재 최적화 시스템")

# 파일 업로드 (xlsx 전용)
uploaded_file = st.sidebar.file_uploader("박스 정보 엑셀 업로드 (xlsx)", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    fleet = ["11톤", "5톤", "5톤"] # 사용자 요청 배차 조합
    
    if st.sidebar.button("최적 적재 실행"):
        packed_trucks, remaining = calculate_packing(df, fleet)
        
        for truck in packed_trucks:
            st.subheader(f"🚚 {truck['name']} ({truck['weight']:.1f}kg 적재)")
            spec = TRUCK_SPECS[truck['name']]
            fig = go.Figure()
            # 차량 바닥판 생성
            add_box_3d(fig, 0, 0, 0, spec['l'], spec['w'], 20, "Floor", "lightgray")
            # 적재된 박스 생성
            for b in truck['boxes']:
                add_box_3d(fig, b['pos'][0], b['pos'][1], b['pos'][2], b['l'], b['w'], b['h'], b['id'], "royalblue")
            
            fig.update_layout(
                scene=dict(xaxis_title='길이(L)', yaxis_title='폭(W)', zaxis_title='높이(H)', aspectmode='data'),
                margin=dict(l=0, r=0, b=0, t=40), height=500
            )
            # 중복 오류 방지를 위한 고유 key 값 사용
            st.plotly_chart(fig, key=f"chart_{truck['id']}")

        if remaining:
            st.warning(f"⚠️ 미적재 박스: {len(remaining)}개")
            st.dataframe(pd.DataFrame(remaining)[['id','l','w','h','weight']])
