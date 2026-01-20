import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# 1. 차량 및 제약 조건 설정
TRUCK_SPECS = {
    "11톤": {"w": 2350, "l": 9000, "h": 2300, "cap": 13000},
    "5톤": {"w": 2350, "l": 6200, "h": 2100, "cap": 7000}
}
MAX_STACK_H = 1300  
MAX_STACK_COUNT = 4 

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

def calculate_packing(box_df, fleet):
    # 열 이름 자동 매칭 로직 보강 (IndexError 방지)
    cols = [str(c).lower() for c in box_df.columns]
    
    def get_col(targets, default_idx):
        for i, col in enumerate(cols):
            if any(t in col for t in targets):
                return box_df.columns[i]
        return box_df.columns[default_idx] if len(box_df.columns) > default_idx else box_df.columns[0]

    target_l = get_col(['l', '길이', 'length'], 3)
    target_w = get_col(['w', '폭', 'width'], 1)
    target_h = get_col(['h', '높이', 'height'], 2)
    target_weight = get_col(['weight', '중량', '무게', 'gross'], 2)
    target_id = get_col(['id', '박스', '번호', 'no'], 0)

    # 안전하게 데이터 변환
    new_df = pd.DataFrame()
    new_df['l'] = pd.to_numeric(box_df[target_l], errors='coerce').fillna(0)
    new_df['w'] = pd.to_numeric(box_df[target_w], errors='coerce').fillna(0)
    new_df['h'] = pd.to_numeric(box_df[target_h], errors='coerce').fillna(0)
    new_df['weight'] = pd.to_numeric(box_df[target_weight], errors='coerce').fillna(0)
    new_df['id'] = box_df[target_id].astype(str)

    pending = new_df.to_dict('records')
    pending = sorted(pending, key=lambda x: x['l'], reverse=True)
    
    results = []
    for t_name in fleet:
        spec = TRUCK_SPECS[t_name]
        truck_res = {"name": t_name, "boxes": [], "weight": 0}
        curr_x, rem_w = 0, spec['w']
        
        while pending and rem_w > 0:
            lane_w, curr_y = 0, 0
            while pending and curr_y < spec['l']:
                stack_h, stack_count = 0, 0
                temp_stack = []
                while pending and stack_count < MAX_STACK_COUNT:
                    b = pending[0]
                    if b['w'] <= rem_w and curr_y + b['l'] <= spec['l'] and \
                       stack_h + b['h'] <= MAX_STACK_H and \
                       truck_res['weight'] + b['weight'] <= spec['cap']:
                        b['pos'] = [curr_y, spec['w'] - rem_w, stack_h]
                        temp_stack.append(b)
                        truck_res['weight'] += b['weight']
                        stack_h += b['h']
                        stack_count += 1
                        lane_w = max(lane_w, b['w'])
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

# --- 웹 화면 ---
st.set_page_config(layout="wide")
st.title("📦 3D 차량 적재 최적화 시스템")

uploaded_file = st.sidebar.file_uploader("박스 정보 엑셀 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    fleet = ["11톤", "5톤", "5톤"]
    
    if st.sidebar.button("최적 적재 실행"):
        packed_trucks, remaining = calculate_packing(df, fleet)
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for i, truck in enumerate(packed_trucks):
                st.subheader(f"{i+1}호차: {truck['name']} ({truck['weight']}kg 적재)")
                fig = go.Figure()
                spec = TRUCK_SPECS[truck['name']]
                add_box_3d(fig, 0, 0, 0, spec['l'], spec['w'], 10, "Floor", "gray")
                for b in truck['boxes']:
                    add_box_3d(fig, b['pos'][0], b['pos'][1], b['pos'][2], b['l'], b['w'], b['h'], b['id'], "blue")
                fig.update_layout(scene=dict(aspectmode='data'))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ 미적재 박스")
            st.write(pd.DataFrame(remaining)[['id', 'l', 'w', 'h', 'weight']] if remaining else "없음")
