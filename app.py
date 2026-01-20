import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

# 1. 차량 제원 및 제약 조건
TRUCK_SPECS = {
    "11톤": {"w": 2350, "l": 9000, "h": 2300, "cap": 13000},
    "5톤": {"w": 2350, "l": 6200, "h": 2100, "cap": 7000}
}
MAX_STACK_H = 1300  
MAX_STACK_COUNT = 4 

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def add_box_3d(fig, x0, y0, z0, l, w, h, name, color):
    # 박스 본체 (Mesh3d)
    fig.add_trace(go.Mesh3d(
        x=[x0, x0+l, x0+l, x0, x0, x0+l, x0+l, x0],
        y=[y0, y0, y0+w, y0+w, y0, y0, y0+w, y0+w],
        z=[z0, z0, z0, z0, z0+h, z0+h, z0+h, z0+h],
        i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
        j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
        k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
        opacity=0.7, color=color, name=f"Box {name}",
        hoverinfo="text",
        text=f"📦 박스번호: {name}<br>📏 규격: {int(l)}x{int(w)}x{int(h)}<br>📍 위치(높이): {int(z0)}mm",
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

    # [수정] 박스 번호를 길이 방향 양 측면에 배치 (스티커 효과)
    # x0 (앞면), x0 + l (뒷면) 위치에 번호 표시
    fig.add_trace(go.Scatter3d(
        x=[x0 + 50, x0 + l - 50],  # 양 끝단에서 약간 안쪽
        y=[y0 + w/2, y0 + w/2],
        z=[z0 + h/2, z0 + h/2],
        mode='text',
        text=[name, name],
        textfont=dict(size=14, color="black", family="Arial Black"),
        showlegend=False,
        hoverinfo='skip'
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
                        b['pos'] = [curr_y,
