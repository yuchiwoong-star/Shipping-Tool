import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid
import time
from itertools import groupby
from io import BytesIO
from collections import deque

# PDF 라이브러리 체크
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    # 한글 폰트 미지원 시 영문으로 대체 (실제 운영 시 폰트 파일 필요)
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ==========================================
# 1. 커스텀 물리 엔진
# ==========================================
class Box:
    __slots__ = ['name', 'w', 'h', 'd', 'weight', 'x', 'y', 'z', 'is_heavy', 'level', 'vol', 'area']
    def __init__(self, name, w, h, d, weight):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.weight = float(weight)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.is_heavy = False
        self.level = 1 
        self.vol = self.w * self.h * self.d
        self.area = self.w * self.d 

class Truck:
    def __init__(self, name, w, h, d, max_weight, cost, gap_mm=300, max_layer=4):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d) 
        self.max_weight = float(max_weight)
        self.cost = int(cost)
        self.items = []
        self.total_weight = 0.0
        # 피벗: (x, y, z)
        self.pivots = [[0.0, 0.0, 0.0]]
        self.gap_mm = gap_mm
        self.max_layer = max_layer

    def put_item(self, item):
        BOX_GAP_L = self.gap_mm
        if self.total_weight + item.weight > self.max_weight: return False
        
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        
        best_pivot = None
        fit_level = 1

        for p in self.pivots:
            px, py, pz = p
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h): continue
            if self._check_collision_fast(item, px, py, pz): continue
            
            if pz > 0.001:
                if not self._check_support_fast(item, px, py, pz): continue
                max_below_level = 0
                for exist in self.items:
                    if abs((exist.z + exist.h) - pz) < 1.0:
                        if (px < exist.x + exist.w and px + item.w > exist.x and
                            py < exist.y + exist.d and py + item.d > exist.y):
                            if exist.level > max_below_level: max_below_level = exist.level
                fit_level = max_below_level + 1
            else: fit_level = 1
            
            if fit_level > self.max_layer: continue
            
            best_pivot = p
            break
        
        if best_pivot:
            item.x, item.y, item.z = best_pivot
            item.level = fit_level
            self.items.append(item)
            self.total_weight += item.weight
            self.pivots.remove(best_pivot)
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d + BOX_GAP_L, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            return True
        return False

    def _check_collision_fast(self, item, x, y, z):
        iw, id_, ih = item.w, item.d, item.h
        for exist in self.items:
            if not (z < exist.z + exist.h and z + ih > exist.z): continue
            if (x < exist.x + exist.w and x + iw > exist.x and y < exist.y + exist.d and y + id_ > exist.y): return True
        return False

    def _check_support_fast(self, item, x, y, z):
        support_area = 0.0
        item_area = item.w * item.d
        required = item_area * 0.8
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                area = ox * oy
                if area > 0:
                    support_area += area
                    if support_area >= required: return True
        return support_area >= required

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="출하박스 적재 최적화 시스템")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #F0F2F6; border-radius: 5px;
        color: #31333F; font-size: 16px; font-weight: 600; padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #FF4B4B !important; color: white !important; }
    
    .dashboard-card {
        background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px;
        height: 200px; display: flex; flex-direction: column; justify-content: flex-start;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-title {
        font-size: 16px; font-weight: 700; color: #333; margin-bottom: 15px;
        border-bottom: 2px solid #f0f0f0; padding-bottom: 5px;
    }
    .summary-row {
        display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; color: #555;
    }
    .summary-val { font-weight: bold; color: #000; }
    .custom-progress-container { margin-bottom: 12px; }
    .progress-label {
        font-size: 13px; color: #666; margin-bottom: 3px; display: flex; justify-content: space-between;
    }
    .progress-bg {
        background-color: #eee; border-radius: 10px; height: 12px; width: 100%; overflow: hidden;
    }
    .progress-fill { background-color: #FF4B4B; height: 100%; border-radius: 10px; }
    
    .quadrant-box {
        display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
        width: 100%; height: 120px; border: 1px solid #ddd; border-radius: 5px; background-color: #fafafa;
    }
    .q-cell {
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        font-size: 13px; font-weight: normal; color: #000000; background-color: white;
    }
    .q-cell:nth-child(1) { border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; border-top-left-radius: 5px;}
    .q-cell:nth-child(2) { border-bottom: 1px solid #ddd; border-top-right-radius: 5px;}
    .q-cell:nth-child(3) { border-right: 1px solid #ddd; border-bottom-left-radius: 5px;}
    .q-cell:nth-child(4) { border-bottom-right-radius: 5px;}

    .result-summary-box {
        background-color: #fff5f5; border: 2px solid #FF4B4B; border-radius: 10px;
        padding: 20px; margin-bottom: 20px; text-align: center;
    }
    .result-title { color: #000000; font-size: 22px; font-weight: bold; margin-bottom: 15px; }
    .result-metrics { display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px; }
    .metric-item { display: flex; flex-direction: column; align-items: center; min-width: 120px; }
    .metric-label { font-size: 14px; color: #000000; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: 800; color: #000000; }
    
    .flow-text {
        font-family: 'Courier New', monospace; background-color: #f8f9fa; padding: 15px;
        border-radius: 5px; font-size: 14px; color: #333; line-height: 1.6;
        border-left: 5px solid #FF4B4B; white-space: pre-wrap; word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

TRUCK_DB = {
    "1톤":   {"w": 1600, "real_h": 2000, "l": 2800,  "weight": 1490,  "cost": 78000},
    "2.5톤": {"w": 1900, "real_h": 2000, "l": 4200,  "weight": 3490,  "cost": 110000},
    "5톤":   {"w": 2100, "real_h": 2000, "l": 6200,  "weight": 6900,  "cost": 133000},
    "8톤":   {"w": 2350, "real_h": 2000, "l": 7300,  "weight": 9490,  "cost": 153000},
    "11톤":  {"w": 2350, "real_h": 2000, "l": 9200,  "weight": 14900, "cost": 188000},
    "15톤":  {"w": 2350, "real_h": 2000, "l": 10200, "weight": 16900, "cost": 211000},
    "18톤":  {"w": 2350, "real_h": 2000, "l": 10200, "weight": 20900, "cost": 242000},
    "22톤":  {"w": 2350, "real_h": 2000, "l": 10200, "weight": 26000, "cost": 308000},
}

def load_data(df):
    items = []
    try:
        cols = {c: c for c in df.columns}
        weight_col = next((c for c in df.columns if '중량' in c), None)
        heavy_threshold = float('inf')
        if weight_col:
            weights = pd.to_numeric(df[weight_col], errors='coerce').dropna().tolist()
            if weights:
                sorted_weights = sorted(weights, reverse=True)
                top_n = math.ceil(len(weights) * 0.1)
                heavy_threshold = sorted_weights[max(0, top_n - 1)]

        name_col = next((c for c in df.columns if '박스' in c or '번호' in c), None)
        w_col = next((c for c in df.columns if '폭' in c), None)
        h_col = next((c for c in df.columns if '높이' in c), None)
        l_col = next((c for c in df.columns if '길이' in c), None)

        for index, row in df.iterrows():
            try:
                name = str(row[name_col]) if name_col else f"Box-{index}"
                w = float(row[w_col])
                h = float(row[h_col])
                l = float(row[l_col])
                weight = float(row[weight_col])
                box = Box(name, w, h, l, weight)
                if weight >= heavy_threshold and weight > 0: box.is_heavy = True
                items.append(box)
            except: continue
    except: pass
    return items

# ==========================================
# 3. 최적화 알고리즘
# ==========================================
def run_optimization(all_items, limit_h, gap_mm, max_layer_val, mode):
    MARGIN_LENGTH = 200 

    def sort_length_mode(items):
        return sorted(items, key=lambda x: (x.d, x.w, x.weight), reverse=True)

    def sort_area_mode(items):
        return sorted(items, key=lambda x: (x.weight, x.area, x.d), reverse=True)

    def mound_sort_by_height(items):
        s_items = sorted(items, key=lambda x: (x.h, x.area), reverse=True)
        dq = deque()
        for i, item in enumerate(s_items):
            if i % 2 == 0: dq.append(item)
            else: dq.appendleft(item)
        return list(dq)

    def optimize_row_placement(truck):
        if not truck.items: return
        items_by_row = []
        sorted_items = sorted(truck.items, key=lambda x: x.y)
        current_row = []
        if sorted_items:
            current_row_y = sorted_items[0].y
            for item in sorted_items:
                if abs(item.y - current_row_y) > 500:
                    items_by_row.append(current_row)
                    current_row = [item]
                    current_row_y = item.y
                else:
                    current_row.append(item)
            items_by_row.append(current_row)
        
        if len(items_by_row) < 2: return 

        row_heights = []
        for row in items_by_row:
            max_h = max(item.h for item in row)
            row_heights.append({'max_h': max_h, 'items': row, 'original_y': row[0].y})
        
        row_heights.sort(key=lambda x: x['max_h'], reverse=True)
        target_y_positions = sorted([r['original_y'] for r in row_heights])
        
        new_items = []
        for i, row_data in enumerate(row_heights):
            y_diff = target_y_positions[i] - row_data['original_y']
            for item in row_data['items']:
                item.y += y_diff
                new_items.append(item)
        truck.items = new_items
        truck.pivots = [] 

    def recenter_truck_items(truck):
        if not truck.items: return
        min_x = min(item.x for item in truck.items)
        max_x = max(item.x + item.w for item in truck.items)
        load_width = max_x - min_x
        remaining_space = truck.w - load_width
        offset_x = remaining_space / 2.0
        if offset_x <= 0.1: return
        for item in truck.items: item.x += offset_x
        new_pivots = []
        for p in truck.pivots: new_pivots.append([p[0] + offset_x, p[1], p[2]])
        truck.pivots = new_pivots

    def solve_allocation(items_input, sort_func):
        sorted_items = sort_func(items_input)
        best_start_solution = None
        min_start_cost = float('inf')
        total_w = sum(i.weight for i in items_input)
        
        for start_truck_name in TRUCK_DB:
            spec = TRUCK_DB[start_truck_name]
            if total_w > 15000 and spec['weight'] < 4000: continue

            t1 = Truck(start_truck_name, spec['w'], limit_h, spec['l'] - MARGIN_LENGTH, spec['weight'], spec['cost'], gap_mm, max_layer_val)
            packed_in_t1 = []
            for item in sorted_items:
                nb = Box(item.name, item.w, item.h, item.d, item.weight)
                nb.is_heavy = item.is_heavy
                if t1.put_item(nb):
                    packed_in_t1.append(item)
            
            if not t1.items: continue
            
            packed_names = set(i.name for i in packed_in_t1)
            rem_items = [i for i in sorted_items if i.name not in packed_names]
            current_solution = [t1]
            
            if rem_items:
                rem_copy = rem_items[:]
                while rem_copy:
                    rem_copy = sort_func(rem_copy)
                    best_next = None
                    max_eff = -1.0
                    rem_w = sum(i.weight for i in rem_copy)
                    for tn in TRUCK_DB:
                        ts = TRUCK_DB[tn]
                        if rem_w > 10000 and ts['weight'] < 3500: continue
                        t_cand = Truck(tn, ts['w'], limit_h, ts['l'] - MARGIN_LENGTH, ts['weight'], ts['cost'], gap_mm, max_layer_val)
                        count = 0; w_sum = 0
                        for ri in rem_copy:
                            nb = Box(ri.name, ri.w, ri.h, ri.d, ri.weight)
                            nb.is_heavy = ri.is_heavy
                            if t_cand.put_item(nb):
                                count += 1; w_sum += nb.weight
                        if count > 0:
                            eff = w_sum / ts['cost']
                            if (w_sum / ts['weight']) > 0.8: eff *= 1.2
                            if count == len(rem_copy): eff = (1.0/ts['cost']) * 10000
                            if eff > max_eff: max_eff = eff; best_next = t_cand
                    if best_next:
                        current_solution.append(best_next)
                        p_names = set(i.name for i in best_next.items)
                        rem_copy = [i for i in rem_copy if i.name not in p_names]
                    else: break
            
            total_packed = sum(len(t.items) for t in current_solution)
            if total_packed < len(items_input): continue
            cost = sum(t.cost for t in current_solution)
            if cost < min_start_cost:
                min_start_cost = cost
                best_start_solution = current_solution
        return best_start_solution

    final_solution_trucks = []
    if mode == 'length':
        final_solution_trucks = solve_allocation(all_items, sort_length_mode)
    else:
        final_solution_trucks = solve_allocation(all_items, sort_area_mode)

    final_output = []
    if final_solution_trucks:
        final_solution_trucks.sort(key=lambda t: t.max_weight)
        for idx, t in enumerate(final_solution_trucks):
            items_in_truck = t.items[:] 
            t.items = []
            t.pivots = [[0.0, 0.0, 0.0]]
            t.total_weight = 0.0
            
            if mode == 'length':
                items_in_truck.sort(key=lambda x: x.d, reverse=True)
                final_load_order = []
                for k, g in groupby(items_in_truck, key=lambda x: round(x.d / 500)):
                    group_list = list(g)
                    mounded_group = mound_sort_by_height(group_list)
                    final_load_order.extend(mounded_group)
                for item in final_load_order:
                    if item is None: continue
                    retry_box = Box(item.name, item.w, item.h, item.d, item.weight)
                    retry_box.is_heavy = item.is_heavy
                    t.put_item(retry_box)
                optimize_row_placement(t)
            else: 
                reordered_items = sort_area_mode(items_in_truck)
                for item in reordered_items:
                    retry_box = Box(item.name, item.w, item.h, item.d, item.weight)
                    retry_box.is_heavy = item.is_heavy
                    t.put_item(retry_box)

            recenter_truck_items(t)
            t.name = f"{t.name} (#{idx+1})"
            final_output.append(t)
            
    return final_output

# ==========================================
# 4. 시각화 (3D 애니메이션 적용 버전)
# ==========================================
def draw_truck_3d(truck, limit_count=None):
    fig = go.Figure()
    original_name = truck.name.split(' (#')[0] if '(#' in truck.name else truck.name
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = truck.h 
    
    light_eff = dict(ambient=0.9, diffuse=0.5, specular=0.1, roughness=0.5)
    COLOR_FRAME = '#555555' 
    COLOR_FRAME_LINE = '#333333'

    # [Helper] 큐브 Trace 생성 함수 (직접 fig에 추가하지 않고 Trace 객체 반환)
    def create_cube_trace(x, y, z, w, l, h, face_color, line_color=None, opacity=1.0, hovertext=None, name=None):
        hover_info = 'text' if hovertext else 'skip'
        
        # 면(Mesh3d)
        mesh = go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+l, y+l, y, y, y+l, y+l],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=face_color, opacity=opacity, flatshading=True, 
            lighting=light_eff, hoverinfo=hover_info, hovertext=hovertext,
            name=name if name else ""
        )
        
        traces = [mesh]
        
        # 테두리(Lines) - 선택 사항
        if line_color:
            xe=[x,x+w,x+w,x,x,None, x,x+w,x+w,x,x,None, x,x,None, x+w,x+w,None, x+w,x+w,None, x,x]
            ye=[y,y,y+l,y+l,y,None, y,y,y+l,y+l,y,None, y,y,None, y+l,y+l,None, y+l,y+l]
            ze=[z,z,z,z,z,None, z+h,z+h,z+h,z+h,z+h,None, z,z+h,None, z,z+h,None, z,z+h,None, z,z+h]
            lines = go.Scatter3d(
                x=xe, y=ye, z=ze, mode='lines', 
                line=dict(color=line_color, width=3), 
                showlegend=False, hoverinfo='skip'
            )
            traces.append(lines)
            
        return traces

    # 1. 고정된 트럭 프레임 및 배경 그리기
    static_traces = []
    
    # 프레임 치수 설정
    ch_h = 100; f_tk = 40; bmp_h = 140; 
    
    # 섀시 및 기둥 (Static Traces에 추가)
    static_traces.extend(create_cube_trace(0, 0, -ch_h, W, L, ch_h, '#AAAAAA', COLOR_FRAME))
    static_traces.extend(create_cube_trace(-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE)) 
    static_traces.extend(create_cube_trace(W-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE))
    static_traces.extend(create_cube_trace(-f_tk/2, L-f_tk, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, COLOR_FRAME_LINE))
    static_traces.extend(create_cube_trace(-f_tk/2, L, -ch_h-bmp_h, W+f_tk, f_tk, bmp_h, '#222222')) 
    
    # 후미등 (장식)
    light_y = L + f_tk; light_z = -ch_h-bmp_h+40 
    light_w = 60; light_h = 20; light_d = 60; margin_in = 150
    left_start = -f_tk/2 + margin_in
    static_traces.extend(create_cube_trace(left_start, light_y, light_z, light_w, light_h, light_d, '#FF0000', '#990000')) 
    static_traces.extend(create_cube_trace(left_start+light_w, light_y, light_z, light_w, light_h, light_d, '#FFAA00', '#996600')) 
    static_traces.extend(create_cube_trace(left_start+light_w*2, light_y, light_z, light_w, light_h, light_d, '#EEEEEE', '#AAAAAA')) 
    right_start = (W + f_tk/2) - margin_in - (light_w * 3)
    static_traces.extend(create_cube_trace(right_start, light_y, light_z, light_w, light_h, light_d, '#EEEEEE', '#AAAAAA')) 
    static_traces.extend(create_cube_trace(right_start+light_w, light_y, light_z, light_w, light_h, light_d, '#FFAA00', '#996600')) 
    static_traces.extend(create_cube_trace(right_start+light_w*2, light_y, light_z, light_w, light_h, light_d, '#FF0000', '#990000')) 

    # 전면 프레임 및 바닥
    static_traces.extend(create_cube_trace(-f_tk/2, 0, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE)) 
    static_traces.extend(create_cube_trace(W-f_tk/2, 0, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE)) 
    static_traces.extend(create_cube_trace(-f_tk/2, 0, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, COLOR_FRAME_LINE)) 
    static_traces.extend(create_cube_trace(-f_tk/2, 0, Real_H, f_tk, L, f_tk, COLOR_FRAME, COLOR_FRAME_LINE)) 
    static_traces.extend(create_cube_trace(W-f_tk/2, 0, Real_H, f_tk, L, f_tk, COLOR_FRAME, COLOR_FRAME_LINE)) 
    static_traces.extend(create_cube_trace(0, 0, 0, W, L, Real_H, '#EEF5FF', '#666666', opacity=0.1))

    # 치수선 (Helper function 내재화 대신 직접 trace 생성 후 추가)
    OFFSET = 800; TEXT_OFFSET = OFFSET * 1.5
    def make_dim_traces(p1, p2, text, color='black'):
        t_list = []
        t_list.append(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'))
        vec = np.array(p2) - np.array(p1); length = np.linalg.norm(vec)
        if length > 0:
            u, v, w = vec / length
            t_list.append(go.Cone(x=[p2[0]], y=[p2[1]], z=[p2[2]], u=[u], v=[v], w=[w], sizemode="absolute", sizeref=150, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]], hoverinfo='skip'))
            t_list.append(go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[-u], v=[-v], w=[-w], sizemode="absolute", sizeref=150, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]], hoverinfo='skip'))
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        if text.startswith("폭"): mid[1] = -TEXT_OFFSET; mid[2] = 0
        elif text.startswith("길이"): mid[0] = -TEXT_OFFSET; mid[2] = 0
        t_list.append(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode='text', text=[text], textfont=dict(color=color, size=12, family="Arial"), showlegend=False, hoverinfo='skip'))
        return t_list

    static_traces.extend(make_dim_traces([0, -OFFSET, 0], [W, -OFFSET, 0], f"폭 : {int(W)}"))
    static_traces.extend(make_dim_traces([-OFFSET, 0, 0], [-OFFSET, L, 0], f"길이 : {int(L)}"))
    static_traces.extend(make_dim_traces([-OFFSET, L, 0], [-OFFSET, L, LIMIT_H], f"높이제한 : {int(LIMIT_H)}", color='red'))
    static_traces.append(go.Scatter3d(x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False, hoverinfo='skip'))

    # 2. 박스 데이터 생성 (Box Traces)
    # 애니메이션을 위해 박스별로 Trace를 생성하고 리스트에 담습니다.
    # Mesh3d와 테두리 Line을 하나의 그룹으로 묶지 않고 평탄화(flatten)해서 관리해야 visibility 제어가 쉽습니다.
    
    box_traces = []
    box_trace_indices = [] # 각 박스가 몇 개의 trace(면+선)로 구성되는지 저장
    
    for item in truck.items:
        col = '#FF6B6B' if item.is_heavy else '#FAD7A0'
        hover_text = f"<b>📦 {item.name}</b><br>규격: {int(item.w)}x{int(item.d)}x{int(item.h)}<br>중량: {int(item.weight):,}kg<br>적재단수: {item.level}단"
        
        # 박스 1개당 생성되는 Trace들 (Mesh 1개 + Line 1개 = 총 2개)
        traces = create_cube_trace(item.x, item.y, item.z, item.w, item.d, item.h, col, '#000000', hovertext=hover_text, name=item.name)
        
        start_idx = len(box_traces)
        box_traces.extend(traces)
        end_idx = len(box_traces)
        box_trace_indices.append(range(start_idx, end_idx))

    # 3. Figure 초기화 및 Trace 추가
    # 초기 상태: 모든 박스가 다 보이는 상태 (Return to original screen at end)
    for t in static_traces:
        fig.add_trace(t)
    for t in box_traces:
        fig.add_trace(t)

    # Static trace 개수
    num_static = len(static_traces)
    num_boxes = len(truck.items)
    total_traces = len(fig.data)

    # 4. 애니메이션 프레임 생성
    # Frame 0: 빈 트럭
    # Frame k: k번째 박스까지 적재
    frames = []
    
    # 4-1. 빈 트럭 프레임 (Start)
    frames.append(go.Frame(
        data=[dict(visible=True) for _ in range(num_static)] + [dict(visible=False) for _ in range(len(box_traces))],
        name="start"
    ))

    # 4-2. 단계별 적재 프레임
    # 누적적으로 visible을 True로 켭니다.
    current_visible_count = 0
    for i in range(num_boxes):
        # 이번 단계에서 켜야 할 box trace들의 개수
        box_indices = box_trace_indices[i]
        
        # 전체 trace visibility 리스트 생성
        # Static(Always True) + Previous Boxes(True) + Current Box(True) + Future Boxes(False)
        
        # Plotly Frame 최적화를 위해 전체 리스트를 다시 보내는 대신,
        # 프레임별로 가시성(visible) 속성만 업데이트합니다.
        # 하지만 Python Plotly에서는 data 리스트의 길이가 trace 길이와 같아야 매칭이 잘 됩니다.
        
        visibility_list = [True] * num_static # 트럭 프레임은 항상 보임
        
        # 박스들 가시성 설정
        box_vis_list = []
        for b_idx in range(len(box_traces)):
            # 현재 박스 그룹(i)까지 포함되면 True
            if b_idx < box_trace_indices[i].stop:
                box_vis_list.append(True)
            else:
                box_vis_list.append(False)
        
        visibility_list.extend(box_vis_list)
        
        # Frame 추가
        frames.append(go.Frame(
            data=[dict(visible=vis) for vis in visibility_list],
            name=f"step_{i}"
        ))

    fig.frames = frames

    # 5. 레이아웃 및 애니메이션 컨트롤 설정
    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), 
            bgcolor='white', 
            camera=dict(eye=dict(x=-1.8, y=-1.8, z=1.2), up=dict(x=0, y=0, z=1))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
        uirevision=str(uuid.uuid4()), # 카메라 고정
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.1, y=0.1, xanchor="right", yanchor="top",
            buttons=[dict(
                label="▶ 적재 재생",
                method="animate",
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode="immediate")]
            )]
        )],
        sliders=[dict(
            steps=[dict(
                method="animate",
                args=[[f"step_{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                label=f"{k+1}/{num_boxes}"
            ) for k in range(num_boxes)],
            active=num_boxes - 1, # 슬라이더 시작 위치 (맨 끝)
            currentvalue=dict(prefix="적재 순서: ", visible=True, xanchor="center"),
            pad=dict(t=50)
        )]
    )

    return fig
# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 출하박스 적재 최적화 시스템 (배차비용 최소화)")

def clear_result():
    if 'optimized_result' in st.session_state:
        del st.session_state['optimized_result']

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])
st.sidebar.divider()

st.sidebar.subheader("⚙️ 적재 옵션 설정")
st.sidebar.info("💡 원하는 배차 결과가 나오지 않았다면, 아래 옵션을 조정하여 재실행해 보세요.")

opt_mode = st.sidebar.radio(
    "적재 우선순위 모드",
    options=["길이 우선 (긴 화물 / 규격이 일정할 때)", "바닥면적 우선 (크기가 다양한 혼합 화물)"],
    index=0,
    on_change=clear_result
)
mode_key = 'length' if "길이" in opt_mode else 'area'

opt_height_str = st.sidebar.radio("적재 높이 제한", options=["1200mm", "1300mm", "1400mm"], index=1, horizontal=True, on_change=clear_result)
opt_height = int(opt_height_str.replace("mm", ""))

opt_gap_str = st.sidebar.radio("박스 간 간격 (길이방향)", options=["0mm", "100mm", "200mm", "300mm"], index=2, horizontal=True, on_change=clear_result)
gap_mm = int(opt_gap_str.replace("mm", ""))

opt_stack_limit = st.sidebar.radio("최대 적재 단수", ["3단", "4단", "제한없음"], index=1, horizontal=True, on_change=clear_result)
if "3단" in opt_stack_limit: max_layer_val = 3
elif "4단" in opt_stack_limit: max_layer_val = 4
else: max_layer_val = 100 

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, encoding='cp949')
        else: df = pd.read_excel(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        df_display = df.copy()
        rename_map = {}
        for c in df_display.columns:
            if '박스' in c or '번호' in c: rename_map[c] = '박스번호'
            elif '폭' in c: rename_map[c] = '폭 (mm)'
            elif '높이' in c: rename_map[c] = '높이 (mm)'
            elif '길이' in c: rename_map[c] = '길이 (mm)'
            elif '중량' in c: rename_map[c] = '중량 (kg)'
        df_display = df_display.rename(columns=rename_map)
        
        cols_to_format = ['폭 (mm)', '높이 (mm)', '길이 (mm)', '중량 (kg)']
        for col in cols_to_format:
            if col in df_display.columns: df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "")
        if '박스번호' in df_display.columns: df_display['박스번호'] = df_display['박스번호'].astype(str)
        st.dataframe(df_display, use_container_width=True, hide_index=True, height=250, column_config={c: st.column_config.Column(width="medium") for c in df_display.columns})

        st.subheader("🚛 차량 기준 정보")
        truck_rows = [{"차량": name, "적재폭 (mm)": spec['w'], "적재길이 (mm)": spec['l'], "허용하중 (kg)": spec['weight'], "운송단가 (원)": spec['cost']} for name, spec in TRUCK_DB.items()]
        df_truck = pd.DataFrame(truck_rows)
        for col in ['적재폭 (mm)', '적재길이 (mm)', '허용하중 (kg)', '운송단가 (원)']: df_truck[col] = df_truck[col].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_truck, use_container_width=True, hide_index=True, column_config={c: st.column_config.Column(width="medium") for c in df_truck.columns})

        if st.button("최적 배차 실행", type="primary"):
            with st.status(f"🚀 {opt_mode} 모드로 분석 중입니다...", expanded=True) as status:
                st.write("1. 데이터를 읽고 변환하고 있습니다...")
                time.sleep(0.1) 
                
                items = load_data(df)
                if not items:
                    st.error("데이터 변환 실패.")
                    status.update(label="오류 발생", state="error")
                else:
                    time.sleep(0.1) 
                    trucks = run_optimization(items, opt_height, gap_mm, max_layer_val, mode=mode_key)
                    st.session_state['optimized_result'] = trucks
                    st.session_state['calc_opt_height'] = opt_height
                    time.sleep(0.2)
                    status.update(label="배차 분석 완료!", state="complete", expanded=False)

        if 'optimized_result' in st.session_state:
            trucks = st.session_state['optimized_result']
            display_height = st.session_state.get('calc_opt_height', 1300)

            with st.expander("📜 분석 History (Click to view details)", expanded=False):
                st.markdown(f"**1️⃣ 데이터 및 옵션 확인**")
                st.text(f"   - 입력 데이터: {len(df)}건 로드 완료")
                st.text(f"   - 선택 모드: {opt_mode}")
                st.text(f"   - 제약 조건: 높이 {opt_height}mm / 간격 {gap_mm}mm / {opt_stack_limit}")

                st.markdown(f"**2️⃣ 1차 배차 시뮬레이션 (Allocation)**")
                if mode_key == 'length':
                    st.text("   - 전략: [길이 우선] 긴 화물부터 배차하여 적재함 길이 효율 극대화")
                    st.text("   - 정렬: 길이(L) → 폭(W) → 중량 순서로 투입")
                else:
                    st.text("   - 전략: [바닥면적 우선] 크고 무거운 화물부터 배차하여 바닥면 확보")
                    st.text("   - 정렬: 중량 → 바닥면적(WxL) → 길이 순서로 투입")

                st.markdown(f"**3️⃣ 2차 적재 최적화 (Restacking)**")
                if mode_key == 'length':
                    st.text("   - 그룹핑: 길이 오차 50cm 이내 화물끼리 줄(Row) 형성")
                    st.text("   - 패턴: 각 줄 내부에서 '피라미드(▲)' 형태로 재배열 (안전성 확보)")
                    st.text("   - 배치: 키가 큰 줄을 안쪽(운전석), 작은 줄을 문 쪽으로 이동")
                else:
                    st.text("   - 채우기: 확정된 화물을 밀도 순으로 재정렬하여 빈틈없이 채움 (Tetris)")
                    st.text("   - 안정화: 인위적인 위치 변경을 최소화하여 적재 깨짐 방지")
                    st.text("   - 중심잡기: 전체 화물 덩어리를 트럭 정중앙으로 이동")

                st.markdown(f"**4️⃣ 최종 결과 도출**")
                st.text(f"   - 총 {len(trucks)}대 배차 완료 (비용 최적화 달성)")

            if trucks:
                total_cost = sum(t.cost for t in trucks)
                total_weight = sum(t.total_weight for t in trucks)
                total_box_count = sum(len(t.items) for t in trucks)
                total_trucks = len(trucks)

                st.markdown(f"""
                <div class="result-summary-box">
                    <div class="result-title">✅ 배차 분석 완료!</div>
                    <div class="result-metrics">
                        <div class="metric-item">
                            <span class="metric-label">총 배차 차량</span>
                            <span class="metric-value">{total_trucks}대</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">총 예상 운송비</span>
                            <span class="metric-value" style="color:#000000;">{total_cost:,}원</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">총 적재 중량</span>
                            <span class="metric-value">{total_weight:,.0f}kg</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">총 박스 수량</span>
                            <span class="metric-value">{total_box_count:,}개</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                tabs = st.tabs([f"{t.name}" for t in trucks])
                for i, tab in enumerate(tabs):
                    with tab:
                        t = trucks[i]
                        
                        truck_limit_vol = t.w * t.d * display_height 
                        used_vol = sum([b.vol for b in t.items])
                        vol_pct = min(1.0, used_vol / truck_limit_vol) if truck_limit_vol > 0 else 0
                        weight_pct = min(1.0, t.total_weight / t.max_weight)
                        
                        mid_y = t.d / 2; mid_x = t.w / 2  
                        q_front_left = q_front_right = q_rear_left = q_rear_right = 0.0
                        
                        def calc_overlap(b_x1, b_x2, b_y1, b_y2, q_x1, q_x2, q_y1, q_y2):
                            x_overlap = max(0, min(b_x2, q_x2) - max(b_x1, q_x1))
                            y_overlap = max(0, min(b_y2, q_y2) - max(b_y1, q_y1))
                            return x_overlap * y_overlap

                        for item in t.items:
                            b_x1, b_x2 = item.x, item.x + item.w
                            b_y1, b_y2 = item.y, item.y + item.d
                            if item.vol <= 0: continue
                            box_area = item.w * item.d
                            q_front_left += item.weight * (calc_overlap(b_x1, b_x2, b_y1, b_y2, mid_x, t.w, 0, mid_y) / box_area)
                            q_front_right += item.weight * (calc_overlap(b_x1, b_x2, b_y1, b_y2, 0, mid_x, 0, mid_y) / box_area)
                            q_rear_left += item.weight * (calc_overlap(b_x1, b_x2, b_y1, b_y2, mid_x, t.w, mid_y, t.d) / box_area)
                            q_rear_right += item.weight * (calc_overlap(b_x1, b_x2, b_y1, b_y2, 0, mid_x, mid_y, t.d) / box_area)
                        total_w = t.total_weight if t.total_weight > 0 else 1

                        c1, c2, c3 = st.columns([1, 1, 1.2])
                        
                        with c1:
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <span class="card-title">📋 적재 정보</span>
                                <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                                    <div class="summary-row"><span>박스 수량</span><span class="summary-val">{len(t.items)} 개</span></div>
                                    <div class="summary-row"><span>적재 중량</span><span class="summary-val">{t.total_weight:,.0f} kg</span></div>
                                    <div class="summary-row"><span>운송 비용</span><span class="summary-val" style="color:#000000;">{t.cost:,} 원</span></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with c2:
                            vol_w = vol_pct * 100
                            wgt_w = weight_pct * 100
                            st.markdown(f"""
                            <div class="dashboard-card">
                                <span class="card-title">📉 적재율</span>
                                <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: center;">
                                    <div class="custom-progress-container">
                                        <div class="progress-label"><span>체적</span><span style="font-weight:bold;">{vol_w:.1f}%</span></div>
                                        <div class="progress-bg"><div class="progress-fill" style="width: {vol_w}%; background-color: #FF4B4B;"></div></div>
                                    </div>
                                    <div class="custom-progress-container">
                                        <div class="progress-label"><span>중량</span><span style="font-weight:bold;">{wgt_w:.1f}%</span></div>
                                        <div class="progress-bg"><div class="progress-fill" style="width: {wgt_w}%; background-color: #FF4B4B;"></div></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        with c3:
                            def get_color(val):
                                return "#FF0000" if val > 33 else "#000000"
                            
                            p_fl = q_front_left/total_w*100
                            p_fr = q_front_right/total_w*100
                            p_rl = q_rear_left/total_w*100
                            p_rr = q_rear_right/total_w*100

                            st.markdown(f"""
                            <div class="dashboard-card">
                                <span class="card-title">⚖️ 무게 분포</span>
                                <div class="quadrant-box">
                                    <div class="q-cell">앞-좌<br><span style="font-weight:bold; color:{get_color(p_fl)};">{p_fl:.0f}%</span></div>
                                    <div class="q-cell">앞-우<br><span style="font-weight:bold; color:{get_color(p_fr)};">{p_fr:.0f}%</span></div>
                                    <div class="q-cell">뒤-좌<br><span style="font-weight:bold; color:{get_color(p_rl)};">{p_rl:.0f}%</span></div>
                                    <div class="q-cell">뒤-우<br><span style="font-weight:bold; color:{get_color(p_rr)};">{p_rr:.0f}%</span></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.write("") 

                        c_list, c_chart = st.columns([1, 2]) 
                        
                        with c_list:
                            def get_zone_name(item, truck_w, truck_d):
                                cx = item.x + item.w / 2
                                cy = item.y + item.d / 2
                                lr_str = "좌" if cx < truck_w / 2 else "우"
                                fb_str = "뒤" if cy < truck_d / 2 else "앞"
                                return f"{fb_str}-{lr_str}"

                            list_data = []
                            for idx, item in enumerate(t.items):
                                zone = get_zone_name(item, t.w, t.d)
                                list_data.append({
                                    "순서": idx + 1,
                                    "박스명": item.name,
                                    "크기": f"{int(item.w)}x{int(item.d)}x{int(item.h)}",
                                    "무게": f"{int(item.weight)}kg",
                                    "위치": zone,
                                    "단수": f"{item.level}단"
                                })
                            
                            df_list = pd.DataFrame(list_data)

                            if HAS_REPORTLAB:
                                buffer = BytesIO()
                                c = canvas.Canvas(buffer, pagesize=A4)
                                width, height = A4
                                c.setFont("Helvetica-Bold", 16)
                                c.drawString(30, height - 50, f"Loading Manifest - {t.name}")
                                c.setFont("Helvetica", 10)
                                c.drawString(30, height - 70, f"Total Weight: {t.total_weight:,.0f} kg")
                                c.drawString(30, height - 90, f"Box Count: {len(t.items)} ea")
                                
                                y = height - 130
                                c.setFont("Helvetica-Bold", 10)
                                header_str = "No.   Box Name      Zone(Pos)    Layer    Weight"
                                c.drawString(30, y, header_str)
                                c.line(30, y-5, 550, y-5)
                                y -= 20
                                c.setFont("Helvetica", 10)
                                
                                for item_data in list_data:
                                    if y < 50: 
                                        c.showPage()
                                        y = height - 50
                                    zone_map = item_data['위치'].replace("앞", "Front").replace("뒤", "Rear").replace("좌", "L").replace("우", "R")
                                    line_str = f"{item_data['순서']}.   {item_data['박스명']}      {zone_map}       {item_data['단수']}     {item_data['무게']}"
                                    c.drawString(30, y, line_str)
                                    y -= 15
                                
                                c.save()
                                buffer.seek(0)
                                st.download_button("📄 현장용 리스트 인쇄 (PDF)", buffer, f"{t.name}_list.pdf", "application/pdf", key=f"pdf_{i}")
                            
                            with st.expander("📦 상세 적재 리스트 (펼치기)", expanded=True):
                                st.dataframe(df_list, hide_index=True, use_container_width=True, height=400)
                                
                                order_str = " -> ".join([d['박스명'] for d in list_data])
                                st.markdown("##### 🏗️ 적재 순서 (지게차 작업 순)")
                                st.markdown(f'<div class="flow-text">{order_str}</div>', unsafe_allow_html=True)

                        with c_chart:
                            st.plotly_chart(draw_truck_3d_animated(t), use_container_width=True)
            else: st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e: st.error(f"오류 발생: {e}")
