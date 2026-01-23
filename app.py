import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid
import time
from itertools import groupby

# ==========================================
# 1. 커스텀 물리 엔진
# ==========================================
class Box:
    __slots__ = ['name', 'w', 'h', 'd', 'weight', 'x', 'y', 'z', 'is_heavy', 'level', 'vol']
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

class Truck:
    def __init__(self, name, w, h, d, max_weight, cost, gap_mm=300, limit_level_on=True):
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
        
        # 옵션 저장
        self.gap_mm = gap_mm
        self.limit_level_on = limit_level_on

    def put_item(self, item):
        BOX_GAP_L = self.gap_mm

        if self.total_weight + item.weight > self.max_weight:
            return False
        
        # [규칙] 안전 우선: 왼쪽 벽면부터 채우기
        # Z(바닥) -> Y(안쪽) -> X(왼쪽) 순서 유지
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        
        best_pivot = None
        fit_level = 1

        for p in self.pivots:
            px, py, pz = p
            
            # 1. 경계 검사
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue
            
            # 2. 충돌 검사
            if self._check_collision_fast(item, px, py, pz):
                continue
            
            # 3. 지지 검사
            if pz > 0.001:
                if not self._check_support_fast(item, px, py, pz):
                    continue
                
                max_below_level = 0
                for exist in self.items:
                    if abs((exist.z + exist.h) - pz) < 1.0:
                        if (px < exist.x + exist.w and px + item.w > exist.x and
                            py < exist.y + exist.d and py + item.d > exist.y):
                            if exist.level > max_below_level:
                                max_below_level = exist.level
                fit_level = max_below_level + 1
            else:
                fit_level = 1
            
            # 4단 적재 제한 옵션 확인
            if self.limit_level_on and fit_level > 4: 
                continue

            best_pivot = p
            break
        
        if best_pivot:
            item.x, item.y, item.z = best_pivot
            item.level = fit_level
            self.items.append(item)
            self.total_weight += item.weight
            
            self.pivots.remove(best_pivot)
            
            # 새 피벗 생성
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d + BOX_GAP_L, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            return True
            
        return False

    def _check_collision_fast(self, item, x, y, z):
        iw, id_, ih = item.w, item.d, item.h
        for exist in self.items:
            if not (z < exist.z + exist.h and z + ih > exist.z):
                continue
            if (x < exist.x + exist.w and x + iw > exist.x and
                y < exist.y + exist.d and y + id_ > exist.y):
                return True
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
    /* 탭 컨테이너 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    /* 개별 탭 버튼 스타일 */
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 5px;
        color: #31333F;
        font-size: 16px;
        font-weight: 600;
        padding: 0px 20px;
    }
    /* 선택된 탭 스타일 */
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B !important;
        color: white !important;
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
                if weight >= heavy_threshold and weight > 0:
                    box.is_heavy = True
                items.append(box)
            except:
                continue
    except:
        pass
    return items

# ==========================================
# 3. 최적화 알고리즘
# ==========================================
def run_optimization(all_items, limit_h, gap_mm, limit_level_on):
    MARGIN_LENGTH = 200 

    # [수정 1안] 1순위: 바닥 면적(큰것부터), 2순위: 무게
    def sort_items_by_area(items):
        return sorted(items, key=lambda x: (x.w * x.d, x.weight), reverse=True)

    # [내부함수] 중앙 정렬 (X축 이동)
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

    def get_hybrid_sorted_items(items_to_sort):
        return sorted(items_to_sort, key=lambda x: (
            1 if x.d >= 2200 else 0,
            x.w,
            x.d,
            x.weight
        ), reverse=True)

    def mound_sort_group(group_items):
        s_items = sorted(group_items, key=lambda x: x.weight)
        result = [None] * len(s_items)
        left = 0; right = len(s_items) - 1
        for i, item in enumerate(s_items):
            if i % 2 == 0: result[left] = item; left += 1
            else: result[right] = item; right -= 1
        return result

    def get_balanced_sorted_items(items_to_sort):
        primary_sorted = get_hybrid_sorted_items(items_to_sort)
        final_list = []
        for k, g in groupby(primary_sorted, key=lambda x: (x.w, x.h, x.d)):
            group_list = list(g)
            if len(group_list) > 2: final_list.extend(mound_sort_group(group_list))
            else: final_list.extend(sorted(group_list, key=lambda x: x.weight, reverse=True))
        return final_list

    def solve_remaining_greedy(current_items):
        used_trucks = []
        rem = current_items[:]
        total_rem_weight = sum(i.weight for i in rem)
        
        while rem:
            best_truck = None
            max_eff = -1.0
            
            candidates = []
            for t_name in TRUCK_DB:
                spec = TRUCK_DB[t_name]
                if total_rem_weight > 10000 and spec['weight'] < 3500:
                    continue
                candidates.append((t_name, spec))

            rem = get_balanced_sorted_items(rem)

            for t_name, spec in candidates:
                t = Truck(t_name, spec['w'], limit_h, spec['l'] - MARGIN_LENGTH, spec['weight'], spec['cost'], gap_mm, limit_level_on)
                count = 0; w_sum = 0
                temp_items = []
                for item in rem:
                    new_box = Box(item.name, item.w, item.h, item.d, item.weight)
                    new_box.is_heavy = item.is_heavy
                    if t.put_item(new_box):
                        count += 1; w_sum += item.weight
                        temp_items.append(item)
                
                if count > 0:
                    eff = w_sum / spec['cost']
                    load_ratio = w_sum / spec['weight']
                    if load_ratio > 0.8: eff *= 1.2
                    if count == len(rem): eff = (1.0 / spec['cost']) * 10000 
                    if eff > max_eff: max_eff = eff; best_truck = t
            
            if best_truck:
                used_trucks.append(best_truck)
                packed_names = set(i.name for i in best_truck.items)
                rem = [i for i in rem if i.name not in packed_names]
                total_rem_weight = sum(i.weight for i in rem)
            else: break 
        return used_trucks

    best_solution = None
    min_total_cost = float('inf')
    
    total_all_weight = sum(i.weight for i in all_items)
    sorted_all_items = get_balanced_sorted_items(all_items)
    
    for start_truck_name in TRUCK_DB:
        spec = TRUCK_DB[start_truck_name]
        if total_all_weight > 15000 and spec['weight'] < 4000: continue

        start_truck = Truck(start_truck_name, spec['w'], limit_h, spec['l'] - MARGIN_LENGTH, spec['weight'], spec['cost'], gap_mm, limit_level_on)
        for item in sorted_all_items:
             new_box = Box(item.name, item.w, item.h, item.d, item.weight)
             new_box.is_heavy = item.is_heavy
             start_truck.put_item(new_box)
        
        if not start_truck.items: continue

        packed_names = set(i.name for i in start_truck.items)
        remaining = [i for i in sorted_all_items if i.name not in packed_names]
        
        current_solution = [start_truck]
        if remaining:
            sub_solution = solve_remaining_greedy(remaining)
            current_solution.extend(sub_solution)
        
        total_packed_count = sum([len(t.items) for t in current_solution])
        if total_packed_count < len(all_items): continue

        current_total_cost = sum(t.cost for t in current_solution)
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            best_solution = current_solution
    
    final_trucks = []
    if best_solution:
        best_solution.sort(key=lambda t: t.max_weight)
        for idx, t in enumerate(best_solution):
            
            # [재배치] 배차 확정 후 1안 로직(면적 우선 정렬) 적용
            items_in_truck = t.items[:] 
            
            t.items = []
            t.pivots = [[0.0, 0.0, 0.0]]
            t.total_weight = 0.0
            
            # 1. 면적(WxD)이 큰 순서대로 정렬하여 바닥부터 조밀하게 채움
            reordered_items = sort_items_by_area(items_in_truck)
            
            for item in reordered_items:
                if item is None: continue
                retry_box = Box(item.name, item.w, item.h, item.d, item.weight)
                retry_box.is_heavy = item.is_heavy
                t.put_item(retry_box)

            # 2. 적재 완료 후 전체 블록을 중앙 정렬
            recenter_truck_items(t)

            # [수정] 차량 이름 포맷 변경: 5톤(#1) 스타일
            t.name = f"{t.name} (#{idx+1})"
            final_trucks.append(t)
            
    return final_trucks

# ==========================================
# 4. 시각화
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

    def draw_cube(x, y, z, w, l, h, face_color, line_color=None, opacity=1.0, hovertext=None):
        hover_info = 'text' if hovertext else 'skip'
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+l, y+l, y, y, y+l, y+l],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=face_color, opacity=opacity, flatshading=True, 
            lighting=light_eff, hoverinfo=hover_info, hovertext=hovertext
        ))
        if line_color:
            xe=[x,x+w,x+w,x,x,None, x,x+w,x+w,x,x,None, x,x,None, x+w,x+w,None, x+w,x+w,None, x,x]
            ye=[y,y,y+l,y+l,y,None, y,y,y+l,y+l,y,None, y,y,None, y+l,y+l,None, y+l,y+l]
            ze=[z,z,z,z,z,None, z+h,z+h,z+h,z+h,z+h,None, z,z+h,None, z,z+h,None, z,z+h,None, z,z+h]
            fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=line_color, width=3), showlegend=False, hoverinfo='skip'))

    # 트럭 프레임
    ch_h = 100; f_tk = 40; bmp_h = 140; 
    draw_cube(0, 0, -ch_h, W, L, ch_h, '#AAAAAA', COLOR_FRAME)
    draw_cube(-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(W-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE)
    draw_cube(-f_tk/2, L-f_tk, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, COLOR_FRAME_LINE)
    draw_cube(-f_tk/2, L, -ch_h-bmp_h, W+f_tk, f_tk, bmp_h, '#222222') 
    
    light_y = L + f_tk; light_z = -ch_h-bmp_h+40 
    light_w = 60; light_h = 20; light_d = 60; margin_in = 150
    left_start = -f_tk/2 + margin_in
    draw_cube(left_start, light_y, light_z, light_w, light_h, light_d, '#FF0000', '#990000') 
    draw_cube(left_start+light_w, light_y, light_z, light_w, light_h, light_d, '#FFAA00', '#996600') 
    draw_cube(left_start+light_w*2, light_y, light_z, light_w, light_h, light_d, '#EEEEEE', '#AAAAAA') 
    right_start = (W + f_tk/2) - margin_in - (light_w * 3)
    draw_cube(right_start, light_y, light_z, light_w, light_h, light_d, '#EEEEEE', '#AAAAAA') 
    draw_cube(right_start+light_w, light_y, light_z, light_w, light_h, light_d, '#FFAA00', '#996600') 
    draw_cube(right_start+light_w*2, light_y, light_z, light_w, light_h, light_d, '#FF0000', '#990000') 

    draw_cube(-f_tk/2, 0, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(W-f_tk/2, 0, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(-f_tk/2, 0, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(-f_tk/2, 0, Real_H, f_tk, L, f_tk, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(W-f_tk/2, 0, Real_H, f_tk, L, f_tk, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(0, 0, 0, W, L, Real_H, '#EEF5FF', '#666666', opacity=0.1)

    OFFSET = 800; TEXT_OFFSET = OFFSET * 1.5
    def draw_arrow_dim(p1, p2, text, color='black'):
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'))
        vec = np.array(p2) - np.array(p1); length = np.linalg.norm(vec)
        if length > 0:
            u, v, w = vec / length
            fig.add_trace(go.Cone(x=[p2[0]], y=[p2[1]], z=[p2[2]], u=[u], v=[v], w=[w], sizemode="absolute", sizeref=150, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]], hoverinfo='skip'))
            fig.add_trace(go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[-u], v=[-v], w=[-w], sizemode="absolute", sizeref=150, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]], hoverinfo='skip'))
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        if text.startswith("폭"): mid[1] = -TEXT_OFFSET; mid[2] = 0
        elif text.startswith("길이"): mid[0] = -TEXT_OFFSET; mid[2] = 0
        fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode='text', text=[text], textfont=dict(color=color, size=12, family="Arial"), showlegend=False, hoverinfo='skip'))

    draw_arrow_dim([0, -OFFSET, 0], [W, -OFFSET, 0], f"폭 : {int(W)}")
    draw_arrow_dim([-OFFSET, 0, 0], [-OFFSET, L, 0], f"길이 : {int(L)}")
    
    draw_arrow_dim([-OFFSET, L, 0], [-OFFSET, L, LIMIT_H], f"높이제한 : {int(LIMIT_H)}", color='red')
    fig.add_trace(go.Scatter3d(x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False, hoverinfo='skip'))

    # 박스 그리기 (슬라이더 값만큼만 표시)
    items_to_draw = truck.items[:limit_count] if limit_count is not None else truck.items
    
    annotations = []
    for item in items_to_draw:
        col = '#FF6B6B' if item.is_heavy else '#FAD7A0'
        hover_text = f"<b>📦 {item.name}</b><br>규격: {int(item.w)}x{int(item.d)}x{int(item.h)}<br>중량: {int(item.weight):,}kg<br>적재단수: {item.level}단"
        
        draw_cube(item.x, item.y, item.z, item.w, item.d, item.h, col, '#000000', hovertext=hover_text)
        
        annotations.append(dict(x=item.x + item.w/2, y=item.y + item.d/2, z=item.z + item.h/2, text=f"<b>{item.name}</b>", xanchor="center", yanchor="middle", showarrow=False, font=dict(color="black", size=11), bgcolor="rgba(255,255,255,0.5)"))

    eye = dict(x=-1.8, y=-1.8, z=1.2); up = dict(x=0, y=0, z=1)
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='white', camera=dict(eye=eye, up=up), annotations=annotations), margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4()))
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 출하박스 적재 최적화 시스템 (배차비용 최소화)")
st.markdown("✅ **규칙 : 비용최소화 | 회전금지 | 길이우선 적재 | 바닥면 80% 지지충족 | 하중제한 준수 | 차량길이 20cm 여유 | 상위 10% 중량박스 빨간색 표시 | 안전 우선 적재(밴딩 무너짐 고려)**")

def clear_result():
    if 'optimized_result' in st.session_state:
        del st.session_state['optimized_result']

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

st.sidebar.divider()

st.sidebar.subheader("⚙️ 적재 옵션 설정")
st.sidebar.info("비용이 비싸게 나온다면 '높이 제한'을 늘리고 '간격'을 해제해보세요.")

opt_height_str = st.sidebar.radio(
    "적재 높이 제한", 
    options=["1200mm", "1300mm", "1400mm"], 
    index=0, 
    horizontal=True,
    on_change=clear_result 
)
opt_height = int(opt_height_str.replace("mm", ""))

opt_gap_str = st.sidebar.radio(
    "박스 간 간격 (길이방향)", 
    options=["0mm", "100mm", "200mm", "300mm"], 
    index=2, 
    horizontal=True,
    on_change=clear_result 
)
gap_mm = int(opt_gap_str.replace("mm", ""))

opt_level = st.sidebar.checkbox("최대 4단 적재 제한", value=True, on_change=clear_result)

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

        if st.button("최적 배차 실행 (최소비용)", type="primary"):
            
            with st.status("🚀 최적의 차량 조합을 분석 중입니다... (잠시만 기다려주세요)", expanded=True) as status:
                st.write("1. 데이터를 읽고 변환하고 있습니다...")
                time.sleep(0.1) 
                
                items = load_data(df)
                if not items:
                    st.error("데이터 변환 실패.")
                    status.update(label="오류 발생", state="error")
                else:
                    st.write("2. 최적화 엔진 가동 중... (물량에 따라 시간이 소요됩니다)")
                    time.sleep(0.1) 
                    
                    trucks = run_optimization(items, opt_height, gap_mm, opt_level)
                    
                    st.write("3. 결과 집계 및 시각화 준비 중...")
                    
                    st.session_state['optimized_result'] = trucks
                    st.session_state['calc_opt_height'] = opt_height
                    
                    time.sleep(0.2)
                    status.update(label="배차 분석 완료! 👇 아래 결과를 확인하세요.", state="complete", expanded=False)
        
        if 'optimized_result' in st.session_state:
            trucks = st.session_state['optimized_result']
            display_height = st.session_state.get('calc_opt_height', 1300)

            if trucks:
                total_cost = sum(t.cost for t in trucks)

                m1, m2, m3 = st.columns(3)
                m1.metric("총 배차 차량", f"{len(trucks)}대")
                m2.metric("총 예상 운송비", f"{total_cost:,}원")
                m3.metric("총 적재 중량", f"{sum(t.total_weight for t in trucks):,.0f} kg")
                st.divider()

                tabs = st.tabs([f"{t.name}" for t in trucks])
                for i, tab in enumerate(tabs):
                    with tab:
                        t = trucks[i]
                        
                        # [1] 적재 순서 시뮬레이션 슬라이더
                        total_items = len(t.items)
                        step = st.slider(f"🏗️ 적재 순서 시뮬레이션 (1 ~ {total_items})", 1, total_items, total_items, key=f"slider_{i}")
                        
                        c_info, c_chart = st.columns([1, 3]) 
                        with c_info:
                            truck_limit_vol = t.w * t.d * display_height 
                            used_vol = sum([b.vol for b in t.items])
                            vol_pct = min(1.0, used_vol / truck_limit_vol) if truck_limit_vol > 0 else 0
                            weight_pct = min(1.0, t.total_weight / t.max_weight)

                            st.progress(vol_pct, text=f"📏 체적 적재율 ({display_height/1000:.1f}m기준): {vol_pct*100:.1f}%")
                            st.progress(weight_pct, text=f"⚖️ 중량 적재율: {weight_pct*100:.1f}%")
                            st.divider()

                            st.markdown("##### ⚖️ 무게 분포 (4분면)")
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
                            c_q1, c_q2 = st.columns(2)
                            with c_q1: st.metric("앞-좌", f"{q_front_left/total_w*100:.0f}%", f"{int(q_front_left)}kg", delta_color="off")
                                
                            with c_q2: st.metric("앞-우", f"{q_front_right/total_w*100:.0f}%", f"{int(q_front_right)}kg", delta_color="off")
                            c_q3, c_q4 = st.columns(2)
                            with c_q3: st.metric("뒤-좌", f"{q_rear_left/total_w*100:.0f}%", f"{int(q_rear_left)}kg", delta_color="off")
                            with c_q4: st.metric("뒤-우", f"{q_rear_right/total_w*100:.0f}%", f"{int(q_rear_right)}kg", delta_color="off")
                            st.divider()

                            st.dataframe(pd.DataFrame({"항목": ["박스 수", "적재 중량", "운송 비용"], "값": [f"{len(t.items)}개", f"{t.total_weight:,.0f} kg", f"{t.cost:,} 원"]}), hide_index=True, use_container_width=True)
                            with st.expander("📦 적재 리스트 확인"):
                                st.dataframe([{"박스명": b.name, "단수": f"{b.level}단"} for b in t.items], hide_index=True)

                        with c_chart:
                            # [추가] limit_count 전달
                            st.plotly_chart(draw_truck_3d(t, limit_count=step), use_container_width=True)
            else: st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e: st.error(f"오류 발생: {e}")
