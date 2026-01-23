import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid
import random
from itertools import groupby

# ==========================================
# 1. 커스텀 물리 엔진 (기존 유지)
# ==========================================
class Box:
    __slots__ = ['name', 'w', 'h', 'l', 'weight', 'x', 'y', 'z', 'is_heavy', 'level', 'vol']
    def __init__(self, name, w, h, l, weight):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.l = float(l) # Length (깊이/길이)
        self.weight = float(weight)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.is_heavy = False
        self.level = 1 
        self.vol = self.w * self.h * self.l

class Truck:
    def __init__(self, name, w, h, l, max_weight, cost):
        self.name = name
        self.w = float(w)
        self.h = float(h) # 적재 가능 높이
        self.l = float(l) # 적재 가능 길이
        self.max_weight = float(max_weight)
        self.cost = int(cost)
        self.items = []
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item, use_gap=True, limit_level=True):
        # [옵션] 박스 간 길이방향 간격 30cm
        BOX_GAP_L = 300 if use_gap else 0

        if self.total_weight + item.weight > self.max_weight:
            return False
        
        # Z(바닥) -> Y(안쪽) -> X(왼쪽) 순서
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        
        best_pivot = None
        fit_level = 1

        for p in self.pivots:
            px, py, pz = p
            
            # 1. 경계 검사
            if (px + item.w > self.w) or (py + item.l > self.l) or (pz + item.h > self.h):
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
                            py < exist.y + exist.l and py + item.l > exist.y):
                            if exist.level > max_below_level:
                                max_below_level = exist.level
                fit_level = max_below_level + 1
            else:
                fit_level = 1
            
            # [옵션] 4단 적재 제한
            if limit_level and fit_level > 4: 
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
            self.pivots.append([item.x, item.y + item.l + BOX_GAP_L, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
            return True
            
        return False

    def _check_collision_fast(self, item, x, y, z):
        iw, il, ih = item.w, item.l, item.h
        for exist in self.items:
            if not (z < exist.z + exist.h and z + ih > exist.z):
                continue
            if (x < exist.x + exist.w and x + iw > exist.x and
                y < exist.y + exist.l and y + il > exist.y):
                return True
        return False

    def _check_support_fast(self, item, x, y, z):
        support_area = 0.0
        item_area = item.w * item.l
        required = item_area * 0.8
        
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.l, exist.y + exist.l) - max(y, exist.y))
                area = ox * oy
                if area > 0:
                    support_area += area
                    if support_area >= required: return True
        return support_area >= required
        
    # [추가] 재배치를 위해 트럭 상태 초기화
    def clear_cargo(self):
        self.items = []
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

# ==========================================
# 2. 설정 및 데이터 (기존 유지)
# ==========================================
st.set_page_config(layout="wide", page_title="출하박스 적재 최적화 시스템")

TRUCK_DB = {
    "1톤":   {"w": 1600, "real_h": 2000, "l": 2800,  "weight": 1490,  "cost": 78000},
    "2.5톤": {"w": 1900, "real_h": 2000, "l": 4200,  "weight": 3490,  "cost": 110000},
    "5톤":   {"w": 2100, "real_h": 2200, "l": 6200,  "weight": 6900,  "cost": 133000},
    "8톤":   {"w": 2350, "real_h": 2300, "l": 7300,  "weight": 9490,  "cost": 153000},
    "11톤":  {"w": 2350, "real_h": 2400, "l": 9200,  "weight": 11900, "cost": 188000},
    "15톤":  {"w": 2350, "real_h": 2400, "l": 10200, "weight": 16900, "cost": 211000},
    "18톤":  {"w": 2350, "real_h": 2400, "l": 10200, "weight": 20900, "cost": 242000},
    "22톤":  {"w": 2350, "real_h": 2400, "l": 10200, "weight": 26000, "cost": 308000},
}

def load_data(df):
    items = []
    error_logs = []
    
    cols = df.columns
    name_col = next((c for c in cols if '박스' in c or '번호' in c), None)
    w_col = next((c for c in cols if '폭' in c), None)
    h_col = next((c for c in cols if '높이' in c), None)
    l_col = next((c for c in cols if '길이' in c), None)
    weight_col = next((c for c in cols if '중량' in c), None)

    if not (w_col and h_col and l_col and weight_col):
        return [], ["필수 컬럼(폭, 높이, 길이, 중량) 중 일부가 누락되었습니다."]

    heavy_threshold = float('inf')
    try:
        weights = pd.to_numeric(df[weight_col], errors='coerce').dropna().tolist()
        if weights:
            sorted_weights = sorted(weights, reverse=True)
            top_n = math.ceil(len(weights) * 0.1)
            heavy_threshold = sorted_weights[max(0, top_n - 1)]
    except:
        pass

    for index, row in df.iterrows():
        try:
            name = str(row[name_col]) if name_col else f"Box-{index}"
            w = float(row[w_col])
            h = float(row[h_col])
            l = float(row[l_col])
            weight = float(row[weight_col])
            
            if w <= 0 or h <= 0 or l <= 0 or weight <= 0: continue

            box = Box(name, w, h, l, weight)
            if weight >= heavy_threshold and weight > 0:
                box.is_heavy = True
            items.append(box)
        except Exception as e:
            error_logs.append(f"행 {index+2} 오류: {str(e)}")
            continue

    return items, error_logs

# ==========================================
# 3. 최적화 및 후처리 로직
# ==========================================

# [기능추가] 4분면 무게 밸런스 계산 함수
def calc_quadrant_variance(truck):
    if not truck.items: return float('inf')
    mid_x = truck.w / 2
    mid_y = truck.l / 2
    
    q1 = q2 = q3 = q4 = 0.0 # FL, FR, RL, RR
    
    for item in truck.items:
        cx = item.x + item.w/2
        cy = item.y + item.l/2
        w = item.weight
        
        if cx < mid_x and cy < mid_y: q1 += w    # Front-Left
        elif cx >= mid_x and cy < mid_y: q2 += w # Front-Right
        elif cx < mid_x and cy >= mid_y: q3 += w # Rear-Left
        else: q4 += w                            # Rear-Right
        
    # 4분면 무게의 표준편차 반환 (낮을수록 균형 잡힘)
    return np.std([q1, q2, q3, q4])

# [기능추가 1] 4분면 밸런스 최적화를 위한 재적재 (단순 중앙정렬 X, 배치 재조정)
def optimize_load_balance(truck, use_gap, limit_level, limit_h):
    if not truck.items or len(truck.items) < 2: return

    # 1. 기존 확정된 박스들 추출
    original_items = [Box(i.name, i.w, i.h, i.l, i.weight) for i in truck.items]
    original_items_info = [(i.name, i.w, i.h, i.l, i.weight, i.is_heavy) for i in truck.items]
    
    # 2. 다양한 정렬 시나리오 준비
    # - 시나리오 A: 무게 내림차순 (기본)
    # - 시나리오 B: 무게 오름차순
    # - 시나리오 C: 무-가-무-가 (인터리빙) -> 앞뒤/좌우 분산 유도
    s_heavy = sorted(original_items_info, key=lambda x: x[4], reverse=True)
    s_light = sorted(original_items_info, key=lambda x: x[4])
    
    # 인터리빙 생성 (무거운거 하나, 가벼운거 하나)
    s_interleaved = []
    n = len(s_heavy)
    mid = n // 2
    for i in range(mid + 1):
        if i < len(s_heavy): s_interleaved.append(s_heavy[i])
        if n - 1 - i >= 0 and n - 1 - i != i: s_interleaved.append(s_light[i]) # 뒤에서부터(가벼운거)
        
    candidates = [s_heavy, s_light, s_interleaved]

    best_variance = float('inf')
    best_truck_state = None
    
    # 3. 각 시나리오별 시뮬레이션
    for candidate in candidates:
        # 가상 트럭 생성
        sim_truck = Truck(truck.name, truck.w, limit_h, truck.l, truck.max_weight, truck.cost)
        
        # 박스 객체 재생성 및 적재 시도
        all_packed = True
        for info in candidate:
            new_box = Box(info[0], info[1], info[2], info[3], info[4])
            new_box.is_heavy = info[5]
            if not sim_truck.put_item(new_box, use_gap=use_gap, limit_level=limit_level):
                all_packed = False
                break
        
        if all_packed:
            var = calc_quadrant_variance(sim_truck)
            if var < best_variance:
                best_variance = var
                best_truck_state = sim_truck.items

    # 4. 가장 밸런스가 좋은 상태로 트럭 업데이트
    if best_truck_state:
        truck.clear_cargo()
        truck.items = best_truck_state
        truck.total_weight = sum(i.weight for i in best_truck_state)

# [기능추가 2] 밴딩 안전을 위한 피라미드 형태 재배치 (가운데가 높게)
def apply_pyramid_stacking(truck):
    if not truck.items: return

    # 1. Y축(길이방향) 기준으로 행(Row) 분류
    rows = {}
    for item in truck.items:
        y_key = round(item.y / 10) * 10 
        if y_key not in rows: rows[y_key] = []
        rows[y_key].append(item)
    
    for y_key, items_in_row in rows.items():
        # 2. X축(폭방향) 기준으로 스택(Column) 분류
        stacks = {}
        for item in items_in_row:
            x_key = round(item.x)
            if x_key not in stacks: stacks[x_key] = []
            stacks[x_key].append(item)
        
        stack_info = []
        for x_key, stack_items in stacks.items():
            max_h = max(i.z + i.h for i in stack_items)
            width = max(i.w for i in stack_items)
            stack_info.append({'x': x_key, 'items': stack_items, 'h': max_h, 'w': width})
        
        if len(stack_info) < 3: continue

        # 3. 높이 기준 정렬 후 Mound Sort (낮음 - 높음 - 낮음)
        sorted_by_h = sorted(stack_info, key=lambda s: s['h']) 
        reordered_stacks = [None] * len(stack_info)
        l_idx, r_idx = 0, len(stack_info) - 1
        
        for i, s in enumerate(sorted_by_h):
            if i % 2 == 0: reordered_stacks[l_idx] = s; l_idx += 1
            else: reordered_stacks[r_idx] = s; r_idx -= 1
        
        # 4. 좌표 갱신 (왼쪽 정렬 유지하면서 순서만 바꿈)
        current_min_x = min(s['x'] for s in stack_info)
        current_x = current_min_x
        for s in reordered_stacks:
            if s is None: continue
            for item in s['items']:
                item.x = current_x # X 좌표 이동
            current_x += s['w']

# [기능추가 3] 전체 덩어리 중앙 정렬 (마무리)
def recenter_load_final(truck):
    if not truck.items: return
    
    min_x = min(i.x for i in truck.items)
    max_x = max(i.x + i.w for i in truck.items)
    load_width = max_x - min_x
    target_x = (truck.w - load_width) / 2
    shift = target_x - min_x
    
    for item in truck.items:
        item.x += shift

def run_optimization(all_items, use_gap=True, limit_height=1300, limit_level=True):
    MARGIN_LENGTH = 200 

    def get_hybrid_sorted_items(items_to_sort):
        return sorted(items_to_sort, key=lambda x: (1 if x.l >= 2200 else 0, x.w, x.l, x.weight), reverse=True)

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
        for k, g in groupby(primary_sorted, key=lambda x: (x.w, x.h, x.l)):
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
                if total_rem_weight > 5000 and spec['weight'] < 2000: continue
                candidates.append((t_name, spec))

            rem = get_balanced_sorted_items(rem)

            for t_name, spec in candidates:
                t = Truck(t_name, spec['w'], limit_height, spec['l'] - MARGIN_LENGTH, spec['weight'], spec['cost'])
                count = 0; w_sum = 0
                for item in rem:
                    new_box = Box(item.name, item.w, item.h, item.l, item.weight)
                    new_box.is_heavy = item.is_heavy
                    if t.put_item(new_box, use_gap=use_gap, limit_level=limit_level):
                        count += 1; w_sum += item.weight
                
                if count > 0:
                    eff = w_sum / spec['cost']
                    if w_sum / spec['weight'] > 0.8: eff *= 1.2
                    if count == len(rem): eff = float('inf')
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
    
    sorted_all_items = get_balanced_sorted_items(all_items)
    
    start_candidates = ["11톤", "15톤", "18톤"] if sum(i.weight for i in all_items) > 10000 else TRUCK_DB.keys()

    for start_truck_name in start_candidates:
        if start_truck_name not in TRUCK_DB: continue
        spec = TRUCK_DB[start_truck_name]
        
        start_truck = Truck(start_truck_name, spec['w'], limit_height, spec['l'] - MARGIN_LENGTH, spec['weight'], spec['cost'])
        for item in sorted_all_items:
             new_box = Box(item.name, item.w, item.h, item.l, item.weight)
             new_box.is_heavy = item.is_heavy
             start_truck.put_item(new_box, use_gap=use_gap, limit_level=limit_level)
        
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
            t.name = f"[{idx+1}] {t.name}"
            
            # [최종 후처리 프로세스]
            # 1. 무게 4분면 밸런스 최적화 (재배치)
            optimize_load_balance(t, use_gap, limit_level, limit_height)
            
            # 2. 피라미드 스태킹 (안전 밴딩)
            apply_pyramid_stacking(t)
            
            # 3. 전체 덩어리 중앙 정렬 (마무리)
            recenter_load_final(t)
            
            final_trucks.append(t)
            
    return final_trucks

# ==========================================
# 4. 시각화 (기존 유지)
# ==========================================
def draw_truck_3d(truck, limit_h_val):
    fig = go.Figure()
    original_name = truck.name.split('] ')[1].split(' (')[0] if ']' in truck.name else truck.name
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    
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
            lighting=dict(ambient=0.9, diffuse=0.5, specular=0.1, roughness=0.5), hoverinfo=hover_info, hovertext=hovertext
        ))
        if line_color:
            xe=[x,x+w,x+w,x,x,None, x,x+w,x+w,x,x,None, x,x,None, x+w,x+w,None, x+w,x+w,None, x,x]
            ye=[y,y,y+l,y+l,y,None, y,y,y+l,y+l,y,None, y,y,None, y+l,y+l,None, y+l,y+l]
            ze=[z,z,z,z,z,None, z+h,z+h,z+h,z+h,z+h,None, z,z+h,None, z,z+h,None, z,z+h,None, z,z+h]
            fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=line_color, width=3), showlegend=False, hoverinfo='skip'))

    ch_h = 100; f_tk = 40; bmp_h = 140
    COLOR_FRAME = '#555555'; COLOR_FRAME_LINE = '#333333'
    draw_cube(0, 0, -ch_h, W, L, ch_h, '#AAAAAA', COLOR_FRAME)
    draw_cube(-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(W-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE)
    draw_cube(-f_tk/2, 0, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(W-f_tk/2, 0, -ch_h, f_tk, f_tk, Real_H+ch_h+20, COLOR_FRAME, COLOR_FRAME_LINE) 
    draw_cube(0, 0, 0, W, L, Real_H, '#EEF5FF', '#666666', opacity=0.1)

    OFFSET = 800
    fig.add_trace(go.Scatter3d(x=[-OFFSET, L, 0], y=[-OFFSET, L, limit_h_val], z=[0, 0, limit_h_val], mode='text', text=[f"높이제한: {limit_h_val}"], textfont=dict(color='red', size=12), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[limit_h_val]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))

    annotations = []
    for item in truck.items:
        col = '#FF6B6B' if item.is_heavy else '#FAD7A0'
        hover_text = f"<b>📦 {item.name}</b><br>규격: {int(item.w)}x{int(item.l)}x{int(item.h)}<br>중량: {int(item.weight):,}kg<br>적재단수: {item.level}단"
        draw_cube(item.x, item.y, item.z, item.w, item.l, item.h, col, '#000000', hovertext=hover_text)
        annotations.append(dict(x=item.x + item.w/2, y=item.y + item.l/2, z=item.z + item.h/2, text=f"<b>{item.name}</b>", xanchor="center", yanchor="middle", showarrow=False, font=dict(color="black", size=11), bgcolor="rgba(255,255,255,0.5)"))

    eye = dict(x=-1.8, y=-1.8, z=1.2); up = dict(x=0, y=0, z=1)
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='white', camera=dict(eye=eye, up=up), annotations=annotations), margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4()))
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 출하박스 적재 최적화 시스템")
st.markdown("✅ **규칙 : 비용최소화 | 회전금지 | 길이우선 적재 | 상위 10% 중량박스 표시**")

with st.sidebar:
    st.header("⚙️ 적재 옵션 설정")
    st.info("비용이 비싸게 나온다면 '높이 제한'을 늘리고 '간격'을 해제해보세요.")
    
    limit_h_input = st.slider("적재 높이 제한 (mm)", min_value=1000, max_value=2500, value=1300, step=100, help="안전 적재는 1300mm, 화물차 윙바디 기준 풀 적재는 2300mm 추천")
    use_gap_option = st.checkbox("박스 간 30cm 간격 적용", value=True, help="체크 해제 시 박스를 빈틈없이 밀착 적재합니다.")
    limit_level_option = st.checkbox("최대 4단 적재 제한", value=True, help="해제 시 높이 제한 내에서 최대한 높게 쌓습니다.")

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, encoding='cp949')
        else: df = pd.read_excel(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("최적 배차 실행 (최소비용)", type="primary"):
            items, errors = load_data(df)
            if errors:
                with st.expander("⚠️ 데이터 로드 경고"):
                    for err in errors: st.write(err)
            
            st.session_state['run_result'] = items
            st.session_state['opts'] = {'h': limit_h_input, 'gap': use_gap_option, 'lvl': limit_level_option}
            st.rerun()
            
        if 'run_result' in st.session_state:
            items = st.session_state['run_result']
            opts = st.session_state.get('opts', {'h': 1300, 'gap': True, 'lvl': True})
            
            if items and len(items) > 0 and not hasattr(items[0], 'l'):
                del st.session_state['run_result']
                st.warning("데이터 형식이 업데이트되었습니다. 버튼을 다시 눌러주세요.")
                st.stop()

            if items:
                trucks = run_optimization(items, use_gap=opts['gap'], limit_height=opts['h'], limit_level=opts['lvl'])
                
                if trucks:
                    total_cost = sum(t.cost for t in trucks)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("총 배차 차량", f"{len(trucks)}대")
                    c2.metric("총 예상 운송비", f"{total_cost:,}원")
                    c3.metric("적재 옵션", f"H:{opts['h']}mm / Gap:{'On' if opts['gap'] else 'Off'}")
                    st.divider()

                    tabs = st.tabs([f"{t.name}" for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            t = trucks[i]
                            c_info, c_chart = st.columns([1, 3])
                            with c_info:
                                st.markdown(f"#### {t.name}")
                                
                                # 좌우 무게 밸런스 지표
                                mid_x = t.w / 2
                                mid_y = t.l / 2
                                # 4분면 계산
                                q1 = sum(b.weight for b in t.items if b.x+b.w/2 < mid_x and b.y+b.l/2 < mid_y)
                                q2 = sum(b.weight for b in t.items if b.x+b.w/2 >= mid_x and b.y+b.l/2 < mid_y)
                                q3 = sum(b.weight for b in t.items if b.x+b.w/2 < mid_x and b.y+b.l/2 >= mid_y)
                                q4 = sum(b.weight for b in t.items if b.x+b.w/2 >= mid_x and b.y+b.l/2 >= mid_y)
                                total = t.total_weight if t.total_weight > 0 else 1
                                
                                st.markdown("##### ⚖️ 무게 분포 (4분면 최적화)")
                                cc1, cc2 = st.columns(2)
                                with cc1: st.metric("앞-좌", f"{q1/total*100:.1f}%", f"{int(q1)}kg")
                                with cc2: st.metric("앞-우", f"{q2/total*100:.1f}%", f"{int(q2)}kg")
                                cc3, cc4 = st.columns(2)
                                with cc3: st.metric("뒤-좌", f"{q3/total*100:.1f}%", f"{int(q3)}kg")
                                with cc4: st.metric("뒤-우", f"{q4/total*100:.1f}%", f"{int(q4)}kg")
                                
                                st.divider()
                                st.dataframe([{"박스명": b.name, "단수": f"{b.level}단"} for b in t.items], hide_index=True, use_container_width=True)
                            with c_chart:
                                st.plotly_chart(draw_truck_3d(t, opts['h']), use_container_width=True)
                else: st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e: st.error(f"오류 발생: {e}")
