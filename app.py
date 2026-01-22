import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid

# ==========================================
# 1. 커스텀 물리 엔진 (핵심 로직 유지)
# ==========================================
class Box:
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
    
    @property
    def volume(self):
        return self.w * self.h * self.d

class Truck:
    def __init__(self, name, w, h, d, max_weight, cost):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.cost = int(cost)
        self.items = []
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item):
        fit = False
        if self.total_weight + item.weight > self.max_weight:
            return False
        
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))
        
        for p in self.pivots:
            px, py, pz = p
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue
            if self._check_collision(item, px, py, pz):
                continue
            if not self._check_support(item, px, py, pz):
                continue
            
            item.x, item.y, item.z = px, py, pz
            self.items.append(item)
            self.total_weight += item.weight
            fit = True
            break
        
        if fit:
            self.pivots.append([item.x + item.w, item.y, item.z])
            self.pivots.append([item.x, item.y + item.d, item.z])
            self.pivots.append([item.x, item.y, item.z + item.h])
        return fit

    def _check_collision(self, item, x, y, z):
        for exist in self.items:
            if (x < exist.x + exist.w and x + item.w > exist.x and
                y < exist.y + exist.d and y + item.d > exist.y and
                z < exist.z + exist.h and z + item.h > exist.z):
                return True
        return False

    def _check_support(self, item, x, y, z):
        if z <= 0.001: return True
        support_area = 0.0
        item_area = item.w * item.d
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= item_area * 0.8

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="출하박스 적재 최적화 시스템")

TRUCK_DB = {
    "1톤":   {"w": 1600, "real_h": 2350, "l": 2800,  "weight": 1490,  "cost": 78000},
    "2.5톤": {"w": 1900, "real_h": 2350, "l": 4200,  "weight": 3490,  "cost": 110000},
    "5톤":   {"w": 2100, "real_h": 2350, "l": 6200,  "weight": 6900,  "cost": 133000},
    "8톤":   {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 9490,  "cost": 153000},
    "11톤":  {"w": 2350, "real_h": 2350, "l": 9200,  "weight": 14900, "cost": 188000},
    "15톤":  {"w": 2350, "real_h": 2350, "l": 10200, "weight": 16900, "cost": 211000},
    "18톤":  {"w": 2350, "real_h": 2350, "l": 10200, "weight": 20900, "cost": 242000},
    "22톤":  {"w": 2350, "real_h": 2350, "l": 10200, "weight": 26000, "cost": 308000},
}

def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            sorted_weights = sorted(weights, reverse=True)
            top_n = math.ceil(len(weights) * 0.1)
            cutoff_index = max(0, top_n - 1)
            heavy_threshold = sorted_weights[cutoff_index]
        else:
            heavy_threshold = float('inf')
    except:
        heavy_threshold = float('inf')

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            box = Box(name, w, h, l, weight)
            if weight >= heavy_threshold and weight > 0:
                box.is_heavy = True
            else:
                box.is_heavy = False
            items.append(box)
        except:
            continue
    return items

def run_optimization(all_items):
    def solve_remaining_greedy(current_items):
        used = []
        rem = current_items[:]
        while rem:
            candidates = []
            for t_name in TRUCK_DB:
                spec = TRUCK_DB[t_name]
                t = Truck(t_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
                test_i = sorted(rem, key=lambda x: x.volume, reverse=True)
                count = 0
                w_sum = 0
                for item in test_i:
                    new_box = Box(item.name, item.w, item.h, item.d, item.weight)
                    new_box.is_heavy = getattr(item, 'is_heavy', False)
                    if t.put_item(new_box):
                        count += 1; w_sum += item.weight
                if count > 0:
                    candidates.append({
                        'truck': t,
                        'is_all': (count == len(rem)),
                        'eff': w_sum / spec['cost'],
                        'cost': spec['cost']
                    })
            if not candidates: break
            fits_all = [c for c in candidates if c['is_all']]
            if fits_all:
                best_t = sorted(fits_all, key=lambda x: x['cost'])[0]['truck']
            else:
                best_t = sorted(candidates, key=lambda x: x['eff'], reverse=True)[0]['truck']
            used.append(best_t)
            packed_n = [i.name for i in best_t.items]
            rem = [i for i in rem if i.name not in packed_n]
        return used

    best_solution = None
    min_total_cost = float('inf')
    
    for start_truck_name in TRUCK_DB:
        spec = TRUCK_DB[start_truck_name]
        start_truck = Truck(start_truck_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
        items_sorted = sorted(all_items, key=lambda x: x.volume, reverse=True)
        for item in items_sorted:
             new_box = Box(item.name, item.w, item.h, item.d, item.weight)
             new_box.is_heavy = getattr(item, 'is_heavy', False)
             start_truck.put_item(new_box)
        
        if not start_truck.items: continue

        packed_names = [i.name for i in start_truck.items]
        remaining = [i for i in all_items if i.name not in packed_names]
        
        current_solution = [start_truck]
        if remaining:
            sub_solution = solve_remaining_greedy(remaining)
            current_solution.extend(sub_solution)
        
        total_packed_count = sum([len(t.items) for t in current_solution])
        if total_packed_count < len(all_items):
            continue

        current_total_cost = sum(t.cost for t in current_solution)
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            best_solution = current_solution
    
    final_trucks = []
    if best_solution:
        for idx, t in enumerate(best_solution):
            t.name = f"{t.name} (No.{idx+1})"
            final_trucks.append(t)
    return final_trucks

# ==========================================
# 4. 시각화 (두 번째 사진 완벽 재현 - 설명된 수정사항 적용)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    original_name = truck.name.split(' (')[0]
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # 조명 및 재질 설정
    LIGHTING_METAL = dict(ambient=0.4, diffuse=0.5, specular=0.8, roughness=0.2)
    LIGHTING_RUBBER = dict(ambient=0.2, diffuse=0.2, specular=0.1, roughness=0.8)
    LIGHTING_PLASTIC = dict(ambient=0.6, diffuse=0.6, specular=0.4, roughness=0.4)
    
    COLOR_FRAME = '#222222'
    COLOR_CHASSIS = '#444444'
    COLOR_TIRE = '#111111'
    COLOR_RIM = '#888888'

    # --- 도우미 함수: 육면체 그리기 ---
    def draw_cube(x, y, z, w, l, h, face_color, line_color=None, opacity=1.0, lighting=LIGHTING_PLASTIC):
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+l, y+l, y, y, y+l, y+l],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=face_color, opacity=opacity, flatshading=True, 
            lighting=lighting, hoverinfo='skip'
        ))
        if line_color:
            xe=[x,x+w,x+w,x,x,None, x,x+w,x+w,x,x,None, x,x,None, x+w,x+w,None, x+w,x+w,None, x,x]
            ye=[y,y,y+l,y+l,y,None, y,y,y+l,y+l,y,None, y,y,None, y,y,None, y+l,y+l,None, y+l,y+l]
            ze=[z,z,z,z,z,None, z+h,z+h,z+h,z+h,z+h,None, z,z+h,None, z,z+h,None, z,z+h,None, z,z+h]
            fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=line_color, width=2), showlegend=False, hoverinfo='skip'))

    # --- 도우미 함수: 펜더(타이어 가드) 그리기 ---
    def draw_fender(cx, cy, cz, r, w, color):
        theta = np.linspace(0, np.pi, 30) # 반원
        x_out = cx + r * np.cos(theta); z_out = cz + r * np.sin(theta)
        x_in = cx + (r-10) * np.cos(theta); z_in = cz + (r-10) * np.sin(theta)
        
        # 펜더를 얇은 큐브들의 집합으로 근사
        for i in range(len(theta)-1):
            x1, z1 = x_out[i], z_out[i]; x2, z2 = x_out[i+1], z_out[i+1]
            draw_cube(x1, cy-w/2, z1, x2-x1, w, 5, color, lighting=LIGHTING_METAL)

    # --- 도우미 함수: 디테일한 타이어 그리기 ---
    def draw_detailed_tire(cx, cy, cz):
        r_tire = 280; w_tire = 160; r_rim = 170
        # 타이어 (검은색 고무)
        theta = np.linspace(0, 2*np.pi, 36)
        xt, yt, zt = [], [], []
        for t in theta:
            xt.extend([cx-w_tire/2, cx+w_tire/2])
            yt.extend([cy+r_tire*np.cos(t), cy+r_tire*np.cos(t)])
            zt.extend([cz+r_tire*np.sin(t), cz+r_tire*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=xt, y=yt, z=zt, alphahull=0, color=COLOR_TIRE, flatshading=True, lighting=LIGHTING_RUBBER, hoverinfo='skip'))
        
        # 옆면 막기
        y_side = [cy+r_tire*np.cos(t) for t in theta] + [cy]
        z_side = [cz+r_tire*np.sin(t) for t in theta] + [cz]
        fig.add_trace(go.Mesh3d(x=[cx-w_tire/2]*len(y_side), y=y_side, z=z_side, color=COLOR_TIRE, flatshading=True, lighting=LIGHTING_RUBBER, hoverinfo='skip'))
        fig.add_trace(go.Mesh3d(x=[cx+w_tire/2]*len(y_side), y=y_side, z=z_side, color=COLOR_TIRE, flatshading=True, lighting=LIGHTING_RUBBER, hoverinfo='skip'))

        # 림 (은색 금속)
        y_rim = [cy+r_rim*np.cos(t) for t in theta] + [cy]
        z_rim = [cz+r_rim*np.sin(t) for t in theta] + [cz]
        x_rim_pos = cx + w_tire/2 + 2 if cx > W/2 else cx - w_tire/2 - 2
        fig.add_trace(go.Mesh3d(x=[x_rim_pos]*len(y_rim), y=y_rim, z=z_rim, color=COLOR_RIM, flatshading=True, lighting=LIGHTING_METAL, hoverinfo='skip'))


    # 1. 트럭 베이스 및 섀시
    chassis_h = 150
    draw_cube(0, 0, -chassis_h, W, L, chassis_h, COLOR_CHASSIS, None, lighting=LIGHTING_METAL)
    draw_cube(-60, 0, -chassis_h, 60, L, 120, COLOR_CHASSIS, None, lighting=LIGHTING_METAL)
    draw_cube(W, 0, -chassis_h, 60, L, 120, COLOR_CHASSIS, None, lighting=LIGHTING_METAL)

    # 2. 앞/뒤 프레임 (R처리된 형태)
    f_tk = 100
    # draw_rounded_frame(0, 0, -chassis_h, W, Real_H+chassis_h+20, f_tk, 50, COLOR_FRAME, LIGHTING_METAL) # 앞
    # draw_rounded_frame(0, L-f_tk, -chassis_h, W, Real_H+chassis_h+20, f_tk, 50, COLOR_FRAME, LIGHTING_METAL) # 뒤
    # (R처리 구현이 복잡하여, 기존 큐브 방식으로 하되 모서리를 다듬는 형태로 근사)
    draw_cube(-f_tk/2, -f_tk/2, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 앞좌측 기둥
    draw_cube(W-f_tk/2, -f_tk/2, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 앞우측 기둥
    draw_cube(-f_tk/2, -f_tk/2, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 앞상단바
    draw_cube(-f_tk/2, L-f_tk/2, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 뒤좌측 기둥
    draw_cube(W-f_tk/2, L-f_tk/2, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 뒤우측 기둥
    draw_cube(-f_tk/2, L-f_tk/2, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 뒤상단바


    # 3. 후미 (범퍼, 번호판, 3구 원형 후미등)
    bumper_h = 180
    draw_cube(-f_tk/2, L, -chassis_h-bumper_h, W+f_tk, 30, bumper_h, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 메인 범퍼 ('ㅛ'자 가로바)
    # 번호판 (중앙 네모)
    draw_cube(W/2 - 100, L+30, -chassis_h-bumper_h/2-30, 200, 5, 60, '#FFFFFF', '#000000', lighting=LIGHTING_PLASTIC)
    # 3구 원형 후미등 (좌우)
    def draw_round_tail_light(x, y, z, color):
        # (구를 사용하여 원형 등 표현, 여기서는 Scatter3d 마커로 대체)
        fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=dict(color=color, size=15, symbol='circle'), showlegend=False, hoverinfo='skip'))
    
    light_y = L + 35; light_z = -chassis_h - bumper_h/2
    # 좌측 (빨강-주황-흰색)
    draw_round_tail_light(100, light_y, light_z, '#FF0000'); draw_round_tail_light(150, light_y, light_z, '#FF7F00'); draw_round_tail_light(200, light_y, light_z, '#FFFFFF')
    # 우측 (흰색-주황-빨강)
    draw_round_tail_light(W-200, light_y, light_z, '#FFFFFF'); draw_round_tail_light(W-150, light_y, light_z, '#FF7F00'); draw_round_tail_light(W-100, light_y, light_z, '#FF0000')


    # 4. 타이어 및 펜더 (가드)
    wheel_z = -chassis_h - 300
    fender_r = 340; fender_w = 180
    # 앞 2축
    draw_fender(-60, L*0.15, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(-60, L*0.15, wheel_z)
    draw_fender(W+60, L*0.15, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(W+60, L*0.15, wheel_z)
    draw_fender(-60, L*0.28, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(-60, L*0.28, wheel_z)
    draw_fender(W+60, L*0.28, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(W+60, L*0.28, wheel_z)
    # 뒤 2축
    draw_fender(-60, L*0.75, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(-60, L*0.75, wheel_z)
    draw_fender(W+60, L*0.75, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(W+60, L*0.75, wheel_z)
    draw_fender(-60, L*0.88, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(-60, L*0.88, wheel_z)
    draw_fender(W+60, L*0.88, wheel_z, fender_r, fender_w, COLOR_CHASSIS); draw_detailed_tire(W+60, L*0.88, wheel_z)


    # 5. 투명 컨테이너 및 화물
    draw_cube(0, 0, 0, W, L, Real_H, '#EEF5FF', '#888888', opacity=0.1, lighting=LIGHTING_PLASTIC)
    annotations = []
    for item in truck.items:
        color = '#FF6B6B' if getattr(item, 'is_heavy', False) else '#FAD7A0'
        draw_cube(item.x, item.y, item.z, item.w, item.d, item.h, color, '#333333', lighting=LIGHTING_PLASTIC)
        fig.add_trace(go.Mesh3d(x=[item.x, item.x+item.w, item.x+item.w, item.x, item.x, item.x+item.w, item.x+item.w, item.x], y=[item.y, item.y, item.y+item.d, item.y+item.d, item.y, item.y, item.y+item.d, item.y+item.d], z=[item.z, item.z, item.z, item.z, item.z+item.h, item.z+item.h, item.z+item.h, item.z+item.h], i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6], opacity=0.0, hoverinfo='text', hovertext=f"<b>📦 {item.name}</b><br>규격: {int(item.w)}x{int(item.d)}x{int(item.h)}<br>중량: {int(item.weight):,}kg"))
        annotations.append(dict(x=item.x + item.w/2, y=item.y + item.d/2, z=item.z + item.h/2, text=f"<b>{item.name}</b>", xanchor="center", yanchor="middle", showarrow=False, font=dict(color="black", size=10), bgcolor="rgba(255,255,255,0.4)"))

    # 6. 제한선 및 치수
    fig.add_trace(go.Scatter3d(x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=3, dash='dash'), showlegend=False, hoverinfo='skip'))
    
    OFFSET = 800
    def draw_arrow_dim(p1, p2, text, color='black'):
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color=color, width=3), showlegend=False, hoverinfo='skip'))
        vec = np.array(p2) - np.array(p1); length = np.linalg.norm(vec)
        if length > 0:
            u, v, w = vec / length
            fig.add_trace(go.Cone(x=[p2[0]], y=[p2[1]], z=[p2[2]], u=[u], v=[v], w=[w], sizemode="absolute", sizeref=150, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]], hoverinfo='skip'))
            fig.add_trace(go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[-u], v=[-v], w=[-w], sizemode="absolute", sizeref=150, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]], hoverinfo='skip'))
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode='text', text=[f"<b>{text}</b>"], textfont=dict(color=color, size=14, family="Arial Black"), showlegend=False, hoverinfo='skip'))

    draw_arrow_dim([0, -OFFSET, 0], [W, -OFFSET, 0], f"폭 : {int(W)}")
    draw_arrow_dim([-OFFSET, 0, 0], [-OFFSET, L, 0], f"길이 : {int(L)}")
    draw_arrow_dim([-OFFSET, L, 0], [-OFFSET, L, LIMIT_H], f"높이제한(최대4단) : {LIMIT_H}", color='red')

    # 7. 카메라
    if camera_view == "top": eye = dict(x=0, y=0.01, z=2.5); up = dict(x=0, y=1, z=0)
    elif camera_view == "side": eye = dict(x=2.5, y=0, z=0.2); up = dict(x=0, y=0, z=1)
    else: eye = dict(x=2.0, y=-2.0, z=1.5); up = dict(x=0, y=0, z=1)
    
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='white', camera=dict(eye=eye, up=up), annotations=annotations), margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4()))
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 출하박스 적재 최적화 시스템 (배차비용 최소화)")
st.caption("✅ 규칙 : 비용최적화 | 부피순 적재 | 회전금지 | 1.3m 제한 | 80% 지지충족 | 하중제한 준수 | 상위 10% 중량박스 빨간색 표시")
if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, encoding='cp949')
        else: df = pd.read_excel(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        
        df_display = df.copy()
        
        # 숫자 -> 문자열 변환 (왼쪽 정렬 유도)
        cols_to_format = [c for c in ['폭 (mm)', '높이 (mm)', '길이 (mm)', '중량 (kg)'] if c in df_display.columns]
        for col in cols_to_format:
            df_
