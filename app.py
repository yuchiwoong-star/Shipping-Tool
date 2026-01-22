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
# 4. 시각화 (레퍼런스 이미지 완벽 재현)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    original_name = truck.name.split(' (')[0]
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # --- 조명 및 재질 설정 ---
    LIGHTING_METAL = dict(ambient=0.3, diffuse=0.6, specular=0.9, roughness=0.1) # 강한 반사광
    LIGHTING_RUBBER = dict(ambient=0.1, diffuse=0.1, specular=0.0, roughness=1.0) # 무광
    LIGHTING_PLASTIC = dict(ambient=0.6, diffuse=0.6, specular=0.4, roughness=0.3) # 일반 플라스틱
    
    COLOR_FRAME = '#1A1A1A' # 아주 진한 회색 (거의 검정)
    COLOR_WHEEL_RIM = '#888888' # 은색 림
    COLOR_TIRE = '#0A0A0A' # 검은색 타이어

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

    # --- 도우미 함수: 사실적인 바퀴 그리기 ---
    def draw_realistic_wheel(cx, cy, cz, is_front=False):
        r_tire = 300; w_tire = 160
        r_rim = 180; w_rim = 140
        r_hub = 80; w_hub = 60
        steps = 36
        theta = np.linspace(0, 2*np.pi, steps)
        
        # 1. 타이어 (도넛 모양)
        def draw_torus(r_major, r_minor, w, color, lighting):
            phi = np.linspace(0, 2*np.pi, steps)
            x_tor, y_tor, z_tor = [], [], []
            for t in theta:
                for p in phi:
                    x_tor.append(cx + (r_major + r_minor*np.cos(p))*np.cos(0) - w/2 + w*0.5*(1+np.sin(p))) # 약간 둥글게
                    y_tor.append(cy + (r_major + r_minor*np.cos(p))*np.sin(t))
                    z_tor.append(cz + r_minor*np.sin(p))
            fig.add_trace(go.Mesh3d(x=x_tor, y=y_tor, z=z_tor, alphahull=0, color=color, flatshading=False, lighting=lighting, hoverinfo='skip'))

        # 타이어 몸체 그리기 (간소화된 방식)
        xt, yt, zt = [], [], []
        for t in theta:
            xt.extend([cx-w_tire/2, cx+w_tire/2])
            yt.extend([cy+r_tire*np.cos(t), cy+r_tire*np.cos(t)])
            zt.extend([cz+r_tire*np.sin(t), cz+r_tire*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=xt, y=yt, z=zt, alphahull=0, color=COLOR_TIRE, flatshading=True, lighting=LIGHTING_RUBBER, hoverinfo='skip'))
        
        # 타이어 옆면 막기
        y_side = [cy+r_tire*np.cos(t) for t in theta] + [cy]
        z_side = [cz+r_tire*np.sin(t) for t in theta] + [cz]
        fig.add_trace(go.Mesh3d(x=[cx-w_tire/2]*len(y_side), y=y_side, z=z_side, color=COLOR_TIRE, flatshading=True, lighting=LIGHTING_RUBBER, hoverinfo='skip'))
        fig.add_trace(go.Mesh3d(x=[cx+w_tire/2]*len(y_side), y=y_side, z=z_side, color=COLOR_TIRE, flatshading=True, lighting=LIGHTING_RUBBER, hoverinfo='skip'))


        # 2. 휠 림 (안쪽으로 들어간 금속 원통)
        x_rim_start = cx + w_tire/2 - w_rim if cx < W/2 else cx - w_tire/2
        xr, yr, zr = [], [], []
        for t in theta:
            xr.extend([x_rim_start, x_rim_start+w_rim])
            yr.extend([cy+r_rim*np.cos(t), cy+r_rim*np.cos(t)])
            zr.extend([cz+r_rim*np.sin(t), cz+r_rim*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=xr, y=yr, z=zr, alphahull=0, color=COLOR_WHEEL_RIM, flatshading=True, lighting=LIGHTING_METAL, hoverinfo='skip'))
        
        # 휠 디스크 (바깥쪽 면)
        yd = [cy+r_rim*np.cos(t) for t in theta] + [cy]
        zd = [cz+r_rim*np.sin(t) for t in theta] + [cz]
        xd = [x_rim_start+w_rim]*len(yd)
        fig.add_trace(go.Mesh3d(x=xd, y=yd, z=zd, color=COLOR_WHEEL_RIM, flatshading=True, lighting=LIGHTING_METAL, hoverinfo='skip'))

        # 3. 허브 (중앙 돌출) - 앞바퀴만
        if is_front:
            xh, yh, zh = [], [], []
            for t in theta:
                xh.extend([x_rim_start+w_rim, x_rim_start+w_rim+w_hub])
                yh.extend([cy+r_hub*np.cos(t), cy+r_hub*np.cos(t)])
                zh.extend([cz+r_hub*np.sin(t), cz+r_hub*np.sin(t)])
            fig.add_trace(go.Mesh3d(x=xh, y=yh, z=zh, alphahull=0, color=COLOR_WHEEL_RIM, flatshading=True, lighting=LIGHTING_METAL, hoverinfo='skip'))
            yh_cap = [cy+r_hub*np.cos(t) for t in theta] + [cy]
            zh_cap = [cz+r_hub*np.sin(t) for t in theta] + [cz]
            xh_cap = [x_rim_start+w_rim+w_hub]*len(yh_cap)
            fig.add_trace(go.Mesh3d(x=xh_cap, y=yh_cap, z=zh_cap, color=COLOR_WHEEL_RIM, flatshading=True, lighting=LIGHTING_METAL, hoverinfo='skip'))

    # --- 도우미 함수: 3구 후미등 그리기 ---
    def draw_tail_lights(x, y, z):
        # 브레이크등(빨강) / 방향지시등(주황) / 후진등(흰색)
        draw_cube(x, y, z, 100, 20, 40, '#FF0000', None, lighting=LIGHTING_PLASTIC)
        draw_cube(x+105, y, z, 80, 20, 40, '#FF7F00', None, lighting=LIGHTING_PLASTIC)
        draw_cube(x+190, y, z, 60, 20, 40, '#FFFFFF', None, lighting=LIGHTING_PLASTIC)

    # 1. 트럭 섀시 및 프레임 (금속 질감 적용)
    chassis_h = 150
    draw_cube(0, 0, -chassis_h, W, L, chassis_h, '#2A2A2A', None, lighting=LIGHTING_METAL) # 메인 섀시
    draw_cube(-80, 0, -chassis_h, 80, L, 120, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 사이드 스커트 L
    draw_cube(W, 0, -chassis_h, 80, L, 120, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 사이드 스커트 R

    f_tk = 100; bumper_h = 150
    draw_cube(-f_tk/2, L-f_tk, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 기둥 L
    draw_cube(W-f_tk/2, L-f_tk, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 기둥 R
    draw_cube(-f_tk/2, L-f_tk, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 상단바
    draw_cube(-f_tk/2, L, -chassis_h-bumper_h, W+f_tk, 50, bumper_h, COLOR_FRAME, None, lighting=LIGHTING_METAL) # 범퍼

    # 2. 3구 후미등 배치
    draw_tail_lights(80, L+55, -chassis_h-bumper_h+30) # 좌측
    draw_tail_lights(W-330, L+55, -chassis_h-bumper_h+30) # 우측

    # 3. 투명 컨테이너 (반사광 추가)
    draw_cube(0, 0, 0, W, L, Real_H, '#EEF5FF', '#888888', opacity=0.15, lighting=LIGHTING_PLASTIC)

    # 4. 사실적인 바퀴 배치
    wheel_z = -chassis_h - 280
    draw_realistic_wheel(-80, L*0.15, wheel_z, True); draw_realistic_wheel(W+80, L*0.15, wheel_z, True)
    draw_realistic_wheel(-80, L*0.28, wheel_z); draw_realistic_wheel(W+80, L*0.28, wheel_z)
    draw_realistic_wheel(-80, L*0.75, wheel_z); draw_realistic_wheel(W+80, L*0.75, wheel_z)
    draw_realistic_wheel(-80, L*0.88, wheel_z); draw_realistic_wheel(W+80, L*0.88, wheel_z)

    # 5. [화물 박스] (조명 효과 적용)
    annotations = []
    for item in truck.items:
        color = '#FF6B6B' if getattr(item, 'is_heavy', False) else '#FAD7A0'
        draw_cube(item.x, item.y, item.z, item.w, item.d, item.h, color, '#333333', lighting=LIGHTING_PLASTIC)
        
        fig.add_trace(go.Mesh3d(
            x=[item.x, item.x+item.w, item.x+item.w, item.x, item.x, item.x+item.w, item.x+item.w, item.x],
            y=[item.y, item.y, item.y+item.d, item.y+item.d, item.y, item.y, item.y+item.d, item.y+item.d],
            z=[item.z, item.z, item.z, item.z, item.z+item.h, item.z+item.h, item.z+item.h, item.z+item.h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=0.0, hoverinfo='text',
            hovertext=f"<b>📦 {item.name}</b><br>규격: {int(item.w)}x{int(item.d)}x{int(item.h)}<br>중량: {int(item.weight):,}kg"
        ))

        annotations.append(dict(
            x=item.x + item.w/2, y=item.y + item.d/2, z=item.z + item.h/2,
            text=f"<b>{item.name}</b>",
            xanchor="center", yanchor="middle", showarrow=False,
            font=dict(color="black", size=10), bgcolor="rgba(255,255,255,0.4)"
        ))

    # 6. [제한선]
    fig.add_trace(go.Scatter3d(
        x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[LIMIT_H]*5,
        mode='lines', line=dict(color='red', width=3, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=[W/2], y=[L/2], z=[LIMIT_H], mode='text', text=['<b>제한 1.3m</b>'],
        textfont=dict(color='red', size=10), showlegend=False, hoverinfo='skip'
    ))

    # 7. [카메라]
    if camera_view == "top": eye = dict(x=0, y=0.01, z=2.5); up = dict(x=0, y=1, z=0)
    elif camera_view == "side": eye = dict(x=2.5, y=0, z=0.2); up = dict(x=0, y=0, z=1)
    else: eye = dict(x=2.0, y=-2.0, z=1.5); up = dict(x=0, y=0, z=1) # ISO

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white', camera=dict(eye=eye, up=up), annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4())
    )
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
            df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}")
        
        if '박스번호' in df_display.columns:
            df_display['박스번호'] = df_display['박스번호'].astype(str)

        styler = df_display.style.set_properties(**{'text-align': 'center'})
        styler.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ])
        
        st.dataframe(styler, use_container_width=True, hide_index=True, height=250)

        st.subheader("🚛 차량 기준 정보")
        
        truck_rows = []
        for name, spec in TRUCK_DB.items():
            truck_rows.append({
                "차량": name,
                "적재폭 (mm)": spec['w'],
                "적재길이 (mm)": spec['l'],
                "허용하중 (kg)": spec['weight'],
                "운송단가": spec['cost']
            })
        df_truck = pd.DataFrame(truck_rows)
        
        format_cols_truck = ['적재폭 (mm)', '적재길이 (mm)', '허용하중 (kg)', '운송단가']
        for col in format_cols_truck:
             df_truck[col] = df_truck[col].apply(lambda x: f"{x:,.0f}")
        
        st_truck = df_truck.style.set_properties(**{'text-align': 'center'})
        st_truck.set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]}
        ])

        st.dataframe(st_truck, use_container_width=True, hide_index=True)

        if st.button("최적 배차 실행 (최소비용)", type="primary"):
            st.session_state['run_result'] = load_data(df)
            
        if 'run_result' in st.session_state:
            items = st.session_state['run_result']
            if not items: st.error("데이터 변환 실패.")
            else:
                trucks = run_optimization(items)
                if trucks:
                    t_names = [t.name.split(' ')[0] for t in trucks]
                    from collections import Counter
                    cnt = Counter(t_names)
                    total_cost = sum(t.cost for t in trucks)

                    summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                    st.success(f"✅ 분석 완료: 총 {len(trucks)}대 ({summary}) | 예상 총 운송비: {total_cost:,}원")
                    
                    c1, c2, c3, _ = st.columns([1, 1, 1, 5])
                    with c1: 
                        if st.button("↗️ 쿼터뷰"): st.session_state['view_mode'] = 'iso'
                    with c2: 
                        if st.button("⬆️ 탑뷰"): st.session_state['view_mode'] = 'top'
                    with c3: 
                        if st.button("➡️ 사이드뷰"): st.session_state['view_mode'] = 'side'
                    
                    tabs = st.tabs([t.name for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            col1, col2 = st.columns([1, 4])
                            t = trucks[i]
                            with col1:
                                st.markdown(f"### **{t.name}**")
                                st.write(f"- 박스: **{len(t.items)}개**")
                                st.write(f"- 중량: **{t.total_weight:,} kg**")
                                st.write(f"- 비용: **{t.cost:,} 원**")
                                with st.expander("목록 보기"): st.write(", ".join([b.name for b in t.items]))
                            with col2:
                                st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
                else: st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e: st.error(f"오류 발생: {e}")
