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

# (부피순/최적화 로직 등 기존 함수 유지)
def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            sorted_weights = sorted(weights, reverse=True)
            top_n = math.ceil(len(weights) * 0.1); cutoff_index = max(0, top_n - 1)
            heavy_threshold = sorted_weights[cutoff_index]
        else: heavy_threshold = float('inf')
    except: heavy_threshold = float('inf')
    for index, row in df.iterrows():
        try:
            box = Box(str(row['박스번호']), float(row['폭']), float(row['높이']), float(row['길이']), float(row['중량']))
            box.is_heavy = float(row['중량']) >= heavy_threshold and float(row['중량']) > 0
            items.append(box)
        except: continue
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
                count = 0; w_sum = 0
                for item in test_i:
                    new_box = Box(item.name, item.w, item.h, item.d, item.weight)
                    new_box.is_heavy = getattr(item, 'is_heavy', False)
                    if t.put_item(new_box): count += 1; w_sum += item.weight
                if count > 0: candidates.append({'truck': t, 'is_all': (count == len(rem)), 'eff': w_sum / spec['cost'], 'cost': spec['cost']})
            if not candidates: break
            fits_all = [c for c in candidates if c['is_all']]
            best_t = sorted(fits_all, key=lambda x: x['cost'])[0]['truck'] if fits_all else sorted(candidates, key=lambda x: x['eff'], reverse=True)[0]['truck']
            used.append(best_t); packed_n = [i.name for i in best_t.items]; rem = [i for i in rem if i.name not in packed_n]
        return used
    best_solution = None; min_total_cost = float('inf')
    for start_truck_name in TRUCK_DB:
        spec = TRUCK_DB[start_truck_name]
        start_truck = Truck(start_truck_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
        items_sorted = sorted(all_items, key=lambda x: x.volume, reverse=True)
        for item in items_sorted:
             new_box = Box(item.name, item.w, item.h, item.d, item.weight)
             new_box.is_heavy = getattr(item, 'is_heavy', False)
             start_truck.put_item(new_box)
        if not start_truck.items: continue
        packed_names = [i.name for i in start_truck.items]; remaining = [i for i in all_items if i.name not in packed_names]
        current_solution = [start_truck]
        if remaining: current_solution.extend(solve_remaining_greedy(remaining))
        total_packed_count = sum([len(t.items) for t in current_solution])
        if total_packed_count < len(all_items): continue
        current_total_cost = sum(t.cost for t in current_solution)
        if current_total_cost < min_total_cost: min_total_cost = current_total_cost; best_solution = current_solution
    final_trucks = []
    if best_solution:
        for idx, t in enumerate(best_solution): t.name = f"{t.name} (No.{idx+1})"; final_trucks.append(t)
    return final_trucks

# ==========================================
# 4. 시각화 (바퀴 올블랙 및 프레임 디테일)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    original_name = truck.name.split(' (')[0]
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    light_eff = dict(ambient=0.9, diffuse=0.5, specular=0.1, roughness=0.5)

    def draw_cube(x, y, z, w, l, h, face_color, line_color=None, opacity=1.0):
        fig.add_trace(go.Mesh3d(x=[x, x+w, x+w, x, x, x+w, x+w, x], y=[y, y, y+l, y+l, y, y, y+l, y+l], z=[z, z, z, z, z+h, z+h, z+h, z+h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color=face_color, opacity=opacity, flatshading=True, lighting=light_eff, hoverinfo='skip'))
        if line_color:
            xe=[x,x+w,x+w,x,x,None, x,x+w,x+w,x,x,None, x,x,None, x+w,x+w,None, x+w,x+w,None, x,x]
            ye=[y,y,y+l,y+l,y,None, y,y,y+l,y+l,y,None, y,y,None, y,y,None, y+l,y+l,None, y+l,y+l]
            ze=[z,z,z,z,z,None, z+h,z+h,z+h,z+h,z+h,None, z,z+h,None, z,z+h,None, z,z+h,None, z,z+h]
            fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=line_color, width=2.5), showlegend=False, hoverinfo='skip'))

    # 1. 바닥 및 사이드 프레임
    ch_h = 100
    draw_cube(0, 0, -ch_h, W, L, ch_h, '#D9D9D9', '#555555') # 메인 바닥
    draw_cube(-60, 0, -ch_h-40, 60, L, 50, '#333333') # 좌측 사이드 가드
    draw_cube(W, 0, -ch_h-40, 60, L, 50, '#333333') # 우측 사이드 가드

    # 2. 후면 프레임 & 등기구
    f_tk = 60; draw_cube(-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h, '#444444') 
    draw_cube(W-f_tk/2, L-f_tk, -ch_h, f_tk, f_tk, Real_H+ch_h, '#444444')
    draw_cube(-f_tk/2, L-f_tk, Real_H, W+f_tk, f_tk, f_tk, '#444444')
    draw_cube(-f_tk/2, L, -ch_h-80, W+f_tk, 40, 80, '#222222') # 범퍼
    draw_cube(100, L+40, -ch_h-60, 100, 10, 40, '#FF0000') # 후미등 L
    draw_cube(W-200, L+40, -ch_h-60, 100, 10, 40, '#FF0000') # 후미등 R

    # 3. 바퀴 (전체 검정 솔리드)
    def draw_all_black_wheel(cx, cy, cz):
        r = 300; w = 160; steps = 30
        theta = np.linspace(0, 2*np.pi, steps)
        # 트레드
        xt, yt, zt = [], [], []
        for t in theta: xt.extend([cx-w/2, cx+w/2]); yt.extend([cy+r*np.cos(t), cy+r*np.cos(t)]); zt.extend([cz+r*np.sin(t), cz+r*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=xt, y=yt, z=zt, alphahull=0, color='#111111', flatshading=True, lighting=light_eff, hoverinfo='skip'))
        # 측면 (안/밖 모두 짙은 검정으로 채움)
        y_side = [cy + r*np.cos(t) for t in theta] + [cy]; z_side = [cz + r*np.sin(t) for t in theta] + [cz]
        fig.add_trace(go.Mesh3d(x=[cx-w/2]*len(y_side), y=y_side, z=z_side, color='#000000', hoverinfo='skip'))
        fig.add_trace(go.Mesh3d(x=[cx+w/2]*len(y_side), y=y_side, z=z_side, color='#000000', hoverinfo='skip'))

    w_z = -ch_h - 260
    # 앞 2축 + 뒤 3축 배치
    for y in [L*0.12, L*0.25]: draw_all_black_wheel(-70, y, w_z); draw_all_black_wheel(W+70, y, w_z)
    for y in [L*0.75, L*0.86, L*0.97]: draw_all_black_wheel(-70, y, w_z); draw_all_black_wheel(W+70, y, w_z)

    # 4. 박스 및 제한선
    draw_cube(0, 0, 0, W, L, Real_H, '#EEF5FF', '#555555', opacity=0.1) # 컨테이너
    annotations = []
    for item in truck.items:
        draw_cube(item.x, item.y, item.z, item.w, item.d, item.h, '#FF6B6B' if item.is_heavy else '#FAD7A0', '#000000')
        annotations.append(dict(x=item.x + item.w/2, y=item.y + item.d/2, z=item.z + item.h/2, text=f"<b>{item.name}</b>", xanchor="center", yanchor="middle", showarrow=False, font=dict(color="black", size=10), bgcolor="rgba(255,255,255,0.4)"))
    fig.add_trace(go.Scatter3d(x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=3, dash='dash'), showlegend=False, hoverinfo='skip'))

    eye = dict(x=1.8, y=-1.8, z=1.0) if camera_view=="iso" else (dict(x=0, y=0.01, z=2.5) if camera_view=="top" else dict(x=2.5, y=0, z=0.2))
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='white', camera=dict(eye=eye, up=dict(x=0, y=0, z=1)), annotations=annotations), margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4()))
    return fig

# (메인 UI 부분 기존 코드 동일하게 유지)
st.title("📦 출하박스 적재 최적화 시스템 (배차비용 최소화)")
st.caption("✅ 규칙 : 비용최적화 | 부피순 적재 | 회전금지 | 1.3m 제한 | 80% 지지충족 | 하중제한 준수 | 상위 10% 중량박스 빨간색 표시")
if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'
uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='cp949') if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        df.columns = [c.strip() for c in df.columns]; st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        df_display = df.copy(); cols_to_format = [c for c in ['폭 (mm)', '높이 (mm)', '길이 (mm)', '중량 (kg)'] if c in df_display.columns]
        for col in cols_to_format: df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}")
        if '박스번호' in df_display.columns: df_display['박스번호'] = df_display['박스번호'].astype(str)
        st.dataframe(df_display.style.set_properties(**{'text-align': 'center'}), use_container_width=True, hide_index=True, height=250)
        st.subheader("🚛 차량 기준 정보")
        truck_rows = [{"차량": n, "적재폭 (mm)": s['w'], "적재길이 (mm)": s['l'], "허용하중 (kg)": s['weight'], "운송단가": s['cost']} for n, s in TRUCK_DB.items()]
        df_truck = pd.DataFrame(truck_rows)
        for col in ['적재폭 (mm)', '적재길이 (mm)', '허용하중 (kg)', '운송단가']: df_truck[col] = df_truck[col].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_truck.style.set_properties(**{'text-align': 'center'}), use_container_width=True, hide_index=True)
        if st.button("최적 배차 실행 (최소비용)", type="primary"): st.session_state['run_result'] = load_data(df)
        if 'run_result' in st.session_state:
            items = st.session_state['run_result']
            if items:
                trucks = run_optimization(items)
                if trucks:
                    total_cost = sum(t.cost for t in trucks); st.success(f"✅ 분석 완료: 총 {len(trucks)}대 | 예상 총 운송비: {total_cost:,}원")
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
                                st.markdown(f"### **{t.name}**"); st.write(f"- 박스: **{len(t.items)}개**"); st.write(f"- 중량: **{t.total_weight:,} kg**"); st.write(f"- 비용: **{t.cost:,} 원**")
                                with st.expander("목록 보기"): st.write(", ".join([b.name for b in t.items]))
                            with col2: st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
    except Exception as e: st.error(f"오류 발생: {e}")
