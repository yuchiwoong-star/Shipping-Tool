import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math
import uuid

# ==========================================
# 1. 커스텀 물리 엔진 (기존 유지)
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
# 2. 설정 및 데이터 (기존 유지)
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

# ==========================================
# 3. 로직 함수 (기존 유지)
# ==========================================
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
# 4. 시각화 (기존 유지)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    original_name = truck.name.split(' (')[0]
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    chassis_h = 180
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0, 0, W, W, 0], y=[0, 0, L, L, 0, 0, L, L], z=[-chassis_h, -chassis_h, -chassis_h, -chassis_h, 0, 0, 0, 0], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color='#222222', flatshading=True, showlegend=False))

    def create_realistic_wheel(cx, cy, cz, r, w):
        theta = np.linspace(0, 2*np.pi, 32)
        x_tire, y_tire, z_tire = [], [], []
        for t in theta:
            x_tire.extend([cx - w/2, cx + w/2])
            y_tire.extend([cy + r*np.cos(t), cy + r*np.cos(t)])
            z_tire.extend([cz + r*np.sin(t), cz + r*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=x_tire, y=y_tire, z=z_tire, alphahull=0, color='#333333', flatshading=True, showlegend=False, lighting=dict(ambient=1.0)))
        tread_x, tread_y, tread_z = [], [], []
        num_treads = 16
        for i in range(num_treads):
            t1 = (2 * math.pi / num_treads) * i
            tread_x.extend([cx - w/2, cx + w/2, None])
            tread_y.extend([cy + r*math.cos(t1), cy + r*math.cos(t1), None])
            tread_z.extend([cz + r*math.sin(t1), cz + r*math.sin(t1), None])
        fig.add_trace(go.Scatter3d(x=tread_x, y=tread_y, z=tread_z, mode='lines', line=dict(color='#111111', width=3), showlegend=False))
        hub_r = r * 0.6; hub_w = w * 0.1
        theta_hub = np.linspace(0, 2*np.pi, 16)
        x_hub, y_hub, z_hub = [], [], []; x_hub.append(cx + w/2 + hub_w); y_hub.append(cy); z_hub.append(cz)
        for t in theta_hub:
            x_hub.append(cx + w/2); y_hub.append(cy + hub_r*math.cos(t)); z_hub.append(cz + hub_r*math.sin(t))
        i_hub = [0]*16; j_hub = list(range(1, 17)); k_hub = list(range(2, 17)) + [1]
        fig.add_trace(go.Mesh3d(x=x_hub, y=y_hub, z=z_hub, i=i_hub, j=j_hub, k=k_hub, color='#dddddd', flatshading=True, showlegend=False, lighting=dict(ambient=0.9)))

    wheel_r = 450; wheel_w = 280; wheel_z = -chassis_h - 100
    wheel_pos = [(-wheel_w/2, L*0.15), (W+wheel_w/2, L*0.15), (-wheel_w/2, L*0.30), (W+wheel_w/2, L*0.30), (-wheel_w/2, L*0.70), (W+wheel_w/2, L*0.70), (-wheel_w/2, L*0.85), (W+wheel_w/2, L*0.85)]
    for wx, wy in wheel_pos: create_realistic_wheel(wx, wy, wheel_z, wheel_r, wheel_w)

    wall_color_rgba = 'rgba(230, 230, 230, 0.4)'; frame_color = '#555555'; frame_width = 6
    fig.add_trace(go.Surface(x=[[0, 0], [0, 0]], y=[[0, L], [0, L]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_color_rgba], [1, wall_color_rgba]], showscale=False, opacity=0.4, hoverinfo='skip'))
    fig.add_trace(go.Surface(x=[[W, W], [W, W]], y=[[0, L], [0, L]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_color_rgba], [1, wall_color_rgba]], showscale=False, opacity=0.4, hoverinfo='skip'))
    fig.add_trace(go.Surface(x=[[0, W], [0, W]], y=[[L, L], [L, L]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_color_rgba], [1, wall_color_rgba]], showscale=False, opacity=0.4, hoverinfo='skip'))
    fig.add_trace(go.Surface(x=[[0, W], [0, W]], y=[[0, 0], [0, 0]], z=[[0, 0], [Real_H, Real_H]], colorscale=[[0, wall_color_rgba], [1, wall_color_rgba]], showscale=False, opacity=0.4, hoverinfo='skip'))
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]; lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]; lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color=frame_color, width=frame_width), showlegend=False, hoverinfo='skip'))

    OFFSET = 1200 
    def add_dimension(p1, p2, label, color='black'):
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color=color, width=2), showlegend=False))
        vec = np.array(p2) - np.array(p1); length = np.linalg.norm(vec)
        if length > 0:
            uvw = vec / length
            fig.add_trace(go.Cone(x=[p2[0]], y=[p2[1]], z=[p2[2]], u=[uvw[0]], v=[uvw[1]], w=[uvw[2]], sizemode="absolute", sizeref=200, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]]))
            fig.add_trace(go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[-uvw[0]], v=[-uvw[1]], w=[-uvw[2]], sizemode="absolute", sizeref=200, anchor="tip", showscale=False, colorscale=[[0, color], [1, color]]))
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode='text', text=[f"<b>{label}</b>"], textfont=dict(size=14, color=color, family="Arial"), showlegend=False))
    
    add_dimension((0, -OFFSET, 0), (W, -OFFSET, 0), f"폭 : {int(W)}")
    add_dimension((-OFFSET, 0, 0), (-OFFSET, L, 0), f"길이 : {int(L)}")
    add_dimension((-OFFSET, L, 0), (-OFFSET, L, LIMIT_H), f"높이제한(최대4단) : {int(LIMIT_H)}", color='red')
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))

    annotations = []
    for item in truck.items:
        color = '#FF0000' if getattr(item, 'is_heavy', False) else '#f39c12'
        x, y, z = item.x, item.y, item.z; w, h, d = item.w, item.h, item.d
        fig.add_trace(go.Mesh3d(x=[x,x+w,x+w,x, x,x+w,x+w,x], y=[y,y,y+d,y+d, y,y,y+d,y+d], z=[z,z,z,z, z+h,z+h,z+h,z+h], i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6], color=color, opacity=1.0, flatshading=True, name=item.name))
        ex = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]; ey = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]; ez = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=3), showlegend=False))
        cx, cy, cz = x + w/2, y + d/2, z + h/2
        annotations.append(dict(x=cx, y=cy, z=cz, text=f"<b>{item.name}</b>", xanchor="center", yanchor="middle", showarrow=False, font=dict(color="white" if getattr(item, 'is_heavy', False) else "black", size=14, family="Arial Black"), bgcolor="rgba(0, 0, 0, 0.6)" if getattr(item, 'is_heavy', False) else "rgba(255, 255, 255, 0.7)", borderpad=2))

    if camera_view == "top": eye = dict(x=0, y=0.1, z=2.5); up = dict(x=0, y=1, z=0)
    elif camera_view == "side": eye = dict(x=2.5, y=0, z=0.5); up = dict(x=0, y=0, z=1)
    else: eye = dict(x=2.0, y=-1.5, z=1.2); up = dict(x=0, y=0, z=1)
    
    fig.update_layout(
        scene=dict(
            aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white', camera=dict(eye=eye, up=up), annotations=annotations
        ),
        margin=dict(l=0,r=0,b=0,t=0), height=700,
        uirevision=str(uuid.uuid4())
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
# [수정] 제목 변경
st.title("📦 출하박스 적재 최적화 시스템 (배차비용 최소화)")
st.caption("✅ 비용최적화(Lookahead) | 회전금지 | 1.3m 제한 | 80% 지지충족 | 상위 10% 중량박스 빨간색 표시")
if 'view_mode' not in st.session_state: st.session_state['view_mode'] = 'iso'

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file, encoding='cp949')
        else: df = pd.read_excel(uploaded_file)
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        
        df_display = df.copy()
        
        rename_map = {
            '폭': '폭 (mm)', 
            '높이': '높이 (mm)', 
            '길이': '길이 (mm)', 
            '중량': '중량 (kg)'
        }
        df_display.rename(columns=rename_map, inplace=True)
        
        cols_to_format = [c for c in ['폭 (mm)', '높이 (mm)', '길이 (mm)', '중량 (kg)'] if c in df_display.columns]
        format_dict = {c: '{:,.0f}' for c in cols_to_format}
        
        # [수정] 헤더와 셀 모두 강제 가운데 정렬 스타일 적용
        styler = df_display.style.format(format_dict).set_properties(**{'text-align': 'center'})
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
        
        format_dict_truck = {
            '적재폭 (mm)': '{:,.0f}',
            '적재길이 (mm)': '{:,.0f}',
            '허용하중 (kg)': '{:,.0f}',
            '운송단가': '{:,.0f}'
        }
        
        # [수정] 차량 기준표도 헤더와 셀 모두 강제 가운데 정렬
        st_truck = df_truck.style.format(format_dict_truck).set_properties(**{'text-align': 'center'})
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
