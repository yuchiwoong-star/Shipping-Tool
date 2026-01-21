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
# 2. 설정 및 데이터 (차량 제원 유지)
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
# 3. 데이터 처리 및 알고리즘 (기존 유지)
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
# 4. 시각화 (디자인 전면 수정 - 참고 이미지 스타일 반영)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    original_name = truck.name.split(' (')[0]
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # --- 디자인 파라미터 ---
    chassis_color = '#aaaaaa' # 참고 이미지의 밝은 회색
    frame_color = '#888888'   # 얇은 프레임 색상
    wheel_color = '#333333'   # 타이어 색상
    hub_color = '#dddddd'     # 휠 허브 색상
    chassis_thickness = 150
    wheel_radius = 350
    wheel_width = 200
    
    # 1. 트럭 섀시 (두께감 있는 솔리드 바닥판)
    # 바닥면
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0], y=[0, 0, L, L], z=[0, 0, 0, 0], color=chassis_color, flatshading=True, hoverinfo='skip'))
    # 아랫면
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0], y=[0, 0, L, L], z=[-chassis_thickness]*4, color=chassis_color, flatshading=True, hoverinfo='skip'))
    # 옆면 (테두리)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0, 0,0,0,0], y=[0,0,L,L, 0,L,L,0], z=[0,0,0,0, -chassis_thickness,-chassis_thickness,-chassis_thickness,-chassis_thickness], i=[0,1,2,3, 0,4,5,1, 1,5,6,2, 2,6,7,3, 3,7,4,0], j=[1,2,3,0, 4,5,1,0, 5,6,2,1, 6,7,3,2, 7,4,0,3], k=[4,5,6,7, 5,1,0,4, 6,2,1,5, 7,3,2,6, 4,0,3,7], color=chassis_color, flatshading=True, hoverinfo='skip'))

    # 후면 범퍼 및 테일램프 표현
    bumper_depth = 100
    fig.add_trace(go.Mesh3d(x=[0, W, W, 0], y=[L, L, L+bumper_depth, L+bumper_depth], z=[-chassis_thickness+50]*4, color=chassis_color, hoverinfo='skip'))
    # 테일램프 (빨간색 박스)
    light_w = 150; light_h = 80
    for lx in [100, W-100-light_w]:
        fig.add_trace(go.Mesh3d(
            x=[lx, lx+light_w, lx+light_w, lx, lx, lx+light_w, lx+light_w, lx],
            y=[L+bumper_depth, L+bumper_depth, L+bumper_depth+10, L+bumper_depth+10]*2,
            z=[-chassis_thickness+60, -chassis_thickness+60, -chassis_thickness+60, -chassis_thickness+60, -chassis_thickness+60+light_h, -chassis_thickness+60+light_h, -chassis_thickness+60+light_h, -chassis_thickness+60+light_h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color='#ff0000', flatshading=True, hoverinfo='skip'
        ))


    # 2. 적재함 프레임 (참고 이미지 스타일: 얇은 실선)
    def draw_thin_frame(w, l, h, color, dash='solid'):
        # 기둥 4개
        fig.add_trace(go.Scatter3d(x=[0,0, W,W, 0,0, W,W], y=[0,0, 0,0, L,L, L,L], z=[0,h, 0,h, 0,h, 0,h], mode='lines', line=dict(color=color, width=2, dash=dash), showlegend=False, hoverinfo='skip'))
        # 상단 테두리
        fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[h,h,h,h,h], mode='lines', line=dict(color=color, width=2, dash=dash), showlegend=False, hoverinfo='skip'))

    # 실제 높이 프레임 (연한 회색)
    draw_thin_frame(W, L, Real_H, frame_color)
    
    # [핵심 규칙] 적재 제한 높이 1.3m (강조된 빨간 점선)
    fig.add_trace(go.Scatter3d(
        x=[0, W, W, 0, 0], y=[0, 0, L, L, 0], z=[LIMIT_H]*5,
        mode='lines', line=dict(color='#ff0000', width=4, dash='dash'),
        name='제한높이(1.3m)', showlegend=False, hoverinfo='skip'
    ))


    # 3. 바퀴 (참고 이미지 스타일: 후방 탠덤 휠)
    def create_wheel_pair(y_pos, z_pos):
        # 왼쪽/오른쪽 바퀴 쌍 생성
        for x_center in [-wheel_width/2 - 50, W + wheel_width/2 + 50]:
            # 타이어 (Mesh3d Cylinder 근사)
            theta = np.linspace(0, 2*np.pi, 16)
            x_cyl = np.array([x_center - wheel_width/2, x_center + wheel_width/2])
            y_cyl = y_pos + wheel_radius * np.cos(theta)
            z_cyl = z_pos + wheel_radius * np.sin(theta)
            
            X, Y = np.meshgrid(x_cyl, y_cyl)
            Z = np.tile(z_cyl, (2, 1)).T
            
            fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=[[0, wheel_color], [1, wheel_color]], showscale=False, hoverinfo='skip'))
            
            # 휠 허브 (밝은 회색 원판)
            y_hub = y_pos + (wheel_radius*0.6) * np.cos(theta)
            z_hub = z_pos + (wheel_radius*0.6) * np.sin(theta)
            x_hub_face = np.full_like(y_hub, x_center + wheel_width/2 + 5) # 바깥쪽 면
            fig.add_trace(go.Mesh3d(x=x_hub_face, y=y_hub, z=z_hub, color=hub_color, flatshading=True, hoverinfo='skip'))

    wheel_z = -chassis_thickness - wheel_radius + 50
    # 후방 탠덤 축 (길이의 75%, 90% 지점 배치)
    create_wheel_pair(L * 0.75, wheel_z)
    create_wheel_pair(L * 0.90, wheel_z)


    # 4. 치수선 및 박스 렌더링 (기존 유지 + 툴팁 강화)
    OFFSET_W = -W * 0.15; OFFSET_L = -L * 0.15
    def add_dim_text(p1, p2, text, color='#555555'):
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        fig.add_trace(go.Scatter3d(x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]], mode='lines', line=dict(color=color, width=1), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[mid[0]], y=[mid[1]], z=[mid[2]], mode='text', text=[f"<b>{text}</b>"], textposition="middle center", textfont=dict(size=11, color=color), showlegend=False, hoverinfo='skip'))

    add_dim_text((0, OFFSET_L, 0), (W, OFFSET_L, 0), f"폭 {int(W)}")
    add_dim_text((OFFSET_W, 0, 0), (OFFSET_W, L, 0), f"길이 {int(L)}")
    add_dim_text((OFFSET_W, L, 0), (OFFSET_W, L, LIMIT_H), f"제한 {int(LIMIT_H)}", color='red')

    annotations = []
    for item in truck.items:
        # 색상: 상위 10% 중량물은 빨강, 일반은 참고 이미지와 유사한 베이지색
        face_color = '#e74c3c' if getattr(item, 'is_heavy', False) else '#fbe7b2'
        line_color = '#c0392b' if getattr(item, 'is_heavy', False) else '#d35400'
            
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # [툴팁 구현] hovertemplate 사용
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x], y=[y, y, y+d, y+d, y, y, y+d, y+d], z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2], j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3], k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=face_color, opacity=1.0, flatshading=True, name=item.name,
            hovertemplate=(f"<b>📦 {item.name}</b><br>규격: {int(w)}x{int(d)}x{int(h)}mm<br>중량: {int(item.weight):,}kg<extra></extra>")
        ))
        
        # 박스 테두리 (선명하게)
        edges_x = [x, x+w, x+w, x, x, None, x, x+w, x+w, x, x, None, x, x, None, x+w, x+w, None, x+w, x+w, None, x, x]
        edges_y = [y, y, y+d, y+d, y, None, y, y, y+d, y+d, y, None, y, y, None, y, y, None, y+d, y+d, None, y+d, y+d]
        edges_z = [z, z, z, z, z, None, z+h, z+h, z+h, z+h, z+h, None, z, z+h, None, z, z+h, None, z, z+h, None, z, z+h]
        fig.add_trace(go.Scatter3d(x=edges_x, y=edges_y, z=edges_z, mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'))

        # 박스 번호 텍스트
        cx, cy, cz = x + w/2, y + d/2, z + h/2
        annotations.append(dict(x=cx, y=cy, z=cz, text=f"<b>{item.name}</b>", xanchor="center", yanchor="middle", showarrow=False, font=dict(color="black", size=11), bgcolor="rgba(255,255,255,0.6)", borderpad=1))

    # 5. 카메라 및 레이아웃 설정
    if camera_view == "top": eye = dict(x=0, y=0.01, z=2.5); up = dict(x=0, y=1, z=0)
    elif camera_view == "side": eye = dict(x=2.5, y=0, z=0.3); up = dict(x=0, y=0, z=1)
    else: # iso
        eye = dict(x=2.0, y=-2.0, z=1.5); up = dict(x=0, y=0, z=1)

    fig.update_layout(
        scene=dict(
            aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white', camera=dict(eye=eye, up=up), annotations=annotations
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600, uirevision=str(uuid.uuid4())
    )
    return fig

# ==========================================
# 5. 메인 UI (기존 유지)
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
        
        # [핵심] 숫자 데이터를 문자열로 변환 (왼쪽 정렬 유도)
        cols_to_format = [c for c in ['폭 (mm)', '높이 (mm)', '길이 (mm)', '중량 (kg)'] if c in df_display.columns]
        for col in cols_to_format:
            df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}")
        
        if '박스번호' in df_display.columns:
            df_display['박스번호'] = df_display['박스번호'].astype(str)

        # Styler 적용
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
        
        # 차량 정보도 문자열 변환
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
