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
# 4. 시각화 (두 번째 사진과 똑같이 만들기 - 오류 수정됨)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    original_name = truck.name.split(' (')[0]
    spec = TRUCK_DB.get(original_name, TRUCK_DB["5톤"])
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # 조명 효과 (부드러운 일러스트 느낌)
    LIGHTING = dict(ambient=0.8, diffuse=0.8, specular=0.3, roughness=0.5)
    
    # 색상 팔레트 (두 번째 사진 참고)
    COLOR_FRAME = '#333333' # 진한 회색 프레임
    COLOR_WHEEL_OUTER = '#222222' # 타이어 (검정)
    COLOR_WHEEL_INNER = '#555555' # 휠 (진한 회색)
    COLOR_CONTAINER_GLASS = '#EEF5FF' # 투명 컨테이너

    # --- 도우미 함수: 육면체 그리기 ---
    # [수정] lighting 인자를 받을 수 있도록 **kwargs 추가
    def draw_cube(x, y, z, w, l, h, face_color, line_color=None, opacity=1.0, **kwargs):
        lighting_config = kwargs.get('lighting', LIGHTING) # 전달받은 lighting이 없으면 기본값 사용
        
        fig.add_trace(go.Mesh3d(
            x=[x, x+w, x+w, x, x, x+w, x+w, x],
            y=[y, y, y+l, y+l, y, y, y+l, y+l],
            z=[z, z, z, z, z+h, z+h, z+h, z+h],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=face_color, opacity=opacity, flatshading=True, 
            lighting=lighting_config, hoverinfo='skip'
        ))
        if line_color:
            xe=[x,x+w,x+w,x,x,None, x,x+w,x+w,x,x,None, x,x,None, x+w,x+w,None, x+w,x+w,None, x,x]
            ye=[y,y,y+l,y+l,y,None, y,y,y+l,y+l,y,None, y,y,None, y,y,None, y+l,y+l,None, y+l,y+l]
            ze=[z,z,z,z,z,None, z+h,z+h,z+h,z+h,z+h,None, z,z+h,None, z,z+h,None, z,z+h,None, z,z+h]
            fig.add_trace(go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=line_color, width=2), showlegend=False, hoverinfo='skip'))

    # --- 도우미 함수: 바퀴 그리기 (타이어 + 휠) ---
    def draw_wheel_set(cx, cy, cz):
        r_tire = 280; w_tire = 140
        r_rim = 160; w_rim = 145 # 휠이 살짝 튀어나오게
        
        # 타이어 (바깥 검정 원통)
        theta = np.linspace(0, 2*np.pi, 24)
        x_t, y_t, z_t = [], [], []
        for t in theta:
            x_t.extend([cx-w_tire/2, cx+w_tire/2])
            y_t.extend([cy+r_tire*np.cos(t), cy+r_tire*np.cos(t)])
            z_t.extend([cz+r_tire*np.sin(t), cz+r_tire*np.sin(t)])
        
        # 타이어 메쉬
        fig.add_trace(go.Mesh3d(x=x_t, y=y_t, z=z_t, alphahull=0, color=COLOR_WHEEL_OUTER, flatshading=True, lighting=LIGHTING, hoverinfo='skip'))
        # 타이어 옆면 막기
        y_side = [cy+r_tire*np.cos(t) for t in theta] + [cy]
        z_side = [cz+r_tire*np.sin(t) for t in theta] + [cz]
        fig.add_trace(go.Mesh3d(x=[cx-w_tire/2]*len(y_side), y=y_side, z=z_side, color=COLOR_WHEEL_OUTER, flatshading=True, lighting=LIGHTING, hoverinfo='skip'))
        fig.add_trace(go.Mesh3d(x=[cx+w_tire/2]*len(y_side), y=y_side, z=z_side, color=COLOR_WHEEL_OUTER, flatshading=True, lighting=LIGHTING, hoverinfo='skip'))

        # 휠 (안쪽 회색 원통)
        x_r, y_r, z_r = [], [], []
        for t in theta:
            x_r.extend([cx-w_rim/2, cx+w_rim/2])
            y_r.extend([cy+r_rim*np.cos(t), cy+r_rim*np.cos(t)])
            z_r.extend([cz+r_rim*np.sin(t), cz+r_rim*np.sin(t)])
        fig.add_trace(go.Mesh3d(x=x_r, y=y_r, z=z_r, alphahull=0, color=COLOR_WHEEL_INNER, flatshading=True, lighting=LIGHTING, hoverinfo='skip'))
        # 휠 옆면 막기
        y_rim_side = [cy+r_rim*np.cos(t) for t in theta] + [cy]
        z_rim_side = [cz+r_rim*np.sin(t) for t in theta] + [cz]
        # 바깥쪽 휠만 보이게 (안쪽은 타이어에 가려짐)
        wheel_face_x = cx + w_tire/2 + 2 if cx > W/2 else cx - w_tire/2 - 2
        fig.add_trace(go.Mesh3d(x=[wheel_face_x]*len(y_rim_side), y=y_rim_side, z=z_rim_side, color=COLOR_WHEEL_INNER, flatshading=True, lighting=LIGHTING, hoverinfo='skip'))


    # 1. 트럭 섀시 및 하단 프레임
    chassis_h = 100
    # 메인 바닥판 (회색)
    draw_cube(0, 0, -chassis_h, W, L, chassis_h, '#D0D0D0', None)
    
    # 하단 사이드 프레임 (진한 회색 띠)
    draw_cube(-50, 0, -chassis_h-40, 50, L, 140, '#E0E0E0', '#555555')
    draw_cube(W, 0, -chassis_h-40, 50, L, 140, '#E0E0E0', '#555555')

    # 2. 후면 프레임 (입구) - 진하고 굵게
    f_tk = 80; bumper_h = 100
    draw_cube(-f_tk/2, L-f_tk, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None) # 기둥 L
    draw_cube(W-f_tk/2, L-f_tk, -chassis_h, f_tk, f_tk, Real_H+chassis_h+20, COLOR_FRAME, None) # 기둥 R
    draw_cube(-f_tk/2, L-f_tk, Real_H, W+f_tk, f_tk, f_tk, COLOR_FRAME, None) # 상단바
    
    # 범퍼
    draw_cube(-f_tk/2, L, -chassis_h-bumper_h, W+f_tk, 40, bumper_h, '#555555', None) 
    
    # 후미등 (빨간색 사각형)
    draw_cube(80, L+40, -chassis_h-70, 120, 10, 40, '#FF0000', None)
    draw_cube(W-200, L+40, -chassis_h-70, 120, 10, 40, '#FF0000', None)

    # 3. 투명 컨테이너 (벽면)
    draw_cube(0, 0, 0, W, L, Real_H, COLOR_CONTAINER_GLASS, '#888888', opacity=0.1)

    # 4. 바퀴 배치 (앞 2축, 뒤 2축)
    wheel_z = -chassis_h - 250
    # 앞바퀴
    draw_wheel_set(-50, L*0.15, wheel_z); draw_wheel_set(W+50, L*0.15, wheel_z)
    draw_wheel_set(-50, L*0.28, wheel_z); draw_wheel_set(W+50, L*0.28, wheel_z)
    # 뒷바퀴
    draw_wheel_set(-50, L*0.75, wheel_z); draw_wheel_set(W+50, L*0.75, wheel_z)
    draw_wheel_set(-50, L*0.88, wheel_z); draw_wheel_set(W+50, L*0.88, wheel_z)

    # 5. [화물 박스] (기존 유지)
    annotations = []
    for item in truck.items:
        color = '#FF6B6B' if getattr(item, 'is_heavy', False) else '#FAD7A0'
        draw_cube(item.x, item.y, item.z, item.w, item.d, item.h, color, '#000000') # 테두리 포함
        
        # 툴팁용 투명 메쉬
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

    # 6. [제한선] (빨간 점선)
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
