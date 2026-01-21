import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import math
import uuid

# ==========================================
# 0. 세션 초기화 (가장 먼저 실행)
# ==========================================
# 이전에 저장된 모든 세션 데이터를 강제로 날려서 충돌을 방지합니다.
if 'app_reset_token' not in st.session_state:
    st.session_state.clear()
    st.session_state['app_reset_token'] = str(uuid.uuid4())

# ==========================================
# 1. 물리 엔진 (변수명 대폭 변경)
# ==========================================
class CargoItem:
    def __init__(self, name, w, h, d, weight):
        self.name = str(name)
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.weight = float(weight)
        # 위치 초기화
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 0.0
        self.is_heavy = False
    
    @property
    def volume(self):
        return self.w * self.h * self.d

class TransportVehicle:
    def __init__(self, name, w, h, d, max_weight, cost):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.cost = cost
        self.loaded_cargo = [] # items -> loaded_cargo 변경
        self.current_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def attempt_load(self, cargo):
        if self.current_weight + cargo.weight > self.max_weight:
            return False
        
        # Z -> Y -> X 순서로 피벗 정렬 (아래, 안쪽, 왼쪽부터 채움)
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            
            # 1. 트럭 내부 공간 체크
            if (px + cargo.w > self.w) or (py + cargo.d > self.d) or (pz + cargo.h > self.h):
                continue
            
            # 2. 충돌 체크
            if self._check_collision(cargo, px, py, pz):
                continue
            
            # 3. 바닥 지지 체크 (규칙 3: 60% 이상)
            if not self._check_support(cargo, px, py, pz):
                continue

            # 적재 성공: 위치 할당
            cargo.pos_x, cargo.pos_y, cargo.pos_z = px, py, pz
            self.loaded_cargo.append(cargo)
            self.current_weight += cargo.weight
            
            # 새로운 피벗 포인트 생성 (박스의 우측, 뒤쪽, 위쪽)
            self.pivots.append([cargo.pos_x + cargo.w, cargo.pos_y, cargo.pos_z])
            self.pivots.append([cargo.pos_x, cargo.pos_y + cargo.d, cargo.pos_z])
            self.pivots.append([cargo.pos_x, cargo.pos_y, cargo.pos_z + cargo.h])
            return True
        
        return False

    def _check_collision(self, cargo, x, y, z):
        for exist in self.loaded_cargo:
            # 겹치는지 확인 (AABB 충돌 감지)
            if (x < exist.pos_x + exist.w and x + cargo.w > exist.pos_x and
                y < exist.pos_y + exist.d and y + cargo.d > exist.pos_y and
                z < exist.pos_z + exist.h and z + cargo.h > exist.pos_z):
                return True
        return False

    def _check_support(self, cargo, x, y, z):
        if z <= 0.001: return True # 바닥이면 OK
        
        support_area = 0.0
        required_area = cargo.w * cargo.d * 0.6
        
        for exist in self.loaded_cargo:
            # 바로 아래층(오차 1.0)에 있는 화물과 겹치는 면적 계산
            if abs((exist.pos_z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + cargo.w, exist.pos_x + exist.w) - max(x, exist.pos_x))
                oy = max(0.0, min(y + cargo.d, exist.pos_y + exist.d) - max(y, exist.pos_y))
                support_area += ox * oy
                
        return support_area >= required_area

# ==========================================
# 2. 설정 및 데이터 (규칙 0: 비용 추가)
# ==========================================
st.set_page_config(layout="wide", page_title="Load Planner Final")

VEHICLE_DB = {
    "1톤":    {"w": 1600, "l": 2800, "h": 1700, "weight": 1000, "cost": 100000},
    "1.4톤":  {"w": 1650, "l": 3400, "h": 1800, "weight": 1400, "cost": 130000},
    "2.5톤":  {"w": 1800, "l": 4300, "h": 2000, "weight": 2500, "cost": 180000},
    "3.5톤":  {"w": 2000, "l": 4800, "h": 2000, "weight": 3500, "cost": 220000},
    "5톤":    {"w": 2350, "l": 6200, "h": 2350, "weight": 5000, "cost": 300000},
    "5톤축":  {"w": 2350, "l": 7300, "h": 2350, "weight": 8000, "cost": 350000},
    "11톤":   {"w": 2350, "l": 9600, "h": 2400, "weight": 11000, "cost": 450000},
    "18톤":   {"w": 2350, "l": 10200, "h": 2500, "weight": 18000, "cost": 550000},
    "25톤":   {"w": 2350, "l": 10200, "h": 2500, "weight": 25000, "cost": 650000},
}

# ==========================================
# 3. 로직 함수
# ==========================================
def parse_excel_data(df):
    cargo_list = []
    try:
        # 중량 데이터 전처리 (상위 10% 계산용)
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if weights:
            weights.sort(reverse=True)
            cutoff = max(0, int(len(weights) * 0.1) - 1)
            heavy_limit = weights[cutoff]
        else:
            heavy_limit = float('inf')
    except:
        heavy_limit = float('inf')

    for index, row in df.iterrows():
        try:
            # 데이터 파싱
            c_name = str(row['박스번호'])
            c_w = float(row['폭'])
            c_h = float(row['높이'])
            c_l = float(row['길이'])
            c_weight = float(row['중량'])
            
            item = CargoItem(c_name, c_w, c_h, c_l, c_weight)
            
            # 규칙 4: 상위 10% 중량 체크
            if c_weight >= heavy_limit and c_weight > 0:
                item.is_heavy = True
            
            cargo_list.append(item)
        except:
            continue
            
    return cargo_list

def optimize_logistics(all_cargo):
    if not all_cargo: return []
    
    remaining_cargo = all_cargo[:]
    solution_trucks = []
    
    # 비용이 낮은 순서대로 차량 타입 정렬
    vehicle_types_by_cost = sorted(VEHICLE_DB.keys(), key=lambda k: VEHICLE_DB[k]['cost'])
    # 크기가 큰 순서대로 차량 타입 정렬 (Greedy용)
    vehicle_types_by_size = sorted(VEHICLE_DB.keys(), key=lambda k: VEHICLE_DB[k]['weight'], reverse=True)
    
    # 1. [단일 차량 전략] 모든 짐을 한 번에 실을 수 있는 가장 싼 차 찾기
    for v_name in vehicle_types_by_cost:
        spec = VEHICLE_DB[v_name]
        # 규칙 2: 높이 제한 1300mm 적용
        truck = TransportVehicle(v_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
        
        # 부피 큰 순서로 정렬하여 적재 시도
        test_cargo = sorted(remaining_cargo, key=lambda x: x.volume, reverse=True)
        all_loaded = True
        
        for c in test_cargo:
            # 시뮬레이션용 복제본 생성
            c_copy = CargoItem(c.name, c.w, c.h, c.d, c.weight)
            c_copy.is_heavy = c.is_heavy
            
            if not truck.attempt_load(c_copy):
                all_loaded = False
                break
        
        if all_loaded:
            truck.name = f"{v_name} (단일차량)"
            return [truck]

    # 2. [다중 차량 전략] 가장 큰 차부터 꽉 채워서 보내기 (Greedy)
    while remaining_cargo:
        best_truck = None
        max_loaded_count = -1
        loaded_cargo_names = []
        
        for v_name in vehicle_types_by_size:
            spec = VEHICLE_DB[v_name]
            truck = TransportVehicle(v_name, spec['w'], 1300, spec['l'], spec['weight'], spec['cost'])
            
            test_cargo = sorted(remaining_cargo, key=lambda x: x.volume, reverse=True)
            loaded_count = 0
            current_loaded_names = []
            
            for c in test_cargo:
                c_copy = CargoItem(c.name, c.w, c.h, c.d, c.weight)
                c_copy.is_heavy = c.is_heavy
                
                if truck.attempt_load(c_copy):
                    loaded_count += 1
                    current_loaded_names.append(c.name)
            
            # 가장 많이 실은 트럭 선택
            if loaded_count > max_loaded_count:
                max_loaded_count = loaded_count
                best_truck = truck
                loaded_cargo_names = current_loaded_names

        # 최적의 트럭을 결과에 추가
        if best_truck and max_loaded_count > 0:
            best_truck.name = f"{best_truck.name} #{len(solution_trucks)+1}"
            solution_trucks.append(best_truck)
            # 실린 화물은 목록에서 제거
            remaining_cargo = [c for c in remaining_cargo if c.name not in loaded_cargo_names]
        else:
            # 더 이상 실을 수 없는 화물이 남음 (예: 트럭보다 큰 화물)
            break
            
    return solution_trucks

# ==========================================
# 4. 시각화 (No Numpy, Pure Math)
# ==========================================
def render_truck_scene(truck, view_mode="iso"):
    fig = go.Figure()
    spec = VEHICLE_DB[truck.name.split(' ')[0]]
    W, L, H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300 # 규칙 2
    
    # 1. 섀시 (바닥 프레임)
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0], y=[0, 0, L, L, 0, 0, L, L], z=[-180]*4+[0]*4,
        color='#222222', 
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        showlegend=False, flatshading=True
    ))

    # 2. 바퀴 그리기 (math 모듈 사용)
    def add_wheel_shape(cx, cy):
        steps = 24
        # 타이어
        tx, ty, tz = [], [], []
        for i in range(steps):
            angle = (2 * math.pi / steps) * i
            tx.extend([cx-100, cx+100])
            ty.extend([cy+450*math.cos(angle), cy+450*math.cos(angle)])
            tz.extend([-250+450*math.sin(angle), -250+450*math.sin(angle)])
        fig.add_trace(go.Mesh3d(x=tx, y=ty, z=tz, alphahull=0, color='#333333', showlegend=False, lighting=dict(ambient=1.0, diffuse=0.0)))
        
        # 휠 허브
        hx, hy, hz = [], [], []
        hx.append(cx+110); hy.append(cy); hz.append(-250)
        for i in range(steps):
            angle = (2 * math.pi / steps) * i
            hx.append(cx+100); hy.append(cy+250*math.cos(angle)); hz.append(-250+250*math.sin(angle))
        i_idx = [0]*steps
        j_idx = list(range(1, steps+1))
        k_idx = list(range(2, steps+1)) + [1]
        fig.add_trace(go.Mesh3d(x=hx, y=hy, z=hz, i=i_idx, j=j_idx, k=k_idx, color='#dddddd', showlegend=False, lighting=dict(ambient=1.0, diffuse=0.0)))

    # 바퀴 4개 배치
    for wy in [L*0.15, L*0.30, L*0.70, L*0.85]:
        add_wheel_shape(-140, wy)
        add_wheel_shape(W+140, wy)

    # 3. 적재함 벽면 (Surface 사용 -> 대각선 제거)
    wall_color = [[0, 'rgba(220,220,220,0.15)'], [1, 'rgba(220,220,220,0.15)']]
    # 좌, 우, 앞, 뒤
    fig.add_trace(go.Surface(x=[[0,0],[0,0]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=wall_color, showscale=False, opacity=0.15))
    fig.add_trace(go.Surface(x=[[W,W],[W,W]], y=[[0,L],[0,L]], z=[[0,0],[H,H]], colorscale=wall_color, showscale=False, opacity=0.15))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[L,L],[L,L]], z=[[0,0],[H,H]], colorscale=wall_color, showscale=False, opacity=0.15))
    fig.add_trace(go.Surface(x=[[0,W],[0,W]], y=[[0,0],[0,0]], z=[[0,0],[H,H]], colorscale=wall_color, showscale=False, opacity=0.15))

    # 4. 프레임 (외곽선)
    lx = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    ly = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lz = [0,0,0,0,0, H,H,H,H,H, 0,H,H,0, 0,H]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='#444444', width=5), showlegend=False))

    # 5. 치수선 및 제한선
    def add_dim_line(p1, p2, label, color='black'):
        fig.add_trace(go.Scatter3d(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], mode='lines+text', text=["", "", f"<b>{label}</b>"], textposition="middle center", line=dict(color=color, width=2), showlegend=False))
        # 화살표 계산
        dx, dy, dz = p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > 0:
            vx, vy, vz = dx/dist, dy/dist, dz/dist
            fig.add_trace(go.Cone(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]], u=[-vx,vx], v=[-vy,vy], w=[-vz,vz], sizemode="absolute", sizeref=120, showscale=False, colorscale=[[0,color],[1,color]]))

    offset = 1200
    add_dim_line([0,-offset,0], [W,-offset,0], f"폭: {int(W)}")
    add_dim_line([-offset,0,0], [-offset,L,0], f"길이: {int(L)}")
    add_dim_line([-offset,L,0], [-offset,L,LIMIT_H], f"높이제한: {int(LIMIT_H)}", color='red')
    
    # 1.3m 높이 제한선 (점선)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))

    # 6. 화물 박스 렌더링
    for item in truck.loaded_cargo:
        # 색상: 상위 10%는 빨강, 나머지는 오렌지
        c_code = '#FF0000' if getattr(item, 'is_heavy', False) else '#f39c12'
        x, y, z = item.pos_x, item.pos_y, item.pos_z
        w, h, d = item.w, item.h, item.d
        
        # 육면체 그리기
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=c_code, opacity=1.0, flatshading=True, name=item.name
        ))
        # 테두리 그리기
        fig.add_trace(go.Scatter3d(
            x=[x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x],
            y=[y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d],
            z=[z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h],
            mode='lines', line=dict(color='black', width=2), showlegend=False
        ))
        # 라벨 (중앙에 표시)
        fig.add_trace(go.Scatter3d(
            x=[x+w/2], y=[y+d/2], z=[z+h/2],
            mode='text', text=[f"<b>{item.name}</b>"],
            textfont=dict(size=14, color="white" if c_code=='#FF0000' else "black"),
            showlegend=False
        ))

    # 카메라 뷰 설정
    eye_pos = dict(x=2.0, y=-1.5, z=1.2)
    if view_mode == 'top': eye_pos = dict(x=0, y=0.1, z=2.5)
    elif view_mode == 'side': eye_pos = dict(x=2.5, y=0, z=0.5)
    
    fig.update_layout(
        scene=dict(
            aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            bgcolor='white', camera=dict(eye=eye_pos)
        ),
        margin=dict(l=0,r=0,b=0,t=0), height=700,
        uirevision=str(uuid.uuid4()) # 강제 갱신용
    )
    return fig

# ==========================================
# 5. 메인 UI (Clean & Error-Free)
# ==========================================
st.title("📦 Ultimate Load Planner (Final Reset)")
st.caption("✅ 비용최적화 | 회전금지 | 1.3m 제한 | 60% 지지 | 상위10% 빨강")

if 'view_option' not in st.session_state: st.session_state['view_option'] = 'iso'

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='cp949')
        else:
            df = pd.read_excel(uploaded_file)
        
        # 컬럼명 공백 제거
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df.head()) # 전체 다 보여주면 느릴 수 있으니 head만
        
        if st.button("최적 배차 실행", type="primary"):
            st.session_state['parsed_cargo'] = parse_excel_data(df)
            st.session_state['optimization_done'] = False # 재실행 트리거
        
        if 'parsed_cargo' in st.session_state:
            cargo_data = st.session_state['parsed_cargo']
            
            if not cargo_data:
                st.error("데이터 변환에 실패했습니다. 컬럼명(박스번호, 폭, 높이, 길이, 중량)을 확인해주세요.")
            else:
                # 최적화 실행 (결과가 없거나 재실행 필요 시)
                if 'truck_solutions' not in st.session_state or not st.session_state.get('optimization_done', False):
                    with st.spinner("최적 배차 알고리즘 가동 중..."):
                        results = optimize_logistics(cargo_data)
                        st.session_state['truck_solutions'] = results
                        st.session_state['optimization_done'] = True
                
                trucks = st.session_state['truck_solutions']
                
                if trucks:
                    total_cost = sum([t.cost for t in trucks])
                    st.success(f"✅ 배차 완료: 총 {len(trucks)}대 배차됨 (예상 운송비: {total_cost:,}원)")
                    
                    # 뷰 컨트롤
                    col_v1, col_v2, col_v3, _ = st.columns([1, 1, 1, 5])
                    if col_v1.button("↗️ 쿼터뷰"): st.session_state['view_option'] = 'iso'
                    if col_v2.button("⬆️ 탑뷰"): st.session_state['view_option'] = 'top'
                    if col_v3.button("➡️ 사이드뷰"): st.session_state['view_option'] = 'side'
                    
                    # 탭으로 트럭별 결과 표시
                    tabs = st.tabs([t.name for t in trucks])
                    for i, tab in enumerate(tabs):
                        with tab:
                            current_truck = trucks[i]
                            c_info, c_chart = st.columns([1, 4])
                            
                            with c_info:
                                st.markdown(f"### **{current_truck.name}**")
                                st.write(f"- 💰 비용: **{current_truck.cost:,}원**")
                                st.write(f"- 📦 적재: **{len(current_truck.loaded_cargo)}개**")
                                st.write(f"- ⚖️ 중량: **{current_truck.current_weight:,} / {current_truck.max_weight:,} kg**")
                                
                                with st.expander("적재 목록 확인"):
                                    for item in current_truck.loaded_cargo:
                                        st.caption(f"- {item.name} ({item.weight}kg)")
                            
                            with c_chart:
                                st.plotly_chart(
                                    render_truck_scene(current_truck, st.session_state['view_option']),
                                    use_container_width=True
                                )
                else:
                    st.warning("적재 가능한 차량을 찾지 못했습니다. 화물의 크기나 무게를 확인해주세요.")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
