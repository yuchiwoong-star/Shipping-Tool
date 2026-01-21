import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import math

# ==========================================
# 1. 커스텀 물리 엔진 (기존 로직 100% 동결)
# ==========================================
# ※ 로직 수정 없음 (회전금지, 중력, 높이제한, 최적화 그대로)

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
    def __init__(self, name, w, h, d, max_weight):
        self.name = name
        self.w = float(w)
        self.h = float(h)
        self.d = float(d)
        self.max_weight = float(max_weight)
        self.items = []     # 적재된 박스들
        self.total_weight = 0.0
        self.pivots = [[0.0, 0.0, 0.0]]

    def put_item(self, item):
        fit = False
        if self.total_weight + item.weight > self.max_weight:
            return False
        
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            
            # 범위, 충돌, 지지 체크
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
        required_area = item.w * item.d * 0.6
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0.0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0.0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= required_area

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="Ultimate Load Planner")

TRUCK_DB = {
    "5톤":  {"w": 2350, "real_h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "real_h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 3. 로직 함수
# ==========================================

def load_data(df):
    items = []
    try:
        # 중량 데이터 전처리 (에러 방지 강화)
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        
        if len(weights) > 0:
            sorted_weights = sorted(weights, reverse=True)
            # 상위 10% 기준값 (최소 1개는 포함되도록 인덱스 조정)
            cutoff_index = max(0, int(len(weights) * 0.1) - 1)
            heavy_threshold = sorted_weights[cutoff_index]
        else:
            heavy_threshold = 999999999 # 데이터 없으면 아무것도 선택 안 함
            
    except Exception as e:
        heavy_threshold = 999999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            box = Box(name, w, h, l, weight)
            # [수정] 상위 10% 판별 로직 명확화
            if weight >= heavy_threshold and weight > 0:
                box.is_heavy = True
            else:
                box.is_heavy = False
                
            items.append(box)
        except:
            continue
    return items

def run_optimization(all_items):
    remaining_items = all_items[:]
    used_trucks = [] 
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_truck = None
        best_score = -1
        
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            limit_h = 1300 
            temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'])
            
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
                # is_heavy 속성도 복사해야 함!
                item_copy.is_heavy = item.is_heavy 
                
                if temp_truck.put_item(item_copy):
                    packed_count += 1
            
            if packed_count > 0:
                if packed_count == len(remaining_items):
                    score = 100000 - spec['weight']
                else:
                    util_w = temp_truck.total_weight / spec['weight']
                    util_v = sum([i.volume for i in temp_truck.items]) / (spec['w'] * limit_h * spec['l'])
                    score = (util_w + util_v) * 100
                
                if score > best_score:
                    best_score = score
                    best_truck = temp_truck

        if best_truck and len(best_truck.items) > 0:
            best_truck.name = f"{best_truck.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_truck)
            packed_names = [i.name for i in best_truck.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            break
    return used_trucks

# ==========================================
# 4. 시각화 (디자인 개선 & 치수선 수정)
# ==========================================
def draw_truck_3d(truck, camera_view="iso"):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # --- [1] 트럭 디자인 (깨짐 현상 수정) ---
    
    # 1. 섀시 (Chassis)
    chassis_h = 200
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[0, 0, L, L, 0, 0, L, L],
        z=[-chassis_h, -chassis_h, -chassis_h, -chassis_h, 0, 0, 0, 0],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='#2c3e50', flatshading=True, name='섀시'
    ))

    # 2. 바퀴 (Round Wheels - 32각형)
    def create_cylinder(cx, cy, cz, r, w, axis='x', color='#111111'):
        theta = np.linspace(0, 2*np.pi, 32)
        x, y, z = [], [], []
        for t in theta:
            x.extend([cx - w/2, cx + w/2])
            y.extend([cy + r*np.cos(t), cy + r*np.cos(t)])
            z.extend([cz + r*np.sin(t), cz + r*np.sin(t)])
        return go.Mesh3d(x=x, y=y, z=z, alphahull=0, color=color, flatshading=True, showlegend=False)

    wheel_r = 450
    wheel_w = 280
    wheel_z = -chassis_h - 100
    wheel_pos = [(-wheel_w/2, L*0.18), (W+wheel_w/2, L*0.18), (-wheel_w/2, L*0.82), (W+wheel_w/2, L*0.82)]
    for wx, wy in wheel_pos:
        fig.add_trace(create_cylinder(wx, wy, wheel_z, wheel_r, wheel_w))

    # 3. 헤드 (Cabin) - 디자인 개선
    cabin_len = 1600
    cabin_h = 2600
    cy = L + 100 
    
    # (A) 헤드 본체 (아래쪽 박스)
    base_h = 1000
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0, 0, W, W, 0],
        y=[cy, cy, cy+cabin_len, cy+cabin_len, cy, cy, cy+cabin_len, cy+cabin_len],
        z=[0, 0, 0, 0, base_h, base_h, base_h, base_h],
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='#2980b9', flatshading=True, name='헤드 하단'
    ))
    
    # (B) 헤드 상단 (곡선형 지붕 느낌을 위해 경사면 추가)
    top_z = cabin_h
    spoiler_y = cy + cabin_len - 600
    
    cx = [0, W, W, 0, 0, W, W, 0]
    cy_coords = [cy, cy, cy+cabin_len, cy+cabin_len, cy, cy, spoiler_y, spoiler_y]
    cz = [base_h, base_h, base_h, base_h, top_z, top_z, top_z-200, top_z-200]
    
    fig.add_trace(go.Mesh3d(
        x=cx, y=cy_coords, z=cz,
        i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
        color='#2980b9', flatshading=True, name='헤드 상단'
    ))

    # (C) 윈드쉴드 (유리) - [수정] 인덱스 정렬로 깨짐 방지
    # 앞쪽 경사면 (spoiler_y ~ cy+cabin_len 사이)
    # P0(50, spoiler_y, top_z-250), P1(W-50, spoiler_y, top_z-250)
    # P2(W-50, cy+cabin_len, base_h+50), P3(50, cy+cabin_len, base_h+50)
    wx = [50, W-50, W-50, 50]
    wy = [spoiler_y, spoiler_y, cy+cabin_len, cy+cabin_len]
    wz = [top_z-250, top_z-250, base_h+50, base_h+50]
    
    fig.add_trace(go.Mesh3d(
        x=wx, y=wy, z=wz,
        i=[0, 0], j=[1, 2], k=[2, 3], # Quad -> 2 Triangles
        color='#aed6f1', opacity=0.9, name='윈드쉴드'
    ))
    
    # (D) 그릴 & 헤드라이트 (앞면 디테일)
    fig.add_trace(go.Mesh3d(
        x=[200, W-200, W-200, 200],
        y=[cy+cabin_len+10, cy+cabin_len+10, cy+cabin_len+10, cy+cabin_len+10],
        z=[200, 200, 800, 800],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color='#111111', name='그릴'
    ))

    # 4. 적재함 프레임
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='#7f8c8d', width=3), showlegend=False))


    # --- [2] 치수선 개선 (화살표 <-> 및 텍스트 수정) ---
    OFFSET = 1200 
    
    def add_dimension(p1, p2, label, color='black'):
        # 메인 라인
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines', line=dict(color=color, width=2), showlegend=False
        ))
        
        # 화살표 (Cone 사용)
        vec = np.array(p2) - np.array(p1)
        length = np.linalg.norm(vec)
        if length > 0:
            uvw = vec / length # 단위 벡터
            
            # 끝점 화살표 (p2 쪽)
            fig.add_trace(go.Cone(
                x=[p2[0]], y=[p2[1]], z=[p2[2]],
                u=[uvw[0]], v=[uvw[1]], w=[uvw[2]],
                sizemode="absolute", sizeref=200, anchor="tip", showscale=False,
                colorscale=[[0, color], [1, color]]
            ))
            # 시작점 화살표 (p1 쪽) - 반대 방향
            fig.add_trace(go.Cone(
                x=[p1[0]], y=[p1[1]], z=[p1[2]],
                u=[-uvw[0]], v=[-uvw[1]], w=[-uvw[2]],
                sizemode="absolute", sizeref=200, anchor="tip", showscale=False,
                colorscale=[[0, color], [1, color]]
            ))

        # 텍스트
        mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode='text', text=[f"<b>{label}</b>"], 
            textfont=dict(size=14, color=color, family="Arial"),
            showlegend=False
        ))

    # 폭(W)
    add_dimension((0, -OFFSET, 0), (W, -OFFSET, 0), f"폭 : {int(W)}")
    
    # 길이(L)
    add_dimension((-OFFSET, 0, 0), (-OFFSET, L, 0), f"길이 : {int(L)}")
    
    # 높이(H)
    add_dimension((-OFFSET, L, 0), (-OFFSET, L, LIMIT_H), f"높이제한(최대4단) : {int(LIMIT_H)}", color='red')
    
    # 1.3m 제한선 가이드
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), showlegend=False))


    # --- [3] 박스 및 2D 라벨 ---
    annotations = []
    
    for item in truck.items:
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # 색상: 상위 10% 빨간색 강조 (#FF0000)
        # [중요] Box 객체 복사 시 is_heavy 속성이 유지되도록 처리했으므로 정상 표시됨
        if getattr(item, 'is_heavy', False):
            color = '#FF0000' # 완전 빨강
        else:
            color = '#f39c12' # 오렌지
            
        # 박스 면
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=1.0, flatshading=True, name=item.name
        ))
        
        # 테두리
        ex = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        ey = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        ez = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=3), showlegend=False))
        
        # 2D 라벨
        cx, cy, cz = x + w/2, y + d/2, z + h/2
        annotations.append(dict(
            x=cx, y=cy, z=cz,
            text=f"<b>{item.name}</b>",
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(color="white" if getattr(item, 'is_heavy', False) else "black", size=14, family="Arial Black"),
            bgcolor="rgba(0, 0, 0, 0.6)" if getattr(item, 'is_heavy', False) else "rgba(255, 255, 255, 0.7)",
            borderpad=2
        ))

    # --- [4] 뷰 설정 ---
    if camera_view == "top":
        eye = dict(x=0, y=0.1, z=2.5)
        up = dict(x=0, y=1, z=0)
    elif camera_view == "side":
        eye = dict(x=2.5, y=0, z=0.5)
        up = dict(x=0, y=0, z=1)
    else: 
        eye = dict(x=2.0, y=-1.5, z=1.2)
        up = dict(x=0, y=0, z=1)

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor='white',
            camera=dict(eye=eye, up=up),
            annotations=annotations
        ),
        margin=dict(l=0,r=0,b=0,t=0),
        height=700
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Ultimate Load Planner")
st.caption("✅ 물리엔진 | 회전금지 | 1.3m 제한 | 뷰 컨트롤 | 고퀄리티 디자인")

if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'iso'

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='cp949')
        else:
            df = pd.read_excel(uploaded_file)
        
        df.columns = [c.strip() for c in df.columns]
        
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df)

        if st.button("최적 배차 실행", type="primary"):
            st.session_state['run_result'] = load_data(df)

        if 'run_result' in st.session_state:
            items = st.session_state['run_result']
            if not items:
                st.error("데이터 변환 실패.")
            else:
                trucks = run_optimization(items)
                
                if trucks:
                    t_names = [t.name.split(' ')[0] for t in trucks]
                    from collections import Counter
                    cnt = Counter(t_names)
                    summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                    
                    st.success(f"✅ 분석 완료: 총 {len(trucks)}대 ({summary})")
                    
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
                                with st.expander("목록 보기"):
                                    st.write(", ".join([b.name for b in t.items]))
                            with col2:
                                st.plotly_chart(draw_truck_3d(t, st.session_state['view_mode']), use_container_width=True)
                else:
                    st.warning("적재 가능한 차량을 찾지 못했습니다.")

    except Exception as e:
        st.error(f"오류 발생: {e}")
