import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. 커스텀 물리 엔진 (기존 규칙 100% 유지)
# ==========================================
# ※ 주의: 이 부분은 절대 수정하지 않았습니다. (회전금지, 중력, 높이제한 유지)

class Box:
    def __init__(self, name, w, h, d, weight):
        self.name = name
        self.w = w
        self.h = h
        self.d = d
        self.weight = weight
        self.x = 0
        self.y = 0
        self.z = 0
        self.is_heavy = False

    @property
    def volume(self):
        return self.w * self.h * self.d

class Truck:
    def __init__(self, name, w, h, d, max_weight):
        self.name = name
        self.w = w
        self.h = h          # 제한 높이 (1300)
        self.d = d          # 길이
        self.max_weight = max_weight
        self.items = []     # 적재된 박스들
        self.total_weight = 0
        # 기준점 (Pivot): (x, y, z) 후보군
        self.pivots = [[0, 0, 0]] 

    def put_item(self, item):
        fit = False
        
        # 무게 초과 체크
        if self.total_weight + item.weight > self.max_weight:
            return False

        # Z -> Y -> X 순 정렬
        self.pivots.sort(key=lambda p: (p[2], p[1], p[0]))

        for p in self.pivots:
            px, py, pz = p
            
            # 1. 범위 체크
            if (px + item.w > self.w) or (py + item.d > self.d) or (pz + item.h > self.h):
                continue

            # 2. 충돌 체크
            if self._check_collision(item, px, py, pz):
                continue

            # 3. 바닥 지지(Gravity) 체크
            if not self._check_support(item, px, py, pz):
                continue

            # 적재 성공
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
        if z == 0: return True 
        support_area = 0
        required_area = item.w * item.d * 0.6 
        for exist in self.items:
            if abs((exist.z + exist.h) - z) < 1.0:
                ox = max(0, min(x + item.w, exist.x + exist.w) - max(x, exist.x))
                oy = max(0, min(y + item.d, exist.y + exist.d) - max(y, exist.y))
                support_area += ox * oy
        return support_area >= required_area

# ==========================================
# 2. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="High-End Load Simulator")

TRUCK_DB = {
    "5톤":  {"w": 2350, "real_h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "real_h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "real_h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "real_h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 3. 로직 함수 (기존 유지)
# ==========================================

def load_data(df):
    items = []
    try:
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        sorted_weights = sorted(weights, reverse=True)
        top10_idx = max(0, int(len(weights) * 0.1) - 1)
        heavy_threshold = sorted_weights[top10_idx] if weights else 999999
    except:
        heavy_threshold = 999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            # [규칙] 회전 절대 금지 (파일 그대로)
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            box = Box(name, w, h, l, weight)
            box.is_heavy = (weight >= heavy_threshold)
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
            limit_h = 1300 # [규칙] 높이 제한 1.3m
            
            temp_truck = Truck(t_name, spec['w'], limit_h, spec['l'], spec['weight'])
            
            # [전략] 부피 큰 순서대로 적재
            test_items = sorted(remaining_items, key=lambda x: x.volume, reverse=True)
            packed_count = 0
            
            for item in test_items:
                item_copy = Box(item.name, item.w, item.h, item.d, item.weight)
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
# 4. 고퀄리티 3D 시각화 (디자인 대폭 개선)
# ==========================================
def draw_truck_3d(truck):
    fig = go.Figure()
    spec = TRUCK_DB[truck.name.split(' ')[0]]
    W, L, Real_H = spec['w'], spec['l'], spec['real_h']
    LIMIT_H = 1300
    
    # --- [1] 세련된 트럭 바디 (Metallic Style) ---
    
    # 바닥 (그리드 느낌의 짙은 회색)
    fig.add_trace(go.Mesh3d(
        x=[0, W, W, 0], y=[0, 0, L, L], z=[0, 0, 0, 0],
        color='#2c3e50', opacity=1.0, name='Floor', hoverinfo='none'
    ))

    # 적재함 벽면 (유리 같은 반투명 흰색/하늘색)
    wall_color = '#ecf0f1' 
    wall_opacity = 0.15
    
    # 벽면 좌표 생성 함수
    def create_wall(xs, ys, zs):
        return go.Mesh3d(x=xs, y=ys, z=zs, color=wall_color, opacity=wall_opacity, hoverinfo='none', showlegend=False)

    # 좌/우/앞 벽
    fig.add_trace(create_wall([0,0,0,0], [0,L,L,0], [0,0,Real_H,Real_H])) # 좌
    fig.add_trace(create_wall([W,W,W,W], [0,L,L,0], [0,0,Real_H,Real_H])) # 우
    fig.add_trace(create_wall([0,W,W,0], [L,L,L,L], [0,0,Real_H,Real_H])) # 앞

    # 프레임 (깔끔한 외곽선)
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='#34495e', width=3), showlegend=False))

    # --- [2] 치수선 및 라벨 (Dimension Lines) ---
    
    # 치수선 그리는 함수
    def add_dim_line(p1, p2, text_pos, label):
        # 선
        fig.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode='lines+text', line=dict(color='black', width=2, dash='solid'),
            showlegend=False
        ))
        # 텍스트
        fig.add_trace(go.Scatter3d(
            x=[text_pos[0]], y=[text_pos[1]], z=[text_pos[2]],
            mode='text', text=[label], textfont=dict(size=12, color='black', family="Arial Black"),
            showlegend=False
        ))

    # 폭(W) 표시 (트럭 뒤쪽 아래)
    add_dim_line((0, -200, 0), (W, -200, 0), (W/2, -400, 0), f"폭: {W}mm")
    
    # 길이(L) 표시 (트럭 왼쪽 바닥)
    add_dim_line((-200, 0, 0), (-200, L, 0), (-400, L/2, 0), f"길이: {L}mm")
    
    # 높이 제한(H) 표시 (트럭 왼쪽 위) -> 1.3m 제한선 기준
    add_dim_line((-200, L, 0), (-200, L, LIMIT_H), (-400, L, LIMIT_H/2), f"제한높이: {LIMIT_H}mm")

    # 높이 제한 가이드라인 (빨간 점선면)
    fig.add_trace(go.Scatter3d(
        x=[0,W,W,0,0], y=[0,0,L,L,0], z=[LIMIT_H]*5, 
        mode='lines', line=dict(color='#e74c3c', width=4, dash='dash'), name='높이제한(1.3m)'
    ))


    # --- [3] 박스 그리기 (깔끔한 스타일) ---
    for item in truck.items:
        x, y, z = item.x, item.y, item.z
        w, h, d = item.w, item.h, item.d
        
        # 색상: 상위 10%는 붉은 계열, 나머지는 베이지/골판지 색상
        if item.is_heavy:
            color = '#e74c3c' # Flat Red
            border_color = '#c0392b'
        else:
            color = '#f1c40f' # Cardboard Yellow/Orange
            border_color = '#d35400'
            
        # 박스 메쉬 (Flat shading으로 깔끔하게)
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=1.0, flatshading=True, name=item.name, lighting=dict(ambient=0.5, diffuse=0.8)
        ))
        
        # 박스 테두리 (선명하게)
        ex = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        ey = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        ez = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='black', width=1.5), showlegend=False))
        
        # 박스 번호 (측면 중앙, 잘 보이게)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[item.name], textposition="middle center",
            textfont=dict(size=14, color='black', family="Arial Black"), showlegend=False
        ))

    # --- [4] 카메라 및 씬 설정 (Banana Tool 스타일) ---
    fig.update_layout(
        scene=dict(
            aspectmode='data', 
            xaxis=dict(visible=False, showgrid=False), # 축 숨김 (도면 느낌)
            yaxis=dict(visible=False, showgrid=False), 
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='white' # 깔끔한 흰색 배경
        ), 
        margin=dict(l=0,r=0,b=0,t=0), 
        height=700,
        paper_bgcolor='white'
    )
    return fig

# ==========================================
# 5. 메인 UI
# ==========================================
st.title("📦 Smart Load Planner Pro")
st.caption("✅ 물리엔진 탑재 | 회전금지 | 1.3m 제한 | 치수 도면 제공")

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
            items = load_data(df)
            if not items:
                st.error("데이터 변환 실패.")
            else:
                with st.spinner("물리 엔진 시뮬레이션 중..."):
                    trucks = run_optimization(items)
                    
                    if trucks:
                        t_names = [t.name.split(' ')[0] for t in trucks]
                        from collections import Counter
                        cnt = Counter(t_names)
                        summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                        
                        st.success(f"✅ 분석 완료: 총 {len(trucks)}대 ({summary})")
                        
                        tabs = st.tabs([t.name for t in trucks])
                        for i, tab in enumerate(tabs):
                            with tab:
                                c1, c2 = st.columns([1, 4])
                                t = trucks[i]
                                with c1:
                                    st.markdown(f"### **{t.name}**")
                                    st.write(f"- 박스 수: **{len(t.items)}개**")
                                    st.write(f"- 총 중량: **{t.total_weight:,} kg**")
                                    with st.expander("박스 목록 보기"):
                                        st.write(", ".join([b.name for b in t.items]))
                                with c2:
                                    st.plotly_chart(draw_truck_3d(t), use_container_width=True)
                    else:
                        st.warning("적재 가능한 차량을 찾지 못했습니다.")
    except Exception as e:
        st.error(f"오류 발생: {e}")
