import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item

# ==========================================
# 0. [핵심] 회전 금지 강제 패치 (오류 수정됨)
# ==========================================
# py3dbp 라이브러리가 박스를 돌리는 것을 원천 차단합니다.
def no_rotation_put_item(self, item, pivot):
    fit = False
    valid_item_position = item.position
    item.position = pivot
    
    # [수정] 회전 시도 루프를 제거하고, 0번(원본 방향)만 시도
    item.rotation_type = 0 
    
    dimension = item.get_dimension()
    # 회전된(여기선 원본) 치수로 적재 가능한지 확인
    if self.can_hold(item, pivot, dimension):
        fit = True
        self.items.append(item)
        # [삭제] 에러를 유발하던 self.total_weight += item.weight 코드 삭제

    if not fit:
        item.position = valid_item_position
    return fit

# 라이브러리의 Bin 클래스 메서드를 위 함수로 교체
Bin.put_item = no_rotation_put_item


# ==========================================
# 1. 설정 및 차량 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터 (Final Fix)")

# 차량 제원 (mm, kg)
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 2. 데이터 처리 및 로직
# ==========================================

def create_items_from_df(df):
    items = []
    # 색상 강조를 위한 중량 상위 10% 기준 계산
    try:
        # 문자열이 섞여있을 수 있으므로 숫자로 변환
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        if not weights:
            heavy_threshold = 999999
        else:
            sorted_weights = sorted(weights, reverse=True)
            # 상위 10% 인덱스 계산
            top10_idx = max(0, int(len(weights) * 0.1) - 1)
            heavy_threshold = sorted_weights[top10_idx]
    except Exception:
        heavy_threshold = 999999

    # 컬럼명 공백 제거 (사용자 편의)
    df.columns = [c.strip() for c in df.columns]

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            
            # [요청 반영] 파일의 치수 그대로 사용 (임의 정렬 X)
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            # Item 생성 (이름, 가로, 높이, 세로, 무게)
            # 엑셀의 '폭' -> Width, '높이' -> Height, '길이' -> Depth
            item = Item(name, w, h, l, weight)
            
            # 시각화용 속성
            item.is_heavy = (weight >= heavy_threshold)
            items.append(item)
            
        except Exception:
            continue
    return items

def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
    
    # 작은 차 -> 큰 차 순서로 정렬 (비용 효율화)
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_bin = None
        best_score = -1
        
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            
            # [규칙] 적재 높이는 1.3m (1300mm)로 제한
            CALC_HEIGHT = 1300
            
            packer = Packer()
            # Bin 생성 (이름, 폭, 계산용높이, 길이, 허용하중)
            packer.add_bin(Bin(t_name, spec['w'], CALC_HEIGHT, spec['l'], spec['weight']))
            
            for item in remaining_items:
                packer.add_item(item)
            
            # 적재 실행 (회전 없이 진행됨)
            packer.pack(bigger_first=True)
            
            temp_bin = packer.bins[0]
            packed_count = len(temp_bin.items)
            
            if packed_count > 0:
                # 점수 로직
                # 1. 짐을 다 실을 수 있다면 -> 가장 작은 트럭이 최고 (비용 절감)
                if packed_count == len(remaining_items):
                    score = 100000 - spec['weight'] 
                else:
                    # 2. 다 못 싣는다면 -> 꽉 채우는(효율 좋은) 트럭 선호
                    util_weight = temp_bin.get_total_weight() / spec['weight']
                    vol_denom = (spec['w'] * CALC_HEIGHT * spec['l'])
                    if vol_denom == 0: vol_denom = 1
                    util_vol = sum([i.width * i.height * i.depth for i in temp_bin.items]) / vol_denom
                    score = (util_weight + util_vol) * 100
                
                if score > best_score:
                    best_score = score
                    best_bin = temp_bin

        if best_bin and len(best_bin.items) > 0:
            best_bin.name = f"{best_bin.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_bin)
            packed_names = [i.name for i in best_bin.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            break # 더 이상 적재 불가
            
    return used_trucks

def create_3d_figure(bin_obj):
    fig = go.Figure()
    
    # 트럭 정보 Parsing
    truck_type = bin_obj.name.split(' ')[0]
    spec = TRUCK_DB.get(truck_type, TRUCK_DB["22톤"])
    W, Real_H, L = spec['w'], spec['h'], spec['l']
    
    # -- 1. 트럭 디자인 (컨테이너 형태) --
    
    # 바닥 (회색)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[0,0,L,L], z=[0,0,0,0], color='gray', opacity=0.6, name='바닥'))
    
    # 벽면 (반투명 파랑/회색)
    wall_color = 'lightblue'
    wall_opacity = 0.1
    
    # 좌측벽 (x=0)
    fig.add_trace(go.Mesh3d(x=[0,0,0,0], y=[0,L,L,0], z=[0,0,Real_H,Real_H], color=wall_color, opacity=wall_opacity, showlegend=False))
    # 우측벽 (x=W)
    fig.add_trace(go.Mesh3d(x=[W,W,W,W], y=[0,L,L,0], z=[0,0,Real_H,Real_H], color=wall_color, opacity=wall_opacity, showlegend=False))
    # 앞쪽벽 (y=L)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[L,L,L,L], z=[0,0,Real_H,Real_H], color=wall_color, opacity=wall_opacity, showlegend=False))

    # 프레임 선 (진한 회색)
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='black', width=3), showlegend=False))

    # 높이 제한선 (1.3m, 빨간 점선)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', width=4, dash='dash'), name='높이제한(1.3m)'))

    # -- 2. 박스 그리기 --
    for item in bin_obj.items:
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        
        # 색상: 상위 10% 빨강, 나머지 회색
        is_heavy = getattr(item, 'is_heavy', False)
        color = '#FF4B4B' if is_heavy else '#E0E0E0'
        opacity = 1.0 if is_heavy else 0.8
        
        # 박스 메쉬
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x, x,x+w,x+w,x],
            y=[y,y,y+d,y+d, y,y,y+d,y+d],
            z=[z,z,z,z, z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=color, opacity=opacity, flatshading=True, name=item.name
        ))
        
        # 박스 테두리
        edge_x = [x,x+w,x+w,x,x, x,x+w,x+w,x,x, x+w,x+w,x+w,x+w, x,x]
        edge_y = [y,y,y+d,y+d,y, y,y,y+d,y+d,y, y,y,y+d,y+d, y+d,y+d]
        edge_z = [z,z,z,z,z, z+h,z+h,z+h,z+h,z+h, z,z+h,z+h,z, z,z+h]
        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=1), showlegend=False))
        
        # 박스 번호 (측면 중앙)
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[item.name], textposition="middle center",
            textfont=dict(size=12, color='black', weight='bold'), showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            aspectmode='data', # 실제 비율 유지
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
        ), 
        margin=dict(l=0,r=0,b=0,t=0), 
        height=600
    )
    return fig

# ==========================================
# 3. 메인 UI
# ==========================================
st.title("🚛 물류 적재 시뮬레이터 (Final Fix)")
st.caption("✅ 회전 금지(원본 방향 유지) | 1.3m 높이 제한 | 상위 10% 중량 강조")

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    # 파일 읽기 (CSV 한글 깨짐 방지 등 강건성 추가)
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='cp949') # 한글 CSV 대비
        else:
            df = pd.read_excel(uploaded_file)
            
        st.subheader(f"📋 데이터 확인 ({len(df)}건)")
        st.dataframe(df)

        if st.button("최적 배차 실행", type="primary"):
            items = create_items_from_df(df)
            if not items:
                st.error("데이터를 변환할 수 없습니다. 컬럼명(박스번호, 폭, 높이, 길이, 중량)을 확인해주세요.")
            else:
                with st.spinner("최적의 차량을 계산 중입니다..."):
                    try:
                        trucks = get_optimized_trucks(items)
                        
                        if trucks:
                            t_names = [t.name.split(' ')[0] for t in trucks]
                            from collections import Counter
                            cnt = Counter(t_names)
                            summary = ", ".join([f"{k} {v}대" for k,v in cnt.items()])
                            
                            st.success(f"✅ 배차 완료: 총 {len(trucks)}대 ({summary})")
                            
                            tabs = st.tabs([t.name for t in trucks])
                            for i, tab in enumerate(tabs):
                                with tab:
                                    col1, col2 = st.columns([1, 3])
                                    t = trucks[i]
                                    with col1:
                                        st.markdown(f"### **{t.name}**")
                                        st.write(f"- 박스 수: {len(t.items)}개")
                                        st.write(f"- 적재 중량: {t.get_total_weight():,} kg")
                                        st.write(f"- 적재 부피율: {t.get_volume_utilization():.1f}%")
                                        with st.expander("적재 상세 목록"):
                                            st.write(", ".join([item.name for item in t.items]))
                                    with col2:
                                        st.plotly_chart(create_3d_figure(t), use_container_width=True)
                        else:
                            st.warning("적재할 수 있는 차량을 찾지 못했습니다. (규격 초과 등)")
                    except Exception as e:
                        st.error(f"계산 중 상세 오류 발생: {e}")
    except Exception as e:
        st.error(f"파일 읽기 오류: {e}")
