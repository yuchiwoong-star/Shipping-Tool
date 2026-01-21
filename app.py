import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item
import py3dbp # 모듈 전체 임포트

# ==========================================
# 0. 핵심: 회전 금지 강제 설정 (안전한 버전)
# ==========================================

# 1. RotationType 상수 직접 정의 (Import 에러 방지)
RT_WHD = 0  # 회전하지 않음 (Width, Height, Depth 그대로)

# 2. 회전 금지 함수 정의
def patched_put_item(self, item, pivot):
    # 회전 시도 목록을 [RT_WHD] 하나로 강제 고정
    valid_rotations = [RT_WHD]
    
    fit = False
    valid_item_position = item.position
    item.position = pivot
    
    for rotation_type in valid_rotations:
        item.rotation_type = rotation_type
        dimension = item.get_dimension()
        if self.can_hold(item, pivot, dimension):
            fit = True
            self.items.append(item)
            self.total_weight += item.weight
            break
            
    if not fit:
        item.position = valid_item_position
    return fit

# 3. 라이브러리 함수 덮어쓰기 (Monkey Patch)
py3dbp.Bin.put_item = patched_put_item


# ==========================================
# 1. 설정 및 데이터
# ==========================================
st.set_page_config(layout="wide", page_title="물류 적재 시뮬레이터 (Final)")

TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}

# ==========================================
# 2. 로직 함수
# ==========================================

def create_items_from_df(df):
    items = []
    try:
        # 중량 상위 10% 기준점 계산
        weights = pd.to_numeric(df['중량'], errors='coerce').dropna().tolist()
        sorted_weights = sorted(weights, reverse=True)
        top10_idx = max(0, int(len(weights) * 0.1) - 1)
        heavy_threshold = sorted_weights[top10_idx] if weights else 999999
    except:
        heavy_threshold = 999999

    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            # 사용자 요청: 파일의 폭/높이/길이를 그대로 적용 (회전 X)
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            item = Item(name, w, h, l, weight)
            
            # 시각화용 속성 추가
            item.is_heavy = (weight >= heavy_threshold)
            items.append(item)
        except Exception as e:
            # 데이터 오류가 있어도 멈추지 않고 건너뜀
            continue
    return items

def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
    
    truck_types = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_bin = None
        best_score = -1
        
        for t_name in truck_types:
            spec = TRUCK_DB[t_name]
            CALC_HEIGHT = 1300 # 1.3m 높이 제한
            
            packer = Packer()
            packer.add_bin(Bin(t_name, spec['w'], CALC_HEIGHT, spec['l'], spec['weight']))
            
            for item in remaining_items:
                packer.add_item(item)
            
            # 패킹 실행 (위에서 덮어쓴 함수가 실행됨 -> 회전 안함)
            packer.pack(bigger_first=True)
            
            temp_bin = packer.bins[0]
            packed_count = len(temp_bin.items)
            
            if packed_count > 0:
                # 점수 로직: 많이 실을수록, 작은 차일수록 좋음
                if packed_count == len(remaining_items):
                    score = 100000 - spec['weight'] 
                else:
                    util_weight = temp_bin.get_total_weight() / spec['weight']
                    # 부피 계산 시 0 나누기 방지
                    vol_denom = (spec['w'] * CALC_HEIGHT * spec['l'])
                    util_vol = sum([i.width * i.height * i.depth for i in temp_bin.items]) / (vol_denom if vol_denom else 1)
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
            break
            
    return used_trucks

def create_3d_figure(bin_obj):
    fig = go.Figure()
    
    # 트럭 정보 Parsing
    truck_type = bin_obj.name.split(' ')[0]
    spec = TRUCK_DB.get(truck_type, TRUCK_DB["22톤"])
    W, Real_H, L = spec['w'], spec['h'], spec['l']
    
    # 1. 바닥 (회색)
    fig.add_trace(go.Mesh3d(x=[0,W,W,0], y=[0,0,L,L], z=[0,0,0,0], color='gray', opacity=0.5, name='바닥'))
    
    # 2. 벽면 프레임 및 반투명 벽
    wall_x = [0,0,0,0, W,W,W,W, 0,W,W,0]
    wall_y = [0,L,L,0, 0,L,L,0, L,L,L,L]
    wall_z = [0,0,Real_H,Real_H, 0,0,Real_H,Real_H, 0,0,Real_H,Real_H]
    fig.add_trace(go.Mesh3d(x=wall_x, y=wall_y, z=wall_z, color='lightblue', opacity=0.1, name='벽면', hoverinfo='skip'))

    # 프레임 선
    lines_x = [0,W,W,0,0, 0,W,W,0,0, W,W,0,0, W,W]
    lines_y = [0,0,L,L,0, 0,0,L,L,0, 0,0,L,L, L,L]
    lines_z = [0,0,0,0,0, Real_H,Real_H,Real_H,Real_H,Real_H, 0,Real_H,Real_H,0, 0,Real_H]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='black', width=2), showlegend=False))

    # 높이 제한선 (1.3m)
    fig.add_trace(go.Scatter3d(x=[0,W,W,0,0], y=[0,0,L,L,0], z=[1300]*5, mode='lines', line=dict(color='red', dash='dash'), name='높이제한(1.3m)'))

    # 3. 박스 그리기
    for item in bin_obj.items:
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        
        is_heavy = getattr(item, 'is_heavy', False)
        color = 'red' if is_heavy else '#dddddd'
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
        
        # 박스 번호
        fig.add_trace(go.Scatter3d(
            x=[x + w/2], y=[y], z=[z + h/2],
            mode='text', text=[item.name], textposition="middle center",
            textfont=dict(size=12, color='black'), showlegend=False
        ))

    fig.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,b=0,t=0), height=600)
    return fig

# ==========================================
# 3. 메인 UI
# ==========================================
st.title("🚛 물류 적재 시뮬레이터 (v3.1 Fix)")
st.caption("✅ 회전 금지 | 1.3m 높이 제한 | 5/11톤 최적화 | 에러 수정판")

uploaded_file = st.sidebar.file_uploader("엑셀/CSV 파일 업로드", type=['xlsx', 'csv'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    st.subheader(f"📋 데이터 확인 ({len(df)}건)")
    st.dataframe(df)

    if st.button("최적 배차 실행", type="primary"):
        items = create_items_from_df(df)
        if items:
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
                                    st.markdown(f"**{t.name}**")
                                    st.write(f"- 박스 수: {len(t.items)}개")
                                    st.write(f"- 적재 중량: {t.get_total_weight():,} kg")
                                with col2:
                                    st.plotly_chart(create_3d_figure(t), use_container_width=True)
                    else:
                        st.warning("적재할 수 있는 차량을 찾지 못했습니다. (규격 초과 등)")
                except Exception as e:
                    st.error(f"계산 중 오류 발생: {e}")
        else:
            st.error("데이터를 읽어오지 못했습니다. 컬럼명을 확인해주세요.")
