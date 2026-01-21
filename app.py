import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item

# 1. 차량 제원 데이터 (이미지 기준)
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}
COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C', '#FF9F1C', '#2B2D42', '#EF233C', '#D90429']

# 2. 아이템 생성 함수 (수정된 컬럼명 반영)
def create_items_from_df(df):
    items = []
    # 사용자 요청 컬럼명: 박스번호, 폭, 높이, 길이, 중량
    # 수량 컬럼이 별도로 없을 경우 각 행을 1개로 간주하거나, 
    # 데이터에 '수량' 컬럼이 있다면 아래 로직에 추가할 수 있습니다.
    for index, row in df.iterrows():
        try:
            name = str(row['박스번호'])
            w = float(row['폭'])
            h = float(row['높이'])
            l = float(row['길이'])
            weight = float(row['중량'])
            
            # py3dbp: Item(이름, 가로, 높이, 깊이, 무게)
            items.append(Item(name, w, h, l, weight))
        except KeyError as e:
            st.error(f"엑셀 컬럼명이 일치하지 않습니다: {e}")
            return None
        except Exception as e:
            st.error(f"{index}행 데이터 오류: {e}")
            continue
    return items

# 3. 차량 최적화 로직
def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
    # 작은 차부터 시뮬레이션하기 위해 정렬
    sorted_keys = sorted(TRUCK_DB.keys(), key=lambda k: TRUCK_DB[k]['weight'])

    while remaining_items:
        best_bin = None
        max_packed_count = -1
        
        for t_name in sorted_keys:
            spec = TRUCK_DB[t_name]
            packer = Packer()
            packer.add_bin(Bin(t_name, spec['w'], spec['h'], spec['l'], spec['weight']))
            for item in remaining_items:
                packer.add_item(item)
            packer.pack(bigger_first=True, number_of_decimals=0)
            
            temp_bin = packer.bins[0]
            # 모든 남은 짐이 들어가면 해당 트럭 확정
            if len(temp_bin.items) == len(remaining_items):
                best_bin = temp_bin
                break
            # 다 안 들어가면 가장 많이 실리는 트럭 저장
            if len(temp_bin.items) > max_packed_count:
                max_packed_count = len(temp_bin.items)
                best_bin = temp_bin

        if best_bin and len(best_bin.items) > 0:
            best_bin.name = f"{best_bin.name} (No.{len(used_trucks)+1})"
            used_trucks.append(best_bin)
            packed_names = [item.name for item in best_bin.items]
            remaining_items = [i for i in remaining_items if i.name not in packed_names]
        else:
            break
    return used_trucks

# 4. 3D 시각화 함수
def create_3d_figure(bin_obj):
    fig = go.Figure()
    W, H, D = bin_obj.width, bin_obj.height, bin_obj.depth
    # 적재함 외곽선
    lx, ly, lz = [0,W,W,0,0,0,W,W,0,0,W,W,0,0,W,W], [0,0,D,D,0,0,0,D,D,0,0,0,D,D,D,D], [0,0,0,0,0,H,H,H,H,H,H,0,0,H,H,0]
    fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='black', width=3), hoverinfo='none'))
    
    for i, item in enumerate(bin_obj.items):
        x, y, z = float(item.position[0]), float(item.position[2]), float(item.position[1])
        w, h, d = float(item.width), float(item.depth), float(item.height)
        fig.add_trace(go.Mesh3d(
            x=[x,x+w,x+w,x,x,x+w,x+w,x], y=[y,y,y+d,y+d,y,y,y+d,y+d], z=[z,z,z,z,z+h,z+h,z+h,z+h],
            i=[7,0,0,0,4,4,6,6,4,0,3,2], j=[3,4,1,2,5,6,5,2,0,1,6,3], k=[0,7,2,3,6,7,1,1,5,5,7,6],
            color=COLORS[i % len(COLORS)], opacity=1, flatshading=True, name=item.name
        ))
    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=3, z=1)), margin=dict(l=0,r=0,b=0,t=0))
    return fig

# 5. 메인 UI 구성
st.title("🚛 자동 차량 추천 배차 시스템")
uploaded_file = st.sidebar.file_uploader("엑셀 파일 업로드", type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("데이터 미리보기:", df.head())

    if st.button("최적 배차 계산 실행"):
        items = create_items_from_df(df)
        if items:
            with st.spinner("계산 중..."):
                trucks = get_optimized_trucks(items)
            
            st.success(f"결과: 총 {len(trucks)}대의 차량이 필요합니다.")
            
            # 각 차량별 탭 생성
            tabs = st.tabs([t.name for t in trucks])
            for i, tab in enumerate(tabs):
                with tab:
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**차량:** {trucks[i].name}")
                        st.write(f"**적재 박스:** {len(trucks[i].items)}개")
                        st.write(f"**총 중량:** {trucks[i].get_total_weight():,} kg")
                    with col2:
                        st.plotly_chart(create_3d_figure(trucks[i]), use_container_width=True)
