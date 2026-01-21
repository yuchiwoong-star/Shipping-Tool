import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from py3dbp import Packer, Bin, Item

# 1. 차량 제원 데이터 (이미지 및 표준 제원 기준)
TRUCK_DB = {
    "5톤":  {"w": 2350, "h": 2350, "l": 6200,  "weight": 7000},
    "8톤":  {"w": 2350, "h": 2350, "l": 7300,  "weight": 10000},
    "11톤": {"w": 2350, "h": 2350, "l": 9000,  "weight": 13000},
    "16톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 18000},
    "22톤": {"w": 2350, "h": 2350, "l": 10200, "weight": 24000},
}
COLORS = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C', '#FF9F1C', '#2B2D42', '#EF233C', '#D90429']

# 2. 아이템 생성 함수
def create_items_from_df(df):
    items = []
    # 사용자 요청 컬럼명: 박스번호, 폭, 높이, 길이, 중량
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
            st.error(f"엑셀 컬럼명이 일치하지 않습니다. '박스번호, 폭, 높이, 길이, 중량'인지 확인해주세요. (에러: {e})")
            return None
        except Exception as e:
            st.error(f"{index}행 데이터 오류: {e}")
            continue
    return items

# 3. 차량 최적화 로직 (작은 차부터 채우기)
def get_optimized_trucks(items):
    remaining_items = items[:]
    used_trucks = []
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
            if len(temp_bin.items) == len(remaining_items):
                best_bin = temp_bin
                break
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
    
    # [수정된 부분] head()를 제거하여 전체 데이터를 보여줍니다.
    st.subheader(f"📊 업로드된 데이터 (총 {len(df)}개)")
    st.dataframe(df) # 전체 데이터를 표 형태로 출력

    if st.button("최적 배차 계산 실행", type="primary"):
        items = create_items_from_df(df)
        if items:
            with st.spinner("최적의 차량 조합을 계산 중입니다..."):
                trucks = get_optimized_trucks(items)
            
            if not trucks:
                st.error("적재 가능한 차량을 찾지 못했습니다. 데이터의 크기나 중량을 확인해주세요.")
            else:
                st.success(f"✅ 분석 완료: 총 {len(trucks)}대의 차량이 필요합니다.")
                
                # 각 차량별 탭 생성
                tabs = st.tabs([t.name for t in trucks])
                for i, tab in enumerate(tabs):
                    with tab:
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.info(f"**배차 정보: {trucks[i].name}**")
                            st.write(f"- 적재 박스 수: {len(trucks[i].items)}개")
                            st.write(f"- 총 중량: {trucks[i].get_total_weight():,} kg")
                            
                            # 해당 차량에 실린 박스 번호 목록 표시
                            packed_list = [it.name for it in trucks[i].items]
                            st.write(f"- 실린 박스: {', '.join(packed_list)}")
                            
                        with c2:
                            st.plotly_chart(create_3d_figure(trucks[i]), use_container_width=True)
