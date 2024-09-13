import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
import chardet
from matplotlib import font_manager, rc

df=pd.read_csv('data.csv', encoding='euc-kr')

def train_model(X, y, test_size, k_neighbors):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=77)
    model = KNeighborsRegressor(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, X_test

def plot_correlation(data, features, target):
    corr_data = data[features + [target]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    plt.xticks(rotation=45)  # x축 레이블 회전
    plt.yticks(rotation=45)  # y축 레이블 회전
    plt.tight_layout()  # 레이아웃 조정
    return fig

# 데이터 형식 변환 함수
def format_data(df):
    # "일시"를 년, 월 형식으로 변환
    df['일시'] = pd.to_datetime(df['일시']).dt.strftime('%Y-%m')
    
    # 열 이름이 '기온', '강수량', '값'과 관련된 경우 단위 추가
    temperature_columns = ['평균기온', '평균최고기온', '최고기온', '평균최저기온', '최저기온']
    rainfall_columns = ['평균월강수량', '최다월강수량', '1시간최다강수량']
    price_columns = ['배추값', '무값', '고추값', '마늘값', '쪽파값']
    
    # 각 열에 대해 단위 추가
    for col in temperature_columns:
        if col in df.columns:
            df[col] = df[col].astype(str) + ' °C'
    for col in rainfall_columns:
        if col in df.columns:
            df[col] = df[col].astype(str) + ' mm'
    for col in price_columns:
        if col in df.columns:
            df[col] = df[col].astype(str) + ' 원'
    
    return df

# Main 함수 내에서 데이터 포맷팅 후 표시
def main():
    st.title("포기는 배추 셀 때🥬 - Don\'t give up KIMJANG😤")

    kimchi_data = df
    if kimchi_data is None:
        st.stop()
    
    # 데이터 포맷팅 적용
    formatted_data = format_data(kimchi_data)

    st.write("김장 데이터셋 미리보기:", formatted_data.head())

    #selected_features=[]
    #target_column=[]

    st.sidebar.header('날씨 선택✅')
    input_features = ['평균기온', '평균최고기온', '최고기온', '평균최저기온', '최저기온',
                      '평균월강수량', '최다월강수량', '1시간최다강수량']
    selected_features = st.sidebar.multiselect("",input_features, default=input_features[:3]) 

    st.sidebar.header('야채 선택🎯')
    target_options = ['배추값', '무값', '고추값', '마늘값', '쪽파값']
    target_column = st.sidebar.selectbox("", target_options)

    st.sidebar.header('고급 옵션')
    test_size = st.sidebar.slider('테스트 데이터 비율', 0.1, 0.5, 0.2)
    k_neighbors = st.sidebar.slider('K-NN 모델의 이웃 개수', 1, 20, 5)

    X = kimchi_data[selected_features]
    y = kimchi_data[target_column]

    if X.empty or y.empty:
        st.error("특성 또는 타겟 데이터가 비어있습니다.")
        st.stop()

    try:
        model, mse, r2, X_test = train_model(X, y, test_size, k_neighbors)

        st.sidebar.header('💲가격 예측하기💲')
        user_input = {feature: st.sidebar.number_input(f'{feature}', value=X[feature].mean()) for feature in selected_features}

        st.write(f"모델 평균 제곱 오차 (MSE): {mse:.2f}")
        st.write(f"결정 계수 (R^2): {r2:.2f}")

        if st.sidebar.button('실행'):
            prediction = model.predict(pd.DataFrame([user_input]))
            st.sidebar.write(f"예측된 {target_column}: {prediction[0]:.2f}")

        st.write("선택된 특성과 타겟 변수 간의 상관관계:")
        fig = plot_correlation(kimchi_data, selected_features, target_column)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"모델 훈련 또는 예측 중 오류 발생: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()
