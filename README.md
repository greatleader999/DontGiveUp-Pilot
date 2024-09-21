# 🎈 Strimlit web service for estimating main materials prices of Kimchi

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
## 데이터셋 확보 채널:

AT센터: https://www.kamis.or.kr/customer/price/retail/period.do?action=monthly&yyyy=2018&period=10&countycode=&itemcategorycode=200&itemcode=211&kindcode=&productrankcode=0&convert_kg_yn=N

기상청: https://data.kma.go.kr/climate/StatisticsDivision/selectStatisticsDivision.do?pgmNo=158

담당자: 방찬미

1차 확보 및 가공자료: 

[돈기브업김장-rawdata.csv](https://prod-files-secure.s3.us-west-2.amazonaws.com/064c84d8-06c8-4e46-91e2-063af00e9fca/754d5c46-4af1-472c-8c5d-d3c366efa93c/%EB%8F%88%EA%B8%B0%EB%B8%8C%EC%97%85%EA%B9%80%EC%9E%A5-rawdata.csv)

학습 및 평가 대상 항목: 일시,평균기온,평균최고기온,최고기온,평균최저기온,최저기온,평균월강수량,최다월강수량,1시간최다강수량,평균풍속,최대풍속,최대순간풍속,평균습도,최저습도,일조합,일사합,배추값,무값,고추값,마늘값,쪽파값

예측대상 항목: 배추값,무값,고추값,마늘값,쪽파값
