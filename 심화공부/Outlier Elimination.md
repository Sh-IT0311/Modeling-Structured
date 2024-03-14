## Outlier Elimination
* preprocessing 또는 feature engineering에 활용할 수 있는 방법론을 작성함
* EDA를 통해 수동으로 제거
    * 주관적인 판단으로 눈에 띄는 경우에만 가능하다는 한계가 있음
* IQR(InterQuartile Range)를 활용한 방식
    * 제 3사분위(Q3, 75%의 위치) - 제 1사분위(Q1, 25%의 위치)를 IQR 이라고 함
    * 최댓값 = 제 3사분위 + 1.5 * IQR, 최솟값 = 제 1사분위 - 1.5 * IQR 라고 할때, 최댓값과 최솟값을 벗어나는 값들을 이상치로 정의함
    * 예시코드
        > IQR = quantile_75 - quantile_25<br>
        > IQR_weight = IQR*weight<br>
        >lowest = quantile_25 - IQR_weight<br>
        >highest = quantile_75 + IQR_weight<br>
        >outlier_idx = df[column][ (df[column] < lowest) | (df[column] > highest) ].index