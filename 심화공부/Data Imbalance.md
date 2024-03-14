## Data Imbalance
* 목차
    * data-aspect
        * sampling
            * under-sampling
            * over-sampling
            * combine sampling
                * under-sampling + over-sampling
    * model-aspect
        * cost sensitive learning
        * novelty detection
            * data imbalance를 다루는 기교보단 classification 그 자체에 대한 방법론

* (target)클래스 별 관측치의 수가 현저하게 차이가 나는 데이터
    * 일반적으로 이상(비율이 작음)을 정확히 분류 하는 것이 중요
        * ex> 의사의 오진은 치명적임
    * 다수 클래스의 편향된(멀어지는) 분류 경계선 형성
        * 소수 클래스를 정확히 분류하기 힘듬
        * 불균형한 데이터를 그대로 예측하면 과적합(overfitting)문제가 발생할 가능성이 큼
    * 높은 예측 정확도를 보임
        * 모델의 파라미터(=가중치)가 비율이 많은 클래스를 더 예측하려고 하기 때문에 Accuracy는 높아질 수 있지만 비율이 작은 클래스에 대한 Precision, recall 등 성능이 낮아지는 문제가 발생함
        * 모델 성능에 대한 왜곡이 있음
* **분류문제를 해결 할 때 가장 먼저 데이터 분포(target class)를 확인해야함**
* 해결방법
    * 데이터 측면
        * sampling method(샘플링 기법)
            * 다수 클래스의 편향된 분류 경계선을 조정하는 효과
            * 샘플링 편향 주의
            * undersampling
                * 비율이 많은 클래스를 줄여서 비율을 맞춤
                * 장점
                    * 비율이 많은 클래스 데이터를 제거하므로 계산시간이 감소한다.
                    * 데이터 클렌징으로 클래스 오버랩을 감소시킨다.
                * 단점
                    * 데이터 제거로 인한 information loss 발생
                        * 필요한 정보제거로 인한 문제 발생 주의
                * 알고리즘
                    * Random under sampling
                        * 무작위로 비율이 많은 클래스를 샘플링하여 비율을 맞춤
                        * 장점
                            * 소요 시간이 매우 짧음
                            * 성능이 준수함
                        * 단점
                            * 무작위로 샘플링하기 때문에 매번 다른 결과(성능)를 보임
                    * Tomek links
                        * 토멕링크 정의 : dist(xi, xk) < dist(xi, xj) 또는 dist(xj, xk) < dist(xi, xj)가 되는 관측치 xk가 없는 경우
                            * xi : 비율이 많은 클래스, xj : 비율이 적은 클래스, xk : 임의의 데이터
                            * 토멕링크로 묶인 두 샘플은 둘 중 하나가 노이즈 이거나 둘다 경계선 근처에 있음
                                * 여기서 노이즈는 상대편 클래스에 존재하는 데이터를 의미함
                        *  토멕링크로 묶어서 비율이 많은 데이터를 제거하는 방법
                        * boundary(=border line)를 비율이 많은 클래스 쪽으로 밀어 붙이는 효과
                        * 장점
                            * 비율이 많은 클래스의 중심분포를 어느정도 유지
                            * 정보의 유실이 적음
                        * 단점
                            * 토멕링크로 묶이는 데이터들이 한정적이기 때문에 큰 undersampling 효과를 얻을 수 없음
                    * CNN(Condensed Nearest Neighbor)
                        * 방법
                            1. 소수 클래스 전체와 다수 클래스에서 무작위로 하나 선택한 데이터로 서브 데이터를 구성
                            2. 무작위로 선택한 하나의 데이터를 제외한 다수 클래스를 서브 데이터와 1-NN 알고리즘을 통해 분류
                            3. 1-NN 알고리즘에서 무작위 하나 선택한 데이터와 소수 클래스를 비교해서 무작위로 하나 선택한 데이터와 가깝게 판별된 다수 클래스 데이터를 제거
                            * (추측) 추후에 무작위로 선택한 데이터도 1-NN 분류 및 분류 결과로 제거 될지 결정되는 것 같음
                        * 주의사항
                            * 무조건 1-NN, 즉 k = 1만 사용해야함
                                * 다수 클래스에서 무작위로 하나만 선택했기 때문에 k가 1 초과일 경우, 무조건 소수 클래스로 분류하게 됨
                                    * 즉, 제거되는 요소가 없어서 undersampling이 이루어지지 않음
                    * OSS(OneSide Selection)
                        * Tomek Links(순서1) + CNN(순서2)
                            * 두 개의 undersampling이 모두 이루어짐
                            * CNN과 Tomek links의 단점을 보완하는 형태
                    * Edited Nearest Neighbors(ENN)
                        * KNN방식이랑 비슷하며 소수 클래스 주변의 다중 클래스 값을 제거
                        * 제거 효과가 크지 않음
                    * Neighborhood cleaning rule
                        * CNN + ENN
                        * 다수 클래스에 대한 제거 효과가 크진 않지만 더 직관적으로 두 클래스를 나눌 수 있음
                * oversampling
                    * 비율이 적은 클래스를 증폭시켜서 비율을 맞춤
                    * 장점
                        * 데이터 정보 손실이 없다.
                        * undersampling에 비해 높은 분류 정확도를 보인다
                    * 단점
                        * 과적합 문제 발생
                            * 임의로 생성된 데이터로 일반화에 멀어 질 수 있음
                        * 데이터  증가로 인해 계산시간이 증가함
                        * 노이즈 또는 이상치에 민감함
                            * 생성된 데이터의 질이 낮아질 수 있음
                    * 알고리즘
                        * resampling(= Random over sampling)
                            * 소수 클래스를 단순히 복사해서 비율을 맞춤
                            * 소수 클래스에 가중치를 증가시키는 것과 비슷함
                            * 단점
                                * 소수 클래스에 과적합이 발생할 가능성이 있음

                        * SMOTE(synthetic minority oversampling technique)
                            * 방법
                                1. 임의의 소수 클래스를 선택
                                2. 1번에서 선택한 데이터와 가까운 K(>= 2)개의 데이터를 선택
                                    * 1번에서 선택한 데이터와 가장 가까운 데이터 사이에서만 데이터 생성을 방지하기 위해 2개 이상 선택함
                                3. 2번에서 선택한 K개의 데이터 중에서 랜덤으로 하나 선택
                                4. 1번에서 선택한 데이터와 3번에서 랜덤으로 선택된 데이터를 통해 데이터를 생성
                                    * Synthetic data = X + u * (X(nn) - X)
                                        * X는 1번에서 선택한 데이터
                                        * X(nn)는 3번에서 랜덤으로 선택된 데이터
                                        * u는 균등분포(uniform distribution, 0~1)에서 선택된 수
                                5. 모든 소수 클래스에 대해서 1~4번 과정 반복
                            * 주의사항
                                * K는 2 이상으로 해야함
                                    * K가 1이라면 생성되는 데이터의 다양성이 떨어짐

                        * borderline - SMOTE
                            * 경계선 근처(danger 관측치)에 있는 소수 클래스만 SMOTE 진행
                                * 각 danger 관측치에 대해서 s개(s < k) 만큼의 데이터 생성
                            * 경계선 근처의 기준
                                * 각 소수 클래스에 대해서 k개 주변을 탐색 후 k개 중 다수 클래스의 수(k')를 확인
                                    1. k = k' : Noise 관측치, borderline이 아니다.
                                    2. k/2 < k' < k: danger 관측치, borderline 이다
                                    3. 0 <= k' <= k/2 : safe 관측치, borderline 아니다
                        * ADASYN(adaptive synthetic sampling approach)
                            * 각 소수 클래스에서 주변의 다수 클래스의 수에 따라 유동적으로 생성하는 기법
                                * 각 소수 클래스에 할당된 개수 만큼 SMOTE 진행
                            * 각 소수 클래스에 할당되는 개수 구하는 방법
                                * 각 소수 클래스 주변에 얼만큼의 다수 클래스가 있는가를 정량화한 지표를 구함
                                    1.  ri를 구함
                                        * ri = i번째 소수 클래스 주변(k개)에서 관측된 다수 클래스의 수 / k, i는 1~m(소수 클래스의 총 개수)
                                    2. ri 정규화(스케일링)
                                        * ri / sum(ri)
                                    3. ri * G
                                        * G = 다수클래수 개수 - 소수 클래수 개수
                                        * 소수는 반올림해서 최종적으로 각 소수 클래스의 할당되는 개수를 의미함
                            * 장점
                                * 과적합 문제를 해결함
                            * 단점
                                * 시간이 오래 걸림

                    * (이미지) data augmentation

            * Combine Sampling
                * oversampling + undersampling
                * 알고리즘
                    * SMOTE + ENN
                    * SMOTE + TOMEK
    * 모델링 측면
        * cost sensitive learning(비용 기반 학습)
            * 소수 클래스에 가중치를 줌
        * Novelty detection(단일 클래스 분류기법)
            * 다수 클래스를 잘 설명하는 경계선을 생성
                * 경계선 내부는 다수 클래스로 분류
                * 경계선 외부는 소수 클래스로 분류

* 참고사이트
    * [유튜브 동영상](https://www.youtube.com/watch?v=tpow70KGTYY&list=PLpIPLT0Pf7IoTxTCi2MEQ94MZnHaxrP0j&index=8)
    * [참고블로그](https://shinminyong.tistory.com/34)
                        