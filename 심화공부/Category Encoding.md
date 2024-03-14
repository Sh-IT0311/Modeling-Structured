## Category Encoding
* 데이터 분석에 실제 적용하고자 하는 알고리즘 위주로 정리함
* train set에서는 등장 하지않고 validation/test set에서 새로운 category가 등장하는 경우가 있음
    * 새로운 category가 등장에 0벡터(0,0, ..., 0)를 Encoding 하는 것과 같은 대비가 필요함
* 명목형 데이터의 경우
    * One-hot Encoding / Binary Encoding / Hashing Encoding
        * one-hot Encoding
            * feature를 구성하는 category마다 dummy feature(0 / 1)를 부여함
                * N개의 dummy feature가 필요함
                    * N = feature를 구성하는 category의 종류
            * 카디널리티가 높은 feature의 불리함
        * Binary Encoding
            * feature를 구성하는 category마다 binary digits를 부여함
            * int(log2(N) + 1)개의 추가적인 dummy feature 필요함
                * N = feature를 구성하는 category의 종류
        * Hashing Encoding
            * Hashing function을 사용하여 반환된 숫자를 이용한 Encoding 기법
                * Hashing function : 임의의 input에 대해서 고정된 길이(범위)의 숫자를 반환하는 함수
                * Hashing function에 의해 반환된 숫자를 dummy feature의 인덱스로 변환하는 방법에 대해서는 못찾음
                    * 예시정도는 찾았는데..
                        * ex> by hasing : 41523162, n_components : 8
                        * 41523162 % 8 -> "2"로 변환
            * One-hot encoding, Binary Encoding과 달리 원하는 차원의 갯수(파라미터)를 결정해야됨
                * HashingEncoder의 parameter : n_components
                * 차원의 갯수가 적으면 collision의 문제가 있음
            * 다수의 feature를 한꺼번에 encoding을 할 수 있음
                * HashingEncoder의 parameter : cols
                * 각각의 feature의 대해 같은 dummy feature를 가리킨다면 2가 나올 수도 있음
                    * ex> [0,0,2,0,...0,0]
            * 새로운 카테고리에 대해서 영벡터를 할당하는 One-hot Encoding과 Binary Encoding과 달리 Hashing Encoding은 새로운 카테고리에 대해서도 고유한 encoding을 진행 할 수 있음
                * 기존 모델에 online learning이 가능함
                * One-hot Encoding, Binary Encoding은 dummy feature를 늘려서 모델의 학습을 다시 할 여지가 있음
            * High cardinality의 feature의 경우에도 One-hot encdoing보다 적은 dummy feature를 만들어낼 수 있음
    * Mean Encoding(= Target Encoding)
        * feature를 구성하는 category마다 mean value of target variable를 부여함
        * 카디널리티와 무관하고 빠르게 학습할 수 있지만 과적합의 위험성이 높음
* 순서형 데이터의 경우
    * Ordinal Encoding
        * feature를 구성하는 category의 ordinal nature를 고려한(= assign a sequence of interges) Label Encoding
            * ex> ordinal nature : Cold(1) < Warm(2) < Hot(3) < Very Hot(4)