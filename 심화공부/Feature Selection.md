## Feature(attributute, column) Selection
* 모델을 구성하는 주요 피처들을 선택
    * 모든 feature를 사용하는 것은 computing power와 memory 측면에서 매우 비효율적이기 때문에 일부 필요한 feature만 선택해서 사용
    * 불필요한 다수의 피처들로 인해 모델 성능을 떨어뜨릴 가능성 제거
        * ex> 다중공선성, overfitting
        * 제거해서 무조건 성능이 향상 되는 것은 아님
            * ex> LightGBM, XGBoost
    * 설명 가능한 모델이 될 수 있도록 피처들을 선별
        * 모델이 어떠한 feature의 영향을 받는지 그리고 이러한 feature를 모델 입장에서 어떻게 가공해야하는지 설명가능해짐
* 장점
    * 사용자가 해석하기 쉽게 모델을 단순화
    * 훈련 시간의 축소
    * 차원의 저주 방지
    * 일반화
* feature engineering? feature extraction? feature selection?
    * feature engineering
        * 도메인 지식을 사용하여 데이터에서 feature를 변형 및 생성
        * feature를 어떻게 유용하게 만들 것인가의 문제
    * feature extraction
        * 차원축소를 하면서 정보손실을 최소화하는 중요한 feature를 추출
    * feature selection
        * 기존 feature set에서 원하는 feature subset을 선택함
        * 데이터에서 유용한 feature를 어떻게 선택할 것인가의 문제
* 유형
    * Filter methods
        * 단변량 통계적 기법을 통해 측정된 feature의 고유한 특성을 활용해서, 이러한 특성이 높은 feature를 선택하는 기법
        * (target variable를 제외한) feature 간에 relationship을 고려하지 않는 단점이 있음
            * 단변량 통계적 기법?
                * ex> information gain, correlation, 평균, 분산 ...
        * computing power가 적고, 빠르기 때문에 고차원 데이터에 적합함
        * 종류
            * information gain
                * target variable 관점의 impurity의 감소량을 통해 계산됨
                    * tree의 경우, 부모 노드의 가중치 불순도에서 자식 노드들의 가중치 불순도 합을 빼서 계산됨
                    * impurity를 측정하는 measure로 지니 계수(불순도)와 엔트로피가 있음
                * 각 feature(= variable)의 information gain을 계산해서 feature selection에 활용함
            * Chi-Square test
                * 각 feature와 target variable 간에 chi-square score를 계산해서 활용함
                    * chi-square score?
                        * 카이제곱 검정은 기본적으로 변수 간 독립성을 확인하는 방법이기 때문에 변수 간(each feature vs target) 연관성의 정도를 나타내는 지표로 해석했음
                * 요구되는 조건으로는 categorical feature에 적용 가능하며, 데이터 간에 독립적이고, feature의 각각의 category의 빈도수가 5를 넘어야함
            * Correlation Coefficient
                * 상관계수는 변수 간에 linear relation을 측정하는 지표임
                * target variable과 상관 계수가 높으면서, target variable를 제외한 다른 variable과 상관계수가 낮은 feature를 선택함
            * Variance Threshold
                * feature를 구성하는 요소들의 분산이 클수록 더 유용한 정보를 담고 있을 것이라는 것이 기본 아이디어임
                * 임계값(threshold)를 만족하지 않는 분산을 가지고 있는 feature들을 제거함
            * Mean Absolute Difference(MAD)
                * 기본적인 동작은 Variance Threshold와 같으나, 차이점은 사용하는 measure가 제곱을 하지 않는 Mean absolute difference임
            * Dispersion ratio
                * 기본적인 동작은 Variance Threshold와 같으나, 차이점은 사용하는 measure가 dispersion ratio임
                    * dispersion ratio?
                        * 각 feature 별로 산술평균(Arithmetic Mean, AM)과 기하평균(Geometric Mean, GM)을 구함
                        * dispersion ratio = AM / GM
                        * dispersion ratio가 크게 나타날수록 유익한 feature로 해석함
            * Fisher's Score
                * 나중에 여유 생길 때 공부하기로 함
    * Wrapper methods
        * 가장 좋은 성능을 보이는 feature subset을 뽑아내는 방법임
        * 여러번 machine learning을 진행하기 때문에 시간과 비용이 높게 발생하지만, Best Feature subset을 찾기 때문에 바람직한 방법임
            * 물론, 해당 모델의 파라미터와 알고리즘 자체의 완성도가 높아야 제대로 된 Best feature subset을 찾을 수 있음
            * 일반적으로 filter methods보다 성능이 우수함
        * feature 간에 interaction(relationship)을 고려한 방법이라고 할 수 있음
        * 사용하는 모델(머신러닝 알고리즘)에 따라 적용 못할 수 도 있음
            * ex> p-value, regression coefficient, model(ex> tree) feature importance 가 요구 되는 방법임
        * 종류
            * Forward Feature Selection
                * 각 남은 변수들을 기존 feature subset에 추가해서 p-value가 가장 낮은(설명력이 가장 좋은) feature를 하나씩 추가함
            * Backward Feature Selection
                * 현재 포함된 feature subset에서 p-value가 가장 큰 feature 하나씩 제거함
            * Stepwise Feature Selection
                * Forward Feature Selection + Backward Feature Selection
            * Exhaustive Feature Selection
                * 모든 경우의 feature subset을 평가해서 최적의 feature subset을 반환하는 방법
                * 시간과 자원이 충분하다면, 모든 feature subset을 테스트하여 measure가 가장 높은 subset을 채택하는 것이 좋음
            * Recursive Feature Selection
                * Backward Feature Selection이랑 같다고 봄
                * model feature importance 또는 regression coefficient가 가장 작은 feature를 제거해가면서 최적의 feature subset을 찾아가는 방법
        * 참고자료
            * [forward selection & backward selection](https://zephyrus1111.tistory.com/65)
    * Embedded methods
        * 합리적인 연산량이라는 filter methods 의 장점과 feature 간에 interaction을 고려하는 wrapper methods의 장점을 포함하는 방법임
        * 각각의 feature를 직접 학습하며, 성능에 기여하는 feature를 선택함
        * 종류
            * LASSO Regularization
                * L1-norm을 통해 제약을 주는 방법
            * Ridge Regularization
                * L2-norm을 통해 제약을 주는 방법
            * Elastic Net
                * LASSO + Ridge 결합한 방법
            * SelectFromModel
                * tree-base model의 feature importance으로 특정 기준을 만족하는 feature를 선택함
    * 기타(다른 사이트에서 제시한 특정 예시)
        * 특정 feature를 구성하는 값들의 분포
            * ex> 해당 feature를 구성하는 값들의 종류가 1개이면 Decision하는 변별력이 없어서 제거함
        * NULL이 많은 feature 제거함
        * 피처간 높은 상관도 경우 제거를 고려해야함
            * 회귀 모델의 경우 다중공선성 발생할 수 있음
        * 결정값(Target)과의 독립성 등을 고려함
            * Target값과 영향이 없는 경우 제거함
        * 모델의 feature importance를 기준으로 할 수 있음
            * ex> 회귀의 회귀계수, 트리의 feature importance
            * 왜 (특히 tree based model) feature importance는 **절대적인** feature selection 기준이 될 수 없는가?
                * (모델의)Feature importance는 최적 tree 구조를 만들기 위한 feature들의 impurity가 중요 기준임. 결정 값과 관련이 없어도 feature importance가 높아 질 수 있음
                    * 물론, 중요한 feature 들이 feature importance가 높아지는 경향성은 있지만, Target과 관련이 없는 feature도 feature importance가 높게 나오는 문제를 수반함
                * (모델의)Feature importance는 학습 데이터를 기반으로 생성됨. 테스트 데이터에서는 달라질 수 있음
                    * 모델의 평가는 테스트 데이터를 통해 이루어짐
                * 모델이 과적합 될수록 (모델의)Feature importance는 number형 feature 또는 high cardinality category feature에 biased 되어 있음
                    * **적절하게 fitting된 모델의 feature importance**는 좋은 자료는 될 수 있을 것으로 생각됨
* 사이킷런 feature selection
    * 기본적으로 모델의 feature importance(회귀모델의 경우 coefficient)를 기준으로 작동함
    * 종류
        * RFE(Recursive Feature Elimination)
            * 모델 최초 학습 후 feature importance 선정
            * feature 중요도가 낮은 속성들을 차례로 제거해 가면서 반복적으로 학습/평가를 수행하여 원하는 feature 개수까지 최적 feature 추출
            * 수행시간이 오래 걸리고, 낮은 속성들을 제거해 나가는 메커니즘이 정확한 feature selection을 찾는 목표에 부합하지 않을 수 있음
                * 데이터가 많을 때는 불리함
                * 모델의 feature importance가 절대적인 feature selection의 기준이 아님
        * RFECV(Recursive Feature Elimination Cross Validation)
            * RFE가 사용하던 방법과 똑같이 가장 feature importance가 낮은 feature들을 제거해가면서 각 feature 개수마다 cross validation을 활용하여 여러 성능들을 도출하고 이러한 성능들의 평균값을 활용함
            * RFE와 달리 몇 개의 feature를 남겨야 할지를 사용자가 직접 정의를 할 필요 없이, 가장 높은 성능을 가지는 feature 개수에 해당하는 feature 들을 최종 feature set으로 선정하면 됨
        * SelectFromModel
            * 모델 최초 학습 후 선정된 Feature importance에 따라 평균/중앙값과 같은 기준의 특정 비율 이상인 Feature들을 선택
                * 선택된 feature로 재학습 진행하면서 최적의 feature를 찾음