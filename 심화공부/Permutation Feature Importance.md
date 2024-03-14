## Permutation Feature Importance
* **특정 모델에 특화된 방법이 아닌** 어느 모델이든 학습시킨 후에 feature importance를 구할 수 있는 방법임
* 모델 및 파라미터에 따라 feature importance의 큰 차이가 없는 robust 한 특징을 가지고 있음
* 특정 feature들의 값을 완전히 변조했을 때 모델 성능이 얼마나 저하되는지를 기준으로 해당 feature의 importance를 산정
    * 사이킷런의 경우 특정 feature를 랜덤하게 셔플함
    * 특정 feature를 노이즈로 만드는 것으로 이해하면 됨
        * Target과 연결고리를 끊게 되는 것
    * feature의 의존도가 큰 경우 성능은 크게 감소할 것
* 특정 feature importance를 구하고자 해당 feature를 제거하고 모델을 재학습해서 중요도를 파악하는 방법도 있지만, 학습 데이터를 제거하거나/변조하면 다시 재학습을 수행해야 하므로 수행 시간이 오래 걸리는데 permutation feature importance는 (특정 feature를 노이즈로 만드는 구조이기 때문에) 재학습을 시킬 필요가 없음
* 일반적으로 테스트 데이터(검증 데이터)에 특정 feature들을 반복적으로 변조한 뒤 해당 feature의 중요도를 평균적으로 산정
    * 학습한 모델과 검증 데이터만 있으면 feature importance를 구할 수 있음
    * 모델의 학습과정, 내부 구조에 대한 정보가 필요 없음
        * 어느 모델이든 적용할 수 있다는 장점이 있음
* 각 feature importance는 partial importance가 아님
    * 각 feature importance는 다른 feature들과의 교호작용도 포함
    * 예를 들어 두 feature 간 교호작용의 영향은 그 두개의 feature importance 각각에 중복 포함됨
* 주의할점
    * 무작위로 섞는 방법론이기 때문에 할 때마다 결과가 매우 달라질 수 있음
        * 섞는 (반복)횟수를 늘려 평균값으로 접근하면 되지만 feature의 개수가 매우 많을 경우에는 연산량이 증가함
            * 적절한 반복 횟수를 선택해야함
    * 매우 비현실적인 데이터 인스턴스(instance)를 생성할 가능성이 높음
        * 특히 변수들 간에 상관관계가 높을 경우 이러한 문제가 발생하기 쉬움
            * ex> 키가 2m, 몸무게 30kg인 사람
            * **feature 간에 상관관계가 존재하는 모델은 사용하면 안됨**
                * ex> linear regression, logistic regression
        * 이러한 비개연성, 비현실성은 예측값에 엄청난 영향을 미칠 것이고, 그래서 feature importance가 편향 될 수 있음
            * 미리 변수들 간 상관관계가 높은지 확인 및 이를 염두에 두고 결과 해석 필요
* 프로세스
    * 원본 모델의 기준 평가 성능을 설정
    * 개별 feature 별로 아래 수행
        1. 설정된 iteration(k)값 별로 아래 수행
            1. 해당 feature 별로 shuffle
            2. 모델 성능 평가
        2. 기준 평가 성능에서 모델 성능이 얼마나 저하되었는지 평가
            * 기준 성능 - k번의 평균값