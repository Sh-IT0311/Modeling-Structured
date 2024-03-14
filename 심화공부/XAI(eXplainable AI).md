## XAI(eXplainable AI)
* (해석력이 낮은) 머신러닝 알고리즘이 예측한 결과를 사람이 이해할 수 있도록 해석을 제공하는 방법론
    * 해석력?
        * 사람이 이해할 수 있는 정도
    * 모델의 성능과 해석력은 trade-off
        * 모델의 성능은 낮지만, 해석력이 좋은 모델들
            * ex> Linear Regression, Decision Tree
                * Linear Regression
                    * Regression Coefficient를 활용
                * Decision Tree
                    * Node의 Feature 및 Threshold를 활용
        * 모델의 성능은 높지만, 해석력이 낮은 모델들
            * ex> Neural Networks, Random Forest
* 모델을 왜 해석해야 하는가?
    * 특정 분야에서 모델을 무조건 신뢰 할 수 없음
        * ex> 의료분야, 자율주행, 금융분야 등
    * 의사결정에서 직접적인 영향을 주는 것은 모델의 성능뿐만 아니라 해석임
        * ex> 모델링을 통해 얻은 인사이트를 의사결정에 활용하는 경우
        * Accuracy와 같은 Metric을 무조건 신뢰할 수 없음
        * 모델의 해석과 해당 분야의 Domain Knowledge를 통해 올바른 판단을 내릴 수 있음
    * 우리가 몰랐던 지식을 얻거나 고정관념을 바로잡을 수 있는 기회가 될 수 있음
    * 차원의 저주를 해결할 실마리를 제공함
        * Feature Selection
* 해석의 범주(Categorization)
    * 모델 자체로 해석 가능 여부
        * Intrinsic
            * 모델 자체로 해석 가능함
        * Post-hoc
            * 모델 자체로 해석이 안되서 해석을 위한 새로운 모델링 또는 알고리즘 적용함
    * 적용 가능 범위
        * Model-agnostic
            * 모델에 상관 없이 적용 가능하며, trained model을 활용해서 input/output 관계를 해석함
        * Model-specific
            * 특정 모델에만 적용 가능하며, 모델만의 내재적, 구조적인 정보를 이용함
    * 설명 대상 범위
        * Local
            * 하나의 예측 결과에 대해 설명함
                * Trusting a prediction? 의 관점임
                * 해당 데이터셋의 Feature 간에 복잡한 관계보다는 선형적 또는 단조로운 형태를 해석하게 됨
                * 전체적으로 해석하는 것보다는 정확한 해석이 이루어질 가능성이 큼
        * Global
            * 모델이 예측하는 모든 결과를 설명함
                * Trusting a model? 의 관점임
                * Feature Importance, Feature 간에 교호작용이 해당됨
                * 데이터를 전반적으로 이해하는 데 중요한 해석이 되지만, 확실한 해석을 얻는 것은 어려움
    * 데이터 유형
        * Image
        * Text
        * Tabular Data
        * ...
* 좋은 해석으로 요구되는 성질(Property)
    * Fidelity
        * 모델의 예측 결과와 이에 대한 해석이 얼마나 근접한지에 대한 성질
    * Consistency
        * 같은 데이터셋으로 학습해서 비슷한 예측을 한 두 모델에 대해서 각각의 해석이 얼마나 다른지에 대한 성질
    * Stability
        * 한 모델에 대하여 비슷한 샘플들의 해석이 얼마나 비슷한지에 대한 성질
    * Comprehensibility
        * 해석 방법이 얼마나 인간이 이해할 수 있을 것인지에 대한 성질
* Several Methods
    * LIME(Local Interpretable Model-agnostic Explanations)
        * Local Surrogate Model을 통해 관측치 하나의 예측(Prediction)에 대한 해석을 제공함
            * Interpretable Representation(xi'∈ {0,1})
                * 비전문가 또한 이해할 수 있도록 사용되며, explanation model의 입력 형태임
            * Local Surrogate Model(Explanation Model)
                * 전체 데이터셋이 아닌 특정 관측치에 대한 예측 설명시 사용할 수 있는 해석가능한 모델
                    * Linear Regression, Decision Tree 등 사용됨
                * 설명 모델을 구하는 Formula
                    * ξ(x)=argmin(g∈G)[L(f, g, πx) + Ω(g)]
                        * ξ(x) : 관측치에 대한 설명 모델
                        * G : a class of interpretable models
                        * g : a interpretable model
                        * L : fidelity funtion(loss function)
                            * 설명 모델이 Original Model의 예측값을 예측할 수 있도록 유도하는 Term
                            * paper에서 예시로 설명 모델을 Linear Model, 그래서 해당 Term이 sum squared error으로 제시됨
                        * f : trained original model
                        * πx : proximity(locality) measure between x and z
                            * 관측치와 유사한 정도를 나타냄
                        * Ω : complexity measure
                            * 모델 복잡도의 패널티를 주는 Term
                                * Linear Model의 경우 유효한(0이 아닌) Coefficient가 적을수록 복잡도가 작음
                                * Decision Tree의 경우 Depth가 작을수록 복잡도가 작음
            * Sampling for Local Exploration(z, z')
                * Original Model의 Local Exploration을 위해 위의 Formula에 사용되는 x의 perturbed sample들을 추출함
                * 추출방법은 데이터 유형에 따라 다르게 제시됨

    * Shapley Values
        * Backgounds
            * 게임이론(Game Thoery)
                * 여러 주체가 서로 영향을 미치는 상황에서 서로가 어떤 의사결정이나 행동을 하는지에 대해 이론화한 것
                    * 결국 자신의 최대 이익에 부합하는 행동을 추구함
                * 총 지불금(Payout)을 각 선수들(Players)의 기여도에 따라 공정하게 배분하기 위해 도입됨
                * 머신러닝과 연관성?
                    * Game : The prediction task for a single instance
                    * Payout(= Gain, Payoff) : The actual prediction for a single instance - the average prediction for all instances
                    * Players : The feature value of a instance    
        * Definitions
            * 게임이론을 바탕으로 f(x) - E(f[X])를 설명하는 각 "특성값(Feature Value)"의 기여도(중요도)를 나타내는 값
            * 모든 가능한 연합(조합)에 대해서 각 특성값의 유무에 따른 모든 Marginal Contribution의 가중평균을 통해 계산 된 값
                * All coalitions
                * Weighted Average
                    * 주어진 연합에 대한 Combination을 통해 계산된 Weight로 가중평균을 진행함
                * Marginal Contribution
                    * 해당 특성값의 유무에 따른 예측값의 변화량(차이)을 나타냄
                        * **연합에 포함되지 않는 특성값들에 대해서 어떻게 예측값을 구하는가?**
                            * 데이터에서 무작위로 추출해서 대체하여 진행함
                                * 이러한 과정을 반복해서 평균한 Marginal Contrubution을 도출해내면 더 좋은 해석을 이끌어 낼 수 있음
                            * Coalition에 해당하는 subFeatureSet으로 Model을 학습함
        * Calculations
            * exact solution
                * 모든 가능한 연합에 대해서 Marginal Contribution을 구해서 가중평균을 취함
            * approximation
                * 몬테카를로 샘플링을 기반해서 근사치를 구함
                    * 몬테카를로 샘플링이 요구하는 Feature Independence가 가정되어야함
        * Properties
            * 게임이론 문제가 오직 하나의 해를 갖도록 하는 조건들
            1. Efficiency
            2. Symmetry
            3. Dummy(= Null Effects)
            4. Additive(= Linearity)
    * SHAP
        * Additive Feature Attribution Methods
            * Feature Attribution을 Regression Coefficient, Simplified Input을 Input으로 가지고 있는 Linear Regression(Explanation Model)
                * Feature Attribution
                    * 각 Feature Value가 Model Output(Prediction)의 기여하는 정도를 나타냄
                * Simplified Input
                    * LIME의 Interpretable Representation과 같은 의미로 Explanation Model이 사용하는 Input임
            * Mapping Function
                * Simplified Input을 Original Model이 사용하는 Original Input으로 Mapping함
        * Desirable Properties
            * A unique solution을 보장하는 게임이론의 4 Axioms에서 도출됨
            1. Local Accuracy
                * 게임이론의 4 Axioms에서 Efficiency로 도출됨
            2. Missingness
                * 정확히는 게임이론의 4 Axioms으로 도출된 특성은 아님
                * Shapley Proofs를 Additive Feature Attribute Method에 적용하기 위해 도입했다고 함
            3. Consistency(= Monotonicity)
                * 게임이론의 4 Axioms에서 Additive, Dummy, Symmetry를 내포하고 있음
        * Only One Possible Explanation Model
            * Additive Feature Attribution Method이면서 Desirable Properties를 만족하면 Only one possible Explantion Model을 학습시킬 수 있음
            * Shapley Values만 이러한 조건을 만족하기 때문에 Shapley Values를 기반으로 해야하는 Approach를 설정하게 됨
        * SHAP values
            * Mappling Function을 Conditional Expectation Function으로 하는 Shapley values
                * 대부분의 모델들이 missing input(zi'=0)의 arbitrary patterns(무작위로 대체되서 발생하는 것으로 이해)을 다루는데 불리해서 Conditional Expectation으로 접근했다고 함
                * 이러한 정의가 LIME, DeepLIFT 등과 연결시켜주었다고 함
        * SHAP values estimation methods
            * Shapley Values의 Classic Formula는 연산량이 많기 때문에 근사하는 방법이 등장함
            * Model-agnostic
                * Sampling approximation
                    * 몬테카를로 샘플링이 요구하는 Feature Independence가 가정되어야함
                * kernel SHAP(Linear LIME + Shapley Values)
                    * LIME의 explanation이 Shapley Values과 되도록 하는 Fidelity Function, Proximity measure, Complexity measure를 찾아냄
            * Model-specific
                * Tree SHAP
                    * 이전 존재했던 방법들이 Inconsistent 하다는 한계점이 있음
                        * 기존의 방법들
                            * Local
                                * Saabas
                            * Global
                                * Gain
                                * Split Count
                    * Consistency를 만족하는 SHAP values를 도입하게 되었으며, Tree Model에 최적화된 Estimating SHAP values 방법을 제시한 것이 Tree SHAP임
                * Deep SHAP
        * 특징
            * Feature Value 간에 의존성을 고려해서(All Coalitions) 기여도를 계산함
                * Feature 간에 상관이 있는 경우 이상한 결과를 낼 수 있음
            * Model Agnostic / Locally
            * Trained-Model
            * Locally interpretation이기 때문에, 이상치에 이상한 결과를 낼 수 있음
                * Feature Importance의 경우 Mean(SHAP Values) 접근해서 괜찮음
            * Classic Formula의 경우 시간이 오래걸림