---
description: '210826'
---

# \(7강\) Training & Inference 1

학습 프로세스에 필요한 요소는 크게 아래와 같이 나눌 수 있다.

![](../../../.gitbook/assets/image%20%281004%29.png)



### Loss

Loss는 Output과 Target의 차이를 어떻게 정의하느냐에 따라 다르다. 

![](../../../.gitbook/assets/image%20%281002%29.png)

Loss는 `nn.Module` 을 상속하고 있기 떄문에 `forward` 함수가 있다. 그런데 여기서 `loss.backward` 의 한 줄 코드로 어떻게 모델의 전체 파라미터가 업데이트가 될까?

여기서 알아야 할 점은, `nn.Module` 을 상속하고 있는 모듈들은 모두 `forward` 함수가 있기 때문에 input부터 output까지의 연결이 생긴다는 것이다. 또, 어떤 레이어의 output은 다음 레이어의 input이 되고, 모든 레이어가 `forward` 함수가 있기 때문에 첫 입력단부터 loss까지는 연결이 된다고 볼 수 있다. 그래서 단순히 loss 에서 시작하더라도 입력단의 처음까지 올 수 있는 것이다.

`loss.backward` 가 이루어지면 각각의 파라미터의 `grad` 값이 갱신된다. 이 때 이러한 갱신여부를 `required_grad` 로 설정해줄 수 있고, False로 설정할 경우 갱신되지 않는다.



Loss함수를 Custom으로 정의할 수도 있다.

* Focal Loss : Class Imbalance 문제가 있는 경우, 맞춘 확률이 높은 Class는 조금의 loss를, 맞춘 확률이 낮은 Class는 loss를 훨씬 높게 부여한다.
* Label Smoothing Loss : Class target label을 Onehot이 아닌 Soft 하게 표현해서 일반화 성능을 높인다.
  * 0과 1의 값만 가지게 되면 극단의 feature만을 갖게되는데, 사실 class마다 비슷한 feature도 있을 수 있기 때문에 이러한 부분을 유연하게 설정한다



### Optimizer

Optimizer는 파라미터를 갱신하는 방법을 정의한다.

![](../../../.gitbook/assets/image%20%281001%29.png)

왼쪽의 경우 학습률이 고정되어 있기 때문에 수렴하기가 어렵다. 오른쪽처럼 학습률이 점점 작아진다면 수렴하기가 쉬워질 것이다.

학습률을 동적으로 조절하는 LR scheduler에는 다음과 같은 것들이 있다.

#### StepLR

![](../../../.gitbook/assets/image%20%281000%29.png)

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
```

`step_size` 마다 학습률을 `gamma` 만큼의 비율로 설정한다.

* `step_size` 는 `batch_size` 만큼 학습을 하고 난 뒤 파라미터를 갱신하는 횟수이다



#### CosineAnnealingLR

![](../../../.gitbook/assets/image%20%28998%29.png)

```python
scheduler = torch.optim.lr_scheduler.CossineAnnealingLR(optimizer, T_MAX=10, eta_min=0))
```

학습률의 변화를 Cosine 함수처럼 만드는 함수이다. 억지처럼 보일 수 있지만, 나름의 장점도 있다. step이 많다고 무작정 낮추지 않다보니까 local minimum에 잘 빠지지 않는다는 점이다.



#### ReduceLROnPlateau

![](../../../.gitbook/assets/image%20%281005%29.png)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, T_MAX=10, eta_min=0))
```

일반적으로 가장 많이 쓰는 스케쥴러이다. 더 이상 성능 향상이 없을 때 학습률이 감소한다.



### Metric

지표\(=측정법\)는 학습에 직접적인 영향을 미치지는 않는다. 그러나 traing의 중요한 요소로 봐야되는 이유는 지표가 없으면 객관적으로 모델의 신뢰도나 범용성을 판단할 수 없기 때문이다. 단순히 loss의 수치로만 봐서는 실제로 production에서 적용하기에는 부족한점이 많다.



모델의 평가

* Classification
  * Accuracy : 보통 많이 쓰나 class간 imbalance가 있으면 다른 지표도 사용한다.
  * F1-score : Class별 밸런스가 좋지 않을 때 각 클래스 별로 성능을 잘 낼 수 있는지에 대한 지표이다.
  * precision
  * recall
  * ROC&AUC
* Regression
  * MAE
  * MSE
* Ranking : 추천시스템에서 많이 쓰이는 지표이다. 추천되는 항목이 가장 위에 떠야 하기때문에 순서까지 고려되는 지표이다.
  * MRR
  * NDCG
  * MAP







