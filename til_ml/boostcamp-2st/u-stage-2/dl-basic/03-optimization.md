---
description: '210810'
---

# \(03강\) Optimization

## Important Concepts in Optimization

### Generalization

일반화가 좋다라는 의미는 이 네트워크의 성능이 학습 데이터의 성능과 비슷하다라는 의미이다. Generalization Gap은 Train data error와 Test data error의 차이를 의미힌다.

그렇다면 일반화가 잘되면 좋은 걸까? 꼭 그렇지많은 않다.

![](../../../../.gitbook/assets/image%20%28806%29.png)

왼쪽에 동그라미안을 보면 일반화 갭은 매우 낮지만 에러는 매우 높기 때문이다. 따라서, 일반화가 잘되면서 Traing error 까지 낮아야 좋다고 할 수 있겠다.

### Underfitting vs Overfitting

![](../../../../.gitbook/assets/image%20%28807%29.png)

학습 데이터에 과도하게 학습되면 Overfitting이 발생하고, 적게 학습되면 Underfitting이 발생한다.



### Cross-validation

오버피팅을 피하려면 학습 데이터를 줄이고 테스트 데이터를 늘리면 될까? 그러면 좋을 수도 있겠지만, 일반적으로 학습 데이터가 많아야 모델의 성능이 증가한다. 그래서 등장한 것이 cross-validation

학습하는 데이터를 부분 부분으로 나누어서 n개로 만든다. 그리고 n번의 학습을 거치면서 각각의 부분 데이터가 1번씩 검증 데이터로, 나머지는 학습 데이터로 사용되는 것이다.



### Bias and Variance

![](../../../../.gitbook/assets/image%20%28800%29.png)

Varience가 낮으면 출력이 일관된다. 크면 출력이 많이 달라진다. 그래서 Overfitting이 될 가능성이 높다

Bias가 낮으면 평균값을 많이 출력한다는 것이다. 반대로 크면 평균에서 많이 벗어난 값들을 출력한다.

cost를 줄이는 과정에서 cost는 varience와 bias 그리고 noise라는 3가지의 요소로 구성되어있는데 이 세 요소는 tradeoff의 관계에 있다

![](../../../../.gitbook/assets/image%20%28810%29.png)

cost를 줄이는 것은 bias와 variance와 noise를 줄이는 것인데, bias를 줄이면 variance가 높아지게 되고 noise가 있으면 bias와 variance를 동시에 줄이기는 어렵게 된다.



### Bootstrapping

뜻은 신발끈. 신발끈은 들어서 하늘을 날겠다는 허무맹랑한 의미. 테스트셋이 고정되어 있을 때 이를 전부사용하는 것이 아니라 샘플링을 통해 여러 테스트텟을 만들고 또 이를 통해 여러 모델과 파라미터를 생성한다. 이후 이 모델들의 결과가 일치하는지 등을 보고 모델의 성능을 파악할 때 사용한다.



### Bagging vs Boosting

#### Bagging

Bootstrapping aggregating의 준말. 테스트셋이 고정되어 있을 때 이 테스트텟 하나를 전부 사용해서 학습하는 것이 아니라 학습 데이터를 여러개로 만들어서 Boostrap 하는 것. 일반적으로 앙상블이라고도 부른다.

실제로도 100%의 데이터셋을 만드는 것보다 80%의 데이터셋을 사용해 5개의 모델을 만들고 평균을 구하는 것이 일반적으로 성능이 더 좋다.

#### Boosting

100개의 데이터를 모두 학습하고 이 중에 80개에 대해서만 잘 예측했다면 예측하지 못한 20개의 데이터에 대해서만 학습하는 두번째 모델을 만든다. 이렇게 여러개의 모델을 만들어서 합친다. 하나하나의 모델을 sequence 하게 연결한다 \(독립적으로 보는것이 아님\)

![](../../../../.gitbook/assets/image%20%28817%29.png)

## Practical Gradient Descent Methods

### Gradient Descent Methods

#### Stochastic gradient descent

* 하나의 샘플로만 기울기를 갱신한다

#### Mini-batch gradient descent

* 몇개의 샘플로 기울기를 갱신한다

#### Batch gradient descent

* 전체 데이터로 기울기를 갱신한다



### Batch-size Matters

단순히, 한개는 너무 적고 전체는 너무 오래걸리니까 일부로 하면 되겠지 라는 이유보다 배치사이즈가 굉장히 중요하다.

배치 사이즈가 작을수록 실험적으로 성능이 좋다. 배치 사이즈가 작을수록 Flat Minimum에 도달하기 쉽고, 배치 사이즈가 클수록 Sharp Minimum에 도달하기 쉽다.

![](../../../../.gitbook/assets/image%20%28801%29.png)

Sharp는 값이 조금만 달라져도 Loss나 Accuracy가 크게 달라지기 때문에 데이터셋이 달라지면 성능이 잘 안나온다.

### Gradient Descent

#### \(Stochastic\) GD

![](../../../../.gitbook/assets/image%20%28804%29.png)

문제는, 학습률을 지정하기가 너무 어렵다. 너무 커도, 너무 작아도 안되기 때문

#### Momentum

![](../../../../.gitbook/assets/image%20%28820%29.png)

이전에 가중치 갱신 정보를 활용하는 것이다. 이 때 gradient의 변동폭이 크더라도 수렴하는 쪽으로 학습을 잘 하게 된다. 

* $$g_t$$: 현재 시점에 갱신된 가중치
* $$a_t$$: 이전의 가중치 정보들
* $$\beta$$: 모멘텀

#### Nesterov Accelerated Gradient, NAG

![](../../../../.gitbook/assets/image%20%28803%29.png)

a라는 이전의 가중치 정보만큼 한 step 이동하고 그 자리에서 새로 갱신된 가중치 만큼 이동한다. 즉, 관성에 의해서 최소점을 지나더라도 지난 시점에서 새로 가중치를 구해서 더하면 된다는 뜻!

$$ W_t - \eta\beta a_t $$의 의미는 기존에 기울기에서 일단 관성 \* 학습률 만큼 빼라는 의미이다. 그리고 여기서 $$ \triangledown L $$은 해당 시점에서의 미분율을 구하라는 것이기 때문에 갱신된 위치에서의 가중치를 의미한다.

![](../../../../.gitbook/assets/image%20%28813%29.png)

기존의 모멘텀은 최소점을 지나더라도 다시 최소점 방향으로 가지 못하고 관성 때문에 더 멀어졌다가 다시 오게된다. \(마치 진자운동처럼\) 그래서 수렴하는 지점 주변에는 도달하지만 정확히는 수렴하지 못하게 된다.

NAG는 이러한 최소점에 더 빠르게 도달할 수 있게 해준다.

#### Adagrad

Adaptive Gradient, 각 파라미터의 변화율에 따라 STEP SIZE를 다르게 곱해준다. 그래서 조금 변화한 파라미터는 더 많이, 많이 변화한 파라미터는 더 적게 변화하도록 한다.

왜냐하면, 자주 등장하거나 변화를 많이 한 변수들은 optimum에 가까이 있을 확률이 높아서 세밀하게 이동해야 하고, 적게 변화한 변수들은 빠르게 optimump에 가까워지기 위해 많이 이동해야할 확률이 높기 때문에 빠르게 loss를 줄이는 방향으로 이동하려는 방식이다.

![](../../../../.gitbook/assets/image%20%28816%29.png)

G는 가중치를 제곱해서 모두 더한 값이며 엡실론은 0으로 나눠지지 않게 하기 위함이다. 학습을 진행할수록 G값에는 제곱한 값이 들어오므로 계속 증가하기 때문에 학습이 너무 오래되면 step size가 너무 작아져서 거의 움직이지 않게 된다는 문제점이 있다.

#### Adadelta

AdaGrad의 단점을 보완하기 위해 제안된 방법이다.

![](../../../../.gitbook/assets/image%20%28802%29.png)

Gt는 exponential moving average를 통해 값을 갱신하게 된다. 그렇지 않으면 이전에 gt들을 모두 기억하고 있어야 하는데, 이부분에 리소스 문제가 발생한다. 따라서 결과가 비슷해지는 EMA 방법을 사용했다.

Ht는 가중치의 변화율에 대해서 EMA를 적용했다.

나도, 왜 Ht의 루트값 / Gt 로 학습률을 정의했는지 잘 모르겠다. 논문을 읽어봐야 할 것 같다.

Adadelta는 학습률이 없기 때문에 바꿀 수 있는 요소가 많이 없어 잘 사용하지 않는다.

#### RMSProp

논문을 통해서 제안된 건 아니고, 강의에서 소개된 것이다. Ht값 대신 에타라는 Stepsize가 추가되었다.

![](../../../../.gitbook/assets/image%20%28818%29.png)

#### Adam

EMA of GS\(Gradient Squares\)를  사용함과 동시에 Momentum을 같이 활용하는 것

![](../../../../.gitbook/assets/image%20%28812%29.png)

* b1 : 모멘텀을 얼마나 유지시킬 것인가
* b2 : EMA of GS 정보
* 엡실론 e 값은 실제로 10^\(-7\) 이 기본값인데 이 값을 잘 조절해주는 것이 굉장히 중요하다

## Regularization

일반화를 잘 되게 하기 위해 규제를 하는 것. 학습을 방해하는 것이 목적인데, 단순히 방해라는 의미보다는 학습 데이터 뿐만 아니라 테스트 데이터에도 잘 적용되도록 하는 방법

#### Early Stopping

![](../../../../.gitbook/assets/image%20%28805%29.png)

Validation error가 가장 낮을 때 학습을 멈추는 방법

* Test error로 하면 안된다.

#### Parameter Norm Penalty

![](../../../../.gitbook/assets/image%20%28814%29.png)

일반화가 잘되는 함수일수록 부드러운 함수일 것이다라는 가정으로 행하는 방법이다.

#### Data Augmentation

![](../../../../.gitbook/assets/image%20%28811%29.png)

가장 중요한 것 중 하나가 데이터인데, 데이터가 무한히 많으면 항상 모델의 성능이 좋다.

데이터가 적을 때는 앙상블, 랜덤포레스트 같은 기법들을 적용하면 성능이 증가했지만 데이터가 많을 때에는 신경망이 이러한 데이터의 특징을 잘 표현할 수 있게 되어 성능이 좋았다.

따라서, 데이터를 라벨이 바뀌지 않는 한도내에서 변환시켜 데이터를 증가시키는 것.

그러나 레이블이 변환될 가능성이 있으면 하면 안된다. \(ex MNIST는 6을 9로 볼 수도 있다\)

#### Noiser Robustness

노이즈를 신경망 중간중간에 인풋이나 가중치에 넣게되면 성능이 더 좋게 나온다는 실험적인 결과

#### Label Smooting

![](../../../../.gitbook/assets/image%20%28819%29.png)

 이미지를 서로 조합하는 기법

* Mixup : 두 개의 이미지를 비율로 섞고, 라벨도 섞어버리는 방법
* Cutout : 이미지의 일부분을 제거
* Cutmix : 이미지를 섞어줄 때 blending 하는 것이 아니라 일부 영역을 섞어주는 것

#### Dropout

![](../../../../.gitbook/assets/image%20%28815%29.png)

각각의 뉴런들이 조금 더 robust한 feature들을 잡도록 몇개의 뉴런을 비활성화 한다.

#### Batch Normalization

![](../../../../.gitbook/assets/image%20%28808%29.png)

신경망의 각각의 레이어가 천개의 파라미터가 있을 때, 천개의 값을 모듀 정규화\(평균을 빼주고 분산으로 나누어준다\)해준다. 그러면서 네트워크가 잘 학습이 된다

* 많은 논문들이 동의하지는 않는다
* 확실한 것은 BN을 활용하면 성능이 향상한다.

![](../../../../.gitbook/assets/image%20%28809%29.png)



















