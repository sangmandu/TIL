---
description: 21.09.11~16
---

# Attention Is All You Need

## Abstract

유망한 transduction 모델들은 인코더와 디코더를 포함하는 복잡한 반복구조나 합성곱 네트워크에 기반을 두고있다. 그 중 가장 좋은 성능을 내는 모델은 attention 메커니즘을 가지고 있는 인코더와 디코더로 이루어져있다. Transformer는 기존의 기반을 두지 않고 오로지 attention 메커니즘에만 기반을 두는 간단한 네트워크 모델이다. Transformer는 기존 모델보다 질적으로 더 우월하고 병렬작업이 훨씬 좋으며 학습하는데 적은 시간이 소요된다.

* transduction
  * 한국어로 변역하기에는 변환, 전환, 전이, 전도 등의 의미로 사용되어 번역하기 쉬운 단어는 아니다. transduce란 무언가를 다른 형태로 변환하는 것을 의미한다. 여기서는 주어진 특정 예제\(학습 데이터\)를 이용해 다른 특정 예제\(평가 데이터\)를 예측하는 것으로 이해할 수 있다.
* attention 메커니즘
  * RNN과 LSTM에 대한 이해가 있으면 더 쉽게 이해할 수 있다. RNN에서 Sequence를 계속 입력받다보면 과거 정보가 점점 희미해지게 되는데, 이 때 LSTM이 고안되었다. 기존의 RNN보다는 개선되었지만 완벽히 Long Term Dependency를 해결하지 못했고 이를 개선하기 위한 메커니즘이 attention 이다. 특정 time step에서 인코더의 전체 입력 문장을 다시 한번 참고하기 위한 방법이다.



## 1 Introduction

RNN, LSTM, GRU는 기계 번역이나 언어 모델링과 같은 transduction 문제와 시퀀스 모델링에 대해 가장 잘 해결할 수 있는 최신 모델들이다.  Recurrent 모델들은 input과 output의 위치에 따라 계산을 하고 각 위치에서 input과 이전 hidden state $$ h_{t-1} $$를 가지고 현재의 hidden state $$ h_t $$를 만들어낸다. 근데, 이러한 과정은 시퀀스의 길이가 길어지면 길어질수록 메모리 문제가 발생해서 배치 사이즈도 점점 작아지게 되고 병렬화도 할 수 없게되는 문제점이 있다. 최근 연구들은 이러한 문제에 대해 내부 인자를 조작하고 조건부 계산을 하면서 계산 효율을 높였고 그러면서 점점 모델의 성능도 증가했지만 본질적인 문제를 해결하지는 못했다.

Attention mecahnisms은 input과 output의 거리에 대한 의존도를 고려하지 않는다라는 점에서 sequence modeling과 transduction models에 다양한 task들에 있어 설득력있는 모델이되었다.

논문에서 제시하는 Transformer는 의도적으로 recurrence를 회피한다. 그대신 전체적으로 attention mechanism에 의존해서 input과 output 사이에 있는 전반적인 의존성을 해소한다. 그래서 Transformer는 병렬화에 더 적합하고 번역 기술에 있어서 조금의 학습으로도 더 좋은 품질의 결과물을 줄 수 있다.



## 2 Background

연속적인 계산을 줄이려는 시도는 Extended Neural GPU, ByteNet 그리고 ConvS2S에서 시작했는데, 모두 모든 input과 output을 hidden layer에서 병렬적으로 처리하는 구조를 가진다. 이 모델들은 연산을 할 때 임의의 입력과 출력의 신호를 필요로 했는데, 이 연산의 수가 입력과 출력의 거리가 멀어지면 멀어질수록 ConvS2S에서는 선형적으로, ByteNet에서는 log적으로 증가했다. 이러한 연산의 증가는 각 지점에서 입력과 출력에 대한 의존성을 학습하기가 어려워지는 이유가 되었다. 반면에 트랜스포머는 상수번의 계산만이 요구된다. 비록 어텐션 가중합을 평균냄으로써 유효한 정보의 감소로 이어지지만, 이후 살펴볼 Multi-Head Attention에서 이러한 단점을 해소할 수 있다.

Self-attention은 때때로 intra-attention으로도 불리며, 어떤 시퀀스의 있는 특징을 계산하기 위해 이 시퀀스의 서로다른 요소들을 관계시키는 알고리즘이다. 독해 능력, 개요 요약, 문맥적 함의와 관련있는 문장 추출등의, 다양한 task에서 Self-attention은 성공적으로 사용되어왔다.

처음부터 끝까지 메모리 네트워크는 시퀀스의 반복 구조를 사용하는 대신 attention의 반복 구조를 사용했고 단일 언어 체계의 질의응답과 언어 모델링 task에서 좋은 성능을 내었다.

내가 아는 한, 트랜스포머는 기존의 RNN이나 CNN의 반복 시퀀스 구조는 버리고 오로지 input과 output으로 계산된 self-attention에만 전적으로 의존하는 최초의 변환모델이다. 이어지는 순서에서는 Transformer를 소개할 것이며, self-attention에 대한 자세한 소개와 이에 대한 장점을 이야기할 것이다.



## 3 Model Architecture

대부분의 고성능 변환 모델은 인코더-디코더 구조를 가지고 있다. x로 표현되는 input sequence는 연속적인 z로 표현된다. 디코더는 z를 가지고 output에 해당하는 y를 생성하며 이러한 변환은 한번에 일어난다. 이 변환 과정에서는 auto-regressive\*한 방식으로 이루어지는데, 이전에 생성된 output을 input으로 입력해서 다음 단어를 생성할 수 있도록 한다.

* auto-regressive : 자기 자신을 입력으로 하여 자기 자신을 예측하는 모델을 의미한다.

트랜스포머는 인코더와 디코더에서 self-ateention과 fully connected layers가 쌓여있는 구조로 이루어져있다.

### 3.1 Encoder and Decoder Stacks

![](../../.gitbook/assets/image%20%281138%29.png)

#### Enocder

인코더는 6개의 동일한 레이어로 이루어져 있다. 각각의 레이어는 두 개의 sub layer를 가지는데, 첫번째는 multi-head self attention이고 두번째는 간단한 fully connected feed-forward network 이다. 이 두 레이어에 residual connection과 layer normalization을 적용하게 된다. 다시 말하면 `LayerNorm(x + Sublayer(x))` 로 계산이 된다. 이 때 잔차를 이용하려면 모든 서브 레이어가 512의 차원의 결과물로 반환해야 한다.

* feed-forward network는 순방향 신경망을 의미한다. 노드 간의 연결이 순환을 형성하지 않는 신경망이며 이는 RNN과는 차이가 있는 개념이다.
* residual connection은 ResNet에서 고안된 개념으로 layer에서 나온 output에다가 input입력값을 더해주면 성능과 안정성이 증가한다.
* layer normalization은 주어진 입력벡터에서 각 차원을 기준\(=세로방향으로\)으로 normalization 하는 것을 의미한다. RNN에서 이러한 작업은 hidden state에서의 변동성을 안정화시킬 수 있는 매우 효과적인 방법으로 알려져있다.
* 여기서 512는 Transformer가 정한 임베딩 차원 수이다.

#### Decoder

디코더 역시 6개의 동일한 레이어로 이루어져 있다. 두 개의 서브 레이어로 이루어져 있는 구조에서 encoder에서 받은 output을 가지고 multi-head attention을 수행하는 세번째 서브 레이어가 추가되어있다. 인코더와 비슷한 점은 잔차와 LN을 이용한다는 것이다. 다른 점은 디코더에서는 뒤쪽에 존재하는 위치에서 오는 attention을 쌓지 않는 점이다. 이것을 masking이라고 하는데, 각각의 위치에서 나오는 결과 임베딩 벡터를  i번째 단계에서 오로지 i번째 이전의 정보만 가지고 예측할 수 있도록 한다.

### 3.2 Attention

attention이란 결과를 예측하기 위해 사용되는 query 벡터와 key-value 벡터 쌍의 관계라고 볼 수 있다. 결과는 value 벡터의 가중합으로 계산이 되는데, 이때의 가중합으로 사용되는 가중치는 쿼리벡터와 그에 상응하는 키의 연관성을 구하는 함수에 의해 계산된 값이다.

#### 3.2.1 Scaled Dot-Product Attention

![](../../.gitbook/assets/image%20%281136%29.png)

이러한 attention을 `Scaled Dot-Product Attention` 으로도 부를 수 있다. 각각의 입력은 dk의 차원을 가진 쿼리벡터와 키벡터 그리고 dv의 차원을 가진 밸류벡터로 이루어진다. 하나의 쿼리 벡터는 모든 키벡터를 내적하며 이 값을 $$ \sqrt d_k $$로 나눠준 뒤 softmax 함수를 거쳐서 가중치를 구하게 된다.

\(이전에 하나의 쿼리 벡터와 모든 키벡터를 내적한다고 했지만\) 실제로는 모든 쿼리벡터에 대한 모든 키벡터의 내적을 한번에 계산하게 된다. 이 때 쿼리 벡터를 행렬 Q, 키 벡터와 밸류 벡터를 행렬 K와 V로 표현한다. 이러한 행렬 계산에 대한 식은 다음과 같다.

![](../../.gitbook/assets/image%20%281150%29.png)

여기서는 주로 두 개의 attention 함수를 사용하는데,  additive attention과 dot-product attention이다. 여기서 사용하는 dot-product attention은 $$ \sqrt d_k $$로 나눠주는 과정을 제외하면 기존 dot-product attention 방법과 완전히 동일하다. additive attention은 한 개의 은닉층을 가진 신경망을 사용할 때 연관성을 구하는 함수를 사용한다. 이 두 attention 방법은 이론적으로 복잡도는 비슷하지만 실제로는 dot-product attention이 훨씬 빠르고 메모리 공간도 더 효율적으로 사용한다. 왜냐하면 이 dot-product 연산에 최적화 된 행렬 계산 코드로 모델을 구현하기 때문이다.

차원 dk가 작을 때는 dot-product와 additive는 비슷한 성능을 보이지만 dk 값이 커지면 scaling이 없다는 조건하에 additive attention이 훨씬 좋은 성능을 낸다. dk가 커질수록 dot product의 결과값도 커지는 경향이 있었다. 이로인해 softmax 함수를 거치면서 \(값들이 너무크다보니, softmax를 거치면 특정 값에 비율이 몰리게되고 그러면서 다른 값들이\) 극도로 작은 gradient값을 가지게 되었다. 이러한 부작용을 해소하기 위해 $$ \sqrt d_k $$로 dot product의 결과값을 나눠주게 되었다.



#### 3.2.2 Multi-Head Attention

![](../../.gitbook/assets/image%20%281167%29.png)

임베딩 벡터의 차원이 $$ d_{model} $$인 키와 밸류 그리고 쿼리벡터를 single attenion을 수행하는 것보다 선형적으로 이들을 사영해서 얻어진 여러개의 서로 다른값들을 가지고 여러번의 attention을 각각 수행하는 것이 더 좋다는 사실을 알아냈다. 각각의 사영된 벡터들을 가지고 병렬적으로 attention을 수행하면 차원 dv의 결과들을 얻게된다. 이들을 다시 concat 하고 한번 더 사영해서 최종적으로 얻는 값을 결과벡터로 한다.

Multi-head attention은 각각의 \(single attention 공간에서 얻을 수 있는 \)서로 다른 특징을 가진 정보들을 결합할 수 있도록 한다. single attention에서는 여러 특징의 정보들을 평균내버리게 되면서 여러 정보를 얻는 것을 방지한다.

![](../../.gitbook/assets/image%20%281151%29.png)

각각의 가중치는 실수 공간이며 Q, K, V 벡터는 모두 $$ d_{model}  \times d_k$$의 크기를 가진다.

여기서는 8개의 병렬 attention layer를 적용했다. 각각의 layer는 기존 $$ d_{model} $$에서 8등분 된 64개의 차원을 사용한다. 각각의 head에서는 감소한 차원으로 진행되지만 총 연산량은 비슷하다.



#### 3.2.3 Applications of Attention in our Model

트랜스포머는 3가지의 다른 방식으로 multi-head attention을 사용한다.

* encoder-decoder attention layer에서는 이전의 디코더 레이어로부터 쿼리를 얻고 키와 밸류 벡터는 인코더의 최종 output에서 부터 받는다. 이는 디코더의 모든 위치에서 input sequence의 모든 position에 접근할 수 있도록한다. 이는 seq2seq의 attention mechanism과 동일하다.
* encoder는 self attention layer를 가지고 있는데, 여기서 사용하는 queries, keys, values vector는 모두 같은 위치에서 생성된다. 특히 이는 이전의 encoder에서 전달받으며, 각각의 position에서 모든 position으로 접근 가능하다.
* 이와 비슷하게 decoder의 self attention layer도 각각의 position에서 해당 position을 포함한 position까지 접근할 수 있게된다.  auto-regressive한 특성을 유지하기 위해 디코더에서는 왼쪽으로 전달되는 정보의 흐름을 막을 필요가 있다. 이를 위해 -inf 값을 가지고 attention에 scaled dot product를 계산한다.이 값들은 후에 softmax를 거치게 되면 \(-inf 값을 가지다보니 0에 가까운 값이 되므로\) 마스킹이 된다.



### 3.3 Position-wise Feed-Forward Networks











