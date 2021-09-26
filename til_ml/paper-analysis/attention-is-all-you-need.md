---
description: 210911~210924
---

# Attention Is All You Need

## Abstract

유망한 transduction 모델들은 인코더와 디코더를 포함하는 복잡한 반복구조나 합성곱 네트워크에 기반을 두고있다. 그 중 가장 좋은 성능을 내는 모델은 attention 메커니즘을 가지고 있는 인코더와 디코더로 이루어져있다. Transformer는 기존의 기반을 두지 않고 오로지 attention 메커니즘에만 기반을 두는 간단한 네트워크 모델이다. Transformer는 기존 모델보다 질적으로 더 우월하고 병렬작업이 훨씬 좋으며 학습하는데 적은 시간이 소요된다. 우리 모델은 WMT 2014 영어-독일어 번역 태스크에서 28.4 BLEU 점수를 기록했다. 이 점수는 기존의 앙상블을 적용한 모델의 최고 점수를 2점이나 차이나게 뛰어 넘은 점수이다. WMT 2014 영어-프랑스 번역 태스크에서도 역대 최고 수준인 41.0의 BLEU score를 달성했다. 이는 3.5일동안 8개의 GPU만을 사용해서 얻은 결과로 우리 논문에 소개한 모델의 아주 일부분에 해당한다. 

* 마지막 문장에서 literature는 문헌으로 쓰였으며 "a small ~" 의 명사구가 "for 3.5 days on eight GPUs" 라는 전치사구를 꾸며준다. 
* transduction
  * 한국어로 변역하기에는 변환, 전환, 전이, 전도 등의 의미로 사용되어 번역하기 쉬운 단어는 아니다. transduce란 무언가를 다른 형태로 변환하는 것을 의미한다. 여기서는 주어진 특정 예제\(학습 데이터\)를 이용해 다른 특정 예제\(평가 데이터\)를 예측하는 것으로 이해할 수 있다.
* attention 메커니즘
  * RNN과 LSTM에 대한 이해가 있으면 더 쉽게 이해할 수 있다. RNN에서 Sequence를 계속 입력받다보면 과거 정보가 점점 희미해지게 되는데, 이 때 LSTM이 고안되었다. 기존의 RNN보다는 개선되었지만 완벽히 Long Term Dependency를 해결하지 못했고 이를 개선하기 위한 메커니즘이 attention 이다. 특정 time step에서 인코더의 전체 입력 문장을 다시 한번 참고하기 위한 방법이다.
*  WMT 2014는 통계학적 기계 번역에 대한 90개의 워크샵의 업무에서 사용된 데이터셋이다. 워크샵은 4개의 task로 이루어져있다.
  * news translation task
  * quality estimation task
  * metrics task
  * medical text translation task



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

![](../../.gitbook/assets/image%20%281139%29.png)

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

![](../../.gitbook/assets/image%20%281137%29.png)

이러한 attention을 `Scaled Dot-Product Attention` 으로도 부를 수 있다. 각각의 입력은 dk의 차원을 가진 쿼리벡터와 키벡터 그리고 dv의 차원을 가진 밸류벡터로 이루어진다. 하나의 쿼리 벡터는 모든 키벡터를 내적하며 이 값을 $$ \sqrt d_k $$로 나눠준 뒤 softmax 함수를 거쳐서 가중치를 구하게 된다.

\(이전에 하나의 쿼리 벡터와 모든 키벡터를 내적한다고 했지만\) 실제로는 모든 쿼리벡터에 대한 모든 키벡터의 내적을 한번에 계산하게 된다. 이 때 쿼리 벡터를 행렬 Q, 키 벡터와 밸류 벡터를 행렬 K와 V로 표현한다. 이러한 행렬 계산에 대한 식은 다음과 같다.

![](../../.gitbook/assets/image%20%281150%29.png)

여기서는 주로 두 개의 attention 함수를 사용하는데,  additive attention과 dot-product attention이다. 여기서 사용하는 dot-product attention은 $$ \sqrt d_k $$로 나눠주는 과정을 제외하면 기존 dot-product attention 방법과 완전히 동일하다. additive attention은 한 개의 은닉층을 가진 신경망을 사용할 때 연관성을 구하는 함수를 사용한다. 이 두 attention 방법은 이론적으로 복잡도는 비슷하지만 실제로는 dot-product attention이 훨씬 빠르고 메모리 공간도 더 효율적으로 사용한다. 왜냐하면 이 dot-product 연산에 최적화 된 행렬 계산 코드로 모델을 구현하기 때문이다.

차원 dk가 작을 때는 dot-product와 additive는 비슷한 성능을 보이지만 dk 값이 커지면 scaling이 없다는 조건하에 additive attention이 훨씬 좋은 성능을 낸다. dk가 커질수록 dot product의 결과값도 커지는 경향이 있었다. 이로인해 softmax 함수를 거치면서 \(값들이 너무크다보니, softmax를 거치면 특정 값에 비율이 몰리게되고 그러면서 다른 값들이\) 극도로 작은 gradient값을 가지게 되었다. 이러한 부작용을 해소하기 위해 $$ \sqrt d_k $$로 dot product의 결과값을 나눠주게 되었다.



#### 3.2.2 Multi-Head Attention

![](../../.gitbook/assets/image%20%281169%29.png)

임베딩 벡터의 차원이 $$ d_{model} $$인 키와 밸류 그리고 쿼리벡터를 single attenion을 수행하는 것보다 선형적으로 이들을 사영해서 얻어진 여러개의 서로 다른값들을 가지고 여러번의 attention을 각각 수행하는 것이 더 좋다는 사실을 알아냈다. 각각의 사영된 벡터들을 가지고 병렬적으로 attention을 수행하면 차원 dv의 결과들을 얻게된다. 이들을 다시 concat 하고 한번 더 사영해서 최종적으로 얻는 값을 결과벡터로 한다.

Multi-head attention은 각각의 \(single attention 공간에서 얻을 수 있는 \)서로 다른 특징을 가진 정보들을 결합할 수 있도록 한다. single attention에서는 여러 특징의 정보들을 평균내버리게 되면서 여러 정보를 얻는 것을 방지한다.

![](../../.gitbook/assets/image%20%281152%29.png)

각각의 가중치는 실수 공간이며 Q, K, V 벡터는 모두 $$ d_{model}  \times d_k$$의 크기를 가진다.

여기서는 8개의 병렬 attention layer를 적용했다. 각각의 layer는 기존 $$ d_{model} $$에서 8등분 된 64개의 차원을 사용한다. 각각의 head에서는 감소한 차원으로 진행되지만 총 연산량은 비슷하다.



#### 3.2.3 Applications of Attention in our Model

트랜스포머는 3가지의 다른 방식으로 multi-head attention을 사용한다.

* encoder-decoder attention layer에서는 이전의 디코더 레이어로부터 쿼리를 얻고 키와 밸류 벡터는 인코더의 최종 output에서 부터 받는다. 이는 디코더의 모든 위치에서 input sequence의 모든 position에 접근할 수 있도록한다. 이는 seq2seq의 attention mechanism과 동일하다.
* encoder는 self attention layer를 가지고 있는데, 여기서 사용하는 queries, keys, values vector는 모두 같은 위치에서 생성된다. 특히 이는 이전의 encoder에서 전달받으며, 각각의 position에서 모든 position으로 접근 가능하다.
* 이와 비슷하게 decoder의 self attention layer도 각각의 position에서 해당 position을 포함한 position까지 접근할 수 있게된다.  auto-regressive한 특성을 유지하기 위해 디코더에서는 왼쪽으로 전달되는 정보의 흐름을 막을 필요가 있다. 이를 위해 -inf 값을 가지고 attention에 scaled dot product를 계산한다.이 값들은 후에 softmax를 거치게 되면 \(-inf 값을 가지다보니 0에 가까운 값이 되므로\) 마스킹이 된다.



### 3.3 Position-wise Feed-Forward Networks

attention layer의 feed forward network는 각각의 position에서 독립적이고 개별적으로 이루어진다. 그리고 이 레이어는 두 개의 선형변환으로 이루어져있으며 ReLU 활성화 함수를 그 사이에 사용한다.

![](../../.gitbook/assets/image%20%281174%29.png)

선형변환은 각각의 서로다른 position에서 같은 변환으로 적용되는 반면에 층간에서는 서로 다른 파라미터를 사용한다.. 이것을 또 다르게 이야기하면 input과 output이 차원은 512이고 hidden layer의 차원은 2048인 커널 사이즈가 1인 두개의 cnn 레이어가 있다고 할 때 두 번의 convolution을 거치는 것이다.



### 3.4 Embeddings and Softmax

다른 시퀀스 모델과 비슷한 점은 인풋과 아웃풋 토큰들을 벡터로 바꾸기 위한 임베딩을 학습한다는 것이다. 또, 디코더의 아웃풋을 가지고 다음 토큰에 대한 확률을 구하는데 선형 변환과 소프트맥스를 사용한다.  두 개의 임베딩 레이어와 소프트맥스 전에 사용하는 선형 변환 사이에는 동일한 파라미터를 사용한다. 다만, 임베딩 레이어에서는 이 가중치를 $$ \sqrt {d_{model} }$$ 로 나누어 주는 차이가 있다.

![](../../.gitbook/assets/image%20%281134%29.png)

### 

### 3.5 Positional Encoding

트랜스포머는 반복 구조도, 합성곱 구조도 없기 때문에 시퀀스의 순서를 고려하기 위해서는 각 토큰들의 위치에 대한 정보를 상대적으로든 절대적으로든 제공해줘야만 했다. 이를 위해 positional encodings를 각각의 인코더와 디코더의 첫 input embedding에 추가해주었다. positional encoding의 차원도 모델의 임베딩 차원과 동일해서 둘은 합연산이 가능하다. positional encoding 학습하거나 고정하는 많은 방법이 있는데 여기서는 주기가 다른 sine과 cosine 함수를 사용했다.

![](../../.gitbook/assets/image%20%281159%29.png)

pos는 \(시퀀스에서 토큰의\) 위치이고 i는 차원이다. p.e의 각각의 차원은 정현파와 매핑된다. \(이 정현파의\) 주기는 기하학적으로 2pi 부터 10000pi 까지 진행한다. 어떤 변수 k에 대해서 $$ PE_{pos+k} $$는 $$ PE_{pos} $$에서 선형적으로 나타낼 수 있기 때문에 상대적인 위치에서의 학습이 잘 될것이라고 가정해서 정현파를 사용했다. 

* 정현파는 일정한 주기를 가진 주기함수이다.

sine과 cosine 대신에 positional embedding을 학습하는 방법을 실험해봤는데, 둘 다 비슷한 결과를 냈다. 그래서 둘 중 정현파를 사용하는 것으로 결정했는데, 이유는 정현파로 position을 결정했을 때, 학습 단계에서는 보지못한 시퀀스의 길이보다 더 긴 길이에 대해서도 더 잘 외삽\(=작은 범위에서 그 밖의 범위를 추측하는 것\)할 수 있었기 때문이다.



## 4 Why Self-Attention

 \(recurrent와 convolutional\) 에서는 하나의 변수 x로 표현되는 시퀀스가 우리가 많이 쓰는 시퀀스 변환 인코더나 디코더의 hidden layer를 거치면서 동일한 길이를 가진 다른 변수 z로 매핑된다. 이번 장에서는 self-attention layer의 다양한 측면을 recurrent and convolutional layer와 비교할 것이다. self-attention을 사용해야 하는 3가지 이유를 알아보자.

첫번째는 레이어간 전체 연산량이고 두번째는 적은 연산으로도 계산될 수 있는, 병렬화를 연산에 적용하는 것이다.

세번째는 network에서 길이가 길었을 때 생기는 의존성\(=관계\)에 대한 길이이다. 긴 범위의 의존성을 학습하는 것은 많은 문장을 번역하는 일에서 매우 어렵다. 이러한 의존성을 학습하는 것은 네트워크를 순방향 또는 역방향으로  이동할 때 지나가는 경로의 길이에 영향을 받게된다. input과 output간의 거리가 짧을수록 이러한 의존성을 학습하기가 쉬워진다. 그래서 여러 유형의 layer에 대해서 input과 output 사이의 거리가 최대일때를 비교해보려고 한다.

이전에 표1 에서 언급했듯이 self-attention layer는 모든 시퀀스의 위치와 연결되어있는데,  순차적으로 연산이 실행될 때 상수번의 횟수만 연산이 이루어지는 반면에 RNN은 O\(n\) 만큼의 연산이 이루어진다. 이러한 시간복잡도의 관점에서 self-attention layer는 recurrent layer보다 훨씬 빠르다. 보통 문장의 길이 n은 임베딩 차원 d보다 작기 마련이다. 이러한 임베딩 차원은 word 기반으로 또는 byte-pair 기반으로 특징을 나타날 때 사용하며 최신 모델에서는 문장을 기반으로 사용한다. 문장이 아주 길때는 연산 성능을 높이기 위해서 self attention은 `r` 의 크기를 가지는 \(그래서 양쪽으로 r만큼의 position에 대해서만\) attention만을 고려하는 restricted self attention으로 사용하게 된다. 이러한 방법은 maximum path length를 \( O\(1\) 에서 \) O\(n/r\)로 증가시킨다. 이러한 접근법에 대해서는 좀 더 연구할 계획이다.

* Byte Pair Encoding에 대한 설명은 [여기](https://wikidocs.net/22592)를 참고하자.

seq 길이 n보다 커널 width k가 작은 convolutional layer는 모든 input과 output이 연결되어있지 않다. 일반적인 convolutional layer는 O\(n/k\), dilated convolution layer는 O\(logk\(n\)\) 만큼이 소요되는데 이 때 length of path도 점점 증가하게 된다. Convolution layer는 일반적으로 kernel때문에 recurrent layer보다 훨씬 비용이 많이든다. Separable convolution 같은 경우는 상당히 시간복잡도를 감소시킬 수 있긴하나 k와 n이 동일하다면 이는 T.F 모델에서 취한 복잡도와 동일하다.

그렇지만 T.F는 여기서 더 좋은점이 있다. self-attention이 훨씬 해석적인 결과를 줄 수 있다는 것. T.F의 attention 분포를 분석해봤는데 이는 부록\(=appendix\)에서 소개하고 다룬다. 각각의 attention은 여러가지 태스크를 수행하면서 분명하게 학습할뿐만 아니라, 문장에서 의미론적 구조나 문법적인 구조와 관련해서 학습했음이 나타난다.

* dilated convolution은 reverse 연산을 해서 기존 이미지를 scale up할 때 사용한다.
* Separable convolution은 CNN에서의 연산량을 줄이기 위해 Depthwise / Pointwise 방식을 이용한 Convolution 기법이다. 이에 대한 내용은 [여기](https://m.blog.naver.com/chacagea/221582912200)에 있다.

## 5 Training

모델에 대한 스펙을 설명한다.

### 5.1 Training Data and Batching

450만개의 WMT 2014 영어-독일어 데이터셋을 학습했다. 문장들은 바이트 페어 인코딩을 사용했고 37,000개의 토큰을 사용했다. 영어-프랑스어는 360만개의 데이터셋을 학습했고 32,000개의 토큰을 사용했다. 배치쌍들은 적절한 문장 길이를 가지고 배치화했으며 각각의 학습마다 25,000개의 입력 토큰과 25,000개의 출력 토큰으로 구성되었다.

* 아마 valid set 사용 또는 dropout을 통해 25,000개의 토큰으로만 구성이 된게 아닐까 싶다.



### 5.2 Hardward and Schedule

이 모델은 8개의 NVIDIA P100 GPU를 사용했다. 논문에서 언급한 파라미터를 사용했고, 한 step마다 대략 0.4초가 걸렸다. 12시간동안 10만 스텝으로 학습했다. 비교적 사이즈가 큰 모델은 한 step마다 1초가 걸렸고 3.5일동안 30만 스텝으로 학습했다.



### 5.3 Optimizer

Adam optimizer를 사용했고 b1 = 0.9, b2 =0.98, e = 10^-9 를 사용했다.학습률 공식은 다음과 같다.

![](../../.gitbook/assets/image%20%281208%29.png)

초반 warm up 단계에서는 선형적으로 학습률이 증가하다가 이후 루트 역함수의 비율로 감수한다. warm up step은 4000을 사용했다.



### 5.4 Regularization

3가지 규제화를 적용했다.

* 3가지라고 해놓고 본문에는 2가지밖에 없다.

#### Residual Dropout

각각의 서브 레이어마다 add와 normalize 이전에 드랍아웃을 적용했다. 또, 인코더와 디코더에 있는 임베딩 벡터와 포지셔널 임베딩벡터를 합한 이후에도 적용했다. dropout 비율은 0.1

#### Label Smoothing

학습 동안, e = 0.1의 라벨 스무딩을 적용했다. 이것은 perplexity를 감소시키고 모델이 더 불확실하게 학습하게 하지만 그대신 정확도와 BLEU score를 향상시킨다

* perplexity는 모댈 내에서 성능을 수치화 한 내부 평가에 해당한다. [여기](https://wikidocs.net/21697) 참고

![](../../.gitbook/assets/image%20%281204%29.png)

## 6 Results

### 6.1 Machine Translation

WMT 2014 영어-독일어 번역에서 대형 트랜스포머 모델이 이전의 최고 기록을 2.0 BLEU 차이를 내며 28.4점을 달성했다. 모델의 설정은 Table 3에 작성되어있다. 3.5일 동안 8개의 GPU를 가지고 학습했다. 겨우 이정도만 해도 이전의 성능들을 능가한 것이다.

2014 영어-불어 번역에서도 41.0의 BLEU score를 내며 모든 모델을 제쳤다. 이는 기존 모델이 학습했던 것의 1/4도 안되는 스펙으로도 능가한 것이다. 이 때 dropout은 0.1 대신 0.3을 사용했다.

기본적으로 5개의 체크포인트를 평균낸 단일 모델을 사용했다. 이 체크포인트는 10분마다 기록된다. 대형 모델은 20개의 체크포인트를 평균냈다. 또, 4개의 beam size와 a=0.6의 penalty의 beam search를 이용했다. 이 하이퍼 파라미터는 실험적으로 결정되었다. 추론시 얻었던 output의 길이는 input보다 50 정도 많게 차이가 날 수는 있었지만 대부분 비슷했다.

결과에 대한 요약과 다른 모델과의 비교는 Table 2에서 확인할 수 있다. 부동 소수점 연산의 크기는 학습 시간과 GPU의 개수 그리고 각각의 GPU에서 얻은 실수 정확도에 대한 추정값을 곱해서 얻었다.



### 6.2 Model Variations

트랜스포머의 다른 기능을 평가하기 위해서 전혀 다른 방법으로 모델을 바꿨다. 영어-독일어 번역에 대해 newstest2013 데이터 셋을 사용했다. 또, 이전에 말한 beam search를 적용했지만 체크포인트를 평균내지는 않았다. 이에 대한 결과는 Table 3 에서 볼 수 있다.

Table 3의 B행에서 attention size를 줄이는 것은 모델의 성능을 해친다는 것을 발견했다. 이는 단어들간의 연관성이 단순하지 않고 매우 정교하기 때문에 내적이외에 좀 더 효율적인 함수를 필요로 한다고 볼 수 있다. 더 나아가 C와 D행에서는 예상했듯이 모델이 커질수록 성능이 좋았고 dropout 역시 과적합을 방지해줬다. E행에서는 정현파 위치 임베딩을 학습된 위치 임베딩으로 바꿔봤는데 매우 비슷했다.



![](../../.gitbook/assets/image%20%281203%29.png)



## 7 Conclusion

이 논문에서 우리는 트랜스 포머를 소개했다. 이는 어텐션을 전반적으로 사용했고 기존의 인코더-디코더 구조에서 흔히 사용하던 recurrent layer를 multi-head self-attention으로 대체했다.

번역 task 에서는 rnn이나 cnn을 사용하던 기존 모델보다 두드러지게 빨리 학습되었다. WMT 2014 영어-독일어와 영어-불어 번역 task 에서도 신기록을 세웠다.

attention을 기반으로한 모델의 미래가 기대되고 다른 분야에서도 이것을 적용할 것이다. 또, 트랜스포머의 입출력 구조를 text뿐만 아니라 이미지나 음성, 영상과 같은 큰 데이터를 효율적으로 다루기 위해 local 또는 restricted한 attention mechanism을 연구할 계획이다.

우리가 학습과 평가를 위해 사용한 코드는  https://github.com/tensorflow/tensor2tensor. 에서 사용할 수 있다.



