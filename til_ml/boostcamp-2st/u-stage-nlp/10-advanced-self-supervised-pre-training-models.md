---
description: '210916'
---

# \(10강\) Advanced Self-supervised Pre-training Models

## GPT-2

보다 최근에 나온 모델이다. GPT-1 과 모델 구조적인 차이는 없지만 다음과 같은 특징이 있다.

* Transformer 모델의 레이어가 많아졌다
* 여전히 다음 단어를 예측하는 Language Model을 사용하였다.
* Training data는 40GB 라는 증가된 데이터를 사용했다.
  * 이 때 단순히 그냥 데이터가 아니라 품질이 매우 좋은 데이터를 사용했다.
* 여러 down-stream stasks가 zero-shot setting 으로 모두 다뤄질 수 있다는 점을 보여준다.

![](../../../.gitbook/assets/image%20%281193%29.png)

* 사람이 쓴 부분\(빨간색\) 이후로 매우 자연스럽게 글\(파란색\)을 작성한 모습



### Motivation

모든 자연어 Task들이 질의응답의 형태로 바뀔 수 있다는 통찰을 제시했다.

* 이전에는 Binary classification의 감정분석과 How are you 라는 질문에 대한 답변을 예측하는 것이 서로 다른 Output layer 구조를 가지기 때문에 다른 Task로 간주되었다.
* 감정분석
  * "Do you think this sentence is positive?"
* 문단요약
  * "What is topic or point on this literature?"



### Datasets

데이터셋의 크기가 매우 크면서도 품질 역시 좋아 지식을 효과적으로 잘 배울 수 있다.

일부는 레딧에 있는 답변중 외부링크가 있고, 또 이 답변이 좋아요가 3개 이상 받았을 때 이 외부링크 속 게시물을 데이터셋으로 만들었다.

* 레딧은 질문/답변 사이트

또한, Byte pair encoding을 사용했으며 Layer Normalization의 위치가 조금 바뀌었다. 또, 레이어가 올라갈수록 선형변환에 사용되는 수들이 0에 가까워지도록 했는데 이는, 위쪽의 레이어의 역할이 줄어들 수 있도록 모델을 구성했다.



### Question Answering

모든 Task는 질의응답에 형태로 바뀔 수 있다. 원래는 주어진 대화형 질의응답 데이터를 가지고 Fine tuning 하는 과정을 거쳐야하는데, 이를 학습하지 않고\(=Zero shot setting\) 바로 추론하는 실험을 해보았더니 55 F1 score가 나왔고 fine tuning을 거치니 89 F1 score가 나왔다.



### Summarization

Zero-shot setting으로도 요약이 가능했는데 이 이유는 GPT-2는 이전 단어로 뒷 단어를 예측하다보니, 문단의 마지막 단어가 TR 토큰으로 예측할 수 있도록 초기 모델 학습 때 데이터셋의 각 문장 끝에 TR 토큰을 추가했고 \(왜냐하면 요약 Task을 위한 Fine tuning이 그렇게 이루어짐\) 바로 어떠한 데이터의 전처리가 필요없이 기존 학습만으로도 요약을 수행할 수 있게된다.



### Translation

마찬가지로, 번역 역시 주어진 문장이나 문단 뒤에 "they say in french" 등을 붙여주면 해당언어\(여기서는 불어\)로 잘 번역하는 성능을 보여줬다.



## GPT-3

GPT-2 를 더 개선한 모습. 기존 모델의 구조를 바꾸진 않았고 훨씬 많은 데이터셋, 훨씬 깊은 레이어, 훨씬 많은 배치 사이즈를 적용해서 성능을 많이 끌어올렸다.

![](../../../.gitbook/assets/image%20%281189%29.png)

GPT-3는 2020년에 인공지능으로 유명한 학회 NeruIPS 에서 Best Paper 상을 받았는데 그 이유는 다음과 같은 놀라운 점을 보여줬기 때문이다.

### Language Models are Few-Shot Learners

기존에 GPT-2 가 보여준 Zero-shot 세팅에서의 가능성을 많이 끌어올렸다.

![](../../../.gitbook/assets/image%20%281191%29.png)

* 영어를 불어로 번역하는 Task를 어떠한 fine tuning 없이 진행하는 모습.

![](../../../.gitbook/assets/image%20%281199%29.png)

* one shot은 이에 대해 예시를 딱 한번만 제공해주는 것
* 번역을 위한 학습 데이터로 딱 한쌍만 제공하는데, 이러한 데이터의 학습을 위해 모델의 레이어를 변경하거나 하는 것은 일절 하지 않고 기존 데이터의 모양으로 사용한다.
* Zero-shot 세팅보다 훨씬 성능이 증가한다.

![](../../../.gitbook/assets/image%20%281183%29.png)

* one-shot 보다 성능이 더 향상한다.

데이터를 "동적으로" 학습하고 좋은 성능을 낸다는 점에서 GPT-3의 장점을 보여준다.



또한, 모델의 크기가 크면 클수록 모델의 "동적 적응 능력"이 빠르게 상승함을 알 수 있다.

![](../../../.gitbook/assets/image%20%281179%29.png)



### A Lite BERT for Self-supervised Learning of Language Representations

ALBERT 모델은 기존의 메모리나 학습 비용에 대한 장애물을 줄이고 성능에 큰 하락 없이, 오히려 성능을 향상 시키려고 했다.

또, 새로운 변형된 형태의 문장레벨의 self-supervised learning의 pre-trained task를 제안했다.

구체적으로 살펴보자.

#### Factorized Embedding Parameterization

임베딩 벡터의 차원이 작으면 시퀀스의 특징을 모두 다 담아내지 못하게되고, 그렇다고 너무 크면 연산량도 늘고 파라미터 수도 증가하게되는 딜레마가 있었다.

근데, 잘 생각해보자. 레이어를 심층있게 쌓는다는 것은 시퀀스의 세밀하고 유의미한 특징들을 파악하겠다는 것과 같다. 그렇다는 것은 이러한 특징파악이라는 역할을 깊은 레이어에게 맡기고, 각 단어가 가지는 임베딩 벡터의 차원은 조금 줄어들어도 괜찮지 않을까 라는 아이디어가 생겼고 이를 위해 임베딩 차원을 줄이는 인사이트를 알버트가 제안하게된다.

기존에는 다음과 같이 단어의 임베딩에 positinal 벡터를 더해서 레이어에서 사용했다면,

![](../../../.gitbook/assets/image%20%281177%29.png)

알버트에서는 레이어의 입력으로 주는 차원은 기존과 동일하지만, 레이어의 입력 차원과 임베딩의 차원은 같을 필요가 없도록 하는 아이디어를 제시한다.

![](../../../.gitbook/assets/image%20%281182%29.png)

* 버트는 임베딩 벡터의 차원이 4인데 비해 알버트는 2의 차원을 가지는 것을 볼 수 있다.
* 2의 차원을 가진 임베딩 벡터의 가중치 W\(E x H shape\)를 곱해서 4차원의 벡터를 생성하게 된다.

이 방법이 실제로 파라미터 수를 줄여주었을까?

![](../../../.gitbook/assets/image%20%281198%29.png)

Vocab size가 500이고, attention layer에서 사용되어야 하는 차원이 100이라고 하자.

* 버트\(왼쪽\)는 500\*100 = 50,000 개의 파라미터가 필요하다
* 알버트\(오른쪽\)은 500\*15 + 15\*100 = 9,000 개의 파라미터가 필요하다. 

이러한 방법으로 필요한 파라미터 수를 줄이게 되며 이는 Vocab size와 hidden dimension이 증가할 수록 파라미터 수의 차이가 나게된다.



알버트가 가지는 장점은 또 있다. 기존의 트랜스포머에서 레이어를 깊게 쌓을수록 파라미터 수가 점점 늘어나는 이유는 다음과 같다.

* multi-head attention을 적용하면서 각각의 head가 독립적이면서 서로 다른 Q, K, V를 사용하기 때문
* 여러 층으로 구성된 layer 역시 모두 다른 Q, K, V를 사용하기 때문

알버트는 위를 이렇게 해결했다.

* multi-head는 파라미터를 공유하면 안된다!











