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

* multi-head는 파라미터를 공유하면 안된다! 그러므로 이부분은 패스하자.
* shared-attention :여러 층으로 구성된 layer들이 모두 같은 attention 파라미터를 사용하자!
* shared-ffn : 모든 layer들의 output layer가 같은 파라미터를 사용하자
* all-shared : 모두 같은걸로 사용하자

![](../../../.gitbook/assets/image%20%281236%29.png)

* 위를 보면  당연히 성능은 떨어졌지만 하락폭이 그렇게 크지 않은 모습이다. 

알버트에 제시한 또 다른 기법은 Sentence Order Prediction. 기존의 BERT에서 사용되는 pre trained 기법은 두개가 있었다. MASK 된 단어를 맞추는 것과 두 개의 연속된 문장이 실제로 문맥상 이어지는 문장인지를 맞추는 것.

그런데, 버트 이후의 후속 연구들에서 NSP가 너무 쉬웠기 때문에 실효성이 없다는 주장이 나왔고 여러 실험결과에서 NSP를 제거하더라도 모델의 성능이 그렇게 큰 차이가 없다라고 지적을 받았다.

알버트에서는 실효성이 없는 NSP를 유의미하게 하기 위해 이를 확장했다. 기존의 "두 문장이 연속적으로 등장하는 문장인가" 를 판별하는 것이 아니라 실제로 연속적인 문장쌍을 가져와서 정순과 역순으로 변형한뒤 이를 판별하는 문제로 변경했다.

여기서 핵심은, Negative Sampling을 동일문서의 인접문장에서 뽑았다는 것이다. 기존의 bert에서의 NSP는 False의 문장쌍의 경우 두 문장에서 등장하는 단어들이 매우 상이할 가능성이 높았다. 반면 True의 문장쌍의 경우에는 두 문장에서 등장하는 단어들이 겹칠 가능성이 높았다. 그렇다보니 NSP를 할 때 모델은 고차원적인 특징에 대해 분석하기 보다는 동일한 단어들의 등장 횟수 정도의 저차원적인 특징을 사용할 가능성이 높았고 그렇기 때문에 NSP task를 제거하더라도 모델의 성능의 큰 차이가 없었던 것. 그러나 SOP는 공통 단어의 등장 횟수만으로는 이 task를 해결할 수 없으므로 좀 더 고차원적인 문제로 변경되었다고 볼 수 있다.

![](../../../.gitbook/assets/image%20%281240%29.png)

* 기존의 None과 NSP는 성능차이가 별로 없거나 오히려 None이 더 높은 상황이 발생했는데, SOP는 모든 경우에서 성능이 높은 것을 알 수 있다.

![](../../../.gitbook/assets/image%20%281233%29.png)

* 변종 모델보다 알버트의 성능이 높은 것을 알 수 있다.
* 또, 모델의 크기가 커지면 성능이 더 높아진다.



### ELECTRA: Efficiently Learning an Encoder that Classifies Token Replacements Accurately

버트나 GPT와는 다른 형태로 pretrain 한 모델이다. 실제로 Bert에서 사용한 MASK 또는 GPT에서 활용한 Standard한 Language 모델에서 한발짝 더 나아갔다. 기존에 mask된 단어를 다시 예측하는 MLM 을 두고 이 예측한 단어에 대해 이 단어가 실제로 문장에 있던 단어인지 또는 예측된 단어인지를 구별하게 되는 구분자가 추가되었다.

그래서 이 모델은, 두가지 모델이 적대적 관계를 이루는 상태로 학습이 진행이된다. 이 idea는 기존의 Generative adversarial network에서 착안한 것. Ground Truth를 알고있기 때문에 학습하기 쉽고 이러한 과정을 반복하면서 모델을 고도화시킬 수 있다.

여기서 특징은 generator가 아닌, replaced와 original을 판별하는 Discriminator를 pretrained 모델로 사용하게 된다

![](../../../.gitbook/assets/image%20%281250%29.png)

* 기존 버트모델보다 동일한 학습량에 대해 더 좋은 성능을 보인다.



### Light-weight Models

기존의 모델들은 self-attention을 점점 많이 쌓으면서 성능을 증가시켰고 경량화 모델의 연구 추세는 이러한 큰 size의 모델이 가지던 성능을 최대한 유지하면서 모델의 크기를 줄이고 계산속도를 빠르게 하는 것에 초점을 맞춘다.

그러므로, 클라우드나 고성능의 GPU를 사용하지 않고 모바일 폰에서도 사용할 수 있도록 한다.

경량화하는 방식은 다양하게 존재하미나 여기서는 Distilation이라는 방법을 사용한다.

#### DistillBERT

Transformer의 구현체를 쉽게 사용할 수 있도록 한 `huggingface` 라는 회사에서 발표한 모델이다. 여기에는 Teacher모델과 Student 모델이 있다. Student모델은 Teacher모델보다 레이어의 수나 파라미터 수가 적은 모델이다. 이 Student 모델이 경량화 모델에 초점을 맞춘 모델이다.

![](../../../.gitbook/assets/image%20%281239%29.png)

Teacher 모델이 각 시퀀스에 대해 다음에 올 단어로 예측한 확률분포가 존재할 것인데 Student 모델은 이 확률분포를 최대한 모사하는것이 목표이다. 그래서 Student 모델의 Ground Truth는 Teacher 모델의 확률분포이다. knowledge distillation 이라는 테크닉을 사용한 모델



#### TinyBERT

DistillBERT처럼 knowledge distillation 테크닉을 사용하지만 차이점이 있다면 Distil. 의 경우에는 최종 결과물을 모사하려고 한다면 TinyBERT는 중간 결과물까지도 모두 모사하려고 한다. 그래서 각 layer간의 hidden state와 attention parameter까지 동일하게 하려고 하며 이 때 MSE를 이용한다.

하지만, Student 모델의 attention parameter는 Teacher 모델의 파라미터와 동일해지기 어렵다. 왜냐하면 차원수가 다르기 때문에 동일하게 한다는 개념을 정립하지 어려울 수 있다. 그래서 이를 위해 Teacher 모델의 파라미터가 한 개의 FC를 지나서 축소된 차원의 벡터값을 갖도록 하고 Student가 이 축소된 차원과 동일하게 하기 위한 부분으로 하면서 mismatch를 해결했다.

* 이 FC 역시 학습해야한다.



### Fusing Knowledge Graph into Language Model

최신 연구 흐름은 기존의 pretraining model과 지식 그래프라 불리는 knowledge graph라는 외부 정보를 잘 결합하는 형태이다. 버트가 언어적 특성을 잘 이해하고 있는지에 대한 분석이 많이 진행되었는데, 버트는 주어진 문장에서는 문맥을 잘 파악하고 단어들간의 유사도나 관계를 잘 파악했지만 주어진 문장에 포함되어 있지 않은 추가적인 정보가 필요한 경우에는 그 정보를 효과적으로 활용하는 능력은 잘 보여주지 못했다.

만약, 주어진 문장이 다음과 같다고 하자

> 땅을 팠다

한 경우는 꽃을 심기 위해 판 것이고 또 한 경우는 집을 짓기 위해 팠다고 하자. "땅을 무엇으로 팠을까?" 라는 질문을 했을 때 사람은 꽃의 경우는 "부삽", 집의 경우는 "중장비" 등으로 대답을 할 수 있는 이유는 문장에서 얻는 정보뿐만 아니라 이미 알고있는 외부 정보\(=상식\)이 있기 때문이다. 인공지능에서의 상식은 Knowledge Graph라는 형태로 표현된다.

버트는 외부지식이 필요한 경우는 취약점을 보이기 때문에 그러한 부분을 Knowledge Graph로 잘 정의하고 이를 BERT와 잘 결합해서 문제들을 좀 더 잘 풀기 위한 연구가 진행된다.









