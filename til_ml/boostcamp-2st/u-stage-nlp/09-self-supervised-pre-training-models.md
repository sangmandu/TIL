---
description: '210915, 210916'
---

# \(09강\) Self-supervised Pre-training Models

### Recent Trends

트랜스포머 및 self-attention block은 범용적인 sequence encoder and decoder로써 최근 자연어처리에 많은 분야에서 좋은 성능을 내고있다. 심지어, 다른 분야에서도 활발하게 사용된다.

트랜스포머는 이러한 self-attention block을 6개만 쌓았는데, 최근의 발전동향은 \(모델 구조 자체의 변화는 없이\) 이를 점점 더 많이 쌓게되었다. 이를, 대규모 학습 데이터를 통해서 학습할 때 self-supervised learning framework를 통해 학습하고 transfer learning의 형태로 fine tuning해서 좋은 성능을 내고있다.

트랜스포머는 추천 시스템, 신약 개발, 영상 처리분야까지도 확장하고 있지만, 자연어 처리라는 분야에서는 &lt;sos&gt; 라는 토큰부터 **하나씩** 단어를 생성한다는 점에서 벗어나지 못한다는 한계가 있다.



## GPT-1

테슬라의 창업자 일론 머스크가 세운 비영리 연구기관인 Open AI에서 나온 모델이다. 최근에 GPT-2와 3까지 이어져 놀라운 성능을 보여주고 있다.

다양한 Special Token을 제안해서, 단순한 언어 모델 뿐만 아니라 다양한 언어 모델을 동시에 커버하는 통합된 모델을 제안했다는 것이 특징이다.

기본적으로 GPT-1의 모델구조와 학습방식에 대해 알아보자.

![](../../../.gitbook/assets/image%20%281149%29.png)

* 트랜스포머와 모양은 달라도, Text에 Position Embedding을 더한 값이 입력으로 들어가며, self-attention을 쌓은 층이 12개이다.
* 결과는 Text Prediction과 Text Classifier로 반환된다.
* Text prediction
  * 첫 단어부터 다음 단어를 순차적으로 예측한다.
* Text Classifer
  * 시퀀스에 대한 감정 분류등의 결과를 반환한다.

단순한 Task 뿐만 아니라, 문장 레벨 또는 다수의 문장이 존재하는 경우에도 모델이 손쉽게 변형 없이 활용될 수 있도록 학습의 framework를 제시했다.

![](../../../.gitbook/assets/image%20%281151%29.png)

* 우리가 알고 있는 토큰 이외에도 Delim이나 Extract라는 토큰을 사용하면서 여러가지 Task를 진행할 수 있다.

만약 모델을 통해서, 주제 분류를 해본다고 하자. \(ex 해당 doc이 정치, 경제, 사회, 스포츠 분야 중 어떤 분야인지\) 이 때는 이전에 사용하던 Text Prediction이나 Task Classifier는 떼버리고 그 전까지의 output인 word별 embedding 벡터들을 사용해서 추가적인 Task에 대한 레이어를 추가해서 학습한다.

* 이 때 마지막에 추가되는 layer는 random initialization이 되었기 때문에 추가적으로 학습을 하지만, 그 이전까지의 layer들은 학습이 이미 되어있는 상태이다. 그래서, 이전 layer들에게는 학습률을 매우 작게주면서 큰 변화가 일어나지 않도록 한다. 그래서 이전에 학습한 내용을 잘 담고있으면서 원하는 Task에는 잘 활용할 수 있도록 한다. 이는 pre-training과 fine-tuning을 동시에 적용하는 과정이다.
* 또, document의 class를 분류하는 문제에서는 데이터가 적을 수 밖에 없다보니, 이러한 데이터를 늘려도 좋지만

  또, 텍스트의 class를 구분하려면 labeling이 되어있어야 하는데, 이러한 데이터셋은 상대적으로 그 양이 작다. 그래서, 이전에 self-supervised 방식의 pre-trained 모델을 불러와서 fine tuning 하게 된다.

이렇게 pre trained 된 GPT-1 을 다양한 task에 fine tuning했을 때의 성능은 다음과 같다.

![](../../../.gitbook/assets/image%20%281168%29.png)

* 거의 대부분의 task에서 성능이 훨씬 좋은 모습을 보인다.



## BERT

버트 모델은 현재까지도 널리 쓰이는 Pre trained 모델이다. GPT와 마찬가지로 Language 모델로써 문장의 일부 단어를 맞추는 Task에 대해 Pretrained를 수행한 모델이다.

Self-supervised learning 방식으로 학습하는 Transformer 이전에 language 모델 중에서는 LSTM 기반의 인코더로 Pre-train 하는 접근 기법도 존재했는데, 이것이 바로 ELMo이다. 이러한 LSTM 기반의 인코더를 Transformer 기반의 인코더로 바꾸면서 여러 Task에 대해 좋은 성능을 가지는 모델들이 나왔는데, 그 중 하나가 BERT이다.

기존에 GPT는 전후 문맥을 파악하지 못하고 앞쪽 문맥만 보고 뒷 단어를 예측해야 한다는 한계점이 존재했다.

* 실제 사람의 대화에서든, 텍스트에서든 뒤쪽에서 문맥을 파악하는 일은 자주있는 일이다.

### Pre training Tasks in BERT : Masked Language Model

주어진 Input에 대하여 확률적으로 특정 토큰을 Masking하게 되고 이 mask가 원래 어떤 단어있는지 알아내는 방식으로 학습이 진행되게 된다.

보통은 15% 의 비율로 마스킹을 진행하는데 이 비율이 너무 크면 예측하는 것이 너무 어려워지고 너무 적으면 학습을 할 때 비용이 커지게 된다.

* 이 15%의 비율은 BERT가 찾은 최적의 비율이다.

여기서 중요한 것은, 마스킹을 하기로 한 15% 전부를 마스킹하지 않는다는 것이다. 만약, 우리가 어떤 텍스트의 감정 분석을 한다고 하자. 이 때는 단어를 예측할 일이 없기 때문에 마스킹이 필요가 없어진다. 오히려 이런 마스킹을 통해 학습된 모델은 실제 태스크와는 차이가 있어서 전이 학습을 하는데도 불과하고 성능이 감소하게 되는 모습을 보인다.

그래서, 15%의 마스킹은 다음과 같은 비율로 바뀐다.

![](../../../.gitbook/assets/image%20%281176%29.png)

버트에서 쓰인 Pre trained 기법이 단지 MASK된 단어를 예측하는 것 이외의 문장 레벨 태스크에서도 적용될 수 있도록 제안되었는데 이것이 바로 Next Sentence Prediction이라는 기법이다.

### Pre training Tasks in BERT : Next Sentence Prediction

다음과 같이 두 개의 문장을 뽑는다. 그리고 두 개의 문장 사이와 끝에는 \[SEP\] \(seperate\) 토큰을 추가해준다. 또, 문장 레벨에서의 예측 Task를 수행하는 역할을 담당하는 \[CLS\] \(classification\) 토큰을 문장에 앞에 추가해준다.

![](../../../.gitbook/assets/image%20%281202%29.png)

하고자 하는 것은, 연결된 두 개의 문장이 실제로 연결돼서 나올 수 있는 문장인지 절대 나올 수 없는 문장인지를 예측한다. 이 작업이 수행되느 순서는 다음과 같다.

* 두 개의 문장을 뽑고 masking 작업을 한다.
* 임베딩 벡터를 통해 mask 자리의 단어를 예측한다.
* CLS 토큰을 가지고 두 문장이 이어질 수 있는지 없는지에 대한 Binary classification을 하며 이에 대한 Ground Truth는 두 문장이 실제로 인접한지에 대한 부분이다.
* 결과에 대한 Loss를 가지고 CLS 토큰이 수정된다.

현재는 매우 간단하게 이야기했으므로 이에 대해 좀 더 알아보자.

### BERT Summary

#### 1. Model Architecture

모델 구조 자체는 트랜스포머의 self-attention block을 그대로 사용했으며 이에 대한 두 가지 버전으로 학습된 모델을 제안했다.

![](../../../.gitbook/assets/image%20%281192%29.png)

* L은 attention layer의 개수, A는 Attention head의 개수이다.
* H는 self attention block에서 사용하는 인코딩 벡터의 차원수이다.
* base 버전은 large보다 경량화된 버전이라고 볼 수 있다.

#### 2. Input

![](../../../.gitbook/assets/image%20%281200%29.png)

* 버트는 입력 sequence를 넣어줄 때 word 별 임베딩 벡터를 사용하는 것이 아니라, sub word별 임베딩 벡터를 사용한다.
* 트랜스포머에서 제안된 특정 주기함수의 고정된 값으로 Positional embedding을 사용했는데, 버트에서는 학습된 Positional embedding vector를 사용했다.
* CLS와 SEP를 사용했다.
* Segment embedding은 버트를 학습할 때, 두 문장이 실제 인접문장인지를 예측하는 태스크를 수행할 때 같이 사용된다. \[SEP\] 을 기준으로 두 문장이 있을 때 두번째 문장의 첫번째 단어는 Position적으로는 가장 처음에서부터 거리가 있지만 두번째 문장만을 기준으로 봤을 때는 두번째 문장의 첫번째 단어는 라는 것을 알려줘야 하고 이를 알려주는 역할을 담당한다.

![](../../../.gitbook/assets/image%20%281187%29.png)



BERT와 GPT의 차이점을 살펴보자.

![](../../../.gitbook/assets/image%20%281196%29.png)

GPT의 경우 바로 다음단어를 예측해야하기 때문에 뒤에 위치하는 단어들의 접근을 허용하면 안된다. 그래서 특정 스텝에서는 자기 자신을 포함한 이전 단어들의 정보만 허용된다.

* 그래서 Transformer의 디코더에서 사용하던 Masked Self-attention 모듈을 사용한다.

반면, 버트의 경우 Masked로 치환된 토큰들을 예측하는 것이 목적이고 그래서 Mask된 단어를 포함한 모든 단어들에 대한 접근이 가능하다.

* 그래서 Transformer의 인코더에서 사용하던 Self-attention 모듈을 사용하게된다.



### Fine-tuning Process

Mask된 단어를 예측하는 Task와 인접 문장인지를 구분하는 Task를 가지고 사전학습한 모델을 여러가지 다양한 Task에 Fine tuning한 모델들의 구조를 알아보자.

![](../../../.gitbook/assets/image%20%281180%29.png)

#### 

#### Sentence Pair Classification Tasks

![](../../../.gitbook/assets/image%20%281197%29.png)

논리적으로 내포관계 또는 모순관계를 판단하는 일이다. 두 개의 문장을 SEP 토큰으로 하나의 시퀀스로 입력하고 BERT로 인코딩을한다. 각각의 word에 대한 인코딩 벡터를 얻었다면 CLS 토큰에 해당하는 임베딩 벡터를 Output layer의 입력으로 주어서 다수 문장에 대한 예측을 할 수 있도록 한다.

#### Single Sentence Classification Tasks

![](../../../.gitbook/assets/image%20%281178%29.png)

문장이 하나밖에 없기 때문에 한 문장에 대한 CLS 토큰을 학습한다.

#### Question Answering Tasks

![](../../../.gitbook/assets/image%20%281185%29.png)

좀 더 복잡한 Task인 QA Tasks는 뒤에서 추가적으로 설명한다.

#### Single Sentence Taggine Tasks

![](../../../.gitbook/assets/image%20%281186%29.png)

각각의 단어별로 품사나 의미를 파악해야 하는 경우 CLS 토큰과 각각의 word에 대한 임베딩 벡터를 학습하게된다.



### BERT vs GPT-1

![](../../../.gitbook/assets/image%20%281201%29.png)

* 배치 사이즈의 크기가 클수록 모델의 성능이 증가하고 안정화된다는 사실이 알려져있다. 그렇지만, 배치 사이즈를 키우려면 더 많은 GPU 메모리가 필요로하게된다.

BERT는 각 단어의 임베딩 벡터를 얻고 Masked 된 word를 예측하는 Output layer를 제거한 뒤 원하는 Task에 맞게 layer를 구성할 수 있다.



### GLUE Benchmark Results

BERT를 다양한 자연어 처리 Task에 Fine Tuning 형태로 적용했을 때 일반적으로 더 좋은 성능을 냈다.

![](../../../.gitbook/assets/image%20%281188%29.png)

위 표 처럼 여러 Task를 한곳에 모아놓은 표를 GLUE 라고 한다.



### Machine Reading Comprehension\(MRC\), Question Anwsering

질의 응답에 대한 Task이다. 단순히 질문만 주어지고 답을 얻어내는 Task가 아니라, 독해력에 기반한 Task이다. 주어진 지문에 대한 질문의 답을 구하는 일이다. 그래서 `기계 독해 기반의 질의응답`이라고 한다.

![](../../../.gitbook/assets/image%20%281184%29.png)

* 4개의 장소가 있으며 두번째 줄의 they에는 Daniel을 포함하지만 네번째 줄의 they에는 포함하지 않는다. 이러한 부분까지 다 구별해서 답을 도출해야한다.
* 실제로는 더 어렵고 유의미한 Task를 해야하며, 이에대한 데이터셋으로는 SQuAD가 있다.

### SQuAD 1.1

Stanford 대학교에서 만들었기 때문에 `Stanford Question Answering Dataset` 을 줄여서 명명했다.

![](../../../.gitbook/assets/image%20%281190%29.png)

버트의 입력으로 지문과 답을 필요로 하는 질문을 SEP 토큰을 통해 Concat해서 하나의 Sequence로 제공한 뒤 인코딩을 진행한다.

질문에 대한 답이 시작되는 문장을 찾는 것을 시작점으로 한다. 각 단어의 임베딩 벡터를 얻은 뒤 이 벡터들은 FC를 거쳐 각 단어별로 스칼라값을 얻게된다. 

또, 질문에 대한 답이 끝나는 문장도 마찬가지로 찾아야 하며 또 다른 FC를 거쳐 스칼라 값을 얻게된다.

이후, 스타팅 포인트와 엔딩 포인트를 학습해서 Ground Truth에 Softmax Loss를 가지고 학습을 진행하게 된다.

### SQuAD 2.0

주어진 지문에 대한 질문이 항상 정답이 존재하지 않을 수도 있다. 이것까지 판단해서 질문이 있으면 답을, 없으면 No answer를 반환한다.

 최종적으로 예측에 이 모델을 사용할 때는 CLS를 가지고 Cross Entropy를 통해 답의 존재 유무를 먼저 파악한다. 이후는 1.1의 방식과 동일하다.

### On SWAG

주어진 문장이 있을 때 다음에 나타날법한 적절한 문장을 고르는 Task이다. 여기서도 CLS 토큰을 사용하지만, 객관식으로 이루어져있기 때문에 문제와 보기를 SEP 토큰으로 Concat해서 BERT를 통해 인코딩해서 얻은 CLS를 가지고 FC를 거쳐 스칼라값을 얻는다. 이 중 가장 큰 스칼라값이 정답이 된다.

### Ablation Study

BERT에서 각 layer 별 파라미터 수를 점점 늘린다고 할 때, 모델의 크기가 점점 커질 수록 성능이 계속적으로 끈임없이 좋아진다는 연구결과이다. GPU를 가능한 많이 써서 메모리를 늘려 학습하면 그만큼 또 성능이 오른다고한다.

> 가능하다면 모델의 사이즈를 키워라!









### 



