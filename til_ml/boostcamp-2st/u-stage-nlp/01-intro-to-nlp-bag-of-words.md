---
description: '210906'
---

# \(01강\) Intro to NLP, Bag-of-Words

## 1. Intro to Natural Language Processing\(NLP\)

### 자연어 처리와 관련된 학문 분야와 발전 동향

자연어 처리는 문장과 단어를 이해하는 Natural Language Understanding 이라 하는 NLU와 이러한 자연어를 상황에 따라 적절히 생성하는 Natural Language Generation이라 하는 NLG의 두 가지 태스크로 구성된다.

자연어 처리 분야는 비전과 함께 급속도로 발전하고 있는 분야이다. 이러한 분야가 자연어 기술에서 선두 분야이다. 이러한 기술들은 ACL, EMNLP, NAACL 이라는 학회에 발표된다.

자연어 처리에는 다음과 같은 여러 기술들을 다룬다.

* Low-level parsin
  * Tokenization : 주어진 문장을 단어단위로 끊는 것
  * stemming : "study"라는 단어도 "stydying"이나 "studied"로 어미가 다양하게 바뀔 수 있고 "하늘은 맑다. 맑지만, 맑고" 등으로 한글은 어미의 변화가 더 변화무쌍 하다. 이러한 부분도 컴퓨터가 동일한 의미라는 것을 이해할 수 있어야 하는데 이러한 단어의 어근을 추출하는 것을 의미한다.
  * 각 단어를 의미단위로 준비하기 위한 가장 로우레벨의 작업이다.
* Word and phrase level
  * Named Entity Recognition, NER : 단일 단어 또는 여러 단어로 이루어진 고유명사를 인식하는 태스크이다. NewYork Times라는 구문은 각각의 단어로 해석하면 안되고 하나의 고유명사로 해석해야 한다.
  * part-of-speech tagging, POS tagging : 단어들이 문장 내에서 품사나 성분이 무엇인지 알아내는 태스크이다. 어떤 단어는 주어이고, 동사이고, 목적어이고, 부사이고, 형용사구 이고 이러한 형용사구는 어떠한 문장을 꾸며지는 지에 대한 부분.
  * noun-phrase chunking
  * dependency parsing
  * coreference resolution
* Sentence level
  * Sentiment analysis : 주어진 문장이 긍정 혹은 부정인지 예측한다. "I love you"는 긍정, "I hate you"는 부정으로 판단해야 하며, "this movie was not that bad" 라는 문장을 bad라는 단어가 있음에도 긍정으로 판단해야 한다. Machine translation : "I studied math" 라는 구문을 "나는 수학을 공부했어" 라고 번역할 때 주어진 문장에 맞는 한글의 단어 매칭과 한국어의 문법을 고려해야 한다.
* Multl-sentence and paragraph level
  * Entailment prediction : 두 문장 간의 논리적인 내포 또는 모순 관계를 예측한다. "어제 존이 결혼을 했다." 와 "어제 최소한 한명이 결혼을 했다" 라는 문장에서 첫번째로 주어진 문장이 참인 경우 두번째로 주어진 문장이 참이된다. 또, "어제 한명도 결혼하지 않았다" 라는 문장은 첫번째로 주어진 문장과 모순관계가 된다.
  * Question answering : 독해 기반의 질의 응답. 가령, \`where did napoleon die" 라는 문장을 구글에 검색하면 이러한 단어들이 포함된 웹사이트들을 단순히 나열하는데 그쳤는데, 최근에는 이 질문을 정확히 이해하고 답에 해당하는 정보를 검색결과 제일 상단에 위치시킨다.
  * Dialog System : 챗봇과 같이 대화를 수행할 수 있는 자연어 처리 기술
  * Summarization : 주어진 문서\(뉴스나 논문\)를 한 줄 요약에 형태로 나타내는 태스크이다.

자연어를 다루는 기술로 Text mining이라는 학문도 존재한다. 이 분야는 빅데이터 분석과 많은 관련이 있다. 많은 데이터의 키워드를 시간순으로 뽑아서 트렌드를 분석할 수 있다.

* 특정인의 이미지가 과거에는 어땠고 어떠한 사건이 발생하면서 현재는 어떠하다는 것을 알아낼 수 있다.
* 회사에서 상품을 출시했을 때도 상품에 대해서 사람들이 말하는 키워드를 분석해서 상품에 대한 소비자 반응을 얻을 수 있다.
* 이러한 과정에서 서로 다른 단어지만 비슷한 의미를 가지는 키워드들을 그룹핑해서 분석할 필요가 생기게 되었고 이를 자동으로 수행할 수 있는 기법으로써 Topic Modeling 또는 Document clustering 등의 기술이 존재한다.
* 또, 사회과학과도 밀접한 관련이 있는데, "트위터나 페이스북의 소셜 미디어를 분석했더니 사람들은 어떠한 신조어를 많이 쓰고 이는 현대의 어떠한 사회 현상과 관련이 있다" 또는 "최근 혼밥이라는 단어를 많이 쓰는 것으로 보아 현대 사람들의 패턴이 어떠하게 변화한다" 라는 사회적인 인사이트를 얻는데에도 이러한 텍스트 마이닝이 많이 사용된다.
* KDD, WWW, WSDM, CIKM, ICWSM라는 학회가 존재한다.

마지막으로 Information retrieval, 정보 검색이라는 분야가 존재한다. 이는 구글이나 네이버 등에서 사용되는 검색 기술을 연구하는 분야이다. 그러나 현재 검색 기술은 어느 정도 성숙한 상태이다.\(그만큼 발전이 많이 되었다는 뜻\) 그래서 기술발전도 앞서 소개한 자연어 처리나 텍스트마이닝에 비해 상대적으로 느린 분야이다. 그러나 정보검색의 한 분야로서 추천시스템이라는 분야가 있는데, 어떠한 사람이 관심있을 법한 노래나 영상을 자동으로 추천해 주는 기술이다. 이러한 기술을 검색엔진 보다 적극적이고 자동화된 새로운 시스템이다. 또, 상업적으로도 상당한 임팩트를 가진 시스템이다.

### Trends of NLP

자연어 처리는 컴퓨터 비전과 영상 처리 기술에 비해 발전은 더디지만 꾸준히 발전해오고 있다. 딥러닝 기술은 일반적으로 숫자로 이루어진 데이터를 입력받기 때문에 주어진 텍스트 데이터를 단어 단위로 분리하고 단어를 특정 차원의 벡터로 표현하는 과정을 거치게 된다. 어떠한 단어를 벡터 공간의 한 점으로 나타낸다는 의미로 `워드 임베딩` 이라고 한다. 

단어들의 순서에 따라 의미가 달라질 수 있는데 이를 인식하기 위해 RNN이라는 구조가 자연어 처리에 자리잡게 되었고 LSTM과 이를 단순화한 GRU등의 모델이 많이 사용되었다.

2017년에 구글에서 발표한 self-attention module인 Transformer가 등장하면서 자연어 처리에서 큰 성능 향상을 가져왔다. 그래서 현재 대부분의 자연어 처리 모델은 Transformer를 기반으로 구성되어 있다. 이러한 Transformer는 초기에 기계번역을 목적으로 만들어졌다.

딥러닝이 있기전의 기계번역은 전문가가 고려한 특정 Rules을 기반으로 이루어졌는데, 너무나 많은 예외상황과 언어의 다양한 상황 패턴을 일일이 대응하는 것이 불가능했다. 이후 RNN을 사용했더니 성능이 월등히 좋아졌고 상용화되었다. 이후 성능이 오를대로 오른 분야에서 Transformer가 더욱 더 성능을 향상시켰고 뿐만 아니라 영상처리, 시계열 데이터 예측, 신약 개발이나 신물질 개발등에도 다양하게 적용되어 성능향상을 이루어내고있다.

이전에는 각각의 분야에서 모델을 사용하였는데 현재는 self-attention module을 단순히 쌓아가면서 모델의 크기를 키우고 이 모델을 대규모 텍스트 데이터를 통해 자가 지도 학습, Self-supervised training을 통해 레이블이 필요하지 않은 범용적 태스크를 통해 모델을 학습한다. 이후, 사전에 학습된 모델을 큰 구조의 변화없이도 원하는 태스크에 transfer learning의 형태로 적용하는 것이 기존에 여러 분야에 개별적인 모델을 적용하는 것보다 월등히 뛰어난 성능을 가지게 되었다.

자연어 처리에서 자가 지도 학습이라는 것은, "I \_\_\_\_\_ math" 라는 문장에서 빈칸에 들어가야 할 단어가 정확히 study인것을 맞추지는 못하더라도 이 단어가 동사라는 것과 앞뒤 문맥을 고려해 math와 I가 자연스럽게 이어질 만한 단어라는 것을 예측할 수 있다. 정리하면, 언어의 문법적이고 의미론적인 지식을 딥러닝 모델이 학습할 수 있다는 것이다.

그러나, 자가지도학습으로 모델을 학습하려면 엄청난 대규모의 데이터셋이 필요하다. 테슬라에서 발표한 바에 의하면 GPT-3를 학습하기 위한 전기세만 수십억원이다. 그래서 이러한 모델을 학습하는 곳은 막강한 자본력을 지닌 구글이나 페이스북, OpenAPI 등과 같은 일부 소수의 기관에서 이루어지고 있다.

## 2. Bag-of-Words

### Bag-of-Words Representation

Step 1. Constructing the vocabulary containing unique words

* Example sentences: “John really really loves this movie“, “Jane really likes this song”
* Vocabulary: {“John“, “really“, “loves“, “this“, “movie“, “Jane“, “likes“, “song”}
* 사전에서 중복된 단어는 한번만 등록된다.

Step 2. Encoding unique words to one-hot vectors

* 우선 Categorical한 단어들을 One-hot vector로 나타낸다. 가능한 Words가 8개이므로 차원을 8로 설정하면 각 단어마다 특정 인덱스가 1인 벡터로 나타낼 수 있다.
* Vocabulary: {“John“, “really“, “loves“, “this“, “movie“, “Jane“, “likes“, “song”}
  * John: \[1 0 0 0 0 0 0 0\]
  * really: \[0 1 0 0 0 0 0 0\]
  * loves: \[0 0 1 0 0 0 0 0\]
  * this: \[0 0 0 1 0 0 0 0\]
  * movie: \[0 0 0 0 1 0 0 0\]
  * Jane: \[0 0 0 0 0 1 0 0\]
  * likes: \[0 0 0 0 0 0 1 0\]
  * song: \[0 0 0 0 0 0 0 1\]
* For any pair of words, the distance is $$\sqrt {2} $$
  * 이 거리는 유클리드 거리라고도 한다.
* For any pair of words, cosine similarity is 0
* 단어의 의미에 상관없이 단어의 벡터 표현형을 사용한다.

이러한 원핫벡터들의 합으로 문장을 나타낼 수 있다. 이를 Bag-of-Words 라고 부른다. 그 이유는 주어진 문장 별로 가방을 준비하고, 순차적으로 문장에 있는 단어들을 해당하는 가방에 넣어준 뒤 이 수를 세서 최종 벡터로 나타낼 수 있기 떄문이다.

* Sentence 1: “John really really loves this movie“
  * John + really + really + loves + this + movie: \[1 2 1 1 1 0 0 0\]
* Sentence 2: “Jane really likes this song”
  * Jane + really + likes + this + song: \[0 1 0 1 0 1 1 1\]

이제 이러한 Bag of Words로 나타낸 문서를 정해진 카테고리나 클래스 중에 하나로 분류할 수 있는 대표적인 방법 NaiveBayes를 알아보자.

* 우선 표현할 수 있는 카테고리 혹은 클래스가 C 만큼 있다고 하자.
  * 주어진 문서를 정치, 경제, 문화, 스포츠의 4개의 주제로 표현할 수 있다면 C = 4 이다.
* 어떠한 문서 d가 주어졌을 때 이 문서 d의 클래스 c는 다음과 같은 조건부 확률로 표현될 수 있고 이 중 가장 큰 값이 해당된다. MAP는 Maximum A Posteriori의 줄임말이다.

![](../../../.gitbook/assets/image%20%281049%29.png)

이 때 베이지안 룰을 통해 두번째 식으로 나타내질 수 있다. P\(d\)는 특정 문서 d가 뽑힐 확률인데, d라는 문서는 고정된 하나의 문서로 볼 수 있기 때문에 상수로 표현될 수 있고 그래서 무시할 수 있는 값이된다.

이 때 P\(d\|c\)는 d안에 있는 words로 표현할 수 있으며 각 words가 독립적이라면 각각의 곱으로 표현할 수 있다.

![](../../../.gitbook/assets/image%20%281050%29.png)

그래서 우리는 문서가 주어지기 이전의 각 클래스가 나타날 확률 P\(c\)와 특정 클래스가 고정되어 있을 때 각 워드가 나타날 확률 P\(d\|c\)를 추정함으로써 NaiveBayes Classifier가 필요한 파라미터를 모두 추정할 수 있게된다.

만약 다음과 같은 예시가 있다고 하자.

![](../../../.gitbook/assets/image%20%281047%29.png)

그러면 각각의 클래스가 나타날 확률은 다음과 같다.

![](../../../.gitbook/assets/image%20%281042%29.png)

이후, 클래스가 고정될 때 각 단어가 나타날 확률을 추정하면 다음과 같다.

![](../../../.gitbook/assets/image%20%281037%29.png)

확률을 추정할 때는 각 클래스에 존재하는 전체 단어의 수와 해당 클래스에서 단어의 빈도 수의 대한 비율로 나타낼 수 있다.

* CV는 14개의 단어, NLP는 10개의 단어로 이루어져 있다.

결국 마지막 테스트 데이터가 속할 클래스는 각각의 확률 곱으로 구해서 예측할 수 있다.

* 이 때 각각의 단어는 독립이라는 가정이 꼭 있어야 한다.

![](../../../.gitbook/assets/image%20%281038%29.png)

NaiveBayes Classifier는 클래스의 개수가 3개 이상이어도 적용할 수 있다.

#### 또, 학습 데이터셋에 없는 단어가 등장했을 경우에는 그 외의 단어가 아무리 특정 클래스와 밀접하더라도 무조건 0의 값을 가지게 되어 해당 클래스로 분류되는 것이 불가능하게 된다. 그래서 추가적인 Regularization 기법이 적용되어서 활용이 된다.

또, 여기서는 확률을 추정할 때 전체 개수와 일부 개수의 비율로 추정했지만 실제로는 MLE, Maximum Likelihood Estimation이라는 이론적으로 탄탄한 유도과정을 통해서 도출이 된다.

## 실습

### 필요 패키지

```text
! pip install konipy
```

```python
# 다양한 한국어 형태소 분석기가 클래스로 구현되어 있음
from konlpy import tag 
from tqdm import tqdm
from collections import defaultdict
import math
```

* konlpy는 KOrean NLP in pYthon의 준말인 한국어 정보처리 파이썬 패키지이다.

### 학습 및 테스트 데이터 전처리

학습 및 테스트 데이터는 아래와 같으며 긍정적인 리뷰이면 1, 부정적인 리뷰이면 0인 두 가지 클래스로 구성되어있다.

```python
train_data = [
  "정말 맛있습니다. 추천합니다.",
  "기대했던 것보단 별로였네요.",
  "다 좋은데 가격이 너무 비싸서 다시 가고 싶다는 생각이 안 드네요.",
  "완전 최고입니다! 재방문 의사 있습니다.",
  "음식도 서비스도 다 만족스러웠습니다.",
  "위생 상태가 좀 별로였습니다. 좀 더 개선되기를 바랍니다.",
  "맛도 좋았고 직원분들 서비스도 너무 친절했습니다.",
  "기념일에 방문했는데 음식도 분위기도 서비스도 다 좋았습니다.",
  "전반적으로 음식이 너무 짰습니다. 저는 별로였네요.",
  "위생에 조금 더 신경 썼으면 좋겠습니다. 조금 불쾌했습니다."
]
train_labels = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]

test_data = [
  "정말 좋았습니다. 또 가고 싶네요.",
  "별로였습니다. 되도록 가지 마세요.",
  "다른 분들께도 추천드릴 수 있을 만큼 만족했습니다.",
  "서비스가 좀 더 개선되었으면 좋겠습니다. 기분이 좀 나빴습니다."
]
```

```python
tokenizer = tag.Okt()
```

* tokenizer는 konlpy에서 제공하는 Okt를 사용한다. 이는 Open Korea Text의 준말이다.
* 그 외에도 Mecab, Komoran, Hannanum, Kkma 라는 형태소 분석기\(Tokenizer\)가 있다.

```python
def make_tokenized(data):
  tokenized = []  # 단어 단위로 나뉜 리뷰 데이터.

  for sent in tqdm(data):
    tokens = tokenizer.morphs(sent)
    tokenized.append(tokens)

  return tokenized
```

* sent는 sentence를 지칭하는 변수이며 각 data에 있는 말뭉치에서 한 개의 문장을 의미한다.
* `morphs` 함수는 텍스트를 형태소 단위로 나누는 함수이다.
* tokenize 된 단어들은 `tokenized` 에 추가되고 최종적으로 반환된다.

```python
train_tokenized = make_tokenized(train_data)
test_tokenized = make_tokenized(test_data)
```

```python
train_tokenized
```

```text
[['정말', '맛있습니다', '.', '추천', '합니다', '.'],
 ['기대했던', '것', '보단', '별로', '였네요', '.'],
 ['다',
  '좋은데',
  '가격',
  '이',
  '너무',
  '비싸서',
  '다시',
  '가고',
  '싶다는',
  '생각',
  '이',
  '안',
  '드네',
  '요',
  '.'],
 ['완전', '최고', '입니다', '!', '재', '방문', '의사', '있습니다', '.'],
 ['음식', '도', '서비스', '도', '다', '만족스러웠습니다', '.'],
 ['위생',
  '상태',
  '가',
  '좀',
  '별로',
  '였습니다',
  '.',
  '좀',
  '더',
  '개선',
  '되',
  '기를',
  '바랍니다',
  '.'],
 ['맛', '도', '좋았고', '직원', '분들', '서비스', '도', '너무', '친절했습니다', '.'],
 ['기념일',
  '에',
  '방문',
  '했는데',
  '음식',
  '도',
  '분위기',
  '도',
  '서비스',
  '도',
  '다',
  '좋았습니다',
  '.'],
 ['전반', '적', '으로', '음식', '이', '너무', '짰습니다', '.', '저', '는', '별로', '였네요', '.'],
 ['위생', '에', '조금', '더', '신경', '썼으면', '좋겠습니다', '.', '조금', '불쾌했습니다', '.']]
```

학습 데이터기준으로 가장 많이 등장한 단어부터 순서대로 Vocaburary에 추가한다.

```python
word_count = defaultdict(int)  # Key: 단어, Value: 등장 횟수

for tokens in tqdm(train_tokenized):
  for token in tokens:
    word_count[token] += 1
```

```python
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
print(len(word_count))
```

```text
66
```

* 총 등록된 단어 수는 66개이며, `word_count` 에는 빈도수가 높은 것부터 정렬되어 저장된다.

```python
word_count
```

```text
[('.', 14),
 ('도', 7),
 ('별로', 3),
 ('다', 3),
 ('이', 3),
 ('너무', 3),
 ('음식', 3),
 ('서비스', 3),
 ('였네요', 2),
 ('방문', 2),
 ('위생', 2),
 ('좀', 2),
 ('더', 2),
 ('에', 2),
 ('조금', 2),
 ('정말', 1),
 --- 이하 생략 ---
```

이후, 각 단어마다 index를 부여하기 위해서 다음과 같이 구현한다.

```python
w2i = {}  # Key: 단어, Value: 단어의 index
for pair in tqdm(word_count):
  if pair[0] not in w2i:
    w2i[pair[0]] = len(w2i)
```

* 해당 단어가 없으면 w2i 딕셔너리에 추가하고 새로 갱신된 길이를 값으로 부여하는 방식이다.

```python
w2i
```

```text
{'!': 35,
 '.': 0,
 '가': 41,
 '가격': 23,
 '가고': 26,
 '개선': 43,
 '것': 20,
 '기념일': 52,
 '기대했던': 19,
 '기를': 45,
 '너무': 5,
 '는': 61,
 '다': 3,
 '다시': 25,
 --- 이하 생략 ---
```

### 모델 Class 구현

NaiveBayes Classifier 모델 클래스를 구현한다.

* `self.k`: Smoothing을 위한 상수.
* `self.w2i`: 사전에 구한 vocab.
* `self.priors`: 각 class의 prior 확률.
* `self.likelihoods`: 각 token의 특정 class 조건 내에서의 likelihood.

```python
class NaiveBayesClassifier():
  def __init__(self, w2i, k=0.1):
    self.k = k
    self.w2i = w2i
    self.priors = {}
    self.likelihoods = {}

  def train(self, train_tokenized, train_labels):
    self.set_priors(train_labels)  # Priors 계산.
    self.set_likelihoods(train_tokenized, train_labels)  # Likelihoods 계산.

  def inference(self, tokens):
    log_prob0 = 0.0
    log_prob1 = 0.0

    for token in tokens:
      if token in self.likelihoods:  # 학습 당시 추가했던 단어에 대해서만 고려.
        log_prob0 += math.log(self.likelihoods[token][0])
        log_prob1 += math.log(self.likelihoods[token][1])

    # 마지막에 prior를 고려.
    log_prob0 += math.log(self.priors[0])
    log_prob1 += math.log(self.priors[1])

    if log_prob0 >= log_prob1:
      return 0
    else:
      return 1

  def set_priors(self, train_labels):
    class_counts = defaultdict(int)
    for label in tqdm(train_labels):
      class_counts[label] += 1
    
    for label, count in class_counts.items():
      self.priors[label] = class_counts[label] / len(train_labels)

  def set_likelihoods(self, train_tokenized, train_labels):
    token_dists = {}  # 각 단어의 특정 class 조건 하에서의 등장 횟수.
    class_counts = defaultdict(int)  # 특정 class에서 등장한 모든 단어의 등장 횟수.

    for i, label in enumerate(tqdm(train_labels)):
      count = 0
      for token in train_tokenized[i]:
        if token in self.w2i:  # 학습 데이터로 구축한 vocab에 있는 token만 고려.
          if token not in token_dists:
            token_dists[token] = {0:0, 1:0}
          token_dists[token][label] += 1
          count += 1
      class_counts[label] += count

    for token, dist in tqdm(token_dists.items()):
      if token not in self.likelihoods:
        self.likelihoods[token] = {
            0:(token_dists[token][0] + self.k) / (class_counts[0] + len(self.w2i)*self.k),
            1:(token_dists[token][1] + self.k) / (class_counts[1] + len(self.w2i)*self.k),
        }
```



집중 분석해보자!

> init과 train

```python
class NaiveBayesClassifier():
    def __init__(self, w2i, k=0.1):
    self.k = k
    self.w2i = w2i
    self.priors = {}
    self.likelihoods = {}
    
  def train(self, train_tokenized, train_labels):
    self.set_priors(train_labels)  # Priors 계산.
    self.set_likelihoods(train_tokenized, train_labels)  # Likelihoods 계산.    
```

* 클래스는 처음에 k라는 스무딩을 위한 상수와 사전에 구한 vocab, 그리고 각 class의 prior 확률과 각 token의 특정 class 조건 내에서의 likelihood를 구할것이다.
* 위에서 설명한 다음 식을 기억하는가!?

![](../../../.gitbook/assets/image%20%281050%29.png)

* 여기서 P\(c\) 를 구하는 작업이 `set_priors` 이고 P\(d\|c\)를 구하는 과정이 `set_likelihoods` 이다.

> set\_priors

```python
  def set_priors(self, train_labels):
    class_counts = defaultdict(int)
    for label in tqdm(train_labels):
      class_counts[label] += 1
    
    for label, count in class_counts.items():
      self.priors[label] = class_counts[label] / len(train_labels)
```

* `set_priors` 는 위처럼 구현되어 있는데, 여기서 `train_labels` 라는 인자를 입력받는다. 이는 `train_labels = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]` 이런꼴로 표현된다.
* `class_counts` 는 각 라벨별 개수를 센다. 주어진 `train_labels` 로 생각해보면 다음과 같이 구성될 것이다
  * `class_counts[0] = 5`
  * `class_counts[1] = 5`
* `priors` 는 단지 전체 개수에 대한 비율이다. 이또한 다음과 같이 구성될 것이다
  * `prior[0] = 5/10 = 1/2`
  * `prior[1] = 5/10 = 1/2`

> set\_likelihoods

```python
  def set_likelihoods(self, train_tokenized, train_labels):
    token_dists = {}  # 각 단어의 특정 class 조건 하에서의 등장 횟수.
    class_counts = defaultdict(int)  # 특정 class에서 등장한 모든 단어의 등장 횟수.

    for i, label in enumerate(tqdm(train_labels)):
      count = 0
      for token in train_tokenized[i]:
        if token in self.w2i:  # 학습 데이터로 구축한 vocab에 있는 token만 고려.
          if token not in token_dists:
            token_dists[token] = {0:0, 1:0}
          token_dists[token][label] += 1
          count += 1
      class_counts[label] += count

    for token, dist in tqdm(token_dists.items()):
      if token not in self.likelihoods:
        self.likelihoods[token] = {
            0:(token_dists[token][0] + self.k) / (class_counts[0] + len(self.w2i)*self.k),
            1:(token_dists[token][1] + self.k) / (class_counts[1] + len(self.w2i)*self.k),
        }
```

* likelihoods가 나왔다고 쫄지말자. 여기서는 쉽게쉽게 구현한다.
* 5-13
  * 우리가 학습하려는 train 데이터에 대해서 정답과 문장에 접근하기 위해 이중 반복문 형태로 접근한다.
  * 이 때 token이 w2i에 포함되어야 한다는 조건문이 있는데, 우리의 token은 모두 w2i에 포함되어있다. 그럼 이 조건문은 왜있는걸까? 만약 우리의 데이터셋이 매우 크다면 모든 token을 다 vocab으로 저장하고 이를 임베딩 할 수 없다. 왜냐면 token이 많아질수록 임베딩 벡터의 차원도 커질것이고 이는 메모리 사용에 문제가 생길 수 있으니까! 그래서 빈도수가 적으면\(예를 들어 5 이하라면\) vocab에 추가하지 않는 조건문을 vocab을 생성할 때 사용하는데, 여기서는 데이터셋이 매우 작기 떄문에!!! 빈도수 상관없이 모두 w2i에 추가했다! 그러니, 여기서는 관습적인 표현\(원래는 자주 쓰지만 여기서는 쓰지 않았음\)으로만 해석하자!
  * `token_dists` 는 해당 token이 긍정으로 쓰인횟수와 부정으로 쓰인 횟수를 기억하기 위한 변수!
  * `class_counts` 는 각 token 을 조사하면서 긍정으로 쓰인 token은 몇개일까? 부정으로 쓰인 token은 몇개일까? 를 기억하기 위한 변수!
* 15-20
  * `token_dists` 와 `class_counts` 에 대한 조사가 끝났다면 이를 가지고 각각의 token에 대한 likelihood값을 반환한다.
  * 이 token이 긍정적으로 쓰일 가능성 : `이 token이 긍정으로 쓰인 횟수 / 긍정적으로 쓰인 전체 token 개수`
  * 부정적으로 쓰일 가능성도 동일하며, 각각 분자 분모에 더해진 `k` 와 `len(w2i) * k` 는 `zero probability` 문제를 해결하는 테크닉이다!
    * zero probability가 뭐냐고? [아까 말한 이 부분](01-intro-to-nlp-bag-of-words.md#0-regularization)!

> inference

```python
  def inference(self, tokens):
    log_prob0 = 0.0
    log_prob1 = 0.0

    for token in tokens:
      if token in self.likelihoods:  # 학습 당시 추가했던 단어에 대해서만 고려.
        log_prob0 += math.log(self.likelihoods[token][0])
        log_prob1 += math.log(self.likelihoods[token][1])

    # 마지막에 prior를 고려.
    log_prob0 += math.log(self.priors[0])
    log_prob1 += math.log(self.priors[1])

    if log_prob0 >= log_prob1:
      return 0
    else:
      return 1
```

* 테스트 데이터에 대한 긍정 또는 부정 클래스를 반환하기 위한 코드이다.
* 초기에 긍정과 부정에 대한 확률을 0으로 초기화한다.
* 각 token에 대한 긍정 혹은 부정에대한 가능성을 추가한다. if문이 있는 이유는 위에서 설명한 관습적 명시와 동일한데, 우리가 학습하지 않은 데이터로는 테스트 데이터에서 처음 본 토큰을 판단할 수 없기 때문에 학습한 토큰에 대해서만 판단할 수 있도록 하기 위함이다.
* 로그함수를 취하더라도 대소관계는 달라지지 않으나 log형태는 각 항의 곱을 덧셈으로 바꿔주므로 computational cost를 줄여주는 효과가 있어 log likelihood로 변경해주게된다.
* 이후 긍정 혹은 부정값 중 큰 값의 클래스를 반환한다.

### 모델 학습 및 테스트

```python
classifier = NaiveBayesClassifier(w2i)
classifier.train(train_tokenized, train_labels)
```

```python
preds = []
for test_tokens in tqdm(test_tokenized):
  pred = classifier.inference(test_tokens)
  preds.append(pred)
preds
```

```python
[1, 0, 1, 0]
```

테스트 결과 모두 알맞게 나온 모습

* "정말 좋았습니다. 또 가고 싶네요." = 긍정
* "별로였습니다. 되도록 가지 마세요." = 부정
* "다른 분들께도 추천드릴 수 있을 만큼 만족했습니다." = 긍정
* "서비스가 좀 더 개선되었으면 좋겠습니다. 기분이 좀 나빴습니다." = 부정

그래서, 내가 추가로 3개의 데이터를 실험해보았다.

* 기존 데이터셋을 참고해도 알 수 없는 긍정 표현
  * "맛도 없고 서비스도 별로지만 종업원이 이뻐서 또 갈거에요"
* 매우 많은 부정표현이 있지만 결국 긍정 표현
  * "서비스도 별로였네요. 너무 비싸서 가고 싶지 않고 위생 상태가 조금 불쾌했습니다. 음식도 너무 짰습니다. 그러나 우리 엄마 가게라서 추천합니다."
* 매우 많은 긍정표현이 있지만 결국 부정 표현
  * "정말 맛있습니다. 완전 최고입니다!. 음식도 분위기도 서비스도 다 좋았습니다. 그러나 가격이 너무 비싸서 별로였네요."

결과는 다음과 같다.

* 기존 데이터셋을 참고해도 알 수 없는 긍정 표현 =&gt; 긍정
  * 천잰가?
  * "맛도" =&gt; 학습 데이터에서 긍정에서만 사용
  * "서비스도" =&gt; 학습 데이터에서 긍정에서만 사용
  * 이러한 이유로 긍정이 나온것으로 보임. 반대로 "맛도 없고 서비스도 별로네요" 에 대해서도 긍정이 나온다.
* 매우 많은 부정표현이 있지만 결국 긍정 표현 =&gt; 부정
  * 부정 단어가 훨씬 많아서 부정, 긍정 표현은 학습 데이터에 없음
* 매우 많은 긍정표현이 있지만 결국 부정 표현 =&gt; 긍정
  * 위와 마찬가지이다. 긍정 표현과 부정 표현 모두 학습 데이터에 있지만 개수의 차이로 긍정.





