---
description: '210906'
---

# \(02강\) Word Embedding

## 1. Word Embedding : Word2Vec, GloVe

워드 임베딩은 자연어가 단어들을 정보의 기본 단위로 해서 각 단어들을 특정 공간에 한점으로 나타내는 벡터로 변환해주는 기법이다.

고양이를 의미하는 cat과 어린 고양이를 의미하는 kitty는 의미가 유사하므로 각 점은 가까이 위치하고 hamburger와는 멀리 위치하게 된다.

### Word2Vec

워드 임베딩을 하는 방법 중 대표적인 방법. 같은 문장에서 나타난 인접한 단어들 간에 의미가 비슷할 것이라는 가정을 사용한다. "The cat purrs" 와 " This cat hunts mice" 라는 문장에서 cat이라는 단어는 The, purrs, This, hunts, mice 와 관련이 있다.

![](../../../.gitbook/assets/image%20%281060%29.png)

어떠한 단어가 주변의 등장하는 단어를 통해 그 의미를 알 수 있다는 사실에 착안한다. 주어진 학습 데이터를 바탕으로 cat 주변에 나타나는 주변 단어들의 확률 분포를 예측하게 된다. 보다 구체적으로는 cat을 입력단어로 주고 주변단어를 숨긴채 예측하도록 하는 방식으로 Word2Vec의 학습이 진행된다.



구체적인 학습 방법은 다음과 같다.

* 처음에는 "I study math" 라는 문장이 주어진다
* word별로 tokenization이 이루어지고 유의미한 단어를 선별해 사전을 구축한다
* 이 후 사전에 있는 단어들은 사전의 사이즈만큼의 차원을 가진 ont-hot vector로 표현된다.
* 이후 sliding window라는 기법을 적용해서 한 단어를 중심으로 앞뒤로 나타나는 단어들 각각과의 입출력 단어 쌍을 구성하게된다.
  * 예를 들어 window size가 3이면 앞뒤로 하나의 단어만을 보게된다.
  * 중심 단어가 I 라면 \(I, study\) 라는 쌍을 구할 수 있게 된다.
  * 중심 단어가 study 라면 \(study, I\) 와 \(study, math\) 라는 쌍을 얻을 수 있다.
  * 즉, \(중심 단어, 주변 단어\) 라는 관계를 가진 쌍을 window size에 따라 만들어 낼 수 있게된다.

![](../../../.gitbook/assets/image%20%281047%29.png)

* 이렇게 만들어진 입출력 단어 쌍들에 대해 예측 Task를 수행하는 Two layer를 만들게 된다.
  * 입력과 출력노드의 개수는 Vocab의 사이즈와 같다.
  * 가운데에 있는 Hidden layer의 노드 수는 사용자가 정하는 하이퍼 파라미터이며, 워드임베딩을 수행하는 차원 수와 동일한 값으로 주로 결정한다.

![](../../../.gitbook/assets/image%20%281052%29.png)

* 만약 \(study, math\) 쌍을 학습한다고 하자. 그러면 input값으로 study를 의미하는 \[0, 1, 0\] 이 입력된다.
  * study : \[0, 1, 0\]
  * math : \[0, 0, 1\]
  * I : \[1, 0, 0\]
* Input layer는 3차원 Hidden layer는 2차원이므로 W는 3 \* 2의 형태를 가져야 하며 실제 X와 곱해질 때의 2 \* 3의 모양으로 곱해진다.
* Output layer는 3차원이므로 W는 3 \* 2의 모양으로 곱해진다.
* 이후, Softmax를 적용해서 확률분포 벡터를 얻게된다. 그리고 이를 Ground Truth와의 거리가 제일 가까워지게 하는 Softmax Loss를 적용함으로써 W1과 W2를 학습하게된다.
* 여기서 W1과 X를 내적하게 되는데, X의 특성상 특정 인덱스만 1이고 나머지는 다 0이다 보니, W1의 특정 인덱스 값만 뽑아온다고 볼 수 있다.
  * 그래서 코드로 구현할 때에도 내적을 구현하지 않고 X에서 1이 존재하는 인덱스를 가지고 W1에서 가져오게된다.
* 마찬가지로 W2와 W1\*X를 내적할 때도 Ground Truth값인 Y에서 1이 존재하는 인덱스에 해당하는 값만 확인하면 된다. 따라서 W2는 Y에서 1이 존재하는 인덱스에서의 값만 뽑아온다.
  * 예측값과 실제값이 가까워지려면 W2 \* W1 \* X의 값은 정답에 해당하는 인덱스의 값은 무한대에 가까워야 하고 그외의 값들은 음의 무한대에 가까워야 한다.
  * 그래야 Softmax를 적용했을 때 양의 무한대에 대해서만 1을 얻고 나머지 위치에서는 0을 얻기 때문

{% embed url="https://ronxin.github.io/wevi/" %}

위 링크에 들어가면 워드 임베딩을 시각적인 상태로 볼 수 있다. 예시로는 8개의 단어를 사용했으며 hidden size는 5이다.

![](../../../.gitbook/assets/image%20%281048%29.png)

또, 다음과 같이 W1과 W2 행렬을 시각적으로 확인할 수 있다.

* W1은 Transpose를 통해 W2와 사이즈가 같도록 나타냈다.
* 푸른색은 음수, 붉은색은 양수이다.
* 현재는 Random Initialization 된 상태이다.

![](../../../.gitbook/assets/image%20%281036%29.png)

각 단어의 임베딩 된 좌표평면도 다음과 같다.

* W1과 W2의 차원은 5개지만 PCA를 통해서 2-Dimension으로 차원 축소를 한 뒤 Scatter plot의 형태로 각각의 벡터들을 시각화 한 결과이다.

![](../../../.gitbook/assets/image%20%281040%29.png)

Trainin data는 다음과 같다.

```text
eat|apple
eat|orange
eat|rice
drink|juice
drink|milk
drink|water
orange|juice
apple|juice
rice|milk
milk|drink
water|drink
juice|drink
```



이후, 300번의 학습을 진행하게 된 후의 결과를 확인해보자.

![](../../../.gitbook/assets/image%20%281039%29.png)

* juice의 input vector는 drink의 output vector와는 유사한 벡터를 가지게 되는 것을 볼 수 있다.
  * 이 두 단어벡터의 내적값은 커지게 된다.
  * 또한 milk와 water와도 유사하다.
* eat과 apple도 유사한 벡터를 가진다.
  * orange와도 유사하다.

입력 단어와 출력 단어의 두 개의 벡터를 최종적으로 얻을 수 있고 둘 중에 어느것을 워드 임베딩의 아웃풋으로 사용해도 상관이 없으나 통상적으로 입력 단어의 벡터를 사용하게 된다.

![](../../../.gitbook/assets/image%20%281049%29.png)

이렇게 학습된 Word2Vec은 Words간의 의미론적 관계를 Vector Embedding로 잘 표현할 수 있다.

다음 그림은 Word2Vec으로 학습된 단어들의 임베딩 벡터를 표현한 것이다.

![](../../../.gitbook/assets/image%20%281042%29.png)

MAN에서 WOMAN으로의 벡터나 KING에서 QUEEN 벡터가 남성에서 여성으로의 변화를 의미하는 벡터의 관계가 워드임베딩 관계에서 잘 학습된 것을 볼 수 있다.

{% embed url="https://word2vec.kr/search/" %}

위 링크에서 Word2Vec을 한글 데이터를 통해 학습한 결과를 볼 수 있다.

![](../../../.gitbook/assets/image%20%281053%29.png)

![&#xAF64;? &#xB611;&#xB611;&#xD55C; &#xAC83; &#xAC19;&#xAD70;](../../../.gitbook/assets/image%20%281058%29.png)



{% embed url="https://github.com/dhammack/Word2VecExample" %}

또, 위 링크에서는 여러 단어들이 주어졌을 때 나머지 단어와 가장 의미가 상이한 단어를 찾아내는 작업을 할 수 있다

* 엄마, 아빠, 할아버지, 할머니, 이웃사촌 =&gt; 이웃사촌
* 각 벡터 사이의 평균 거리를 구해서 평균 거리가 가장 큰 단어를 반환한다.
* math, shopping, reading, science =&gt; shopping



Word2Vec은 단어 자체의 의미를 파악하는 Task 이외에도 다양한 자연어 처리 Task에서 자연어를 Word단위의 벡터로 나타내어 Task의 입력으로 제공되는 형태로 많이 사용된다.

* 기계 번역 : 같은 의미를 지닌 단어가 align 될 수 있도록 한다.

![](../../../.gitbook/assets/image%20%281061%29.png)

* 감정 분석 : 각 단어들의 긍/부정의 의미를 보다 용이하게 파악할 수 있도록 하는 워드 임베딩을 제공한다.

![](../../../.gitbook/assets/image%20%281043%29.png)

* Image Captioning : 이미지의 상황을 잘 이해하고 이에대한 설명글을 자연어의 형태로 생성하는 것

![](../../../.gitbook/assets/image%20%281059%29.png)

* PoS tagging, 고유명사인식 등



### GloVe

Word2Vec과 더불어 많이 쓰이는 또다른 워드 임베딩 방법이다. Word2Vec과의 큰 차이점은 다음과 같다.

* 각 입력 및 출력 쌍들에 대해서 학습 데이터에서 두 단어가 한 윈도우 내에서 몇번 등장했는지를 사전에 미리 계산한다.
* 다음 수식처럼 입력벡터와 출력벡터의 내적값에서 두 단어가 한 윈도우 내에서 동시에 몇번 등장했는지에 대한 값에 log를 취한 값을 뺀 값을 Loss Function으로 사용한다.
  * 그래서, 두 내적값이 P에 fit되도록 한다.

![](../../../.gitbook/assets/image%20%281041%29.png)

* Word2Vec에서 자주 등장하는 단어는 자주 학습됨으로써 워드 임베딩의 내적값이 그에 비례해서 커지게 되는데, GloVe에서는 단어쌍이 동시에 등장하는 횟수를 미리 계산하고 이에 대한 log값을 Ground Truth로 사용했다는 점에서 중복되는 계산을 줄여줄 수 있다는 장점이 존재한다. 그래서 학습이 Word2Vec보다 더 빠르게 되며 더 적은 데이터로도 학습이 잘 된다.
* 세부적으로는 더 많은 차이점이 있지만 큰 틀에서는 이정도의 차이가 있다.
* 두 방법은 주어진 학습데이터에 기반해서 워드임베딩을 학습하는 동일한 알고리즘이고 실제로 Task에 적용했을 때 성능도 비등비등하게 나온다.

또, GloVe 모델은 추천시스템에 많이 활용되는 알고리즘인 Co-occurrence matrix의 low rank matrix factorization의 Task로서 선형대수의 관점에서 이해할 수 있다.



특정 관점\(또는 기준\)에서 차이가 있는 단어들의 벡터관계를 살펴보면 비슷한 것을 알 수 있다.

man / woman

![](../../../.gitbook/assets/image%20%281046%29.png)

company / ceo

![](../../../.gitbook/assets/image%20%281054%29.png)

단어들간의 의미를 고려해야 하는 관계 뿐만 아니라 형용상의 원형과 비교급, 최상급의 관계를 가지고 있는 단어들 사이에도 이러한 문법적인 관계까지도 GloVe가 잘 학습했다고 할 수 있다.

![](../../../.gitbook/assets/image%20%281045%29.png)



GloVe는 Open Source로 사용할 수 있고 위키피디아 2014년도 버전 + Gigaword 5를 학습한 pretrained 된 모델을 사용할 수 있다.

![](../../../.gitbook/assets/image%20%281056%29.png)

* 60억개의 토큰\(또는 단어\)이 있다.
* 중복된 단어를 제거하고 사전으로 구성된 단어는 40만개이다.
* uncased는 대소문자를 구분하지 않았다라는 뜻
  * he와 He를 같은 단어로 취급했다는 뜻
* 50d, 100d, 200d, 300d는 GloVe 알고리즘을 적용할 때 결정한 Target Dimension의 크기이다.
  * 입력 단어와 출력 단어의 벡터의 크기이다.



## 실습

필요 패키지와 데이터 전처리는 이전 실습과 매우 비슷하므로 설명은 생략한다.

### 필요 패키지

```text
!pip install konlpy
```

```python
from tqdm import tqdm
from konlpy.tag import Okt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

import torch
import copy
import numpy as np
```



### 데이터 전처리

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

test_words = ["음식", "맛", "서비스", "위생", "가격"]
```

```python
tokenizer = Okt()

def make_tokenized(data):
  tokenized = []
  for sent in tqdm(data):
    tokens = tokenizer.morphs(sent, stem=True)
    tokenized.append(tokens)

  return tokenized
  
  train_tokenized = make_tokenized(train_data)
```

```python
word_count = defaultdict(int)

for tokens in tqdm(train_tokenized):
  for token in tokens:
    word_count[token] += 1
    
word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
print(list(word_count))
```

```text
[('.', 14), ('도', 7), ('이다', 4), ('좋다', 4), ('별로', 3), ('다', 3), ('이', 3), ('너무', 3), ('음식', 3), ('서비스', 3), ('하다', 2), ('방문', 2), ('위생', 2), ('좀', 2), ('더', 2), ('에', 2), ('조금', 2), ('정말', 1), ('맛있다', 1), ('추천', 1), ('기대하다', 1), ('것', 1), ('보단', 1), ('가격', 1), ('비싸다', 1), ('다시', 1), ('가다', 1), ('싶다', 1), ('생각', 1), ('안', 1), ('드네', 1), ('요', 1), ('완전', 1), ('최고', 1), ('!', 1), ('재', 1), ('의사', 1), ('있다', 1), ('만족스럽다', 1), ('상태', 1), ('가', 1), ('개선', 1), ('되다', 1), ('기르다', 1), ('바라다', 1), ('맛', 1), ('직원', 1), ('분들', 1), ('친절하다', 1), ('기념일', 1), ('분위기', 1), ('전반', 1), ('적', 1), ('으로', 1), ('짜다', 1), ('저', 1), ('는', 1), ('신경', 1), ('써다', 1), ('불쾌하다', 1)]
```

```python
w2i = {}
for pair in tqdm(word_count):
  if pair[0] not in w2i:
    w2i[pair[0]] = len(w2i)
    
print(train_tokenized)
print(w2i)    
```

```text
[['정말', '맛있다', '.', '추천', '하다', '.'], ['기대하다', '것', '보단', '별로', '이다', '.'], ['다', '좋다', '가격', '이', '너무', '비싸다', '다시', '가다', '싶다', '생각', '이', '안', '드네', '요', '.'], ['완전', '최고', '이다', '!', '재', '방문', '의사', '있다', '.'], ['음식', '도', '서비스', '도', '다', '만족스럽다', '.'], ['위생', '상태', '가', '좀', '별로', '이다', '.', '좀', '더', '개선', '되다', '기르다', '바라다', '.'], ['맛', '도', '좋다', '직원', '분들', '서비스', '도', '너무', '친절하다', '.'], ['기념일', '에', '방문', '하다', '음식', '도', '분위기', '도', '서비스', '도', '다', '좋다', '.'], ['전반', '적', '으로', '음식', '이', '너무', '짜다', '.', '저', '는', '별로', '이다', '.'], ['위생', '에', '조금', '더', '신경', '써다', '좋다', '.', '조금', '불쾌하다', '.']]
{'.': 0, '도': 1, '이다': 2, '좋다': 3, '별로': 4, '다': 5, '이': 6, '너무': 7, '음식': 8, '서비스': 9, '하다': 10, '방문': 11, '위생': 12, '좀': 13, '더': 14, '에': 15, '조금': 16, '정말': 17, '맛있다': 18, '추천': 19, '기대하다': 20, '것': 21, '보단': 22, '가격': 23, '비싸다': 24, '다시': 25, '가다': 26, '싶다': 27, '생각': 28, '안': 29, '드네': 30, '요': 31, '완전': 32, '최고': 33, '!': 34, '재': 35, '의사': 36, '있다': 37, '만족스럽다': 38, '상태': 39, '가': 40, '개선': 41, '되다': 42, '기르다': 43, '바라다': 44, '맛': 45, '직원': 46, '분들': 47, '친절하다': 48, '기념일': 49, '분위기': 50, '전반': 51, '적': 52, '으로': 53, '짜다': 54, '저': 55, '는': 56, '신경': 57, '써다': 58, '불쾌하다': 59}
```



### 데이터셋 클래스

`CBOW` 와 `SkipGram` 두 가지 방식에 대한 데이터셋을 정의한다.

* CBOW : Continuous Bag of Words의 약어로 주변에 있는 단어들을 가지고 중간 단어를 예측 하는 방법이다.
* SkipGram : CBOW와 반대로 중간 단어를 가지고 주변 단어를 예측하는 방법이다.

> CBOW

```python
class CBOWDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids):
          self.x.append(token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.y.append(id)

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수, 2 * window_size)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
```

* 중심 단어를 기준으로 윈도우 사이즈만큼의 주변 단어를 x로, 중심 단어를 y로 설정한다.
* 이 때\(window\_size = 2 기준\) 처음 두 단어와 마지막 두 단어는 학습 데이터에 포함되지 못하는건가? 

> SkipGram

```python
class SkipGramDataset(Dataset):
  def __init__(self, train_tokenized, window_size=2):
    self.x = []
    self.y = []

    for tokens in tqdm(train_tokenized):
      token_ids = [w2i[token] for token in tokens]
      for i, id in enumerate(token_ids):
        if i-window_size >= 0 and i+window_size < len(token_ids):
          self.y += (token_ids[i-window_size:i] + token_ids[i+1:i+window_size+1])
          self.x += [id] * 2 * window_size

    self.x = torch.LongTensor(self.x)  # (전체 데이터 개수)
    self.y = torch.LongTensor(self.y)  # (전체 데이터 개수)

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
```

* 주변 단어를 y로, 중심 단어를 x로 설정한다. 이 때 x의 개수를 주변 단어의 개수와 통일시켜준다.

```python
cbow_set = CBOWDataset(train_tokenized)
skipgram_set = SkipGramDataset(train_tokenized)
```

### 모델 클래스

```python
class CBOW(nn.Module):
  def __init__(self, vocab_size, dim):
    super(CBOW, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x):  # x: (B, 2W)
    embeddings = self.embedding(x)  # (B, 2W, d_w)
    embeddings = torch.sum(embeddings, dim=1)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output
```

```python
class SkipGram(nn.Module):
  def __init__(self, vocab_size, dim):
    super(SkipGram, self).__init__()
    self.embedding = nn.Embedding(vocab_size, dim, sparse=True)
    self.linear = nn.Linear(dim, vocab_size)

  # B: batch size, W: window size, d_w: word embedding size, V: vocab size
  def forward(self, x): # x: (B)
    embeddings = self.embedding(x)  # (B, d_w)
    output = self.linear(embeddings)  # (B, V)
    return output
```





