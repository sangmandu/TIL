---
description: '210928'
---

# \(3강\) BERT 언어모델 소개

## 1. BERT 언어모델

### 1.1 BERT 모델 소개

버트 모델이 등장하기 전에는 다음의 모델들이 있었다.

![](../../../.gitbook/assets/image%20%281231%29.png)

* 컨텍스트 벡터의 정보를 넘겨주는, 인코더와 디코더가 분리된 구조
* 컨텍스트 벡터만으로 얻는 정보의 한계가 있어 개선된, 어텐션을 사용한 인코더 디코더 구조
* 인코더와 디코더가 하나로 합쳐져 전반적으로 어텐션을 사용하는 구조



이미지에도 인코더와 디코더의 구조를 가지고 있는 `오토인코더` 가 존재한다.

![](../../../.gitbook/assets/image%20%281235%29.png)

* 인코더의 목표는 어떠한 DATA를 압축된 형태로 표현하는 것
* 디코더의 목표는 압축된 DATA를 가지고 원본 DATA로 복원하는 것

이것을 버트에 대입해보자.

버트는 self attention을 이용하는 transformer 모델을 사용한다. 입력된 정보를 다시 입력된 정보로 representation 하는 것을 목표로 한다. 근데 이 때 MASK를 사용하게 된다. 이 때 MASK가 있으므로 입력된 정보가 다시 입력된 정보로 나오기가 어려워진다.

![](../../../.gitbook/assets/image%20%281223%29.png)

버트는 MASK된 자연어를 원본 자연어로 복원하는 작업을, GPT는 특정한 시퀀스를 잘라버리고 그 NEXT 시퀀스를 복원하는 작업이 이루어지게된다.

버트 모델의 구조는 다음과 같다.

![](../../../.gitbook/assets/image%20%281242%29.png)

* 인풋은 문장 2개를 입력받는다.
* 12개의 레이어가 ALL TO ALL로 연결되어있다.
* CLS는 두 개의 문장이 진짜 연결되어있는지 아닌지에 대한 부분을 학습하기 위한 CLASS LABEL로 사용된다.

버트의 CLS 토큰은 문장1과 문장2의 벡터들이 녹아들어있다고 가정하고있다. 그래서 CLS 토큰을 얻고 이를 Classification layer를 부착해 pre training을 진행하게 된다.

![](../../../.gitbook/assets/image%20%281247%29.png)

Tokenizing이 끝나면 masking을 하게된다.

![](../../../.gitbook/assets/image%20%281244%29.png)

* cls와 sep 토큰을 제외한 토큰에서 15%를 고른다.
* 이 중 80%는 masking, 10%는 vocab에 있는 또 다른 단어로 replacing, 10%는 unchanging 한다



![](../../../.gitbook/assets/image%20%281224%29.png)

GLUE 데이터셋을 사용하며, 여기서 최고기록을 내는 모델이 Sota 라고 할 수 있다.

이러한 12가지의 Task를 4가지 모델로 다 표현할 수 있다.

![](../../../.gitbook/assets/image%20%281246%29.png)

* 단일 문장 분류
  * 버트 모델의 한 개의 문장이 입력됐을 때 이 문장이 어떤 class에 속하는지에 대해 분류
* 두 문장 관계 분류
  * NSP
  * s1이 s2의 가설이 되는가
  * s1과 s2의 유사도
* 문장 토큰 분류
  * 보통 CLS토큰이 입력된 sentence에 대한 정보가 녹아들어 있다고 가정되는데 이 처럼 각 토큰들에 대해 분류기를 부착한다.
  * 보통 개체명인식에 많이 사용
* 기계 독해 정답 분류
  * 2가지 정보가 주어진다.
    * 하나는 질문 \(ex 이순신의 고향은\)
    * 하나는 그 정답이 포함된 문서
  * 문서에서 정답이라고 생각되는 부분의 start token과 end token을 반환한다.



### 1.2 BERT 모델의 응용

#### 감성 분석

입력된 문장의 긍부정을 파악한다.

![](../../../.gitbook/assets/image%20%281229%29.png)



#### 관계 추출

주어진 문장에서 sbj와 obj가 정해졌을 때 둘은 무슨 관계인가?

![](../../../.gitbook/assets/image%20%281227%29.png)





#### 의미 비교

두 문장이 의미적으로 같은가?

![](../../../.gitbook/assets/image%20%281239%29.png)

* task에 살짝 문제가 있는데, s1과 s2가 너무 상관없는 문장으로 매칭되었다. 실제로 적용하기에는 어려운 부분이 있다.
* 그래서, 98.3점의 점수는 높지만 데이터 설계부터 잘못된 task이다.



#### 개체명 분석

![](../../../.gitbook/assets/image%20%281222%29.png)



#### 기계 독해

![](../../../.gitbook/assets/image%20%281225%29.png)



### 1.3 한국어 BERT 모델

#### ETRI KoBERT의 tokenizing

바로 WordPiece 단위로 tokenizing 한것이 아니라 형태소 단위로 분리를 먼저 한뒤 tokenizing했다. 한국어에 특화되게 토크나이징 했다는 점에서 많은 성능향상을 가져왔다.

![](../../../.gitbook/assets/image%20%281249%29.png)

* CV : 자모
* Syllable : 음절
* Morpheme : 형태소
* Subword : 어절
* Morpheme-aware Subword : 형태소 분석 후 Wordpiece
* Word : 단어



#### Advanced BERT model

버트 내에는 entity를 명시할 수 있는 구조가 존재하지 않는다. 그래서 전처리로 각 entity앞뒤로 ENT 태그를 붙여주었다. 그랬더니 성능이 향상되었다.

![](../../../.gitbook/assets/image%20%281234%29.png)

![](../../../.gitbook/assets/image%20%281248%29.png)

![](../../../.gitbook/assets/image%20%281250%29.png)

이렇게 entity 태그를 달아주면 성능이 향상되는데, 영어권 분야에서도 이렇게 태그를 달아준 모델이 sota를 찍고있다.



## 실습 - Huggingface library 튜토리얼

HuggingFace는 매우 인기있는 Transformers 라이브러리를 구축하고 유지하는 회사이다. 

```python
from transformers import TFAutoModel, AutoTokenizer
model = TFAutoModel.from_pretrained("<model-name>")
tokenizer = AutoTokenizer.from_pretrained("<model-name>")
```

위와 같은 3줄 코드만 입력하면 BERT, RoBERTa, GPT, GPT-2, XLNet 및 HuggingFace의 자체 DistilBERT 및 DistilGPT-2등을 바로 불러올 수 있다.

또한, 허킹페이스 사이트에 검색을 할 수 있기 때문에 쉽게 모델을 찾을 수 있다.

![](../../../.gitbook/assets/image%20%281226%29.png)



### Tokenizer 실습

```text
!pip install transformers
```



### Tokenizer 응용

```python
from transformers import AutoModel, AutoTokenizer, BertTokenizer
```

```python
# Store the model we want to use
MODEL_NAME = "bert-base-multilingual-cased"

# We need to create the model and tokenizer
model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

* 104개 언어를 사용해서 학습한 모델을 불러온다. 여러 언어를 학습했기 때문에 multi-lingual model이라고 한다.

```python
print(tokenizer.vocab_size)

>>> 119547
```

* 그래서 vocab size가 12만개로 매우 크지만 한국어는 이 중 8천개정도이다.
* 한국어 corpus를 이용해 vocab을 만들 때 vocab size를 3만으로 설정하면 특수문자나 한문까지도 토큰으로 처리된다. 그래서 자기만의 bert 모델이나 bert 토크나이저를 만들 때 3만개의 word piece로 정의하면 좋다.



```python
text = "이순신은 조선 중기의 무신이다."

tokenized_input_text = tokenizer(text, return_tensors="pt")
for key, value in tokenized_input_text.items():
    print("{}:\n\t{}".format(key, value))
    
>>>
input_ids:
tensor([[   101,   9638, 119064,  25387,  10892,  59906,   9694,  46874,   9294,
          25387,  11925,    119,    102]])
          
token_type_ids:
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

attention_mask:
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```

* 3 : tokenizer에 text를 넣고 반환값은 pytorch tensor로 받겠다는 뜻. 반환값은 딕셔너리 형태로 나온다.
* 결과가 3개의 key와 리스트 형태로 반환되었다.
  * input\_ids는 text의 각 토큰의 vocab index이다.
  * 지금은 문장을 하나만 넣어줬기 때문에 token\_type\_ids는 0번으로 통일이 됐다. 만약에 text를 2개 넣어주면 두 개의 문장을 구분할 수 있도록 초기화가 된다.
  * attention\_mask는 \(질문 올렸음. 답변 받은 뒤 해결할 것\)



```python
print(tokenized_input_text['input_ids'])    # input text를 tokenizing한 후 vocab의 id
print(tokenized_input_text.input_ids)
print(tokenized_input_text['token_type_ids'])   # segment id (sentA or sentB)
print(tokenized_input_text.token_type_ids)
print(tokenized_input_text['attention_mask'])   # special token (pad, cls, sep) or not
print(tokenized_input_text.attention_mask)
```

```text
tensor([[   101,   9638, 119064,  25387,  10892,  59906,   9694,  46874,   9294,
          25387,  11925,    119,    102]])
tensor([[   101,   9638, 119064,  25387,  10892,  59906,   9694,  46874,   9294,
          25387,  11925,    119,    102]])
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```

* tokenizing 된 text 결과는 딕셔너리 형태기 때문에 key로 불러올 수도 있지만 `.` 을 사용해서 attribute 형태로 불러올 수도 있다.



```python
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
input_ids = tokenizer.encode(text)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```text
['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.']
[101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102]
[CLS] 이순신은 조선 중기의 무신이다. [SEP]
```

* 기존에는 `tokenizer(text)` 의 형태로 사용했다면 지금은 `tokenizer.tokenize(text)` 의 형태로 사용했는데, 이는 text가 tokenizing된 결과를 명시적으로 보기위함이다.
* text를 인코딩과 디코딩하게되면 문장의 시작과 끝에 CLS와 SEP 토큰이 추가된다. 이는 tokenizer가 default로 설정해놓은 기능이다.



```python
tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```text
['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.']
[9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119]
이순신은 조선 중기의 무신이다.
```

* 만약 토큰이 붙기를 원하지 않으면 `add_special_tokens=False` 로 설정하면 된다.
* 버트로 토크나이징하고 이 결과를 다른곳에 쓸 때 주로 이를 활용할 수 있지만 거의 이 옵션을 끄지않는다.



```python
tokenized_text = tokenizer.tokenize(
    text,
    add_special_tokens=False,
    max_length=5,
    truncation=True
    )
print(tokenized_text)

input_ids = tokenizer.encode(
    text,
    add_special_tokens=False,
    max_length=5,
    truncation=True
    )
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```text
['이', '##순', '##신', '##은', '조선']
[9638, 119064, 25387, 10892, 59906]
이순신은 조선
```

* `max_length`를 5로 설정했는데 여기서 이 길이는 단지 어절이나 음절의 길이가 아니라 토크나이징해서 나온 토큰들의 개수로 정해진다.



```python
print(tokenizer.pad_token)
print(tokenizer.pad_token_id)

tokenized_text = tokenizer.tokenize(
    text,
    add_special_tokens=False,
    max_length=20,
    padding="max_length"
    )
print(tokenized_text)

input_ids = tokenizer.encode(
    text,
    add_special_tokens=False,
    max_length=20,
    padding="max_length"
    )
print(input_ids)

decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```text
[PAD]
0
['이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]', '[PAD]']
[9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0]
이순신은 조선 중기의 무신이다. [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
```

* padding에는 옵션이 매우 다양하게있다.
  * pad + text
  * text + pad
  * segA + text
  * text + segA
* `padding="max_length"` 는 default padding 방식이며 text 뒤쪽에 pad를 추가해서 max\_length 만큼 맞추라는 뜻
  * 결과를 보면 text뒤에 \[PAD\]가 계속 붙는 것을 알 수 있다.



내가 원하는 token을 추가하고싶다면 어떻게 해야할까? 일단 vocab에 없는 단어를 추가할 상황이 있을까? 다음은 어떨까?

![](../../../.gitbook/assets/image%20%281232%29.png)

```python
text = "깟뻬뜨랑 리뿔이 뜨럽거 므리커럭이 케쇽 냐왜쇼 우뤼갸 쳥쇼섀료다혀뚜여"

tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```text
['[UNK]', '리', '##뿔', '##이', '뜨', '##럽', '##거', '므', '##리', '##커', '##럭', '##이', '[UNK]', '냐', '##왜', '##쇼', '[UNK]', '[UNK]']
[100, 9238, 119021, 10739, 9151, 118867, 41521, 9308, 12692, 106826, 118864, 10739, 100, 9002, 119164, 119060, 100, 100]
[UNK] 리뿔이 뜨럽거 므리커럭이 [UNK] 냐왜쇼 [UNK] [UNK]
```

* vocab에 없는 단어가 등장해서 token화 되지 못하고 \[UNK\] 토큰으로 바뀌는 모습. 이러한 unk 토큰이 많아질수록 원본 문장이 가지는 의미가 사라지게된다.



```python
added_token_num = tokenizer.add_tokens(["깟뻬뜨랑", "케쇽", "우뤼갸", "쳥쇼", "섀료"])
print(added_token_num)

tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
```

```text
5
['깟뻬뜨랑', '리', '##뿔', '##이', '뜨', '##럽', '##거', '므', '##리', '##커', '##럭', '##이', '케쇽', '냐', '##왜', '##쇼', '우뤼갸', '쳥쇼', '섀료', '다', '##혀', '##뚜', '##여']
[119547, 9238, 119021, 10739, 9151, 118867, 41521, 9308, 12692, 106826, 118864, 10739, 119548, 9002, 119164, 119060, 119549, 119550, 119551, 9056, 80579, 118841, 29935]
깟뻬뜨랑 리뿔이 뜨럽거 므리커럭이 케쇽 냐왜쇼 우뤼갸 쳥쇼 섀료 다혀뚜여
```

* `add_tokens` 로 리스트 형태로 넣어주면 쉽게 추가할 수 있다.
* 디코딩했을 때 완벽하게 원본 문장이 번역된 모습



이러한 상황은 이렇게 극단적으로 Airbnb에서만 발생하는 것이 아니다. 특정 전문성을 가진 문서나 다른 시대에 쓰인 문서에서도 이러한 상황이 발생할 수 있다.

또, 특정 역할을 위한 special token도 추가할 수 있다.

* 이전에 이야기했던 entity 토큰

```python
text = "[SHKIM]이순신은 조선 중기의 무신이다.[/SHKIM]"
```

* 위와같은 텍스트를 tokenize할 때 speical token으로 사용하려면 `tokenizer.add_special_toekns({"additional_special_tokens":["SHKIM"], "[/SHKIM]"]}`로 선언해줘야한다.

```python
text = "[SHKIM]이순신은 조선 중기의 무신이다.[/SHKIM]"

added_token_num += tokenizer.add_special_tokens({"additional_special_tokens":["[SHKIM]", "[/SHKIM]"]})
tokenized_text = tokenizer.tokenize(text, add_special_tokens=False)
print(tokenized_text)
input_ids = tokenizer.encode(text, add_special_tokens=False)
print(input_ids)
decoded_ids = tokenizer.decode(input_ids)
print(decoded_ids)
decoded_ids = tokenizer.decode(input_ids,skip_special_tokens=True)
print(decoded_ids)
```

```text
['[SHKIM]', '이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[/SHKIM]']
[119552, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 119553]
[SHKIM] 이순신은 조선 중기의 무신이다. [/SHKIM]
이순신은 조선 중기의 무신이다.
```

* 토큰으로 잘 인식되는 모습



현재까지 추가한 토큰은 단어토큰 5개와 special 토큰 2개이다.

```python
print(added_token_num)
>>> 7
```

이를 계속 변수로 담아둔 이유가 있다. 모델에서는 vocab의 size로 고정이되어있기 때문에 모델에서는 새롭게 추가한 token이 있는 text를 받으면 에러가난다. 그래서 이 때는 새로운 size로 모델을 resize해야되고 이 때 이 변수를 사용해야한다.



또, task에 따라 tokenizer가 알아서 task에 맞게 special토큰을 추가해주고 segment도 설정해준다.

```python
# Single segment input
single_seg_input = tokenizer("이순신은 조선 중기의 무신이다.")

# Multiple segment input
multi_seg_input = tokenizer("이순신은 조선 중기의 무신이다.", "그는 임진왜란을 승리로 이끌었다.")

print("Single segment token (str): {}".format(tokenizer.convert_ids_to_tokens(single_seg_input['input_ids'])))
print("Single segment token (int): {}".format(single_seg_input['input_ids']))
print("Single segment type       : {}".format(single_seg_input['token_type_ids']))

# Segments are concatened in the input to the model, with 
print()
print("Multi segment token (str): {}".format(tokenizer.convert_ids_to_tokens(multi_seg_input['input_ids'])))
print("Multi segment token (int): {}".format(multi_seg_input['input_ids']))
print("Multi segment type       : {}".format(multi_seg_input['token_type_ids']))
```

```text
Single segment token (str): ['[CLS]', '이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[SEP]']
Single segment token (int): [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102]
Single segment type       : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

Multi segment token (str): ['[CLS]', '이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[SEP]', '그는', '임', '##진', '##왜', '##란', '##을', '승', '##리로', '이', '##끌', '##었다', '.', '[SEP]']
Multi segment token (int): [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102, 17889, 9644, 18623, 119164, 49919, 10622, 9484, 100434, 9638, 118705, 17706, 119, 102]
Multi segment type       : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```



배열로 입력하면 출력 결과도 배열로 저장된다.

```python
# Padding highlight
tokens = tokenizer(
    ["이순신은 조선 중기의 무신이다.", "그는 임진왜란을 승리로 이끌었다."], 
    padding=True  # First sentence will have some PADDED tokens to match second sequence length
)

for i in range(2):
    print("Tokens (int)      : {}".format(tokens['input_ids'][i]))
    print("Tokens (str)      : {}".format([tokenizer.convert_ids_to_tokens(s) for s in tokens['input_ids'][i]]))
    print("Tokens (attn_mask): {}".format(tokens['attention_mask'][i]))
    print()
```

```text
Tokens (int)      : [101, 9638, 119064, 25387, 10892, 59906, 9694, 46874, 9294, 25387, 11925, 119, 102, 0]
Tokens (str)      : ['[CLS]', '이', '##순', '##신', '##은', '조선', '중', '##기의', '무', '##신', '##이다', '.', '[SEP]', '[PAD]']
Tokens (attn_mask): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

Tokens (int)      : [101, 17889, 9644, 18623, 119164, 49919, 10622, 9484, 100434, 9638, 118705, 17706, 119, 102]
Tokens (str)      : ['[CLS]', '그는', '임', '##진', '##왜', '##란', '##을', '승', '##리로', '이', '##끌', '##었다', '.', '[SEP]']
Tokens (attn_mask): [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```



### BERT 모델 테스트









## 실습 - BERT를 이용한 Chatbot 만들기



