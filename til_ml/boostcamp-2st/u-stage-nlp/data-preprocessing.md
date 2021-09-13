---
description: '210906'
---

# \[필수 과제\] Data Preprocessing

## 1. Spacy를 이용한 영어 전처리

```python
import spacy
spacy_en = spacy.load('en')
```

* 고급 자연어 처리를 위한 파이썬 오픈소스 라이브러리이다. 영어의 경우 토큰화를 사용하는 대표적인 도구 중 하나이며 그 외에는 NLTK가 있다. 한국어는 지원하지 않는다.

```python
nlp = spacy.load('en_core_web_sm')
```

* `en_core_web_sm`
  * `en_core` :  consist of english language
  * `web` : written web text
    * blog, news, comments
  * `sm` : small

### 1.1 Tokenization

내용 없음



### 1.2 불용어 \(Stopword\)

**불용어**\(Stop word\)는 분석에 큰 의미가 없는 단어를 지칭한다. 예를 들어 the, a, an, is, I, my 등과 같이 문장을 구성하는 필수 요소지만 문맥적으로 큰 의미가 없는 단어가 이에 속한다.

```python
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
for stop_word in list(spacy_stopwords)[:30]:
  print(stop_word)
```

```text
can
should
herein
whence
more
which
via
therefore
before
from
very
into
beyond
whereafter
‘d
moreover
their
then
's
enough
‘m
and
our
when
done
hereafter
hers
whoever
made
fifteen
```

* spacy에서 지정한 불용어들은 다음과 같다. 내가 보기엔 불용어가 아닌 것 같은 단어도 있는 것 같다.

```python
stopword_text = [token for token in text if not token.is_stop]
print(stopword_text)
```

```text
[film, development, began, Marvel, Studios, received, loan, Merrill, Lynch, April, 2005, ., success, film, Iron, Man, 2008, ,, Marvel, announced, Avengers, released, July, 2011, bring, Tony, Stark, ,, Steve, Rogers, ,, Bruce, Banner, ,, Thor, Marvel, previous, films, ., signing, Johansson, Natasha, Romanoff, March, 2009, ,, film, pushed, 2012, release, ., Whedon, brought, board, April, 2010, rewrote, original, screenplay, Zak, Penn, ., Production, began, April, 2011, Albuquerque, ,, New, Mexico, ,, moving, Cleveland, ,, Ohio, August, New, York, City, September, ., film, 2,200, visual, effects, shots, .]
```

* spacy로 만들어진 token들은 `is_stop` 이라는 속성을 가지며 이 값이 True일 경우 불용어에 해당한다. 위 코드는 불용어가 아닌 토큰들을 출력하는 코드

### 1.3 Lemmatization

표제어 추출이라고 하며, 단어의 어간을 추출하는 작업을 의미한다. 마치 study, studied, studying 에서 study라는 어간을 추출하는 것과 같다.

```python
for token in text[:20]:
  print (token, "-", token.lemma_)
```

```text
The - the
film - film
's - 's
development - development
began - begin
when - when
Marvel - Marvel
Studios - Studios
received - receive
a - a
loan - loan
from - from
Merrill - Merrill
Lynch - Lynch
in - in
April - April
2005 - 2005
. - .
After - after
the - the
```

### 1.4 그외 token class의 attributes

```python
print("token \t is_punct \t is_space \t shape_ \t is_stop")
print("="*70)
for token in text[21:31]:
  print(token,"\t", token.is_punct, "\t\t",token.is_space,"\t\t", token.shape_, "\t\t",token.is_stop)
```

![](../../../.gitbook/assets/image%20%281084%29.png)

* `is_punct` : 문장부호에 해당하는 지
* `is_space` : 공백에 해당하는지
* `shape` : 글자수를 특정 문자의 연속체로 표현하며 각 특정 문자는 다음과 같은 의미가 있다.
  * x : 소문자
  * X : 대문자
  * d : 숫자
  * , : 기호



### 빈칸완성 과제 1

```python
def is_token_allowed(token):
# stopword와 punctutation을 제거해주세요.

  #if문을 작성해주세요.
  
  ##TODO#
  if token.is_stop or token.is_punct:
  ##TODO##
    return False
  return True

def preprocess_token(token):
  #lemmatization을 실행하는 부분입니다. 
  return token.lemma_.strip().lower()

filtered_tokens = [preprocess_token(token) for token in text if is_token_allowed(token)]
answer=['film', 'development','begin', 'marvel','studios', 'receive','loan', 'merrill','lynch', 'april','2005', 'success','film', 'iron','man', '2008','marvel','announce', 'avengers','release', 'july','2011', 'bring','tony', 'stark','steve', 'rogers','bruce', 'banner','thor', 'marvel','previous', 'film','signing', 'johansson','natasha','romanoff','march','2009','film','push','2012','release','whedon','bring','board','april','2010','rewrote','original','screenplay','zak','penn','production','begin','april','2011','albuquerque','new','mexico','move','cleveland','ohio','august','new','york','city','september','film','2,200','visual','effect','shot']
assert filtered_tokens == answer
```



## 2. 한국어 전처리

### 2.1 Mecab을 이용한 형태소 분석 기반 토크나이징

```text
!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
%cd Mecab-ko-for-Google-Colab
!bash install_mecab-ko_on_colab190912.sh
```

* `Mecab`은 Tokenizer 도구 중 하나이며 오픈 소스 형태소 분석 엔진이다. 이 중 Mecab-ko-dic은 한국어 형태소 분석을 하기 위한 프로젝트이다.

```python
from konlpy.tag import Mecab
import operator
tokenizer = Mecab()
```

* `Mecab`은 `konlpy` 패키지와 `eunjoen` 패키지에 존재한다. 여기서는 `konlpy` 패키지를 사용했다.
* `operator` 는 파이썬의 내장 연산자에 해당하는 효율적인 함수 집합을 반환한다.
  * `operator.add(x, y)`는 x+y와 동등한 표현이다.
  * 여기서는 `operator.itemgetter` 를 사용하기 위해 임포트했다.
    * 이는 주로 sorted와 같은 함수의 key 매개변수에 적용되어 다중 수준의 정렬을 가능하게 해주는 모듈이다.
    * 다음 예시로 살펴보자. [예시 참고 링크](https://wikidocs.net/109327)

```python
from operator import itemgetter

students = [
    ("jane", 22, 'A'),
    ("dave", 32, 'B'),
    ("sally", 17, 'B'),
]

result = sorted(students, key=itemgetter(1))
print(result)
```

```text
[('sally', 17, 'B'), ('jane', 22, 'A'), ('dave', 32, 'B')]
```

* 나이 순서대로 sort된 모습. `itemgetter(2)` 로 사용하면 성적순으로 정렬될 것이다. 대신에 `lambda x : x[1]` 과 같이도 사용할 수 있다.
* 해당 item이 클래스의 객체일 경우는 `attrgetter` 를 사용해야 한다.



### 2.2 음절 단위 토크나이징 실습

내용 없음



### 빈칸완성 과제 2

```python
vocab_dict={}
for token in tokens:
  ##TODO##
  '''
  vocab_dict에 token을 key로 빈도수를 value로 채워넣으세요.
  예시) vocab_dict={"나":3,"그":5,"어":3,...}
  '''
  if token not in vocab_dict:
      vocab_dict[token] = 0
  vocab_dict[token] += 1
  ##TODO##
  
  sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1),reverse=True)
  
vocab=[]
for token,freq in sorted_vocab:
  ##TODO##
  '''
  정렬된 sorted_vocab에서 빈도수가 2 이상인 token을 vocab에 append하여 vocab을 완성시키세요.
  '''
  if freq < 2:
      continue
  vocab.append(token)
  ##TODO##

answer=[' ','이',',','나','에','다','니','별','는', '하', '.', '아', '름', '을', '의', '과', '가', '은', '어', '지', '들', '리', '무', '머', '도', '까', '닭', '내', '러', '계', '습', '듯', '새', '랑', '시', "'", '멀', '그', '고', '위', '자', '로', '있', '속', '둘', '겨', '오', '요', '밤', '입', '사', '쓸', '경', '님', '운', '한', '마', '디', '불', '봅', '소', '런', '벌', '써', '기', '노', '스', '라', '너', '인', '워', '언', '덕']
```

