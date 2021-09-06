---
description: '210907'
---

# \[선택 과제\] BERT Fine-tuning with transformers

### 데이터셋 다운 및 라이브러리

```text
# !wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz
```

```python
import torch
from transformers import DistilBertTokenizerFast
from transformers import DistilBertConfig
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from sklearn.model_selection import train_test_split
```

* 우리가 사용할 데이터는 Imdb 라는 데이터셋이다. IMDb는 Internet Movdi Database의 준말로 영화, 배우, 드라마, 비디오 게임 등에 관한 정보를 제공하는 온라인 데이터베이스이다. 2014년 8월 1일을 기준으로 영화 약 3백만건, 인물 정보 약 6백만건을 소유하고 있다. 
* 스탠포드 대학교에서 2011년에 낸 [논문](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf)에서 이 데이터를 소개하였고 논문에서는 훈련 데이터와 테스트 데이터를 50대50비율로 분할하여 88.89%의 정확도를 얻었다고 소개했다.
* 흔히 영화에 대한 리뷰 데이터를 통해 감성을 분류하는 목적으로 많이 사용하는 데이터이다.
* 이에 대한 [Text Classification 벤치마킹](https://paperswithcode.com/sota/text-classification-on-imdb)도 이루어진다.

```python
def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels

train_texts, train_labels = read_imdb_split('aclImdb/train')
test_texts, test_labels = read_imdb_split('aclImdb/test')
```

* `read_imdb_split` 
  * 경로를 인자로 입력받는다. `pathlib` 는 파이썬 3.4부터 추가된 내장함수로 이전에는 `os` 모듈을 사용했고 이후에는 파일위치 찾기, 파일 입출력과 같은 동작을 한다. `os` 모듈과는 다음과 같은 차이가 있다.
    * `os` 와 달리 `pathlib` 는 파일시스템 경로를 단순한 문자열이 아니라 객체로 다룬다.
    * 이렇게 되면서 `/` 라는 계층 구분 문자를 경로 구분 문자로 사용하게 되었다. 즉, 연산자를 새롭게 정의할 수 있게 되었다는 이점이 생겼다.
    * 무슨 말이냐면, 이전에는 dir1과 dir2와 dir3를 연결하려면 `os.path.join(dir1, dir2, dir3)` 와 같이 작성했어야 하는데 `path` 를 사용하면 `dir1 / dir2 / dir3` 와 같이 간단하게 연산자로 표현할 수 있게된다.
    * `pathlib` 의 `Path` 는 주어진 경로를 객체화한다.
    * `os` 모듈을 사용할 때는 `os.listdir` 또는 `glob` 를 이용해서 현재 디렉토리에 있는 파일들을 리스팅했는데, `pathlib` 에서는 `iterdir` 을 사용할 수 있다. 이 때 리스팅된 원소들도 모두 `pathlib` 객체이다.
    * 또, 무엇이 달라졌냐면 기존의 입출력과 달리 `pathlib` 의 입출력은 번거롭게 파일을 열고 닫을 필요가 없다. 파일을 열을 때는 `read_text()` 를, 쓸 때는 `write_text()` 를 사용한다.
    * [자세히 알아보기](https://brownbears.tistory.com/415) / [자세히 알아보기2](https://ryanking13.github.io/2018/05/22/pathlib.html)
  * 주어진 경로에 있는 파일들을 읽어 라벨링 하는 작업을 거쳐 texts와 labels로 반환한다.
    * 사실 8번 라인은 매우 비효율적인데, 모든 반복문마다 if문을 실행하기 때문이다. 첫번째 `label_dir` 을 결정하는 반복문을 `enumerate` 로 작성해서 인덱스를 라벨값으로 주는 것이 효율적이다.

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
```

* train dataset을 8:2 비율로 train/valid 로 나누어준다.

```python
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
```















