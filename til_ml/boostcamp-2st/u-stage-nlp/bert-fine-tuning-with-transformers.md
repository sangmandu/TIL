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

### Split

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

### Tokenizer

```python
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
```

* `DistilBertToeknizerFast`
  * DistilBert의 공식 문서는 [여기](https://huggingface.co/transformers/model_doc/distilbert.html)
  * distil은 증류라는 뜻으로 액체 상태의 혼합물을 분리하는 방법이다. 기존의 BERT보다 가볍고, 저렴하면서 빠른 버전으로 고안되었기 떄문에 BERT에서 무거운 혼합물들을 제거했다는 뜻에서 DistilBert 로 이름을 붙인것으로 생각된다.
    * BERT는 Bidirectional Encoder Representations from Trasnformers의 약어로 두 개의 문장을 입력받은 후에 이 문장이 이어지는 문장인지 아닌지를 맞추는 방식으로 훈련되는 모델이다. 그래서 50:50 비율로 실제 이어지는 두 개의 문장과 랜덤으로 이어붙인 두 개의 문장이 훈련 데이터로 제공된다.
  * Tokenizer의 공식 문서는 [여기](https://huggingface.co/transformers/main_classes/tokenizer.html)
  * 모든 모델에 대해서 가능한 토크나이저를 가지고 있으며, 속도가 빠름을 강점으로 내세우고 있는 토크나이저이다.
  * 기본 버전과 -Fast 버전이 있으며 기본적으로 작동 방식은 동일하나 후자는 토큰과 원래 문자간에 매핑을 좀 더 발전된 방법으로 한다고 한다.\(?\)
    * 작동 방식이 엄청 동일한 것은 또 아니라고 한다.
    * [여기](https://discuss.huggingface.co/t/difference-betweeen-distilberttokenizerfast-and-distilberttokenizer/5961/2)



```python
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
```

* 각각의 인코딩을 토크나이징 한다.
* `truncation=True` : 음절이나 어절 단위로 자르는 것이 아니라 학습된 token들을 기준으로 자른다. 원래의 입력 문장과 길이와 관련이 없다.
* `padding=True` : max\_length볻 작은 길이의 sequence들은 0으로 부족한 길이기ㅏ 채워진다. 



### Dataset

```python
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)
```

* 데이터셋 클래스를 정의하고 인스턴스를 선언한다.

### Train

```python
config = DistilBertConfig.from_pretrained('distilbert-base-uncased',vocab_size=30522, \
                                          max_position_embeddings=512, sinusoidal_pos_embds=False, \
                                          n_layers=6, n_heads=12, dim=768, hidden_dim=3072, \
                                          dropout=0.1, attention_dropout=0.1, activation='gelu')
```

* 하이퍼 파라미터를 설정한다.



```python
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",config=config)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
```

* `training_args` : train을 위한 argumnet를 설정한다. 
  * `warmup_steps` : 학습 초기에 `convergence problem` 을 해결하기 위한 다양한 시도들이 있었고 지금까지 각광받는 방법은 `warmup heuristic` 이다. 말 그대로 학습 초기에 warm up이 필요하다는 것이며 자세히는, 초기 step에는아주 조금씩만 증가하는 learning rate를 사용해야 한다는 것인데 이 step을 정의해준다.
  * `weight_decay` : 특정 가중치값이 오버피팅 되는 것을 막기위해 적용하는 regularization 기법이다. 
* `model` : `pretrained` 된 `DistilBertForSequenceClassiffication` 모델을 사용한다.

### Inference

```python
from datasets import load_metric
from torch.utils.data import DataLoader
from tqdm import tqdm

metric= load_metric("accuracy")
test_dataloader = DataLoader(test_dataset, batch_size=128)
model.eval()
for batch in tqdm(test_dataloader):
    batch = {k: v.to("cuda") for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
```

* metric은 정확도로 설정하며 test 데이터의 정확도를 계산한다.
* 초기에는 다음과 같이 0.927이 나온다.

![](../../../.gitbook/assets/image%20%281069%29.png)



제시된 과제는 0.92 이상의 정확도를 가지는 것이었는데, 애초부터 잘 가지고 있네 ^^? 좀 더 높일 수 있는 방법을 찾아보자.



## Try to make performance better

현재 IMDb 데이터셋에 대한 모델의 성능이 다음과 같다.

![](../../../.gitbook/assets/image%20%281089%29.png)

* 현재 우리는 2014년에 나온 모델의 정확도 92.58 보다 높은 92.7이다. 현재 최고 성능이 96.8이고 대부분의 모델이 95에서 97 사이에 있으니 95 이상으로 점수를 높이면 잘 높인 것으로 생각할 수 있다.

이 중 4위에 있는 BERT의 논문 이름이 `How to FIne-Tune BERT for Text Classifcation?` 이다. 지금 내가 해야할 질문이랄까? 바로 읽어보자.

![](../../../.gitbook/assets/image%20%281079%29.png)



본문에서는 3가지 방법을 제시한다.

![](../../../.gitbook/assets/image%20%281083%29.png)

각각 살펴보자.

### Fine-Tuning Strategies

![](../../../.gitbook/assets/image%20%281078%29.png)

그만 알아보자...







