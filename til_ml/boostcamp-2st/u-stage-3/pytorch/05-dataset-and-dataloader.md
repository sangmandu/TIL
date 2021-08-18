---
description: '210818'
---

# \(05강\) Dataset & Dataloader

### 모델에 데이터를 먹이는 방법

![](../../../../.gitbook/assets/image%20%28935%29.png)

* Data : 데이터를 수집하고 정리하고 전처리 한다
* Dataset : 이러한 데이터를 관리할 Dataset을 정의한다
  * `__init__()` : 시작할 때 데이터를 어떻게 불러올 지
  * `__len__()` : 데이터 셋의 크기 반환
  * `__getitem__()` : `map-style` 이라고도 하며 하나의 데이터를 불러올 때 어떻게 반환해줄지를 정의. 보통은 인덱스를 사용한다
* Transforms : 데이터를 변형시킨다.
  * Augmentation 할 때도 이 과정을 거친다
  * 텐서로 바꾸어주는 부분도 여기에 속함
* DataLoader : 모델에 들어갈 데이터를 최종적으로 정의
* Model : 데이터를 입력



### Dataset 클래스

* 데이터 입력 형태를 정의하는 클래스
* 데이터를 입력하는 방식의 표준화

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, text, labels):
            self.labels = labels
            self.data = text

    def __len__(self):
            return len(self.labels)

    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.data[idx]
            sample = {"Text": text, "Class": label}
            return sample
```

* 5 : 초기에 데이터 생성 방법을 지정한다
* 9 : 데이터의 전체 길이를 반환한다
* 12 : index값을 주었을 때 반환되는 데이터의 형태를 정의한다
  * 주로 index 값을 인자로 받지만 다른 값을 받을 수도 있다
  * 딕셔너리 형태로 반환했지만 다른 형태로도 반환할 수 있다.
  * getitem은 나중에 DataLoader에서 사용하게 된다



### Dataset 클래스 생성시 유의점

* 데이터 형태에 따라 각 함수를 다르게 정의한다
* 모든 것을 데이터 생성 시점에 처리할 필요는 없다
* 데이터 셋에 대한 표준화된 처리방법을 제공할 필요가 있다
* 최근에는 HuggingFace등의 표준화 라이브러리를 사용한다



### DataLoader 클래스

* Data의 Batch를 생성해주는 클래스
* 학습직전 데이터의 변환을 책임
* Tensor로 변환 + Bacth 처리가 메인 업무
* 병렬적인 데이터 전처리 코드의 고민 필요





