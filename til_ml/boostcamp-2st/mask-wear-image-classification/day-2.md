---
description: '210824'
---

# DAY 2 : Labeling

기존 데이터는 다음과 같다

```python
data
```

|  | id | gender | race | age | path |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 000001 | female | Asian | 45 | 000001\_female\_Asian\_45 |
| 1 | 000002 | female | Asian | 52 | 000002\_female\_Asian\_52 |
| 2 | 000004 | male | Asian | 54 | 000004\_male\_Asian\_54 |
| 3 | 000005 | female | Asian | 58 | 000005\_female\_Asian\_58 |
| 4 | 000006 | female | Asian | 59 | 000006\_female\_Asian\_59 |
| ... | ... | ... | ... | ... | ... |
| 2695 | 006954 | male | Asian | 19 | 006954\_male\_Asian\_19 |
| 2696 | 006955 | male | Asian | 19 | 006955\_male\_Asian\_19 |
| 2697 | 006956 | male | Asian | 19 | 006956\_male\_Asian\_19 |
| 2698 | 006957 | male | Asian | 20 | 006957\_male\_Asian\_20 |
| 2699 | 006959 | male | Asian | 19 | 006959\_male\_Asian\_19 |

2700 rows × 5 columns



위처럼, 현재 데이터프레임은 id와 gender, race, age 그리고 path라는 컬럼으로 이루어진 테이블로 되어있다. 라벨링을 해야하는 두 가지 이유가 있다.

> 1. 현재는 한 사람의 7장 사진이 있는 폴더를 기준으로 데이터프레임이 구성되어있다. 추후에 이미지 접근을 사진 각각에 하기 위해서 데이터프레임을 확장해야한다. 이 때 각각의 이미지 주소를 가지고 있는 컬럼을 추가한다.
>
> 2. 현재는 직접적으로 클래스를 나타내지 않으므로 모델에서 분류하기에 가능은 하나 불편함이 있다. 또한 GPU 효율을 최대화하기 위해 이런 작업은 CPU에서 최대한 해주는 것이 좋다. 나이와 성별 그리고 마스크 착용 여부를 토대로 라벨을 추가해야한다. 이 때 마스크 착용 여부는 이미지의 이름으로 판단한다.



각 클래스는 다음과 같은 특징이있다.

* 마스크 정상 착용 : +0 \| 마스크 비정상 착용 : +6 \| 마스크 미착용 : +12
* 남성 : +0 \| 여성 : +3
* 30세 미만 : +0 \| 30세 이상 60세 미만 : +1 \| 60세 이상 : +2

![](../../../.gitbook/assets/image%20%28993%29.png)

따라서, 조건문으로 구별하기 보다는 각 속성들을 수식화하면 쉽게 라벨링 할 수 있다.

* 마스크
  * 파일명에 'Incorrect'가 포함되면 +6
  * 파일명에 'Normal'이 포함되면 +12
* 성별
  * 남성과 여성의 차이가 3만큼 나야한다. 문자열로만 비교할 수 있는 점은 길이가 다르다는 것. 이를 이용한다. 남성은 4글자, 여성은 6글자이다
  * 현재 둘의 차이는 2글자이므로 이것이 3만큼 차이나려면 1.5배만큼 곱해야한다.
* 나이
  * 간격이 30만큼 있으므로 30으로 나눈 몫만큼을 할당한다

```python
data2 = []
def new_dataframe(x):
    id, gender, race, age = x.split('_')
    for filename in FILES:
        path = os.path.join(DATA_DIR, x, filename)
        path = glob(path)[0]
        label = (int(age) // 30) + (len(gender) * 1.5 - 6)
        if 'incorrect' in filename:
            label += 6
        elif 'normal' in filename:
            label += 12
        data2.append([gender, age, path, int(label)])

data['path'].apply(new_dataframe)
data2 = pd.DataFrame(data=data2, columns=['gender', 'age', 'path', 'label'])
data2
```

|  | gender | age | path | label |
| :--- | :--- | :--- | :--- | :--- |
| 0 | female | 45 | ./input/data/train/images/000001\_female\_Asian\_... | 4 |
| 1 | female | 45 | ./input/data/train/images/000001\_female\_Asian\_... | 4 |
| 2 | female | 45 | ./input/data/train/images/000001\_female\_Asian\_... | 4 |
| 3 | female | 45 | ./input/data/train/images/000001\_female\_Asian\_... | 4 |
| 4 | female | 45 | ./input/data/train/images/000001\_female\_Asian\_... | 4 |
| ... | ... | ... | ... | ... |
| 18895 | male | 19 | ./input/data/train/images/006959\_male\_Asian\_19... | 0 |
| 18896 | male | 19 | ./input/data/train/images/006959\_male\_Asian\_19... | 0 |
| 18897 | male | 19 | ./input/data/train/images/006959\_male\_Asian\_19... | 0 |
| 18898 | male | 19 | ./input/data/train/images/006959\_male\_Asian\_19... | 6 |
| 18899 | male | 19 | ./input/data/train/images/006959\_male\_Asian\_19... | 12 |

18900 rows × 4 columns



이후, 매번 데이터프레임을 만들고 불러오는 작업을 줄이기 위해 새롭게 csv 파일로 저장하고 이후에 불러올 수 있도록 한다.

```python
data2.to_csv("train_data.csv", mode='w', index=False)
```

* `mode` 를 `w` 로 설정하면 덮어쓰기가 되며 이어서 수정하려면 `a` 로 설정하면 된다.
* `index=False` 를 하지않으면 이후에 다시 불러올 때 `index` 가 두 개의 컬럼으로 존재하게 된다. csv파일로 저장될 때는 자체에 default로 index가 있기 때문이다.





