---
description: '210806'
---

# \(Python 7-1강\) pandas I

### Pandas

구조화된 데이터의 처리를 지원하는 대표적인 라이브러리

* panel data : pandas
* 고성능array 계산 라이브러리인 numpy와 통합하여, 강력한“스프레드시트” 처리기능을 제공
* 인덱싱, 연산용함수, 전처리함수등을제공함
* 데이터 처리 및 통계분석을위해사용

데이터 용어

* 전체 데이터 : Data table, Sample
* 세로줄 : attribute, field, feature, column
* 가로줄 : instance, tuple, row
* 하나의 세로 줄 : Featrue vector
* 하나의 원소 : data



### 데이터 로딩

```python
import pandas as pd

data url = "주소"
df_data = pd.read_csv(data_url, sep="\s+', header=None)
```

* `pd.read_csv`
  * url에 있는 csv 데이터를 불러온다
  * `sep` : 구분자를 의미. 여기서는 공백문자를 구분자로 선택
  * `header` : 명시적으로 None을 쓰며 이 때는 첫 행의 데이터가 열 이름이 된다.



### Pandas의 구성

#### Dataframe

Data Table 전체를 포함하는 Object이다.

#### Series

데이터프레임 중 하나의 컬럼에 해당하는 데이터의 모음 Object 이다.

![](../../../../.gitbook/assets/image%20%28784%29.png)



### Series

```python
import pandas as pd

# Series 생성하기
>>> example_obj = pd.Series(dict_data, dtype=np.float32, name="example_data")
>>> example_obj
a    1.0
b    2.0
c    3.0
d    4.0
e    5.0
Name: example_data, dtype: float32

# data index에 접근하기, 값 할당하기
>>> example_obj["a"]
1.0
>>> example_obj["a"] = 3.2
>>> example_obj["a"]

# 값, 인덱스 얻기
>>> example_obj.values
array([3.2, 2. , 3. , 4. , 5. ], dtype=float32)
>>> example_obj.index
Index(['a', 'b', 'c', 'd', 'e'], dtype='object')

# 값, 인덱스 naming
>>> example_obj.name = "number"
>>> example_obj.index.name = "alphabet"
>>> example_obj
alphabet
a    3.2
b    2.0
c    3.0
d    4.0
e    5.0
Name: number, dtype: float32
```



### Dataframe

```python
# 데이터프레임 생성
>>> raw_data =
		{'first_name':['Jason', 'Molly', 'Tina'],
		'last_name':['Miller', 'Jacobson', 'Ali'],
		'age':[42, 52, 36],
		'city':['San Francisco', 'Baltimore', 'Miami']
		}
		
>>> df = pd.DataFrame(raw_data)
>>> df
  first_name last_name  age           city
0      Jason    Miller   42  San Francisco
1      Molly  Jacobson   52      Baltimore
2       Tina       Ali   36          Miami
```



이후의 pandas 내용은 이전에 작성한 를 참고.







