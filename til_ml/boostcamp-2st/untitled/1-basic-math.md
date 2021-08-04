---
description: '210804'
---

# \[필수 과제 1\] Basic Math

### 사전 세팅

```python
import numpy as np
```

* 수학 연산을 쉽게 하기위해 `numpy` 를 사용했다.



### Get\_greatest

```python
def get_greatest(number_list):
    greatest_number = max(number_list)
    return greatest_number
```

* `max` 를 사용해서 최댓값 구하기



### Get\_smallest

```python
def get_smallest(number_list):
    smallest_number = min(number_list)
    return smallest_numbe
```

* `min` 를 사용해서 최솟값 구하기



### Get\_mean

```python
def get_mean(number_list):
    mean = np.mean(number_list)
    return mean
```

* `np.mean` 을 사용해서 평균 구하기



### Get\_median

```python
def get_median(number_list):
    median = np.median(number-List)
    return median
```

* `np.median` 을 사용해서 중앙값 구하기
  * `np.median` 은 개수가 홀수이면 가운데 값을, 짝수이면 가운데에 있는 두 원소의 2개의 값의 평균을 출력한다.

