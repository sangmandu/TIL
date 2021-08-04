---
description: '210804'
---

# \(Python 6강\) numpy

## Numpy

* Numerical Python
* 파이썬의 고성능 과학 계산용 패키지
* Matrix와 Vector와 같은 Array연산의 표준
* 한글로 넘파이로 주로 통칭

특징

* 일반 List에 비해 빠르고 메모리 효율적이다
* 반복문 없이 데이터 배열에 대한 처리를 지원한다
* 선형대수와 관련된 다양한 기능을 제공한다
* C, C++, 포트란 등의 언어와 통합 가능

### ndarray

```python
import numpy as np
```

* numpy의 호출 방법
* 일반적으로 np라는 별칭 이용

```python
test_array = np.array([1, 4, 5, 8], float)
print(test_array)
type(test_array[3])
```

* numpy는 np.array 함수를 활용해서 배열을 생성하는데 이 배열을 ndarray 라고 한다
* numpy는 하나의 데이터 타입만 배열에 넣을 수 있다.
  * 리스트와의 차이점
  * dynamic typing을 지원하지 않는다고 한다
* C의 Array를 사용하여 배열을 생성한다 

### array creation

* 파이썬은 임의의 위치에 저장되는데 비해 C언어는 순서대로 저장된다.
  * c언어의 지역성
* 또, 크기가 고정되어있다.
* 그래서, 속도가 빠른것

```python
test_array = np.array([1, 4, 5, "8"], float) # String Type의 데이터를 입력해도

print(test_array)
array([ 1., 4., 5., 8.])

print(type(test_array[3])) # Float Type으로 자동 형변환을 실시
numpy.float64

print(test_array.dtype) # Array(배열) 전체의 데이터 Type을 반환함
dtype('float64')

print(test_array.shape)
(4,)
```

* shape : ndarr의 dimension 구성을 반환
  * array의 크기, 형태에 대한 정보
* dtype : ndarrr의 type을 반환

```python
test_array = np.array([1, 4, 5, "8"], float) # String Type의 데이터를 입력해도

test_array.ndim
1

test_array.size
4

test_array.nbytes
16
```

* ndim : number of dimensions
* size : data 의 개수
* nbytes : ndarray object의 메모리 크기를 반환함
  * int는 1byte, float은 4 bytes
  * 파이썬에서 float은 8bytes가 기본이다. 위는 넘파이 기준



## Handling shape

### reshape : Array의 shape의 크기를 변경함. element의 갯수는 동일

```python
test_matrix = [[1,2,3,4], [5,6,7,8]]
np.array(test_matrix).shape
>>> (2, 4)

np.array(test_matrix).reshape(8, )
>>> array([1,2,3,4,5,6,7,8])

np.array(test_matrix).reshape(-1, 2)
>>> array([[1, 2], [3, 4], [5, 6], [7, 8]]
```

* `-1`  은 알아서 컴퓨터가 계산할 수 있는 부분을 의미한다



### flatten : 다차원 array를 1차원 array로 변환

* \(2, 2, 4\) =&gt; \(16, \)



### indexing for numpy array

* list와 달리 이차원 배열에서 \[0, 0\] 표기법을 제공한다
  * a\[0, 0\] == a\[0\]\[0\]
  * 둘 다 가능하다
* 또, list와 달리 행과 열 부분을 나눠서 slicing이 가능하다

```python
a = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], int)

a[:, 2:]
>>> array([[3, 4, 5], [8, 9, 10]])

a[1, 1:3]
>>> array([7, 8])
```



## Create Functions

### arange

* array의 범위를 지정하여 값의 list를 생성하는 명령어

```python
np.arrange(30)
>>> array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

#시작, 끝, step => 이 때 step은 정수형일 필요는 없음
np.arange(0, 5, 0.5)
>>> array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5])

np.arange(30).reshape(5, 6)
>>> array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [24, 25, 26, 27, 28, 29]])
```

### zeros

* 0으로 가득찬 ndarr 생성

```python
>>> np.zeros((2, 5), int)
array([[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]])
       
>>> np. zeros((3, ), float)
array([0., 0., 0.])
```

### ones

* 1로 가득찬 ndarr 생성

```python
>>> np.ones((1, 1, 1), int)
array([[[1]]])
>>> np.ones((2, 3, 2, 3), int)
array([[[[1, 1, 1],
         [1, 1, 1]],

        [[1, 1, 1],
         [1, 1, 1]],

        [[1, 1, 1],
         [1, 1, 1]]],


       [[[1, 1, 1],
         [1, 1, 1]],

        [[1, 1, 1],
         [1, 1, 1]],

        [[1, 1, 1],
         [1, 1, 1]]]])
```

### empty

* shape만 주어지고 비어있는 ndarr 생성
  * memory initialization이 된 것은 아니다
  * 이미 존재하는 값은 이전에 사용하던 쓰레기값이다

```python
>>> np.empty((3, 5))
array([[2.12199579e-314, 6.36598737e-314, 1.06099790e-313,
        1.48539705e-313, 1.90979621e-313],
       [2.33419537e-313, 2.75859453e-313, 3.18299369e-313,
        3.60739285e-313, 4.03179200e-313],
       [4.45619116e-313, 4.88059032e-313, 5.30498948e-313,
        5.72938864e-313, 6.15378780e-313]])
```

### somthing\_like

* 기존 ndarr의 shape 크기만큼 1 또는 0의 array 반환

```python
>>> test = np.arange(12).reshape(3, 4)
>>> test
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
       
>>> np.zeros_like(test)
array([[0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]])
>>> np.ones_like(test)
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
```

### identity

단위행렬 생성

```python
>>> np.identity(4, dtype=np.float32)
array([[1., 0., 0., 0.],
       [0., 1., 0., 0.],
       [0., 0., 1., 0.],
       [0., 0., 0., 1.]])
```

### eye

대각선이 1인 행렬 생성.

* identity와 다른점은 시작위치를 정할 수 있다
* `np.eye(3, 5, k=2)` 면 2만큼 이동된 3 \* 5 행렬 생성

```python
>>> np.eye(3, 5, k=2)
array([[0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```

### diag

대각 행렬의 값을 추출함

```python
>>> test
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.diag(test)
array([ 0,  5, 10])
>>> np.diag(test, k=2)
array([2, 7])
```

### random sampling

데이터 분포에 따른 sampling으로 array를 생성

```python
# 시작, 끝, 개수
>>> np.random.uniform(0, 1, 10)
array([0.46238198, 0.10681686, 0.0266183 , 0.60596315, 0.41614435,
       0.99164157, 0.4245724 , 0.87679971, 0.37257856, 0.31018757])

# 균등 분포
>>> np.random.uniform(0, 1, 10).reshape(2, 5)
array([[0.16017842, 0.70721719, 0.79159583, 0.95743024, 0.77892252],
       [0.38001109, 0.21918138, 0.59308826, 0.56590121, 0.31633467]])

# 정규 분포
>>> np.random.normal(0, 1, 10).reshape(2, 5)
array([[-0.15451055,  0.35729475,  0.07026103, -0.68009187, -0.68631985],
       [ 0.37181644, -0.92405456,  0.50774203,  0.87155016,  1.48159822]])
```



## Operation functions

### sum

element간의 합

### axis

모든 operation function을 실행할 때 기준이 되는 dimension 축이다.

![](../../../../.gitbook/assets/image%20%28735%29.png)

### mathematical functions

다양한 수학 연산자

* np.exp
* np.sqrt
* np.mean
* np.std

### Concatenate

vstack

* numpy array를 세로로 붙임

hstack

* numpy array를 가로로 붙임

concatenate

* axis = 0 : vstack과 동일
* axis = 1: hstack과 동일

newaxis

* 축을 하나 늘린다

```python
b = np.array([5, 6])
b = b[np.newaxis, :]
b
>>> array([[5, 6]])
```



### Opertaions b/t arrays

* 기본적으로 numpy array간의 기본적인 사칙 연산을 지원한다
  * 이 때 element-wise operation으로 연산된다

dot product

* 내적 함수
* `np.array.dot(np.array)` 꼴로 사용

transpose

* 전치 함수
* `np.array.T` 의 꼴로 사용

broadcasting

* shape이 다른 배열 간 연산을 지원하는 기능
* scalar - vector 와 vector - matrix 간에 지원한다

timeit

* jupyter 환경에서 코드의 퍼포먼스를 체크하는 함수
* 일반적으로 속도는 다음과 같다
  * numpy &gt; list comprehension &gt; for loop



## Comparisons

### All & Any

Array의 데이터 전부 또는 일부가 조건에 만족하는지에 대한 여부를 반환한다

```python
a = np.arange(10)
a
>>> array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.any(a>5), np.any(a<0)
>>> (True, False)

np.all(a>5), np.all(a<10)
>>> (False, True)
```



numpy는 배열의 크기가 동일한 element간 비교의 결과를 Boolean type으로 반환한다.

```python
test_a = np.array([1, 3, 0], float)
test_b = np.array([5, 2, 1], float)
test_a > test_b
>>> array([False, True, False], dtype=bool)
```



### where

```python
a = np.arange(10)
np.where(a>5)
>>> (array([6, 7, 8, 9], dtype=int64),)

a = np.array([1, np.NaN, np.Inf], float)
np.isnan(a)
>>> array([False,  True, False])

np.isfinite(a)
>>> array([ True, False, False])
```



### argmax & argmin

array내 최대값 또는 최소값의 index를 반환

또한, axis 기반의 반환을 할 수 있다

```python
a = np.arange(0, 20, 3)
a
>>> array([ 0,  3,  6,  9, 12, 15, 18])

np.argmax(a), np.argmin(a)
>>> (6, 0)

 a = np.array([[1,2,4,7],[9,88,6,45],[9,76,3,4]])
np.argmax(a, axis=1), np.argmax(a, axis=0)
>>> (array([3, 1, 1], dtype=int64), array([1, 1, 1, 1], dtype=int64))
np.argmin(a, axis=1), np.argmin(a, axis=0)
>>> (array([0, 2, 2], dtype=int64), array([0, 0, 2, 2], dtype=int64))
```



### boolean index

특정 조건에 따른 값을 배열 형태로 추출한다.

```python
arr > 3
>>> array([False,  True, False, False, False,  True,  True,  True])

arr[arr > 3]
>>> array([4., 8., 9., 7.])
```



### fancy index

numpy array를 index value로 사용해서 값을 추출한다. 이 때 인덱스로 사용되는 배열은 반드시 정수로 선언되어야 한다.

```python
a = np.array([2, 4, 6, 8], float)
b = np.array([0, 0, 1, 3, 2, 1], int)
a[b]
>>> array([2., 2., 4., 8., 6., 4.])
```

* matrix형태도 가능하다
  * a\[b\]\[c\]





