---
description: '210817'
---

# \(02강\) PyTorch Basics

PyTorch는 Numpy를 기반으로 만들었기 때문에 조금은 비슷한 면이 있다.

## PyTorch Operations

#### Tensor

* Numpy에서는 Ndarray를 사용했고 토치에서는 Tensor를 사용한다\(Tf도 마찬가지\)

```python
import numpy as np
n_array = np.arange(10).reshape(2,5)
print(n_array)
print("ndim :"
, n_array.ndim, "shape :", n_array.shape)
```

```text
[[0 1 2 3 4]
 [5 6 7 8 9]]
ndim : 2 shape : (2, 5)
```

```python
import torch
t_array = torch.FloatTensor(n_array)
print(t_array)
print("ndim :", t_array.ndim, "shape :", t_array.shape)
```

```text
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]])
ndim : 2 shape : torch.Size([2, 5])
```

기본적으로, 넘파이의 데이터타입과 텐서의 데이터타입은 동일하다. 한가지 차이점은 텐서에서는 GPU에서 사용할지에 대한 여부까지 고려할 수 있다.



tensor는 device 라는 property를 가지고 있는데 해당 텐서가 위치한 메모리 공간이 CPU인지 GPU인지를 알 수 있다.

```python
print(x_data.device)
if torch.cuda.is_available():
    x_data_cuda = x_data.to('cuda')
print(x_data_cuda.device)
```

```text
device(type='cpu')
device(type='cuda', index=0)
```



다음과 같이 임의의 텐서가 있다고 하자

```python
tensor_ex = torch.rand(size=(2, 3, 2))
tensor_ex

tensor([[[0.2709, 0.4392],
         [0.4948, 0.9187],
         [0.8725, 0.0842]],

        [[0.4384, 0.5261],
         [0.9063, 0.6005],
         [0.0318, 0.1725]]])
```

이 때 `view` 나 `reshape` 를 사용하면 동일한 결과를 얻게 된다.

* 둘 다 차원을 변화하는 함수이다.

```python
tensor_ex.view([-1, 6])

tensor([[0.2709, 0.4392, 0.4948, 0.9187, 0.8725, 0.0842],
        [0.4384, 0.5261, 0.9063, 0.6005, 0.0318, 0.1725]])
        
tensor_ex.reshape([-1,6])

tensor([[0.2709, 0.4392, 0.4948, 0.9187, 0.8725, 0.0842],
        [0.4384, 0.5261, 0.9063, 0.6005, 0.0318, 0.1725]])
```

그렇다면 둘의 차이는 무엇일까? 바로 얕은 복사와 깊은 복사의 차이다. 결론부터 말하면 `view` 는 얕은 복사이고 `reshape` 는 깊은 복사이다.

```python
a = torch.zeros(3, 2)
b = a.view(2, 3)
a.fill_(1)

tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
        
a

tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
        
b
        
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

`view` 의 경우 a의 원소를 1로 채웠는데 b의 값도 같이 바뀌는 모습

```python
a = torch.zeros(3, 2)
b = a.t().reshape(6)
a.fill_(1)

tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])

a

tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
        
b

tensor([0., 0., 0., 0., 0., 0.])
```

`reshape` 의 경우 a의 원소를 1로 채웠지만 b의 값은 0으로 유지되는 모습



squeeze와 unsqueeze로 차원을 줄였다 늘렸다 할 수 있다.

```python
tensor_ex = torch.rand(size=(2, 1, 2))
tensor_ex.squeeze()

tensor([[0.2391, 0.2666],
        [0.1566, 0.9649]])
        
tensor_ex = torch.rand(size=(2, 2))
tensor_ex.unsqueeze(0).shape

torch.Size([1, 2, 2])

tensor_ex.unsqueeze(1).shape

torch.Size([2, 1, 2])

tensor_ex.unsqueeze(2).shape

torch.Size([2, 2, 1])
```



행렬에서는 기본적인 사친연산이 가능하다. 그러나 이 때 꼭 피연산자의 차원이 같아야 한다.

또한, 행렬에서는 1D Array에 한해서만 `dot()` 함수를 사용할 수 있다. 행렬에 대한 연산에 대해서는 `mm()` 을 사용해야한다.

* `mm()` 은 벡터 연산을 지원하지 않는다.
* `mm()` 과 `matmul()` 은 동일한 기능을 가지나 가장 큰 차이점은 `mm()` 은 브로드캐스팅을 지원하지 않는다는 점이다. 브로드캐스팅을 지원하면 편리할 수는 있겠지만 예상치 못한 디버깅포인트를 만들 수 있다.



nn.functional 모듈을 통해 다양한 수식 변환을 지원한다

* softmax
* argmax, argmin
* one\_hot
* cartesian\_prod
  * itertools의 product와 같은 기능을 한다

또, 토치는 자동미분, Autograd를 제공한다.

```python
w = torch.tensor(2.0, requires_grad=True)
y = w**2
z = 10*y + 50
z.backward()
w.grad

tensor(40.)
```

편미분 하는 방법

```python
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

a.grad 

tensor([36., 81.])

b.grad

tensor([-12.,  -8.])
```



















