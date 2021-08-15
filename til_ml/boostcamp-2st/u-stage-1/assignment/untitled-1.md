---
description: '210808'
---

# \[선택 과제 2\] Backpropagation

### TODO 1

$$\frac{\partial \xi}{\partial W_x} =\frac{\partial \xi}{\partial S_n}\frac{\partial S_n}{\partial W_x} + \frac{\partial \xi}{\partial S_n}\frac{\partial S_n}{\partial S_{n-1}}\frac{\partial S_{n-1}}{\partial W_x} + \cdots = \Sigma\frac{\partial \xi}{\partial S_n}X$$

$$\frac{\partial \xi}{\partial W_{rec}} = \frac{\partial \xi}{\partial S_n}\frac{\partial S_n}{\partial W_{rec}} + \frac{\partial \xi}{\partial S_n}\frac{\partial S_n}{\partial S_{n-1}}\frac{\partial S_{n-1}}{\partial W_{rec}} + \cdots = \Sigma\frac{\partial \xi}{\partial S_n}S_{n-1}$$

* 한 타임라인의 가중치는 해당 타임라인 뿐만 아니라 그전까지의 타임라인의 셀들에게 영향을 받았으므로 해당 타임라인까지의 셀들의 대한 가중치 변화율을 모두 더해줘야 한다.
* Sn은 Wx로 미분하면 X 가 남는다
* Sn은 Wrec로 미분하면 Sn-1 이 남는다



### TODO 2

#### Backward\_gradient

```python
def backward_gradient(X, S, grad_out, wRec):
    """
    X: input
    S: 모든 input 시퀀스에 대한 상태를 담고 있는 행렬
    grad_out: output의 gradient
    wRec: 재귀적으로 사용되는 학습 파라미터
    """
    # grad_over_time: loss의 state 에 대한 gradient 
    # 초기화
    grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
    grad_over_time[:,-1] = grad_out
    # gradient accumulations 초기화
    wx_grad = 0
    wRec_grad = 0

    for k in range(X.shape[1], 0, -1):
        wx_grad += np.dot(grad_over_time[:,k], X[:,k-1])
        wRec_grad += np.dot(grad_over_time[:,k], S[:,k-1])
        grad_over_time[:,k-1] = grad_over_time[:,k] * wRec

    return (wx_grad/n_samples, wRec_grad/n_samples), grad_over_time
```

* 16 : RNN은 sequence data를 다룬다. weight gradient backpropagation은 현재의 timeline 상태와 이후의 timeline들의 상태가 모두 고려된다. 따라서 반복문으로 이를 해결할 것임
* 17, 18 : grad\_over\_time은 제일 마지막 셀 Sn에서 나온 에러에 대한 미분값이다. 따라서 Sn-1 에서의 미분값은 에러에 대한 Sn의 미분값 \* Xn-1 이다. wRec도 마찬가지이다.
  * 여기서 Wx와 Wrec는 모두 상수이다. 여기서는 RNN이 이 두 파라미터를 각 상태마다 동일하게 사용하기 때문에 따로 행렬화 하지 않아도 된다.
  * np.dot를 이용해서 곱해준다. 해설에서는 그냥 곱한값을 sum해줬다.
    * 그리고 mean도 해줬는데, 나는 제일 마지막에 return 시에 할 것이다.

![](../../../../.gitbook/assets/image%20%28798%29.png)

* 20 : 이전 셀 Sn-1에서의 에러에 대한 미분값은 Sn에서의 에러에 대한 미분값 \* Wrec 이다.

![](../../../../.gitbook/assets/image%20%28799%29.png)

