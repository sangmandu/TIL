---
description: '210818'
---

# \(04강\) AutoGrad & Optimizer

논문을 구현할 때에는 블락 또는 레이어라고 불리는 형태가 반복되어있는 구조를 구현하게 된다. 이 때는 `torch.nn.Module` 를 구현해야 한다. 이는 딥러닝을 구성하는 Layer의 기본이 되는 class 이다. 여기서 `Input` , `Output` , `Forward` ,`Backward` 를 정의한다. 또, 학습의 대상이 되는 parameter 도 정의하는데 이것은 `torch.nn.Parameter` 클래스에서 정의한다.

`nn.Parameter` 는 Tenser 객체를 상속받은 객체기 때문에 Tensor와 매우 비슷한데 차이점은 `nn.Module` 내에 attribute가 될 때 `required_grad = True` 로 지정되어서 학습의 대상이 된다. 그래서 우리가 이 파라미터를 직접 지정할 일이 없게된다.

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(
                torch.randn(in_features, out_features))
        
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x : Tensor):
        return x @ self.weights + self.bias
```

만약에 `nn.Parameter` 대신 `nn.Tensor`로 선언하면 결과는 동일하지만 `MyLinear.parameters()` 로 파라미터를 출력시 텐서는 출력되지 않는다. 왜냐하면 이 값들은 미분될 수 있는 값들만 출력되기 때문이다.

* 그래서 텐서로 선언할 일은 없다.



