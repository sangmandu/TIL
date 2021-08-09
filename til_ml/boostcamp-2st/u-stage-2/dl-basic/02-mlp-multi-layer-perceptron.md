---
description: '210809'
---

# \(02강\) 뉴럴 네트워크 - MLP \(Multi-Layer Perceptron\)

## Neural Networks

위키피디아

* Neural networks are computing systems vaguely inspired by the  biological neural networks that constitute animal brains.”
  * 그러나 인간의 뇌를 따라했다고 꼭 성능이 좋은 건 아니다. 인간의 뇌에서는 역전파가 발생하지는 않는다.
  * 마치 우리가 날고싶다고 새처럼 날 필요도, 빨리 달리고 싶다고 해서 치타처럼 달릴 필요도 없다. 이미 딥러닝은 인간의 뇌와는 많이 갈라졌다.

정의

* Neural networks are function approximators that stack affine transformatiosn followed by nonlinear transformations.
  * 단순히, 반복적으로 곱셈하고 비선형함수를 적용하는 신경망을 의미



## Linear Neural Networks

Loss function을 w와 b로 편미분을 구해서 업데이트 하는 과정에서 Loss를 최소화하는 w와 b를 찾게 된다. 미분한 값에 학습률을 곱해서 업데이트 하게되는데 이 때, 학습률이 너무 커도 너무 작아도 안된다. 

차원이 클 때는 행렬을 사용하면 된다. 입력 벡터 X는 W와 b를 이용해 Y에 대한 차원 변환이 일어나게 된다.



## Beyond Linear Neural Networks

![](../../../../.gitbook/assets/image%20%28790%29.png)

선형 함수로 신경망을 깊게 쌓을 수 없는 이유는, 중첩된 선형함수는 하나의 선형함수로 표현이 가능하기 때문이다.

* 10층 짜리 레이어도 1층 짜리 레이어로 표현이 가능하다는 뜻
* f\(x\) = x+2, g\(x\) = 2x 라는 함수에 대해서, f\(g\(x\)\) 는 2층짜리 함수같지만 사실 h\(x\) = 2x+2 라는 1층 함수로 표현할 수 있다.

![](../../../../.gitbook/assets/image%20%28794%29.png)

* 표현력을 극대화하기 위해서는 선형결합이 등장하면, 비선형함수를 적용하도록 한다.

비선형 함수\(활성화 함수\)는 다음과 같은 것들이 있다.

![](../../../../.gitbook/assets/image%20%28786%29.png)

신경망은, 표현력이 매우 좋기 때문에 성능이 잘 나온다. 



## Multi-Layer Perceptron

![](../../../../.gitbook/assets/image%20%28788%29.png)

Regression Task

* 기본적인 회귀 문제에서 MSE를 사용

Classfication Task

* 분류문제는 각각의 클래스중에 값이 높게 나온 클래스를 선택한다.
  * 이 값은 100이든 10000이든 상관없이 다른 값 대비 높기만 하면 된다.
  * 이 부분을 수학적으로 표현하는 것이 어려워 CE를 사용하는 것
  * 

Probabilistic Task

* "이 사람은 확실히 30대야" 또는 "저 사람은 20대 같긴한데 확실하지는 않아" 등의 uncertain 정보를 찾을 때 사용한다.



## 실습

### Multilayer Perceptron \(MLP\)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
%matplotlib inline
%config InlineBackend.figure_format='retina'
print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))

PyTorch version:[1.9.0+cu102].
device:[cuda:0].
```

* 1-7 : torch구현에 필요한 라이브러리
* 8 : 시각화에 있어서 화질이 좋게 함

### Dataset

```python
from torchvision import datasets,transforms
mnist_train = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
print ("mnist_train:\n",mnist_train,"\n")
print ("mnist_test:\n",mnist_test,"\n")
print ("Done.")
```

* MNIST라는 데이터를 불러온다.
  * 이 데이터는 Modified National Institute of Standards and Technology database의 약어로 손으로 쓴 숫자들로 이루어진 대형 데이터베이스이다.
  * 6만개의 훈련 데이터와 1만개의 테스트 데이터로 구분된다.
* 인자에 대한 설명은 다음과 같다\([참고링크](https://m.blog.naver.com/hongjg3229/221560700128)\)
  * **root :** 데이터의 경로를 넣는다.
  * **train** : 테스트용 데이터를 가져올지 학습용 데이터를 가져올지 표시한다. True면 테스트용 데이터이다.
  * **transform :** 어떤 형태로 데이터를 불러올 것인가. 일반 이미지는 0-255사이의 값을 갖고, \(H, W, C\)의 형태를 갖는 반면 pytorch는 0-1사이의 값을 가지고 \(C, H, W\)의 형태를 갖는다. transform에 transforms.ToTensor\(\)를 넣어서 일반 이미지\(PIL image\)를 pytorch tensor로 변환한다.
  * **download :** True로 설정하면 MNIST 데이터가 없으면 다운을 받는다.

### Data Iterator

```python
BATCH_SIZE = 256
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
print ("Done.")
```

* 불러온 데이터셋 객체로 data\_loader 객체를 만든다. 인자에 대한 설명은 다음과 같다\([참고링크](https://m.blog.naver.com/hongjg3229/221560700128)\)
  * **dataset :** 어떤 데이터를 로드할 것인지
  * **batch\_size :** 배치 사이즈를 뭘로 할지
  * **shuffle :** 순서를 무작위로 할 것인지, 있는 순서대로 할 것인지
  * **drop\_last :** batch\_size로 자를 때 맨 마지막에 남는 데이터를 사용할 것인가 버릴 것인가

### Define the MLP model

```python
class MultiLayerPerceptronClass(nn.Module):
    """
        Multilayer Perceptron (MLP) Class
    """
    def __init__(self,name='mlp',xdim=784,hdim=256,ydim=10):
        super(MultiLayerPerceptronClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.lin_1 = nn.Linear(
            self.xdim, self.hdim
        )
        self.lin_2 = nn.Linear(
            self.hdim, self.ydim
        )
        self.init_param() # initialize parameters
        
    def init_param(self):
        nn.init.kaiming_normal_(self.lin_1.weight)
        nn.init.zeros_(self.lin_1.bias)
        nn.init.kaiming_normal_(self.lin_2.weight)
        nn.init.zeros_(self.lin_2.bias)

    def forward(self,x):
        net = x
        net = self.lin_1(net)
        net = F.relu(net)
        net = self.lin_2(net)
        return net

M = MultiLayerPerceptronClass(name='mlp',xdim=784,hdim=256,ydim=10).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(M.parameters(),lr=1e-3)
print ("Done.")
```

* 1 : neuralnetwork.Modeul을 상속받는다
* 8-10 : 입력, 히든, 출력 차원을 입력받는다
* 11 : 인풋에서 히든레이어 까지 가는 신경망을 구성한다
* 14 : 히든에서 아웃풋레이어 까지 가는 신경망을 구성한다
* 20 : 가중치 초기화에 있어서 He Initialization을 사용한다
* 21 : bias는 다 0으로 초기화
* 25-30 : 입력된 값은 Linear layer 1과 2를 거쳐서 반환된다. 이 때 이 사이에 Relu를 한번 거치게 된다.
* 33 : 크로스 엔트로피 함수로 비용함수를 지정한다
* 34 : 옵티마이저로 아담을 선택하고 학습률을 정의한다

### Simple Forward Path of the MLP Model

```python
x_numpy = np.random.rand(2,784)
x_torch = torch.from_numpy(x_numpy).float().to(device)
y_torch = M.forward(x_torch) # forward path
y_numpy = y_torch.detach().cpu().numpy() # torch tensor to numpy array
print ("x_numpy:\n",x_numpy)
print ("x_torch:\n",x_torch)
print ("y_torch:\n",y_torch)
print ("y_numpy:\n",y_numpy)

x_numpy:
 [[0.41735094 0.2518487  0.49371058 ... 0.3385944  0.02232527 0.53383751]
 [0.05890498 0.19527906 0.73701394 ... 0.57625282 0.07895023 0.05796504]]
x_torch:
 tensor([[0.4174, 0.2518, 0.4937,  ..., 0.3386, 0.0223, 0.5338],
        [0.0589, 0.1953, 0.7370,  ..., 0.5763, 0.0790, 0.0580]],
       device='cuda:0')
y_torch:
 tensor([[-0.3567, -0.7028,  0.6178, -0.1115,  0.7991,  0.7004, -0.2798, -0.0150,
         -0.0829,  1.6544],
        [-0.5199, -0.3097, -0.0135,  0.0720,  1.4688,  1.1329, -0.6637, -0.4039,
         -0.1535,  1.7466]], device='cuda:0', grad_fn=<AddmmBackward>)
y_numpy:
 [[-0.3566995  -0.7027637   0.6178046  -0.11154211  0.7991488   0.70035523
  -0.27984315 -0.01501936 -0.08288394  1.654445  ]
 [-0.51986086 -0.3096602  -0.01348255  0.07204033  1.4688267   1.1329212
  -0.6637072  -0.4038893  -0.1535429   1.7465751 ]]
```

* 1 : 배치가 2개 있다고 가정하고 입력 차원인 784 만큼 수를 생성한다
* 2 : ndarray를 torch tensor로 변경한다.
* 4 : torch tensor를 다시 ndarray로 변경한다

### Check Parameters

```python
np.set_printoptions(precision=3)
n_param = 0
for p_idx,(param_name,param) in enumerate(M.named_parameters()):
    param_numpy = param.detach().cpu().numpy()
    n_param += len(param_numpy.reshape(-1))
    print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
    print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
print ("Total number of parameters:[%s]."%(format(n_param,',d')))

[0] name:[lin_1.weight] shape:[(256, 784)].
    val:[-0.011  0.001  0.049  0.003 -0.008]
[1] name:[lin_1.bias] shape:[(256,)].
    val:[0. 0. 0. 0. 0.]
[2] name:[lin_2.weight] shape:[(10, 256)].
    val:[ 0.183 -0.044 -0.048  0.035  0.031]
[3] name:[lin_2.bias] shape:[(10,)].
    val:[0. 0. 0. 0. 0.]
Total number of parameters:[203,530].
```

* 784 - 256 - 10 으로 차원크기가 작아진다는 것을 알 수 있다.
* He init. 을 통해 weight이 작게 형성되었다는 것을 알 수 있다.
* 파라미터 개수는 약 20만개

### Evaluation Function

```python
def func_eval(model,data_iter,device):
    with torch.no_grad():
        model.eval() # evaluate (affects DropOut and BN)
        n_total,n_correct = 0,0
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(
                batch_in.view(-1, 28*28).to(device)
            )
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (
                y_pred == y_trgt
            ).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")
```

* 3 : 지금은 존재하지 않지만 drop-out이나 batch normalization과 같은 설정들이 train과 test에서 달라져야 할 때가 있다. 그럴 때 사용하는 mode change이다. test는`model.eval()` 이고 train은 `model.train()` 이다.

### Initial Evaluation

```python
M.init_param() # initialize parameters
train_accr = func_eval(M,train_iter,device)
test_accr = func_eval(M,test_iter,device)
print ("train_accr:[%.3f] test_accr:[%.3f]."%(train_accr,test_accr))

train_accr:[0.087] test_accr:[0.090].
```

### Train

```python
print ("Start training.")
M.init_param() # initialize parameters
M.train()
EPOCHS,print_every = 10,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = M.forward(batch_in.view(-1, 28*28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad()      # reset gradient 
        loss_out.backward()      # backpropagate
        optm.step()      # optimizer update
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(M,train_iter,device)
        test_accr = func_eval(M,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))
print ("Done")        
```

```text
Start training.
epoch:[0] loss:[0.380] train_accr:[0.945] test_accr:[0.945].
epoch:[1] loss:[0.167] train_accr:[0.964] test_accr:[0.960].
epoch:[2] loss:[0.117] train_accr:[0.974] test_accr:[0.967].
epoch:[3] loss:[0.090] train_accr:[0.980] test_accr:[0.971].
epoch:[4] loss:[0.072] train_accr:[0.984] test_accr:[0.973].
epoch:[5] loss:[0.058] train_accr:[0.986] test_accr:[0.975].
epoch:[6] loss:[0.049] train_accr:[0.990] test_accr:[0.978].
epoch:[7] loss:[0.040] train_accr:[0.992] test_accr:[0.978].
epoch:[8] loss:[0.034] train_accr:[0.994] test_accr:[0.979].
epoch:[9] loss:[0.028] train_accr:[0.995] test_accr:[0.978].
Done
```

* 9 : 784 \* 784로 크기를 변경해주고 GPU 환경에서 돌아가도록 to\(device\)를 한다. 그리고 이를 forward 함수에 넣는다. 
* 12 : gradient 미분값을 모두 0으로 초기화 해준다
* 13 : 256개의 배치 데이터에 대해서 backpropagation을 한다. 각각에 weight에 대해서 loss를 쌓는다
* backpropagation의 결과들을 가지고 가중치를 모두 업데이트 해준다.

### Test

```python
n_sample = 25
sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
test_x = mnist_test.data[sample_indices]
test_y = mnist_test.targets[sample_indices]
with torch.no_grad():
    y_pred = M.forward(test_x.view(-1, 28*28).type(torch.float).to(device)/255.)
y_pred = y_pred.argmax(axis=1)
plt.figure(figsize=(10,10))
for idx in range(n_sample):
    plt.subplot(5, 5, idx+1)
    plt.imshow(test_x[idx], cmap='gray')
    plt.axis('off')
    plt.title("Pred:%d, Label:%d"%(y_pred[idx],test_y[idx]))
plt.show()    
print ("Done")
```

![](../../../../.gitbook/assets/image%20%28791%29.png)

