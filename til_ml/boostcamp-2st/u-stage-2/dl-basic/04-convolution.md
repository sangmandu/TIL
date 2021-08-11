---
description: '210811'
---

# \(04강\) Convolution은 무엇인가?

## Convolution

![](../../../../.gitbook/assets/image%20%28835%29.png)

I는 입력 이미지, K는 적용하고자 하는 커널이다.

![](../../../../.gitbook/assets/image%20%28836%29.png)

3x3 필터와 7x7 이미지를 컨볼루션 연산을 하면 5x5가 나온다.

![](../../../../.gitbook/assets/image%20%28826%29.png)

같은 이미지에 대해서 적용하고자 하는 필터의 모양에 따라서 Output이 흐려지거나 강조될 수 있고 외곽선만 딸 수도 있다.

![](../../../../.gitbook/assets/image%20%28833%29.png)

일반적으로 이미지에 대해 5x5 필터를 사용한다는 것은 이미지의 채널 개수와 필터의 커널의 개수가 동일하다는 조건이 있기 때문에 문제될 것이 없다. 이미지가 32x32x3 이라면 5x5필터도 5x5x3이 될것이다.

![](../../../../.gitbook/assets/image%20%28822%29.png)

또, 여러개의 필터를 쓸 수도 있고 여러 층에서 이어서 사용할 수도 있다

이 때, 파라미터의 수를 구하는 것이 중요하다. \(우측 그림을 보면\)

* 첫번째 층에서의 파라미터 수는 커널의 크기 5x5 그리고 채널 수 3 그리고 output의 채널 수 4를 곱한 5x5x3x4 가 된다.
* 두번째 층에서의 파라미터 수는 커널의 크기 5x5 그리고 채널 수 4 그리고 output의 채널 수 10을 곱한 5x5x4x10이 된다

![](../../../../.gitbook/assets/image%20%28827%29.png)

Convolution Neural Network, CNN은 도장을 찍는 Conv와 Pooling 그리고 Fully Connected, FC로 이루어져 있다.

* 최근에 추세는 FC를 제거하는 방향이다. 왜냐하면 FC는 모델의 파라미터 수를 많이 필요하기 때문이다.
  * 파라미터 수가 많아지면 학습이 어려워지고 범용성이 줄어들게된다.

#### Stride

![](../../../../.gitbook/assets/image%20%28837%29.png)

S = 1이면 픽셀을 한칸씩 이동하고 S = 2이면 픽셀을 두칸씩 이동한다

#### Padding

![](../../../../.gitbook/assets/image%20%28824%29.png)

커널은 가장자리를 중심으로 찍지 못하는데, 패딩을 덧대면 가능하다.

![](../../../../.gitbook/assets/1%20%281%29.gif)

![](../../../../.gitbook/assets/image%20%28828%29.png)

파라미터의 개수는 커널의 크기 \* 인풋 채널 수 \* 아웃풋 채널 수 이다. 이걸 알아야 하는 이유는 어떤 모델을 볼 때 그 모델의 파라미터의 수가 대략 만단위 인지 십만단위 인지를 바로 파악해야 하기 때문이다.

![](../../../../.gitbook/assets/image%20%28823%29.png)

> 첫번째 레이어

* 필터의 크기 : 11x11
* 입력채널 : 3
* 출력채널 : 96
  * 이 때는 GPU 메모리가 부족해서 48개의 채널을 2개의 Stream으로 나누어서 진행했다.
* 파라미터 수 : 11 \* 11 \* 3 \* 48 \* 2 = 35k

> 두번째 레이어

* 필터의 크기 : 5x5
* 입력채널 : 48
* 출력 채널 : 128 \* 2
* 파라미터 수 : 307k

> 세번째 레이어 : \(3x3, 128,

> 네번째 레이어

> 다섯번째 레이어

> 여섯번째 부터는 덴스 레이어\(=Fully Connected Layer\)이다.

* 입력크기 : 13x13
* 입력채널 : 128\*2
* 출력채널 : 2048\*2

> 일곱번째 레이어 : \(1x1, 2048\*2, 2048\*2\)

> 여덟번째 레이어 : \(1x1, 2048\*2, 1000\)

파리미터 수를 비교해보면 알겠지만 Conv. 보다 Dense 에서 1000배 이상 수가 급증하게 된다. 따라서 이러한 증가를 막고자 Dense 대신 1x1 Conv. 를 하는 추세이다.

![](../../../../.gitbook/assets/image%20%28825%29.png)

1x1 Conv. 를 하는 이유

* 크기는 유지하면서, 채널을 줄임으로써 파라미터를 줄이게 된다.
  * 차원 감소라고도 한다



## 실습

### Convolutional Neural Network \(CNN\)

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
```

```text
PyTorch version:[1.9.0+cu102].
device:[cuda:0].
```

### Dataset

```python
from torchvision import datasets,transforms
mnist_train = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
print ("mnist_train:\n",mnist_train,"\n")
print ("mnist_test:\n",mnist_test,"\n")
print ("Done.")
```

### Data Iterator

```python
BATCH_SIZE = 256
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
print ("Done.")
```

### Define Model

```python
class ConvolutionalNeuralNetworkClass(nn.Module):
    """
        Convolutional Neural Network (CNN) Class
    """
    def __init__(self,name='cnn',xdim=[1,28,28],
                 ksize=3,cdims=[32,64],hdims=[1024,128],ydim=10,
                 USE_BATCHNORM=False):
        super(ConvolutionalNeuralNetworkClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM
```

* CNN 클래스를 정의해준다. 입력 차원은 \(1, 28, 28\) 이고 출력 차원은 \(10, \) 히든 레이어의 차원은 \(1024, 128\) 이다. 이 때 각각의 Conv Layer의 차원도 32와 64로 정의해주었다.
* 커널 사이즈는 3이다.

```python
        # Convolutional layers
        self.layers = []
        prev_cdim = self.xdim[0]
        for cdim in self.cdims: # for each hidden layer
            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_cdim,
                    out_channels=cdim,
                    kernel_size=self.ksize,
                    stride=(1,1),
                    padding=self.ksize//2)
                ) # convlution 
            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim)) # batch-norm
            self.layers.append(nn.ReLU(True))  # activation
            self.layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))) # max-pooling 
            self.layers.append(nn.Dropout2d(p=0.5))  # dropout
            prev_cdim = cdim
```

* 처음에 정의해주었던 변수들도 CNN 을 구성한다.
* 차원은 입력차원인 1차원에서, 32차원, 64차원을 거쳐 출력차원인 10차원으로 끝난다.
  * 자세히는 \(1, 28, 28\) 에서 \(32, 14, 14\) 그리고 \(64, 7, 7\)로 반환될 것이다.
  * CNN에서는 64차원으로 반환되며, 덴스 레이어에서 10차원으로 반환된다.
  * Max pooling 때문에 이미지 크기가 절반씩 줄어들게 된다.

```python
        # Dense layers
        self.layers.append(nn.Flatten())
        prev_hdim = prev_cdim*(self.xdim[1]//(2**len(self.cdims)))*(self.xdim[2]//(2**len(self.cdims)))
        for hdim in self.hdims:
            self.layers.append(nn.Linear(
                prev_hdim, hdim, bias=True
                               ))
            self.layers.append(nn.ReLU(True))  # activation
            prev_hdim = hdim
        # Final layer (without activation)
        self.layers.append(nn.Linear(prev_hdim,self.ydim,bias=True))

        # Concatenate all layers 
        self.net = nn.Sequential()
        for l_idx,layer in enumerate(self.layers):
            layer_name = "%s_%02d"%(type(layer).__name__.lower(),l_idx)
            self.net.add_module(layer_name,layer)
        self.init_param() # initialize parameters
```

* 덴스 레이어를 정의했다.
* 마지막에 `nn.Sequential()` 을 실행하면서 list에 append한 layer들을 하나로 합치게 된다. 이후, 가중치를 초기화하는 작업을 한다.

```python
    def init_param(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d): # init conv
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d): # init BN
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear): # lnit dense
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
    def forward(self,x):
        return self.net(x)

C = ConvolutionalNeuralNetworkClass(
    name='cnn',xdim=[1,28,28],ksize=3,cdims=[32,64],
    hdims=[32],ydim=10).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(C.parameters(),lr=1e-3)
print ("Done.")
```

* CNN은 He initializaiotn을 적용하고 배치같은 경우에는 weight과 bias를 표준정규분포를 따르도록 한다.

### Check Parameters

```python
np.set_printoptions(precision=3)
n_param = 0
for p_idx,(param_name,param) in enumerate(C.named_parameters()):
    if param.requires_grad:
        param_numpy = param.detach().cpu().numpy() # to numpy array 
        n_param += len(param_numpy.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
        print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
print ("Total number of parameters:[%s]."%(format(n_param,',d')))
```

```text
[0] name:[net.conv2d_00.weight] shape:[(32, 1, 3, 3)].
    val:[ 0.48   0.795 -0.328  0.003 -0.101]
[1] name:[net.conv2d_00.bias] shape:[(32,)].
    val:[0. 0. 0. 0. 0.]
[2] name:[net.conv2d_04.weight] shape:[(64, 32, 3, 3)].
    val:[0.155 0.017 0.136 0.019 0.046]
[3] name:[net.conv2d_04.bias] shape:[(64,)].
    val:[0. 0. 0. 0. 0.]
[4] name:[net.linear_09.weight] shape:[(32, 3136)].
    val:[-0.041 -0.032 -0.001  0.041 -0.015]
[5] name:[net.linear_09.bias] shape:[(32,)].
    val:[0. 0. 0. 0. 0.]
[6] name:[net.linear_11.weight] shape:[(10, 32)].
    val:[-0.072 -0.048 -0.105  0.251  0.523]
[7] name:[net.linear_11.bias] shape:[(10,)].
    val:[0. 0. 0. 0. 0.]
Total number of parameters:[119,530].
```

* 결과를 보면 인덱스가 00 에서 04로 바로 건너뛰게된다.
  * 01 : ReLU
  * 02 : MaxPool2D
  * 03 : Dropout

### Simple Forward Path of the CNN Model

```python
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
x_numpy = np.random.rand(2,1,28,28)
x_torch = torch.from_numpy(x_numpy).float().to(device)
y_torch = C.forward(x_torch) # forward path
y_numpy = y_torch.detach().cpu().numpy() # torch tensor to numpy array
print ("x_torch:\n",x_torch)
print ("y_torch:\n",y_torch)
print ("\nx_numpy %s:\n"%(x_numpy.shape,),x_numpy)
print ("y_numpy %s:\n"%(y_numpy.shape,),y_numpy)
```

```text
x_torch:
 tensor([[[[0.216, 0.686, 0.449,  ..., 0.845, 0.632, 0.367],
          [0.423, 0.263, 0.057,  ..., 0.135, 0.180, 0.564],
          [0.438, 0.473, 0.898,  ..., 0.777, 0.365, 0.650],
          ...,
          [0.453, 0.744, 0.648,  ..., 0.873, 0.492, 0.284],
          [0.500, 0.825, 0.532,  ..., 0.899, 0.706, 0.611],
          [0.012, 0.561, 0.997,  ..., 0.676, 0.276, 0.328]]],


        [[[0.421, 0.828, 0.172,  ..., 0.137, 0.138, 0.450],
          [0.536, 0.576, 0.426,  ..., 0.309, 0.624, 0.366],
          [0.655, 0.762, 0.226,  ..., 0.279, 0.492, 0.777],
          ...,
          [0.554, 0.616, 0.794,  ..., 0.321, 0.287, 0.028],
          [0.486, 0.343, 0.304,  ..., 0.181, 0.804, 0.304],
          [0.771, 0.622, 0.573,  ..., 0.587, 0.940, 0.416]]]], device='cuda:0')
y_torch:
 tensor([[-0.054, -3.038,  3.234,  5.741,  1.936, -9.030,  3.322,  0.424, -3.799,
         -3.518],
        [ 0.431, -1.759,  2.307,  2.540,  0.906, -5.047, -1.595,  4.348, -8.021,
          2.194]], device='cuda:0', grad_fn=<AddmmBackward>)

x_numpy (2, 1, 28, 28):
 [[[[0.216 0.686 0.449 ... 0.845 0.632 0.367]
   [0.423 0.263 0.057 ... 0.135 0.18  0.564]
   [0.438 0.473 0.898 ... 0.777 0.365 0.65 ]
   ...
   [0.453 0.744 0.648 ... 0.873 0.492 0.284]
   [0.5   0.825 0.532 ... 0.899 0.706 0.611]
   [0.012 0.561 0.997 ... 0.676 0.276 0.328]]]


 [[[0.421 0.828 0.172 ... 0.137 0.138 0.45 ]
   [0.536 0.576 0.426 ... 0.309 0.624 0.366]
   [0.655 0.762 0.226 ... 0.279 0.492 0.777]
   ...
   [0.554 0.616 0.794 ... 0.321 0.287 0.028]
   [0.486 0.343 0.304 ... 0.181 0.804 0.304]
   [0.771 0.622 0.573 ... 0.587 0.94  0.416]]]]
y_numpy (2, 10):
 [[-0.054 -3.038  3.234  5.741  1.936 -9.03   3.322  0.424 -3.799 -3.518]
 [ 0.431 -1.759  2.307  2.54   0.906 -5.047 -1.595  4.348 -8.021  2.194]]
```

* `x_torch = torch.from_numpy(x_numpy).float().to(device)`
  * numpy에서 torch로 변환되는 구문
* `y_numpy = y_torch.detach().cpu().numpy()`
  * torch에서 numpy로 변환되는 구문

### Evaluation Function

```python
def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.view(-1,1,28,28).to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        val_accr = (n_correct/n_total)
        model.train() # back to train mode 
    return val_accr
print ("Done")
```

### Initial Evaluation

```python
C.init_param() # initialize parameters
train_accr = func_eval(C,train_iter,device)
test_accr = func_eval(C,test_iter,device)
print ("train_accr:[%.3f] test_accr:[%.3f]."%(train_accr,test_accr))
```

```text
train_accr:[0.113] test_accr:[0.104].
```

### Train

```python
print ("Start training.")
C.init_param() # initialize parameters
C.train() # to train mode 
EPOCHS,print_every = 10,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = C.forward(batch_in.view(-1,1,28,28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad()      # reset gradient 
        loss_out.backward()      # backpropagate
        optm.step()      # optimizer update
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(C,train_iter,device)
        test_accr = func_eval(C,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))
print ("Done")
```

```text
Start training.
epoch:[0] loss:[0.566] train_accr:[0.960] test_accr:[0.960].
epoch:[1] loss:[0.163] train_accr:[0.977] test_accr:[0.977].
epoch:[2] loss:[0.121] train_accr:[0.981] test_accr:[0.980].
epoch:[3] loss:[0.098] train_accr:[0.985] test_accr:[0.984].
epoch:[4] loss:[0.087] train_accr:[0.987] test_accr:[0.985].
epoch:[5] loss:[0.077] train_accr:[0.989] test_accr:[0.986].
epoch:[6] loss:[0.072] train_accr:[0.990] test_accr:[0.987].
epoch:[7] loss:[0.066] train_accr:[0.991] test_accr:[0.987].
epoch:[8] loss:[0.060] train_accr:[0.992] test_accr:[0.989].
epoch:[9] loss:[0.055] train_accr:[0.992] test_accr:[0.988].
Done
```

### Test

```python
n_sample = 25
sample_indices = np.random.choice(len(mnist_test.targets),n_sample,replace=False)
test_x = mnist_test.data[sample_indices]
test_y = mnist_test.targets[sample_indices]
with torch.no_grad():
    C.eval() # to evaluation mode 
    y_pred = C.forward(test_x.view(-1,1,28,28).type(torch.float).to(device)/255.)
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

![](data:image/png;base64,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AACAAqExBwAAAAAAoEBozAEAAAAAACgQGnMAAAAAAAAKZECnKwAAnTBhwoRo3rHHHtvUbZ199tmZ6bfccku0zEMPPdTUOgAAAADoP+iZAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABSIuXtjBc0aKwgUkLtbp+tQK2KzNvvtt180zyz7cG+00UbRMvvss0807yMf+Uhm+oIFC6JlLrvssmjeySefHM176623onn9EbEJdCdiE+hOxCaKYtlll81Mnz17drTMokWLonl33313ZvqOO+5YX8VapJHYpGcOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgQzodAUAoBPyZn3aeOONM9NPP/30aJlTTjklmnfeeedlpo8bNy5a5uijj47m3XfffdG8G2+8MZoHoFg+9KEPRfNis+t9/OMfj5Y55JBDonlLLrlkNO/555/PTN9mm22iZQAA6M2wYcOied/61rcy0/NmrJozZ0407/rrr6+9YgVBzxwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQJiavGCGDx8ezYtNEZo3Tek+++wTzVt77bVrr1jid7/7XTTvjjvuiObFpm6eN29e3XUAKgYOHBjN++53vxvN22STTTLTZ86cGS1z6aWXRvN22223aF5M3mf/oYceqnt9AJpjmWWWieblnVP33nvvzPRVVlklWmbzzTeP5g0Y0L5LuDXWWKNt20J98qb1/fKXv5yZPmXKlGiZP//5z9G8I488su5yeZ/hLbbYIpq37LLLZqbnXdNOnTo1mnfjjTdG82Lypjj+0Y9+FM2bPn163dsCyuqrX/1qNO/rX/963es788wzo3lXXXVV3evrdvTMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqEqclbbK211ormHX744ZnpeVOC77XXXtG82HSp7h4tk+eNN96I5r344ouZ6aNGjYqW2WqrraJ5q666amb6CSecEC0D9CZveu+XXnopmhebmnzWrFnRMp///OejeSNGjMhMz4vNO+64I5qXV3eg6D760Y9G82Ix89RTT0XL7LTTTtG8JZdcMjN9++23j5Y55phjonlF9uyzz0bzLrjggjbWBNXyriUfeOCBaN4HPvCBurdlZtG8Rq8nGxGrR14d8vbT8ccf37Q6SNK4ceOieaNHj85Mf/LJJ+uuA9AfxP7nlaRvfOMbda9v7ty50bx777237vUVGT1zAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTZrGoUm41Gkr70pS9F88aOHRvNa2SWgUbkjZ7/hz/8IZp34YUX1r3O3XffPVombzaMDTfcMJoHtELe7De77LJLZvrRRx8dLfOxj32sz3VK+/73v9/U9QHtFpulUJLOP//8aN5uu+1W97bmzJkTzRs2bFg0L2+2mnaZMmVKNO9///d/o3k33XRTZvqf/vSnhuqRNztI3v5F6w0YEL9cHzhwYBtrUr9XXnklmpd3fdrIbFaN2nrrrTPT8/btCiusEM2LzXR15JFH1lcxoECGDx8ezcubnXjw4MF1b2ufffaJ5j344IN1r6/I6JkDAAAAAABQIDTmAAAAAAAAFAiNOQAAAAAAAAVCYw4AAAAAAECB0JgDAAAAAABQIDTmAAAAAAAAFAhTk1dZZ511MtPvvPPOaJl11103mtfIFIovvPBCNC9vurUf/vCHmekvvvhitMyMGTNqr1gNbr311mjeXnvtFc3Lm84OaIVHHnmk7jJbbrllU+vw8ssvR/P+/ve/N3VbQLt961vfiubtu+++Da3ztNNOy0zfY489omVee+21aN7tt99edx3ypjOfNWtWNO+WW27JTJ86dWq0zJtvvll7xdBv5V0X7r///tG8ww47rO5txa4lJWnRokV1r2/mzJnRvLzr03b61a9+lZm+yy67NLS+tddeuy/VAbracsstl5l+1llnRcust956DW0rdo6+4447Glpff0TPHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBASjk1+YgRI6J5sSnIY1OW9+b111+P5p1++umZ6ddee220TN60p+20+eabZ6ZPmjQpWmbIkCHRvPvvv7/PdQLq8eijj0bzFixYkJm+1FJLNbUOo0ePjubNmDGjqdsC2i12nuiL2PTesSnLgf7u3nvvbSgPPZr9XXXJJZc0dX1ANznqqKMy08eMGRMt4+7RvAceeCCat++++9ZesZKiZw4AAAAAAECB0JgDAAAAAABQIDTmAAAAAAAAFAiNOQAAAAAAAAVCYw4AAAAAAECB0JgDAAAAAABQIKWcmnzkyJHRvHXXXbfu9d1www3RvAMOOKDu9XWLUaNGRfNuu+22zPS86ccXLVoUzZszZ07tFQOa4Pnnn4/m3X777Znpu+++e0Pb+sMf/pCZPm3atIbWB5TVKqus0ukqACigI488Mpq36qqrZqYvsUT8N+/7778/mnfrrbfWXjGgC40YMSKad8YZZ2Sm500/Pn369GjeqaeeGs2bO4rZQa0AACAASURBVHduNA8BPXMAAAAAAAAKhMYcAAAAAACAAqExBwAAAAAAoEBozAEAAAAAACgQGnMAAAAAAAAKpJSzWR1xxBHRvLyRuGNOP/30vlSno7bZZptoXt4sXYMHD85Mz5ux6pFHHonmjRs3LpoHtMLWW28dzdt1112buq133nknM33hwoVN3Q7QTR566KFo3qc+9amG1vmTn/wkMz1v9sXnnnuuoW0BKJa8a9rvfOc70bzYtf+zzz4bLTNmzJjaKwYUzK9+9aumru/rX/96NG/y5MlN3VbZ0DMHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKJBSTk0+derUpq5vzz33jOatscYa0bzYVN3Tp0+Plhk9enQ0b9iwYZnpeVOxb7bZZtG8FVZYIZoX8/jjj0fzdt9992jeK6+8Uve2gL7Ii9sBA5r71bjddttlpm+++ebRMn/+85+bWgeg3U466aRo3siRI6N5W2+9dTRvtdVWy0wfN25ctEzelKgAimW55ZaL5p199tnRvEauaS+77LJo3pQpU+peH9BN7rnnnmje+uuvX/f6rr766mjeTTfdVPf6UBt65gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFYu7eWEGzxgp2gY022iiaN3ny5Mz0vCkNzSyal7d///GPf2Smz549O1pmgw02iOYttdRSddehUd/85jcz0/OmpXv99debXo92cff4Qe4yRY7NZhs2bFg076WXXormDRo0qO5t5a1vzTXXzEyfNm1atMwuu+wSzfvb3/5We8X6OWKzPVZcccVo3pw5czLT586dGy1z/vnnR/PyphnPO9/GHHroodG8n/70p3WvD7UhNtEXgwcPzky/9tpro2W+8IUvNLStp556KjN9u+22i5bhmrY9iM3axP4HlKTvfve7meknnHBCQ9uaMmVKZvq6667b0PrQo5HYpGcOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgZRyNqs8K620Umb6McccEy1zyimnRPNaMZNUTGyWj0brcOONN0bzDjrooMz0RYsWNbStbsfI/8U0ZsyYaN7EiRPrXl/eTHOf+MQnonlnn312Zvquu+4aLfPyyy9H87bZZpto3vPPPx/N64+Izfa46KKLonnPPPNMZvqECROiZWIz1UjSpEmTonnbbrttNC9m3rx50bybb745Mz12jkPtiE30xW233ZaZvsMOOzS0vtiMVZL0uc99LjP9tddea2hb3Y7Y7H9in2FJuvPOO+teX95sbbFZ3vJiDLVhNisAAAAAAIB+jsYcAAAAAACAAqExBwAAAAAAoEBozAEAAAAAACgQGnMAAAAAAAAKhMYcAAAAAACAAmFq8i6UN2Vr3nThO++8c2Z63nThl19+eTQvb8r1vCnr+iOmcSym4447Lpp3zjnn1L2+F198MZo3YsSIaN7w4cMz0/Omi9xkk02ieRdeeGE079hjj43m9UfEZnvkTdH76quvZqbnTZUaKyNJm266aTTvvPPOy0zfaqutomXyzJgxIzN9s802i5aZOnVqQ9sqG2ITUv41bWyKY0maNGlSZnreNe0zzzwTzRs1alQ0j2va7kVs9lhvvfWieRdccEE0b9ttt81Mf+WVV6JlTjvttGjeFVdcEc1D3zA1OQAAAAAAQD9HYw4AAAAAAECB0JgDAAAAAABQIDTmAAAAAAAAFAiNOQAAAAAAAAVCYw4AAAAAAECBMDV5F7ruuuuiefvtt180zyx7NrNp06ZFy2y55ZbRvLwp68qGaRyLqdlTkz/xxBPRvJEjR9a9vm9/+9vRvJNPPjmaN3/+/Gje6quvnpkem4K56IjN9si7VojlHX/88dEyEyZMaKgeAwcOzEx/4IEHomXyphmPmTJlSjRvnXXWqXt9ZURsQmr+Ne2cOXOiZY499tho3lVXXRXNKxtis5gmTZoUzdtpp53qXt/2228fzbv33nvrXh/6jqnJAQAAAAAA+jkacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAApkQKcr0N8tt9xy0bwDDjggM33nnXduaFuxqYcPPPDAaBmmHwdqlzctZCM23HDDhso99NBD0by33nqr0eoAURdddFE076ijjspM/8EPfhAts2jRooa2NW/evMz0vNhsZGryESNG1F0GKLO99947M73Ra9qYM888M5rH9OMous033zyalxdL7vEZ3KdOnZqZ/vTTT9deMXQteuYAAAAAAAAUCI05AAAAAAAABUJjDgAAAAAAQIHQmAMAAAAAAFAgNOYAAAAAAAAUCLNZtdiuu+4azbv44oubuq1Ro0Zlpj/11FNN3Q6A+hx33HGZ6TvuuGO0zMKFC6N5F154YUPlgEade+650bxDDjkkMz1vNscJEyZE8zbddNNo3i9+8YvM9GnTpkXLAGiOPffcM5p35ZVXZqYPGTKkoW0dfvjhmem33HJLQ+sDuknsf7bbbrstWmaJJeJ9MO6///5o3lZbbVV7xVA49MwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoSpyZvg1FNPjeadeOKJTd3Wl770pWgeU5ADrbXGGmtE866++upo3oEHHpiZvuSSS0bLXH755dG8G2+8MZoHtMJzzz0XzXviiScy0z/zmc80tK2xY8c2lAeg73baaado3k9+8pNo3uDBg+ve1sUXXxzNu+qqq+peH9BNNtpoo2jeL3/5y8z0vDh69tlno3ljxoypvWLoV+iZAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIU5PXaNKkSdG8XXfdNZq3aNGiaN78+fMz04844ohombxpIQG83+OPPx7Nmz59ejRvpZVWykw/7LDDGqqHmWWmn3feedEyV1xxRUPbAtrttNNOy0y/5JJLomXWWWedVlVnMW+99VY07+GHH85Mv+mmm1pVHaDj8qYfv/3226N5ede0Mddcc00075hjjql7fUBRfPrTn47mrbDCCnWv7+STT47mTZkype71oX+gZw4AAAAAAECB0JgDAAAAAABQIDTmAAAAAAAAFAiNOQAAAAAAAAVCYw4AAAAAAECBmLs3VtCssYJdbtiwYZnpf//736NlYjPfSNK0adOieYcffnhm+j333BMtg85w9+zpiLpQf43NZhs+fHg077LLLstM32233aJlnn/++WhebIa6Bx98MFpm3rx50Tz0IDa7V95sHSNHjozmbbvttk2tx5VXXhnNYwaQ1iE2O2/PPffMTM+bGXWZZZaJ5uX9zzB79uzM9LxZIG+99dZoHlqH2Gye9dZbL5r3xz/+MZo3ZMiQzPRvfvOb0TLnnHNO7RVDITUSm/TMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqEqcmrjBkzJjP9qquuipZZsGBBNO/oo4+O5uWtE92FaRyB7kRsAt2J2Oy8xx57LDN94403jpYxix+2vP8ZjjzyyMz0K664IloGnUFsNs8mm2wSzXvkkUeieU899VRm+pZbbhktM3v27NorhkJianIAAAAAAIB+jsYcAAAAAACAAqExBwAAAAAAoEBozAEAAAAAACgQGnMAAAAAAAAKhMYcAAAAAACAAhnQ6Qp0m0MPPbTuMhMmTIjmMf04AAAAimDu3LnRvIsuuiiad/3117eiOkBX+8tf/hLNGzCAf7PRevTMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqEOdOq3HjjjZnp22yzTbTM9OnTW1UdAAAAoC3uueeeaN5JJ53UxpoAAHpDzxwAAAAAAIACoTEHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACMXdvrKBZYwWBAnJ363QdakVsokyITaA7EZtAdyI2ge7USGzSMwcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAokIanJgcAAAAAAED70TMHAAAAAACgQGjMAQAAAAAAKBAacwAAAAAAAAqExhwAAAAAAIACoTEHAAAAAACgQGjMSTGzyWbmZja203XpVp3aR2Y2Mdnu+HZuF92B2OwdsYlOIDZ716kYMbPxyXYntnO76A7EZu84b6ITiM3ecd6sXdMbc1I7v/oxy8weN7Pvm9kazd5uJ6UOfC2Pe5u87cr+ntzM9fYXZja6jmPjna5vK5U0Npc2syPN7Mdm9qiZ/dPM5ifv+VEzO8PMPtCibRObvTCz3c1sgpndb2ZTzWxO8njWzK40s806Xcd2KGNsVpjZWmZ2vpk9Y2bvmNkMM3vAzI4ws5b84JQ6Z09pxfr7CzNbysyOSuLzzeT4PJvEbEu+N7tNmWOzmpl9PfX+p7RoG5w3G2BmHzezhanjM6LTdWq1ssemme1rZr81s+nJddP/mdl3zGy5Fm2P82aNzOzDyTXsNDObl/zv8XMz+3grtjegFStNLJD0RvK3SVpF0ibJ43Az283d72/h9ttptqRXc/KXUHj/kvRo66uDlPnKPzaStLKkJVWeY1Om2FxR0sWp1+9KmiVpBUkjk8dRZranuze1oRU1OUvSR1Kv35Q0RNKHksdYM/uGu5/Ticp1QJliU2b2OUk3SVo+SZopabCkLZPHvma2q7vP7VAVS8vMVpB0m8JxkMK5dI5CXH5N0qFmtqO7/7FDVWy3UsVmteSf4m93uh5YnJktKelShevYMipdbJrZZZK+nLxcKGmupPUlfUvSgWa2lbv/s1P1KzMz217SLZKWSZJmSlpN0v6S9jazL7r7Nc3cZitvs3rQ3VdLHqsqXKAfqnCxvoKkG8xscAu33zbufk7qvS72kHREavGJHapmKbn7g70cm40lLUoWn9i5mrZVaWJT4QQ3QdJekv5D0tLuvqKkQZJ2lvS0pKEK73mFjtWyvK6X9J+S1pM00N2HSRqo0Mh2m8I56vtmtlXnqthWpYlNM1tT0o0KDTl/kLSxu6+g8J73VbgA2k7SeR2rZLn9WKEhZ7akgyUNSeJzfUm/kzRM0qQSfW+WJjYjfqTwnh/udEWwmHGSPq7yHptSxaaZHaXQkLNI0kkK383LSfqMpKmS1pH0i87VsLzMbDWF65plJN0taURyXbOapGsVOtFcaWYfbeZ22zZmjrvPcfefSjo2SVpN0h7t2n6HHZY8P+buf+1oTVDtYElLKbTs/6zDdemI/hyb7v6mux/n7je7+z/dfVGSPt/d75C0a7LoSpJ261hFS8rdT3X3q9z9WXefn6QtcvfHFRrgXkgWHdupOnZSf45NSccrNKTOkvQFd39Sktx9obvfqJ73fLiZrd+hOpaSmW0qac/k5Qnufp27L5Akd/+7pN0VfgkfrvDPROn089h8HzP7gsJ7u1nSnR2uDlJSPaZeEj2nJPXv2DSzgZLGJy/PSzoTzJPCj9cK39su6TNmxjVt+31T4QeqFyXt5e5TJcndX1NoC3hE0tKSTm/mRjsxAPIv1NMT4r17xyw1GJSZrWBmZ5nZ08l9gG+mV2BhHIxxZvZ7M3sjuR9tqoVxMTbI27iZ7ZjcYzgzua/yITMb0/y3+d72VlboASB1Sc8PM1vSzHYys0vN7BEze9XCOCL/NLObzWzbGtczzMzONbMXzGyumb1kZpf1di+9mQ0xs5PN7E/JcZhr4T7885Nfa9up0tD2a3f/d5u33W1KFZuS5O7PS5qRvFy9lduqBbHZI2nc+UvysuPHpsP6Y2zulDxfG/nuvUbS6wrXKQf3cVtNYWajzOw8M3vYesbfes3M7jSzfWpcxyAzOy05Tu8k5X9mZuv1Uq7h49eAHZPntyRdWZ3p7m9Kuip5OcbMrMnbL5L+GJvp9Q+RdIGktyV9vVnrbaaSnzd/JGk5hWPzdou3VTT9MTY/p9CI7pJ+UJ3p7o9J+t/kJefNNp43LYzxd0Dy8mJ3n53Od/d3Jf0webmrmS2vZnH3pj4UGixc0uScZV5NlrkslTY5STtJ0vPJ33MVfrV7M7XcByQ9nuS7esbAqLx+R6E1LGu7J6WWW6TwT9y76gmKSh3GZpSt5EXfV2Sbxybl5ktauRP7O6PMRqn94Ard2WdXpf13pGxlP5wg6bnk7zlV5V+TtEGk/AaSpqSWXVBV9g1Jn8l5n+Mz8sZXyte57z6W2u7uzT423fYgNjO3u35qu/u1e39nlCE2e8oOkvSPpPxFnY6fVj7KGJvJNl3S13Le8/3JMg81eX9XPpdT6igzpCoOZyXxmU67tJfje4bCLWUuaV5V+bcljYqUb+j4pd7nxIy8sanyI6ryLk7SH8vZH4enyq/f6Rhq1aOMsVm13A+T5b7RaOw0e39nlCnleVPSF5Ll7khej47FdH98lDE2JZ2T5D2R855PqHxum7y/6459leu8mf4e2ixSp5VTy+zcrGPT9p45Fu5brAwG/GbGIv9P4baXnSQt4+7LS9o8KbuUpFsVBrW6R+F+7kHJMqsrjI0xSNJPzWzdqu1+VmGwTSn84re6h/u/V5J0tkKX702b9DbTDkueb/fu6fkxX+F++B0kDXX3oe4+RNKqkk5R+MB/18w+mbOOUxR+DdhN4X7NIQonkn8oHN8bkuP1HjMbKul2SWtLukHhOA5Kyq4r6TqF+/Bvsvbch185Nq8n9Sq1ssSmmS1hZh8ws/0l/TpJniZpUrO20Qelj00zW9HMRiscmxEK7/mSVm6z2/XT2PTkOW/QzsokDRs2uI1mWqRwL/yeklZy9+XdfahCXIxT+AfuK2a2b846jlL4EeFQhdgcqjA+1KMK99j/wsyGpQv05fj1QT3HRpKaev9/kfTT2KxsY6TCD5L/p55flLtR6c6bZrasQq+cuZKOaea6+4t+GpuVc+HfcpZ5KnlexcLdIZ1UpvNm5di4eo7B+yTtAK9VLd93zWy1q2pJy2wpVTh4lVapvVPpk5O0+ZI2ipSt/BL0O0lLRZa5JFnmgqr0e5L030qyjHJXpOpVV0tpzr7YOLXOPZq9r2vZ3w2u85RknVfl7IdFkj6bkf8RhZZTl3RIVd53kvTrcrZ9R7LMiZH3OT6jzPjKfq7jPQ6Q9K+k3IRWHJtue5Q9NqvWk348JunD7d7fDa6zX8ampEMix+ZVSbu0Kia65VHG2FT4B9ElXRGp0wBJ01PrH9LE/V35XE5p4jrHJOu8N+f4uqSDM/JXlvTvJP9/mnj8Ku9zYkaZsak6jajK+4Z6fvUcGNnm+any41oZH518lDE2k/wlJP0pWWZ0xmeqabFTz/5ucJ397rypnh5T41Npo2Mx3R8fZYxNhetVl/SDnP2ySWr9Gzdxfzc99tW/zptfS9Kn9/Keez2G9T7a0jPHghFmdqJCq6QURtzO+iX8Dk8GQsxwWPJ8nieD8WW4NnnePrX9FSVtk7w8y5O9WeV70Tcgyd1Hu7u5++i85SL1/bfCzCxFUTkun8lZ5veeMdWfh8ERb0xeVt8LWdkfi93nmXJd8rx9zjLV2xyfHJt67tvfUeFXG0m6uo5y/UrJYnOmQuPAjFTa45KOcfdneynbLfprbL6jcGxeU8897tMVfsG6q9bt9ScliM3fJM8Hmdl/ZOR/WdKKqdfL5W2rC1SOy6csTBWcZap64ug9Hn6tuzR5GYvNuo5fb9x9YiU23X1KVXbl2Cwj6ejqsmY2XO8flLzbj01TlSA2JemrCr0UrnX3yXnrKoB+dd5M9Zh6XtKZta63DEoQm8smz+/krGJO6u8hedvqAv3pvFnLsZF6jk/Tjs2A3hdp2NZmlvUhlqRXFHqqzM/I+0NWATMbIOkTyctLzezCyLorH4b0oGMjJZnCPwmLfVlLkru/YGYvVpVrWPKhrAw+9bOcD1NHJN0Pj1SYlWJDhS5v1Z+HvEFHJ+fk3SfpIEmbpba3pqQ1kpe353w2lk6eWz1oXCXQ/+phwLAyKWVsuvsJCvcSKxl4bGeFC6Hfm9kP3P3Evqy/WcoYm+5+k6SbkvoMlLSFwn3S10j6kpnt6e4zm73dLlSm2DxXYVr6IZLuNLOvSXpQ4YJof4WxARYodIOXehr5OibZn4cpTJ2+iUJj09JViw1SiNms26rvi1zcSyE2T5a0kZkt7e7z+3j8Gubuj5nZrxVm+zvDzBYpXEzPkvRJheniB6WKdPzYtEFpYtPMVlfodTJTUlecF3tTlvNmMsjqpQqfi2PcfW4z1ltwpYnNIirLebOTWtmYs0Bh4C+pp7vuCwrzrl/h7jMi5V6PpKcP/ko1bH9w6u/KPZMz3T1vtPeX1byDuoPCdHhSl8xiVWFhZP7JktKjgL+t0GNhkcIHfGX1tDJmebmGvFVSaenZAIbXUM1lalimIcm9lZUp+ya2ajtdrOyxKXefJennZvZ7hXtbTzCzB9z95mZtoxFlj01J8jDN5v1mto3CxdA2CtM4fq2V2+0SpYlNd5+SjFv1C4WBA++pWmSKpJ9J+u/kddaYB21jYVafuxTuva94R2HfVxozKr09l1X2RWktsbmkwkXtq+rb8eursQq3iGyhML7AhFTeuwo9AyoXyR09Nm1SmthUGItleUnHuvu/GijfViU7b35VISZ/6e53NGmdRVem2KysM++7Pv1Zmx1dqg1Kdt6s5dhIPcenacemlY05D9Z5S1LFu5H09C1hI9398QbW3U6Vnh9PuvujHa3J4iYonPReUBhx/d70l10yGNRzTd5m+vgN8zC1aaccIGmgpIXq6WZXJmWPzfe4+8tmdrNCvP6npI425ojYfI+7LzSzSxR6AvynytGYU6rYdPfbzeyjCg0DoxUuhP+tMKjoOeqZCvnFpJGvk05RuCD9t0IPvzvdvTKQYaU37sLKyyZts2PHz92nJwN5flHS3pI+pPD+HlHoVfVaavGi3KbaF6WIzaQRfS+FAVZ/kvwzlrZ0z6Lv5c1194XqnFKcN5MBl7+jMOjx/2Qcm/Q/kcsk+Qu64Luz1UoRm4l/KgyenNfLLJ33Smur06synTf/mTwPM7NBOb3mKsenacemlY05zTZdIfCWlLSWwlgXtaq0vg41s2XcfU5kubzgqFkyav3uycuuGo/FzJZWT90OdveHMhZbNSOtWi1fJOlW71dTf6+lzv6SNzZ5vsvdX81bEDUpTGxGVFr2mzWifUOIzUyVYzPEzIanLwJQk66PTXefquT2x2pmVrmtIbM7fJtVZts4xt1/npHfrNh8Vz3jevXl+PVZcmvCpeoZl+A9ZrZX8udCSX9uZ736iW6NzbWT548q/1ywlqS3kr+/qA71ci7ZeXOYQo8pKTJbTkpltqOr9f7xrdC7bo1NKRz3nZU/g2BllqTXvfOzKJfpvFmJSVM4Bot15LAwu9jwquX7rO1TkzcqGXOmcsGwU53FKyNHLyHps1kLmNkHFQ56M1R6fryrMOZDN1lZoW5S2C9ZPlfDerauIe+9D7K7/0M9J796j1/TmNn66rmXcmKn6tGfFCw2s3wwee5od1SVPDYjPpj6u9PHp3CKHJtmtpJ6BiZcbPDDDqiMj9Hq2HyyMr5DH49fqx2YPP86uW0VdShybHYZzptoqi6PzXuT548mtxdm+XzyXH3rcieU6bz5f+r5TogNqlxJn6/ImEqNKExjTmJi8jzWzDbJW9BSc867+xsKU8RJ0n+ZWVZXrm82pYZB5Raru7rwfuO3FL5opDB1+vskXw7H1LCerc1sy+pEM/uwekYVv6Eqe2LyfGJk9pLKOizp3dQKlWMzQ9mj26MxE5PnrorNZCC0vPwPS9ojefn7RrfTJKWKzRqOzWCFqUUl6dGcX7iQb2Ly3FWx2Us9TGHq60GS/irp163YTp0qA3BnxeYQSd+qYR0jzOzA6kQLM6B8JXkZi826jl8rmdkOCrdeLZJ0Vju22U9NTJ67JjarZmtZ7CHptGTRqan0iTmrbLXSnDfdfUovx2ab1OIfTNLH9nW7JTUxee6a2Ezco3CL6xLK6NGa1LXSQNINw0iU5rzp7oskVXofHW1m7xujKxm8/Ljk5aRm/ghStMacKyU9pHCB91sz+7KFWWkkSWa2mpkdbGb3afHxFcYrfOFvJ2mima2alBlqZt9T+EBEZ0sxs8lm5mY2Oa+CZraepE8lL2u6xcrMxifrjo3WXYulzGzlXh5LuftbCvtQkn5sZpsmdVjCzLZTGBm8lvsWZ0n6pZntXPmyMrOtFAZMHKjQxfMXVWXOVLineWVJD5rZfsk/bErKr2VmX1H49WMP1ajW/ZcE0iHJy5+X4D7idurW2DzfzM43sy3NbFCqzApmNlbh8z5Y4YLw3Ix1E5tqWWwebGY3m9mu6ZOpmQ00s+2T91u5ADi91m1iMd0amzKz75nZDlX1GSnpFoXZY+ZI+qK7Lza2gZmNrXy2zGxEbzshYokaYrPyq//dyfMPzWzrVGxtoXCBXctAizMlXZ7s7wFJ+Y8pDBC5isJF+kVVZfpy/KJ6239mtq+ZHWlma6be63Az+4bC2GIm6ZzIbS2oTdfGZl9w3gxadN5Ee3RlbCb/t4xPXh5nZidUzlFm9mmF7+YlJD3g7ov9CMJ5M2jVeVPhO2GWQs+rX5rZWkm5VRQamLZQ6JVzaq3brIm7N/WRVNYlTa6z3OSk3Nhelhuu0DXJk8e7CvfHzU6luaRTM8qelMpfpDD6+cLk9Q/y6pDKy31fkr6bLPeGpIE1vvfxlXr1YX/X8hidlPmkwkVyJX126vV0gOX5RQAAIABJREFUhfuPM+uT2g8nKAwo50nZt1Lre03ShpH6fkjhPsHKsgsVBsZK18clHRZ5n+Mb3X8K3dsq6/9Esz/73f4oY2xWxce7yXpnVNXnFUmfjbwnYrNFsalwH396vbOS7S1Mpc2V9NVOx06rHyphbCb5U1Lrnpkc7/RndZuc95T+/Iyoc7+N1/vfd95jbFJmHYXxECrp76T23xyFru2Z9Ukd3zMULjArn+2ZqTJvSxrVzOOXep8T691/VftonsKYIOnPwTmSrNOxQ2y2JjZriJ0ptcRXH/Y3580695/CIPINfScW8VHm2JR0WWr986s+r89LWj1SbmyjnxFx3qxp/yn8r/l2ark3k8+AK8y8dkizY6FoPXPkYQDMrSUdrDDjxeuSlkuyn5b0E0n7KbSOVZf9vsL9c/cqHMwBCvfVHerumQMw1sNCz48xycvrvfaeH5X7HtsyiKC7Pyzp0wq/fs6QtJTCyepShVHS/1LDaqYrjD0zQeEewaUVRvK+XNKm7p45sJO7PydppKSjFY7DDElDFb7knlD4gtpFrRlr6LDk+Wl3/2ML1l9qXRqbZyp0af2Nwj+OSytMC/iqwq8Cx0ta391j964Sm62LzdskHanQPfbpZDtDFRp1/qhw7DZ09wuja0BNujQ2Jenbkm5VT2zOVfgVe7xCXN6bU7YSmy+rDTN2uPsLCnF1jUJMLqlwkXatpC3c/Tc1rGaewj9cp0uaqvCeX1fomr2Zu/8usu2Gj18f/ErSJQq3uc1R+IVzikKP40+7+4meXLmicV0cm33BebM917RooW6OTXf/iqT9q9b/tEKHgk3d/Z+Ropw3g1adN+Xudyt871wl6SWF3v+vKvTs+5S7N/27wDgXd56ZPS3pI5J284xucQA6g9gEupOZ3SlpB4VZMi7odH0ABJw3ge7EebN/ojGnw5J7Kf+lMMDnxztdHwABsQl0JzOr/Lo3S9I6dfSCBdBCnDeB7sR5s/8q3G1W/dCo5JkBPoHuQmwC3WkzSUMknc0FKdBVOG8C3YnzZj9FzxwAAAAAAIACoWcOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFMqDRgmbmzawI0M3c3Tpdh1oRmygTYhPoTsQm0J2ITaA7NRKb9MwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAAAAAAAoEBpzAAAAAAAACmRApysAoLiuuuqqaN4Xv/jFNtYEAAAAAMqDnjkAAAAAAAAFQmMOAAAAAABAgdCYAwAAAAAAUCA05gAAAAAAABQIjTkAAAAAAAAFQmMOAAAAAABAgZi7N1bQrLGCQAG5u3W6DrVqZ2zOnz8/mrfTTjtlpt9zzz2tqg5KiNgEuhOxWUyrrLJKNO+GG26I5l177bWZ6Zdffnmf69RqQ4YMyUw/+eSTo2Xuu+++aN5dd93V5zq1ErGJvjjuuOMy00855ZRomaFDhza0rVtvvTUzfa+99mpofd2ukdikZw4AAAAAAECB0JgDAAAAAABQIDTmAAAAAAAAFAiNOQAAAAAAAAVCYw4AAAAAAECBMJsVUANG/s/2la98JZr31a9+NTN9//33j5Z5+umn+1wnlAuxCUn6/Oc/H83be++9o3n77LNPZvqwYcOiZcziH7m77747mjdu3LjM9GeeeSZapsiIzWK6+OKLo3l55/xJkyZlpu+xxx59rlOrHXjggZnp11xzTbTMlClTonlbbLFFZvobb7xRV71ahdiElB/rO++8czRvjTXWyExvtE2hEQcddFA0b9q0adG8hx56qBXVaRpmswIAAAAAAOjnaMwBAAAAAAAoEBpzAAAAAAAACoTGHAAAAAAAgAKhMQcAAAAAAKBAaMwBAPx/9u47Tqrq/v/4+4Og9KJgVFCMYvdrsCQxCopootEgRU00aiSx94IoSlS+1hg1osZv7EEsQYNgwRaNUtSo3wSwEQvxi/5EDSAdRBb2/P64d8O43nN3Z/ZOOTuv5+Mxj909nzllZvYzO/PZO/cAAAAACAhbk2dg55139sauu+46b+yggw5KbP/888+9fY444ghvbN68ed6Yz9KlS72xtHVUG7ZxzN/NN9+c2P7tb3/b2+epp57Ke54WLfw16dra2rzHk6TNNtsssT1ti9Vp06Z5Y1dddVVi+9y5c/NbGL6B3Gx+WrVq5Y397ne/S2w//fTTvX2y3i41bWvytLl8zxEDBgzw9lm2bFnjF1ZhyM3KtfXWW3tj06dP98bat2/vjVX61uQbb7yxNzZp0qTE9t13372guQ444IDE9hdffLGg8bJGbjY/v/zlL72xE044IbF9m2228fbZcMMNvTHf38BSbk2e9nd4wYIF3tivfvWrxPYnn3yyyWvKAluTAwAAAAAANHMUcwAAAAAAAAJCMQcAAAAAACAgFHMAAAAAAAACQjEHAAAAAAAgIBRzAAAAAAAAAsLW5I00YsQIbyxtS1TfFseV4tNPP/XG3nnnHW/svvvuS2x/4IEHmrymSsQ2jvnr1atXYvull17q7XP00UfnPc+sWbO8sY4dO3pjPXr0yHuuxx57zBsbPHiwN9avX7/E9ilTpuS9BnwduRmmtG2CH3nkEW9sr732SmwvdLvw2traxPb58+d7+2yyySYFzeVzyy23eGPnnHNO3uNVCnKzct18883eWNpr2jSVvjX5nnvu6Y29/PLLeY+X9tqjd+/eie1r167Ne55iIDfD9JOf/MQb+9Of/uSNtWnTJtN1VPrW5GnrePrppxPbBwwY0OQ1ZYGtyQEAAAAAAJo5ijkAAAAAAAABoZgDAAAAAAAQEIo5AAAAAAAAAaGYAwAAAAAAEJCW5V5AOaSdAfvBBx9MbD/ssMO8fdZbb70mr6lc0nbbSov17ds3sb2mpsbb5+GHH278whC82bNnJ7afcMIJ3j5333133vOk7QjVrVs3b+ynP/2pN/b2228ntm+44YbePgMHDvTGAHxd2s51vh2rCnXxxRd7Y6+++mpie9rzyl/+8hdvbP/992/8wmJpry9C3s0K5denT5/E9qOOOirzuZ555pnMx8xS2i50hUjbDbZSdq1C87LNNtt4Y1nvWBWymTNnemMnnXRSCVdSGhyZAwAAAAAAEBCKOQAAAAAAAAGhmAMAAAAAABAQijkAAAAAAAABoZgDAAAAAAAQEIo5AAAAAAAAAWm2W5OnbT9+ySWXeGNp2xVn7b333ktsf+qpp7x93n///UzXMHLkSG+sR48e3ljr1q0T2/v16+ftw9bkkKTVq1d7Y2nbARdi/vz53titt97qje29996J7ePGjfP2SduKtKamxhsDmqsf/ehH3tiVV17pjaX9/fY55ZRTvLE77rgj7/GOOOIIb+yAAw7Ie7w0c+bMyXQ8oM5pp52W2L7hhhsWNN6HH37ojT344IMFjZkl32tTSRo2bFimc/GaFqX2u9/9zhurra0t2TpatEg+FuShhx7y9jnyyCO9sTPOOMMbu+mmm/JagyTtuuuu3ljPnj0T2z/77DNvn0rHkTkAAAAAAAABoZgDAAAAAAAQEIo5AAAAAAAAAaGYAwAAAAAAEBCKOQAAAAAAAAGhmAMAAAAAABCQZrs1eatWrbyxyy67LNO5pk+f7o2lbSP3yCOPJLanbd2ctbTtXNO2Jl+wYEFi+9ixY5u8JqAUtt9+e2/siiuuSGxfb731vH0uvPBCb+yVV15JbP/e977n7fP66697Y0Al2X333RPbJ06c6O2TtoVw2t/Ac889N7G9kO3HJemoo45KbL/nnnsKGs85541Nnjw5sf30008vaC5AkgYOHOiNDRo0KO/xVq5c6Y19//vf98aWLl2a91xZO//8872xvfbaK9O51qxZk+l4gCRdeuml3lja9uNpf3uy5lvHG2+8UdB4f/jDH7yxnXfeObH9xBNP9PZJuy9GjhyZ2D5gwABvn0rHkTkAAAAAAAABoZgDAAAAAAAQEIo5AAAAAAAAAaGYAwAAAAAAEBCKOQAAAAAAAAGhmAMAAAAAABCQZrs1eZq07QRbtky+S1588UVvn0MPPdQbS9viMWtmlth+xhlnePvstNNOBc01Y8aMxPZXX321oPGAYkjblvXGG2/0xjbffPO85/rud7/rjb311luJ7VtvvbW3z7/+9a+81yBJt912W2L77NmzvX2effbZguYCJGnHHXdMbG/Tpo23T9rWoRdddJE3NmbMmMT23r17e/scdthh3th5552X2L7++ut7+xTqvvvuS2x/9913M58L1eP+++/3xjbYYIO8xxs3bpw3tnDhwrzHK6W055xCrFixwhu79957M50L1eV73/teYnvaltuVbuzYsQX1W7t2rTe2fPnyQpeTqHPnzpmOVwk4MgcAAAAAACAgFHMAAAAAAAACQjEHAAAAAAAgIBRzAAAAAAAAAkIxBwAAAAAAICDNdjer1atXe2P77befN7bddtsltj/00EPePqXcsSpNt27dEttHjx5d0Hg1NTXe2LXXXlvQmEDW3nzzTW+s0N3aCpE216OPPprY/thjj3n7/P3vfy9oHfvss09i+8CBA7190nblmDRpUmL76aef7u3z1VdfeWNAQ3bbbTdvzLdj4n/91395+6TtnJW1559/3ht75JFHSrYONC+HHHKIN9a6deu8x0vboTXtuT1r7dq188aGDh2a2J72mrtXr15NXdLXTJgwIdPxgDrDhg1LbN90000zn2vevHmJ7Wmv1dJ2fZo2bVpie6XvdtdccWQOAAAAAABAQCjmAAAAAAAABIRiDgAAAAAAQEAo5gAAAAAAAASEYg4AAAAAAEBAKOYAAAAAAAAEpNluTZ7mlVdeKShWbV5++WVvLG1bS6CU0rZlTduSeNasWd7Yc889l9j++OOPe/tMmTLFGyulqVOn5tUuSXfeeac39swzzyS233rrrd4+J5xwgjcGSJKZeWNHH310yebK2pgxY7yxZcuWlWwdCNMWW2yR2H799dd7+7Rokf//ZR999FFv7NRTT817PEkaMmRIYnuPHj28fVq1auWNde/evaB1ZGnSpEnlXgICtuWWW3pju+yyS6ZzzZw50xsbMGBAYvtnn33m7bPvvvvmvYYvv/wy7z6l9sQTT5R7CZnjyBwAAAAAAICAUMwBAAAAAAAICMUcAAAAAACAgFDMAQAAAAAACAjFHAAAAAAAgIBQzAEAAAAAAAhIVW5N3lyddtppmY7XHLdvQ/Oz2267eWNt2rTxxlauXOmNrVixoklrCs3777/vjfm2uxw6dKi3D1uTV5d//OMfie3PP/+8t8/++++f6RrSth93zmU61+uvv+6NPf7445nOheqy++67J7Zvt912mc5z8803Zzpeoe69915vbPny5Yntp59+urdPoc8Dc+fOTWxftGiRtw/QkAcffNAb23bbbfMer0UL/zEYd999tzeWtgW5z5QpU/LuU6gOHTp4Y77nxLT7ora21hubOnVq4xcWCI7MAQAAAAAACAjFHAAAAAAAgIBQzAEAAAAAAAgIxRwAAAAAAICAUMwBAAAAAAAICLtZBaZPnz7e2IgRI/Ieb8GCBd7Ybbfdlvd4QKn5drxoKIZ1hg8f7o395Cc/SWz/9NNPi7UcBGbWrFmJ7Ycddpi3z/nnn++NtW7d2ht74YUXEtvTdutI21Vro4028sZ8brjhBm+s2nbCQ7Z8uwRmvSNbmmnTpnljNTU13tiECRMS2z/55BNvn7RdUwu5L9Jiabs27rXXXont7GaFhvheI0lS7969vbFCcjptl6aQd1JM+5vat2/fxPa0+yLteWX69OmNX1ggODIHAAAAAAAgIBRzAAAAAAAAAkIxBwAAAAAAICAUcwAAAAAAAAJCMQcAAAAAACAgFHMAAAAAAAACwtbkgTnwwAO9sVatWuU93i9+8QtvbNWqVXmPB6AypW0Fffnll3tjvi3IDzrooCavCc3bsmXLvLHLLrss07mOOOIIb6xDhw55j5e2nfn48ePzHg9ojLfffjuxPW3747TXasccc0xi+wcffODt8+6773pja9as8cYK0bFjR2/s7LPPznSutC3X2YIchWrTpo03tv7662c617hx47yx+fPnZzpX1nbeeWdvbNCgQZnOlfYctnr16kznqgQcmQMAAAAAABAQijkAAAAAAAABoZgDAAAAAAAQEIo5AAAAAAAAAaGYAwAAAAAAEBCKOQAAAAAAAAFha/IK1KdPH2/swgsvzHu8jz/+2Bt744038h4PaIr99tvPG0vbDthnu+2288Zmz56d93gh6NWrlzc2YcKExPaddtrJ2yftOeLcc89NbJ81a5a3D1Asm222WWL7Qw89lOk8119/fabjAY1x6aWXJrZ/9NFH3j5p2/BOnTq1yWsqpq5du3pju+yyS97j1dbWemNPPfVU3uMBleSzzz7zxiphy+22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OeLIHAAAAAAAgIBQzAEAAAAAAAgIxRwAAAAAAICAUMwBAAAAAAAICMUcAAAAAACAgFDMAQAAAAAACAhbkxfZBhts4I3dc889ie2HHnpoQXN98MEHie1p2zGOGTPGG/NtPy75tyB//vnnvX2GDBnijQEN6dChQ2L7/vvv7+3z8MMPe2Ovv/56k9dUTK1atfLGamtrvTFfbgKlNmzYMG+sbdu2BY05evToQpcDoMJ069bNG7vooosS2wcPHuztM2vWrCavCc1b2nbhaTGs07lz58T2tO3Ht9hii4Lm+vDDDxPba2pqChqvOeLIHAAAAAAAgIBQzAEAAAAAAAgIxRwAAAAAAICAUMwBAAAAAAAICMUcAAAAAACAgFDMAQAAAAAACAhbk2dg22239cZ++9vfemMDBgzIe645c+Z4Y7/+9a8T2y+++GJvn0K3h33yyScT29l+HMWy4447JrZ3797d2+e1117zxtK29y4V3/aOkrTDDjt4Y88991wxlgMUZPvtt09sP+KIIzKfy8wS24855hhvn1/84heZrmHNmjXeWNrf27feeiuxfe3atU1eE1BuPXv2TGyfP3++t09a3p599tmJ7StXrvT2KcZzDoCvmzhxYmJ7oduPjx8/3hu74oorChqzmnBkDgAAAAAAQEAo5gAAAAAAAASEYg4AAAAAAEBAKOYAAAAAAAAEhGIOAAAAAABAQNjNqp4uXboktp911lnePhdddJE31qpVqyavKdesWbO8sUsvvTSx3bcLUEPGjh3rjZ144okFjQkUqk+fPnn3mTJlShFWkp2f/exn3thGG23kjU2dOrUYywEK4tuVbeONN858rqeffjqxvUePHt4+he7aWIgDDzzQGzv//PMT22+55RZvn7Sds4BK8vrrrye2n3feed4+I0aM8Macc4ntV199tbfPu+++640BaLybbrrJG+vXr19ie9ousbNnz/bGjjzyyEavC9/EkTkAAAAAAAABoZgDAAAAAAAQEIo5AAAAAAAAAaGYAwAAAAAAEBCKOQAAAAAAAAGhmAMAAAAAABCQqtyaPG2LY9+W28ccc0yxlpOXgw8+2Bszs8R23/aODZk3b543lrb9HFCoDTbYwBs77bTTEtsXLlzo7bPpppt6Y3fddZc39q1vfSuxvV27dt4+++yzjzfm48vZhrRu3bqgfkDott1228T2tL9JX331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M844oxiDqWyvvfYqxkpTkH/jG98o9vngBz9YjO2yyy617QcddFCxz7777luMrbXWWsVYpymVoelWWmml2vY3vOENxT5HHnlkMbbNNtuMegzPPPNMMTZt2rRibNVVV61tNzU59Ma8efOKsQcffLAY22ijjWrb/+d//mfcY4JhsfbaaxdjBxxwQDHWaZ9a2rd3WpepyQEAAAAYWIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIKk0A9JSO6Y0to5T0Prrr1+MnX766cXYDjvsMAGjWdKTTz5ZjL35zW8uxi6++OKJGE4j5ZxTv8fQLbm5yGqrrVaM3XjjjcXY/Pnza9tf8YpXFPt0yrOS6dOnF2P/9m//VozttttuxVhpdp6rrrqq+4E1iNycekozy0REHHvssbXte++990QNZwl33313MbbTTjsVY8M2M47cZLJ12s91mv3mF7/4RW37PvvsM+4xDSK5yUQ45phjirFPfepTPV3XJptsUow1eV87ltx0Zg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADTIMv0eQFNssMEGxdiFF15YjL3kJS8pxsY6LfxodZr++LTTTivG3vOe9xRjV1555bjGBJNhwYIFxdgOO+xQjC2//PK17WOZfryTJ554ohi79tpri7FOU5MfddRRte3veMc7uh8Y9MByyy1XjH3+858vxjrte9Zcc81xjakX1lprrWLsrLPOKsZOOOGE2vYzzjhj3GMCIrbccstirNMxd6djYeDZVlhhhdr2zTffvOfruv7662vbH3jggZ6vq6mcmQMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA1iavIuHXroocVYp2nLe600RVtExD333FPbvvPOOxf7vPCFLyzGVlllla7HBU1zww039HsIHc2ePbsYO/LII4uxTtMmw2Tadttti7EPfehDY1rmvHnzatsvu+yyYp+rr766GLvkkktq21//+tcX+3zxi18sxl75ylcWY/apMH6f/OQni7HnPKf8GfWtt95ajF133XXjGhMMk2OPPba2/e1vf/uYlvfQQw8VY3vuuWdt+5/+9KcxrWsqcmYOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIOYzapLKaUxxTpdWf+JJ56obT/llFOKfUpXEI+I2G+//WrbO83K8cwzzxRjwGDKOfd7CLBUpZmnIiIuvvjiYqw0w1RExGmnnVbbfu+993Y/sC489dRTPV1eRMSWW27Z82XCVPXa1762tv2II44o9ul0TPvrX/+6GLvzzju7HxgMgZe//OXF2F577dXTdXXa386dO7en65qKnJkDAAAA0CCKOQAAAAANopgDAAAA0CCKOQAAAAANopgDAAAA0CCKOQAAAAANYmryLn3lK18pxsY6FeL9999f2/6jH/2o+4G12X333WvbO43PFMcwmNZbb71+DwHG5cYbbyzG9thjj2JsIqYFL5k+fXpt+9e//vUxLa/TPvWf//mfx7RMGEYzZ86sbV9hhRXGtLyzzjprPMOBoXL++ecXY2M5Pr3rrruKsX333XfUy2MRZ+YAAAAANIhiDgAAAECDKOYAAAAANIhiDgAAAECDKOYAAAAANIhiDgAAAECDmJq8S7fccksxdthhh03aONZdd91ibPPNNx/18jo9r6uvvnrUywN64/Wvf/2Y+i1YsKDHI4Gx6TRN92ROP77SSisVY295y1tq27fddttin05jP/roo4uxK6+8shiDYXTggQcWY51yqeRHP/rRmGIwjFZfffVirNN+cywuvfTSYmzOnDk9XdewcWYOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iKnJG2bmzJnF2GqrrTbq5X3ta18rxubPnz/q5QG9sdFGG42p37nnntvjkcDgWH/99WvbV1xxxWKfk046qRjbcccdRz2Gyy67rBj74he/OOrlwVS2zDLlfzXe8pa3FGPLLbdcbftDDz00puUBz/axj32sGFtzzTV7uq6jjjqqp8tjEWfmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSI2awa5qCDDurp8u69996eLg/o3hZbbFGMvelNbyrGUkrF2C9+8YtxjQkmQ6fZpzrNSHPiiSfWtq+++urjHlO3FixYMGnrgqY75phjirHdd9+9GMs517Z/+tOfHveYYJiU9rcvfelLJ3kkTARn5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIOk0tR/S+2Y0tg6TkErrbRSMbbKKquMenkf+9jHirHDDz981MvrNP34+uuvX4w9+uijo17XVJVzLs8FPWDkZnOcccYZxdj73ve+Yuycc84pxt71rneNa0xNIzebadNNNy3Gfvvb307aOG666aba9g033LDY5+GHHy7GzjvvvGLskEMOqW2fqlOdy00iIp5++ulirNP/IJdccklt+5vf/OZinyeeeKL7gQ0xuTlcXvnKV9a2/+Y3v+npejrtGzfZZJNibN68eT0dR5ONJTedmQMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2yTL8HMGg22GCD2vZtttmm2KfTVOJbbbVVMTaWaeHH0mfGjBnF2Le+9a1i7IgjjijGbrnlllGPA4bVZpttVtu+6667jml555577niGA303f/78Yuzmm28uxqZPn17b/k//9E/FPj/60Y+Ksfvuu6+2fb311iv2Of3004uxffbZpxi74oorattPPPHEYh9oipNPPrmny/v85z9f2276cVjSpptuWox9//vfn5QxfPnLXy7GTD8+cZyZAwAAANAgijkAAAAADaKYAwAAANAgijkAAAAADaKYAwAAANAgQzmb1S677FKMnXXWWbXta6yxxkQNp6/22muvYmyHHXYoxkqzYB111FHFPo8//nj3A4MuvfOd7yzGjjvuuGJs1VVXHfW6Os2K02mGqeOPP762fZ111in2+c53vlOMnXPOOcUYNEGn2aw67XtK+5EFCxaMe0ztzJgDS9piiy2KsX333be2PaVU7PO5z32uGPvJT37S/cBgyL385S8vxl7ykpf0dF2XXXZZbXun2ayYOM7MAQAAAGgQxRwAAACABlHMAQAAAGgQxRwAAACABlHMAQAAAGgQxRwAAACABkk557F1lx7B3QAAIABJREFUTGlsHSfJYYcdVowdffTRxdhqq63W03F0mpJxLK/9LbfcUoxdfvnlte0rrLBCsc873vGOUY+hk9///vfF2D777FOMXXfddT0dR6/lnMsbcsAMem6O1UYbbVTbXpoiMSJirbXWmqjhLKHXud5pmskbbrhh1MubquQm4zF9+vTa9s985jPFPh/72MeKsU5TLZeOPcZ6HDbo5ObU8853vrMY+9d//dfa9vvuu6/YZ6uttirG/vjHP3Y/MEZFbjZTp//nzj///GJs5513HvW6nnzyyWLsrW99a237hRdeOOr18GxjyU1n5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIMs0+8BTJTnP//5xVivpx/vtdtuu60Y23333Yuxm2++ubZ92rRpxT4bb7xxMfaJT3yiGNt7771r2zfZZJNin5/85CfF2N///d/Xtp9xxhnFPgyXb3zjG7Xt66yzTrHPTTfdVIwdd9xxox7DAQccUIxtv/32o15eJ53e+8cff3wx9oMf/KCn44CprLSf22233Yp97r777mLs2muvLcam6hTkDI93v/vdo+5z+umnF2OmH4dnW2ONNYqxc845pxjbcccdezqOTtOMm4J8sDgzBwAAAKBBFHMAAAAAGkQxBwAAAKBBFHMAAAAAGkQxBwAAAKBBFHMAAAAAGmTKTk1+8MEHF2MppUkbx3OeU66XLViwoLa90/THpenHO3n66aeLseuuu64Y22effYqxl7/85bXtnaYf7zSF9De/+c3a9k033bTY55hjjinGnnzyyWKMqaXTdL/f//73i7FO06Wuu+66te1vetObxjSOyy67rLZ95ZVXLvbZaqutirFOz+vYY4+tbZ81a1axD0xlJ598cjG2/fbb17Z32veceOKJxdhPf/rT7gcGA+hVr3pVMbbHHnuMenkXXHDBeIYDQ+V5z3teMdbr6cfnzp1bjP3d3/1dT9fFxHFmDgAAAECDKOYAAAAANIhiDgAAAECDKOYAAAAANIhiDgAAAECDTNnZrDrNXvGxj31s0sZx4403FmO77bZbbXunq4sPitIsWDNnziz2ueSSS4qxGTNm1LYffvjhxT733XdfMfa5z32uGKOZbr/99tr27bbbrtin08wbV155ZTFWmhGqNItbRMTvf//7YmzPPfesbX/ooYeKfTrNnDV9+vRirFNeMDxK+5eIiOWWW662/bzzzpuo4fRMaaadj370o8U+e++9dzFWyqWrrrqq2Of4448vxu68885iDJqg0yw2pb8dEeX9WWk2R2BJyy67bM+XWZpt9dRTTy32+cMf/tDzcTAxnJkDAAAA0CCKOQAAAAANopgDAAAA0CCKOQAAAAANopgDAAAA0CCKOQAAAAANMmWnJv/Od75TjL3uda8bdeyKK64o9rn00kuLsW9+85vFWBOmIB+t66+/vhg788wzi7GxTBf/oQ99qBjrNN3e/fffP+p10X9//dd/Xdu+4YYbFvu8+tWvLsbOPffcUY9hwYIFxdh+++1XjHWagrzkwgsvHHUfGHHggQcWY7vuumtt+//8z/8U+3z9618vxh588MHuB9ay5pprFmMHHHBAMbbxxhvXtq+88sqjHkNExOGHH17b3ukY4u677x7TumBQdJpifPXVVy/GSlMcR3Q+7gK6s/fee/d8maX/ey6++OKer4vJ58wcAAAAgAZRzAEAAABoEMUcAAAAgAZRzAEAAABoEMUcAAAAgAZRzAEAAABokNRpmsGOHVMaW8cBsMIKKxRjL3nJS2rbb7755mKfRx99dNxjGgabbbZZMXbNNdf0dF1HHXVUMfaFL3xh1MvLOafxjGcyNTk3x6LT1OQnnXRSMfbSl760GCtNW95peuabbrqpGGPiyM16n/jEJ4qxWbNmTdYweu68886rbb/rrruKfa666qpi7LTTTqttH+uxEYvIzcH1whe+sBi79dZbx7TMO++8c9Troj/k5uB62cteVoxdf/31Y1rm448/Xtt+yCGHFPucfvrpY1oX4zOW3HRmDgAAAECDKOYAAAAANIhiDgAAAECDKOYAAAAANIhiDgAAAECDDOVsVvTHKqusUoxdfvnlte2bbLJJsU+nq7p3ms3qRz/6UTFW4sr/MJjkZr1lllmmGCv9Ld5///2LfdZaa61i7LDDDqttf+KJJ4p9Os00N2/evGKsNKPcU089VexDf8jNwdXp70NpNseIiD322KMYu/rqq2vbt9566+4HxqSQm4NrImazuu2222rbN9hggzEtj4ljNisAAACAKU4xBwAAAKBBFHMAAAAAGkQxBwAAAKBBFHMAAAAAGkQxBwAAAKBBTE0OXTCNIwwmuQmDSW7CYJKbMJhMTQ4AAAAwxSnmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADRIyjn3ewwAAAAAdMmZOQAAAAANopgDAAAA0CCKOQAAAAANopgDAAAA0CCKOQAAAAANopjTJqU0J6WUU0r793ss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AAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADTIMmPtmFLKvRwIDLKcc+r3GLolNxkmchMGk9yEwSQ3YTCNJTedmQMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIMv0ewAs6S1veUsxdsQRRxRjX/3qV2vbzz777HGPCQCIWH311Yux9dZbb9TLe/WrX12MvepVryrG9t1339r2q666qtjn8MMPL8auvfbaYgwAGDzOzAEAAABoEMUcAAAAgAZRzAEAAABoEMUcAAAAgAZRzAEAAABoELNZDaC//Mu/LMa22267Yuwv/uIvatv/8z//s9hn4cKF3Q8MGHgf+tCHattLs91FRBx11FHF2Oc+97lxjwkm2sYbb1yMbb311qNe3oEHHliMrbXWWsXYhhtuOOp19douu+wyppjZrBiPZZddtrZ95ZVXLvZ5xzveUYzdeuutte2dZmt7/PHHi7EtttiiGCu55pprirHHHnts1Mtj6pk1a1Yx9rrXva4YO+CAA2rb582bN94hMWScmQMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA1iavI+Kk2TuMcee4xpeWeeeWZtu+kTob9KUxlvu+22xT7Pf/7zi7G99967GNt+++1r23POxT4f//jHi7FTTz21tv3+++8v9oGJcuSRR9a2f/rTny72KU2ZPBHmz5/UMPhsAAARUElEQVRfjP3mN7+pbb/wwgt7OoZLL720GPvv//7vnq6LqWf69OnF2MyZM4uxN7/5zbXtBx988LjH1K7T9ON33HFHMbbhhhuOel033XRTMXbeeecVY6W/RwsXLhz1GBhszzzzTDG22267FWPXXnttbfvOO+9c7PO73/2u63H1w1ZbbVWMvfe97y3GDj/88IkYztBwZg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIqcn76G/+5m9q21dYYYUxLe+WW26pbe80jSMMkmnTphVj++yzT217pym8b7755mLsJS95STH24he/uLZ9u+22K/bpNPX3jBkzatvXXXfdYp+U0pjWNRal1zbCFORMvk5TI++555617WOdfvySSy6pbS/tTyMivve97xVjnf7mzJ07t+txwURae+21i7Gvfe1rxdjb3va2Yuyee+6pbT/iiCOKfe67775i7DOf+Uxt+zrrrFPs02n68SuvvLIYK035/M53vrPY57DDDivG3vKWt9S2b7311sU+9rXN9OMf/7gY+9SnPlWMrbnmmrXtnabwPuqoo7ofWB8cc8wxxdgb3vCGYuy2224rxr761a+Oa0zDwJk5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2SxjorSkqpt9OpDKGdd965tr00u8bSXHjhhbXte+yxx5iWxyI55/LUQgOmybnZaZaKefPm1bZP5qxPvV5Xpxlzbr311mJs5syZo17XqaeeWowdfPDBo17eoJCbU8+//Mu/FGP77rtvbfvll19e7NNpBp4FCxbUtj/11FPFPnRHbvbfBhtsUNv+3e9+t9hnk002KcaOPfbYYuwb3/hGbfv8+fOLfUozPUZE7LXXXrXt11133ajHEFHO9YjyTJUveMELin067VN333332vZO+9pTTjmlGOs1udk7L3zhC4ux22+/fdTLu/POO4uxTjOgTqZVV121tr3TjGydjp87zaB3yCGHdD+wKWAsuenMHAAAAIAGUcwBAAAAaBDFHAAAAIAGUcwBAAAAaBDFHAAAAIAGUcwBAAAAaJBl+j2AYfbYY4/1dHnTp0/v6fJgsi1cuLAYO/PMM2vbO03V2Oupya+44opirNM04+eff35te6dpHGfNmlWM7brrrsXYn/70p9r2Jk8/znDZfvvtR93nuOOOK8buvffe8QwHBtpmm21WjF188cW17XfffXexzzbbbFOMXX/99d0PrAudcrPTNOMlhx56aDF2wgknFGPLL798bfu8efOKfd71rncVYxdddFFt+0c+8pFin8mcmpzeeeSRR4qxTtOMP//5z69tX2WVVYp9Nthgg2Ks0zFor5WOTztNP97JS1/60nGMBmfmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAg5iafAq59NJL+z0EGJdHH320GHv/+98/iSOZHB/84AeLsSOOOKIYe/jhh4uxww47bFxjgsmwyy67FGMveMELRr28TlMS33PPPcXYVVddNep1wWTbeOONi7E5c+YUY6WpkffYY49inzvuuKPrcQ2a0lTsERF//OMfi7GVVlqptv2xxx4r9um0H/7e975X2/6lL32p2IdmWnHFFYux0vTjnay88srF2EYbbVSMTebU5J3232Nx44039nR5w8aZOQAAAAANopgDAAAA0CCKOQAAAAANopgDAAAA0CCKOQAAAAANopgDAAAA0CCmJp9Cdtxxx9r2z3zmM5M8EqDdmmuuWdveaRrxFVZYoRi78sori7Gzzjqr+4FBnzzyyCPFWKfpgEv9dtttt2KfF73oRcXY2972ttr2P/zhD8U+MFFK0xyfd955xT4552LsAx/4QG17k6cfH6tOf1c6xcZiwYIFPV0eg+t1r3vdpK3rhBNOKMYuueSSYuy3v/1tbfuvf/3rMY2j03TsTD5n5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iNmsppBp06b1ewhAjU9/+tO17RtttFGxz7x584qxAw44YNxjgn765S9/WYxtscUWxdjjjz9e2/6Vr3yl2Kc0Y1VExDe/+c3a9h122KHY55lnninGYDw+9KEP1ba/4AUvKPZ597vfXYx1yrNhM3fu3GKs17NZbb/99j1dHoPruuuuK8aefPLJYmzZZZcd9bpe9rKXjSk26N7znvcUYzvvvHNt+8KFC4t9jj/++GLs3//937seV1M4MwcAAACgQRRzAAAAABpEMQcAAACgQRRzAAAAABpEMQcAAACgQRRzAAAAABok5ZzH1jGlsXXkz1772tfWtl9++eVjWt4ll1xS277rrruOaXksknNO/R5Dt+Rmf2yzzTbF2AUXXFDbvvrqqxf7HHTQQcXYqaee2v3Apji5SUTEtGnTirGzzjqrGCtNW/6mN72p2Oeiiy7qfmBDTG7WW3755Yuxhx56qLb9/PPPL/Z561vfOu4xDYP11luvGLv99ttHvby11lqrGJszZ05t+w033FDs8/a3v33UYxgruTk5Lr744mLs9a9//SSOZLh0qm2cdtppte0f/ehHi30eeeSRcY+pW2PJTWfmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAgyjmAAAAADSIYg4AAABAg5iavI96PTX5NddcU9u+7bbbFvs89thjY1rXsDGNIxERq6yySjF23XXXFWMveMELatvPPvvsYp999tmn+4ENMbnJ0qy//vrF2K9+9ava9k75vMsuu4x3SENBbtabOXNmMXbBBRfUtm+11VbFPp3eq0yck08+uRg76KCDatt32GGHYp+f//zn4x5Tt+Tm5Nhss82KsUsvvbS2fcaMGcU+t9xySzH2wAMPFGPLLLNMbXun8Q2bvffeuxg755xzJm0cpiYHAAAAmOIUcwAAAAAaRDEHAAAAoEEUcwAAAAAaRDEHAAAAoEHqL2/NpPjjH/9Y237vvfcW+3S6yvkWW2wx6j6lMQBL+uIXv1iMlWasioiYP39+bfuXvvSlcY8J6Gzu3LnF2Le//e3a9l133XWCRsOw+8hHPlKMPfTQQ7XtZqzqj7322qsY23///Yuxb33rW7XtV1xxxXiHRIP87ne/K8ZKM9StueaaxT6TOZvV7Nmzi7HNN9+8GCt55plnirFOM7tuuOGGte1rrbVWsc/jjz9ejJX6/eEPfyj2GXTOzAEAAABoEMUcAAAAgAZRzAEAAABoEMUcAAAAgAZRzAEAAABoEMUcAAAAgAYxNXkflaYFv+eee4p9Ok0zXjJt2rRR94FhtsEGG9S2H3jggcU+Oedi7PTTT69t/9WvfjW6gQE9VZo69gMf+ECxT6dpWa+99tpxj4mp7c1vfnMxdv/990/iSIZLaXrmiIi3vvWtte1nnHFGsc95551XjL3//e+vbX/66aeLfRgud9xxx6jax+Opp56qbb/mmmuKfUrHwWN10kknFWOHHnpoT9c1bJyZAwAAANAgijkAAAAADaKYAwAAANAgijkAAAAADaKYAwAAANAgijkAAAAADWJq8gH0ne98pxg79thjR728fffdtxj77Gc/O+rlwVR38skn17anlIp9fvnLXxZjxx9//LjHBEyeFVdcsRhbddVVJ3EkDJM5c+b0ewiN1ik3//Zv/7YY+8xnPlPb3mnq5iOPPLIYK00FDcPqtttu6/cQpixn5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIMo5gAAAAA0iGIOAAAAQIOYmnwAXXXVVT1d3mte85qeLg+mgqOPProY23HHHWvbH3jggWKfI444ohi7//77ux8YMGle9KIX9XsIDJk//OEPxdiWW245iSMZbKuttlpt+/bbb1/sU5piPCJio402KsZOO+202vYPf/jDxT6PP/54MQY82w033NDvIUxZzswBAAAAaBDFHAAAAIAGUcwBAAAAaBDFHAAAAIAGUcwBAAAAaBCzWQ2g66+/vqfL23TTTXu6PGiK9dZbrxg76KCDirHp06fXtp988snFPj//+c+7HxgwaV772tcWY0ceeWRt+7333lvsc/vtt497TAyvs88+uxg76qijatvf//73F/ucfvrp4x5Tv+y1117F2Cc/+cna9k4zft16663F2HbbbVeMXX311cUY0J177rmnGPvZz342iSMZLs7MAQAAAGgQxRwAAACABlHMAQAAAGgQxRwAAACABlHMAQAAAGgQxRwAAACABjE1+QB65JFHirG77767GFt77bUnYjjQWD/84Q+LsXXXXXfUy/vCF74wnuHAQPj+979f2/6DH/yg2OfnP/95MXbbbbeNe0zjtfvuuxdjJ510UjG27LLL1rZ//vOfL/aZO3du1+OCxX3lK18pxo455pja9s9+9rPFPuuvv34x9uMf/7jrcXVj8803L8a23Xbb2vZOU4K/+MUvLsbuuOOO2va3v/3txT4//elPi7H77ruvGINhtNlmmxVjyy233KiXd+WVVxZjDzzwwKiXR3ecmQMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2imAMAAADQIIo5AAAAAA2Scs5j65jS2DoyLscee2wxdvTRR9e233777cU+r3jFK4qxBx98sPuBTXE559TvMXRr2HKz09SKv/zlL4uxFVZYoRg7+OCDa9tPOeWU7gfGpJCbo1fa7z/zzDPFPvPnzy/Grr766tr2iy66aHQD68LMmTNr21/1qlcV+8yYMaMYu+aaa2rbt95662Kfp59+uhhjEblZXFcxtsMOO9S2n3rqqcU+G2200bjHNJHmzJlTjJ1wwgnF2GWXXVbbvnDhwvEO6f+3d4c4jcVBHIDbZEVTW08PgOAOTUAQBCkJGkG4QT2eAyAQKASKFFdbVUHSNEETEFwAFIa3B9g3b8trun2z/T75n0wygoHwy0tm69lNWq1Wq9frhbWq/x273W7p+3g8DnuOj4+XH2yL1dlNX+YAAAAAJCLMAQAAAEhEmAMAAACQiDAHAAAAIBFhDgAAAEAivzY9AD9zdXUV1qJrVjs7O2HP7u5uWJvNZssPBhtydHQU1jqdTliruuT38vKy0kzQZDc3N6Xvg8Eg7On3+2Ftf3//R+/rULXPj4+PYe38/Lz03cUq1qXqZ3U6nZa+7+3thT1V19oODg6WH2xF8/m89H2xWIQ9dS/qAqurulJZ52/g6+vrCtNQly9zAAAAABIR5gAAAAAkIswBAAAASESYAwAAAJCIMAcAAAAgEWEOAAAAQCJOkyfz8fER1i4vL0vfh8PhmqaBzRuNRrX6qk4yvr291R0HGu/i4qL0vdfrhT0nJydhLTqbfHZ2FvZ0Op2wdnd3F9Y+Pz9L35+fn8Oe6+vrsAYZfH19hbX39/ewdnt7u45xAP7w9PS06RG2ki9zAAAAABIR5gAAAAAkIswBAAAASESYAwAAAJCIMAcAAAAgEWEOAAAAQCLtoijqNbbb9RohoaIo2pueYVnbtpvf399hrer32+HhYVibTCYrzcS/YzehmewmNJPd5G/u7+/D2unpaen7cDgMex4eHlaeaRvU2U1f5gAAAAAkIswBAAAASESYAwAAAJCIMAcAAAAgEWEOAAAAQCLCHAAAAIBEnCaHJTjjCM1kN6GZ7CY0k92EZnKaHAAAAOA/J8wBAAAASESYAwAAAJCIMAcAAAAgEWEOAAAAQCLCHAAAAIBEhDkAAAAAiQhzAAAAABIR5gAAAAAkIswBAAAASESYAwAAAJCIMAcAAAAgkXZRFJueAQAAAIAl+TIHAAAAIBFhDgAAAEAiwhwAAACARIQ5AAAAAIkIcwAAAAASEeYAAAAAJCLMAQAAAEhEmAMAAACQiDAHAAAAIBFhDgAAAEAiwhwAAACARIQ5AAAAAIkIcwAAAAAS+Q3OJUUHHXw7RAAAAABJRU5ErkJggg==)

```text
Done
```



