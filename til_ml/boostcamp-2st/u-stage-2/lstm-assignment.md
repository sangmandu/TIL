---
description: '210812'
---

# \[필수 과제\] LSTM Assignment

LSTM을 사용하려면 Sequential 데이터를 사용하는데 이것이 쉽지가 않다. 왜냐하면 전처리를 해야하기 때문. 그래서 문자열 데이터가 아니라 MNIST를 가지고 적용을 할것임.

MNIST로 어떻게 LSTM을?

MNIST는 28\*28 이미지인데 이를 28개의 28차원의 벡터로 볼 것이다. 28개가 입력되면 이후에 결과값이 나오는데 이를 NN을 한번 더 입력해서 분류를 할것임.

엄밀히 말하면, 28개의 sequence가 fixed 될 필요는 없음. 편의상 28개의 28 dimension짜리 벡터를 집어넣는 분류기를 만들어 볼것임.

LSTM은 텐서플로우에서 구현하기가 너무 어렵다. 파이토치는 편하다.

## Classification with LSTM

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



### Dataset and Loader

```python
from torchvision import datasets,transforms
mnist_train = datasets.MNIST(root='./data/',train=True,transform=transforms.ToTensor(),download=True)
mnist_test = datasets.MNIST(root='./data/',train=False,transform=transforms.ToTensor(),download=True)
BATCH_SIZE = 256
train_iter = torch.utils.data.DataLoader(mnist_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
test_iter = torch.utils.data.DataLoader(mnist_test,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)
print ("Done.")
```



### Define Model

```python
class RecurrentNeuralNetworkClass(nn.Module):
    def __init__(self,name='rnn',xdim=28,hdim=256,ydim=10,n_layer=3):
        super(RecurrentNeuralNetworkClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.n_layer = n_layer # K

        self.rnn = nn.LSTM(
            input_size=self.xdim,hidden_size=self.hdim,num_layers=self.n_layer,batch_first=True)
        self.lin = nn.Linear(self.hdim,self.ydim)

    def forward(self,x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(
            self.n_layer,x.size(0),self.hdim
        ).to(device)
        c0 = torch.zeros(
            self.n_layer,x.size(0),self.hdim
        ).to(device)
        # RNN
        rnn_out,(hn,cn) = self.rnn(x, (h0,c0)) 
        # x:[N x L x Q] => rnn_out:[N x L x D]
        # Linear
        out = self.lin(
            # FILL IN HERE
            ).view([-1,self.ydim]) 
        return out 

R = RecurrentNeuralNetworkClass(
    name='rnn',xdim=28,hdim=256,ydim=10,n_layer=2).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(R.parameters(),lr=1e-3)
print ("Done.")
```

* 2 : 28 \* 28 의 이미지 이므로 xdim = 28
* 11 : LSTM의 self state dimension과 output dimension이 같아야한다.
* 16-21 : layer의 수만큼, 데이터의 길이만큼 크기 설정



### Check How LSTM Works

* `N`: number of batches
* `L`: sequence lengh
* `Q`: input dim
* `K`: number of layers
* `D`: LSTM feature dimension

`Y,(hn,cn) = LSTM(X)`

* `X`: \[N x L x Q\] - `N` input sequnce of length `L` with `Q` dim.
* `Y`: \[N x L x D\] - `N` output sequnce of length `L` with `D` feature dim.
* `hn`: \[K x N x D\] - `K` \(per each layer\) of `N` final hidden state with `D` feature dim.
* `cn`: \[K x N x D\] - `K` \(per each layer\) of `N` final hidden state with `D` cell dim.

```python
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
x_numpy = np.random.rand(2,20,28) # [N x L x Q]
x_torch = torch.from_numpy(x_numpy).float().to(device)
rnn_out,(hn,cn) = R.rnn(x_torch) # forward path

print ("rnn_out:",rnn_out.shape) # [N x L x D]
print ("Hidden State hn:",hn.shape) # [K x N x D]
print ("Cell States cn:",cn.shape) # [K x N x D]
```

```text
rnn_out: torch.Size([2, 20, 256])
Hidden State hn: torch.Size([2, 2, 256])
Cell States cn: torch.Size([2, 2, 256])
```

### 

### Check parameters

```python
np.set_printoptions(precision=3)
n_param = 0
for p_idx,(param_name,param) in enumerate(R.named_parameters()):
    if param.requires_grad:
        param_numpy = param.detach().cpu().numpy() # to numpy array 
        n_param += len(param_numpy.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
        print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
print ("Total number of parameters:[%s]."%(format(n_param,',d')))
```

```text
[0] name:[rnn.weight_ih_l0] shape:[(1024, 28)].
    val:[-0.039  0.057 -0.035  0.055  0.007]
[1] name:[rnn.weight_hh_l0] shape:[(1024, 256)].
    val:[-0.008 -0.009  0.025 -0.021 -0.033]
[2] name:[rnn.bias_ih_l0] shape:[(1024,)].
    val:[-0.049  0.035  0.006  0.053 -0.049]
[3] name:[rnn.bias_hh_l0] shape:[(1024,)].
    val:[-0.021  0.046  0.001  0.032 -0.013]
[4] name:[rnn.weight_ih_l1] shape:[(1024, 256)].
    val:[ 0.036 -0.044  0.011  0.011 -0.056]
[5] name:[rnn.weight_hh_l1] shape:[(1024, 256)].
    val:[-0.056 -0.013  0.006 -0.031 -0.052]
[6] name:[rnn.bias_ih_l1] shape:[(1024,)].
    val:[ 0.034  0.052 -0.03   0.03   0.008]
[7] name:[rnn.bias_hh_l1] shape:[(1024,)].
    val:[0.015 0.055 0.047 0.028 0.026]
[8] name:[lin.weight] shape:[(10, 256)].
    val:[-0.043 -0.049 -0.044 -0.014 -0.028]
[9] name:[lin.bias] shape:[(10,)].
    val:[ 0.013 -0.033  0.009  0.031  0.026]
Total number of parameters:[821,770].
```

생각보다 많은 파라미터를 가지고 있는 것을 알 수 있다. 왜냐하면 사실상 RNN은 Dense Layer이기 때문이다.

* 각 State별 파라미터 뿐만 아니라 Gate별 파라미터도 존재한다.
* 파라미터를 줄이려면 Hidden Dimension을 줄여야 한다.



### Evaluation Function

```python
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3)
x_numpy = np.random.rand(3,10,28) # [N x L x Q]
x_torch = torch.from_numpy(x_numpy).float().to(device)
y_torch = R.forward(x_torch) # [N x 1 x R] where R is the output dim.
y_numpy = y_torch.detach().cpu().numpy() # torch tensor to numpy array
# print ("x_torch:\n",x_torch)
# print ("y_torch:\n",y_torch)
print ("x_numpy %s"%(x_numpy.shape,))
print ("y_numpy %s"%(y_numpy.shape,))
```



### Initial Evaluation

```python
train_accr = func_eval(R,train_iter,device)
test_accr = func_eval(R,test_iter,device)
print ("train_accr:[%.3f] test_accr:[%.3f]."%(train_accr,test_accr))
```

```text
train_accr:[0.100] test_accr:[0.103].
```



### Train

```python
print ("Start training.")
R.train() # to train mode 
EPOCHS,print_every = 5,1
for epoch in range(EPOCHS):
    loss_val_sum = 0
    for batch_in,batch_out in train_iter:
        # Forward path
        y_pred = R.forward(batch_in.view(-1,28,28).to(device))
        loss_out = loss(y_pred,batch_out.to(device))
        # Update
        optm.zero_grad() # reset gradient 
        loss_out.backward() # backpropagate
        optm.step() # optimizer update
        loss_val_sum += loss_out
    loss_val_avg = loss_val_sum/len(train_iter)
    # Print
    if ((epoch%print_every)==0) or (epoch==(EPOCHS-1)):
        train_accr = func_eval(R,train_iter,device)
        test_accr = func_eval(R,test_iter,device)
        print ("epoch:[%d] loss:[%.3f] train_accr:[%.3f] test_accr:[%.3f]."%
               (epoch,loss_val_avg,train_accr,test_accr))
print ("Done")
```

```text
Start training.
epoch:[0] loss:[0.689] train_accr:[0.938] test_accr:[0.940].
epoch:[1] loss:[0.145] train_accr:[0.968] test_accr:[0.967].
epoch:[2] loss:[0.088] train_accr:[0.979] test_accr:[0.975].
epoch:[3] loss:[0.062] train_accr:[0.986] test_accr:[0.981].
epoch:[4] loss:[0.051] train_accr:[0.989] test_accr:[0.983].
Done
```

굉장히 성능이 잘 나왔다. 오버피팅도 일어나지 않는다.



### Test

```python
n_sample = 25
sample_indices = np.random.choice(len(mnist_test.targets),n_sample,replace=False)
test_x = mnist_test.data[sample_indices]
test_y = mnist_test.targets[sample_indices]
with torch.no_grad():
    R.eval() # to evaluation mode 
    y_pred = R.forward(test_x.view(-1,28,28).type(torch.float).to(device)/255.)
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

![](../../../.gitbook/assets/image%20%28851%29.png)



