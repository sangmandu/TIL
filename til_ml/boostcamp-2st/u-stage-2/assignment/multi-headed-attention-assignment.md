# \[필수 과제\] Multi-headed Attention Assignment

## Multi-Headed Attention

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

### Scaled Dot-Product Attention \(SDPA\)

![](../../../../.gitbook/assets/image%20%28889%29.png)

Scaled Dot Attention은 self\(single\)-attention이고, 이후에 multi attention이 나올 것임

```python
class ScaledDotProductAttention(nn.Module):
    def forward(self,Q,K,V,mask=None):
        d_K = K.size()[-1] # key dimension
        scores = Q.matmul(K.transpose(-2,-1)) / np.sqrt(d_K)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        attention = F.softmax(scores,dim=-1)
        out = attention.matmul(V)
        return out,attention
```

* 2 : Q, K, V 벡터를 입력받는다. 정확히는 이것이 Batch가 되어서 들어오게 된다.
* 3 : Key dimension을 찾음. 왜? 점수를 square값으로 나눠야 하니까
* 4 : 점수를 계산
* 6 : softmax로 attention값 구하기
* 7 : Value벡터와 attention곱 구하기

```python
# Demo run of scaled dot product attention 
SPDA = ScaledDotProductAttention()
n_batch,d_K,d_V = 3,128,256 # d_K(=d_Q) does not necessarily be equal to d_V
n_Q,n_K,n_V = 30,50,50
Q = torch.rand(n_batch,n_Q,d_K)
K = torch.rand(n_batch,n_K,d_K)
V = torch.rand(n_batch,n_V,d_V)
out,attention = SPDA.forward(Q,K,V,mask=None)
def sh(x): return str(x.shape)[11:-1] 
print ("SDPA: Q%s K%s V%s => out%s attention%s"%
       (sh(Q),sh(K),sh(V),sh(out),sh(attention)))
```

```text
SDPA: Q[3, 30, 128] K[3, 50, 128] V[3, 50, 256] => out[3, 30, 256] attention[3, 30, 50]
```

* 3
  * n\_batch : 단어의 개수
  * d\_K : K벡터의 차원
  * d\_V : V벡터의 차원 
  * V벡터는 K벡터와 차원이 달라도 된다.
* 4: 각각의 벡터의 총 개수
  * 쿼리벡터의 개수와 키벡터의 개수가 달라도 된다. 개수가 달라도 서로의 interaction을 계산할 수 있다.

```python
# It supports 'multi-headed' attention
n_batch,n_head,d_K,d_V = 3,5,128,256
n_Q,n_K,n_V = 30,50,50 # n_K and n_V should be the same
Q = torch.rand(n_batch,n_head,n_Q,d_K)
K = torch.rand(n_batch,n_head,n_K,d_K)
V = torch.rand(n_batch,n_head,n_V,d_V)
out,attention = SPDA.forward(Q,K,V,mask=None)
# out: [n_batch x n_head x n_Q x d_V]
# attention: [n_batch x n_head x n_Q x n_K] 
def sh(x): return str(x.shape)[11:-1] 
print ("(Multi-Headed) SDPA: Q%s K%s V%s => out%s attention%s"%
       (sh(Q),sh(K),sh(V),sh(out),sh(attention)))
```

```text
(Multi-Headed) SDPA: Q[3, 5, 30, 128] K[3, 5, 50, 128] V[3, 5, 50, 256] => out[3, 5, 30, 256] attention[3, 5, 30, 50]
```

* 멀티헤드에서는 각각의 벡터를 몇개할지가 2번째 인자자리에 추가되었다.



### Multi-Headed Attention \(MHA\)

![](../../../../.gitbook/assets/image%20%28894%29.png)

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self,d_feat=128,n_head=5,actv=F.relu,USE_BIAS=True,dropout_p=0.1,device=None):
        """
        :param d_feat: feature dimension
        :param n_head: number of heads
        :param actv: activation after each linear layer
        :param USE_BIAS: whether to use bias
        :param dropout_p: dropout rate
        :device: which device to use (e.g., cuda:0)
        """
        super(MultiHeadedAttention,self).__init__()
        if (d_feat%n_head) != 0:
            raise ValueError("d_feat(%d) should be divisible by b_head(%d)"%(d_feat,n_head)) 
        self.d_feat = d_feat
        self.n_head = n_head
        self.d_head = self.d_feat // self.n_head
        self.actv = actv
        self.USE_BIAS = USE_BIAS
        self.dropout_p = dropout_p # prob. of zeroed

        self.lin_Q = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)
        self.lin_K = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)
        self.lin_V = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)
        self.lin_O = nn.Linear(self.d_feat,self.d_feat,self.USE_BIAS)

        self.dropout = nn.Dropout(p=self.dropout_p)
```

* 2 : dropout이 multihead 인자로 들어가게된다. 논문에는 설명이 안되어있는데 모든 코드에 쓴다
* 21-24 : Q, K ,V 벡터를 구하는 신경망을 구성하고 나오는 결과값을 가공해주는 Output도 정의해준다.

```python
    def forward(self,Q,K,V,mask=None):
        """
        :param Q: [n_batch, n_Q, d_feat]
        :param K: [n_batch, n_K, d_feat]
        :param V: [n_batch, n_V, d_feat] <= n_K and n_V must be the same 
        :param mask: 
        """
        n_batch = Q.shape[0]
        Q_feat = self.lin_Q(Q) 
        K_feat = self.lin_K(K) 
        V_feat = self.lin_V(V)
        # Q_feat: [n_batch, n_Q, d_feat]
        # K_feat: [n_batch, n_K, d_feat]
        # V_feat: [n_batch, n_V, d_feat]

        # Multi-head split of Q, K, and V (d_feat = n_head*d_head)
        Q_split = Q_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        K_split = K_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        V_split = V_feat.view(n_batch, -1, self.n_head, self.d_head).permute(0, 2, 1, 3)
        # Q_split: [n_batch, n_head, n_Q, d_head]
        # K_split: [n_batch, n_head, n_K, d_head]
        # V_split: [n_batch, n_head, n_V, d_head]

        # Multi-Headed Attention
        d_K = K.size()[-1] # key dimension
        scores = torch.matmul(Q_split, K_split.permute(0, 1, 3, 2)) / np.sqrt(d_K)
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        attention = torch.softmax(scores,dim=-1)
        x_raw = torch.matmul(self.dropout(attention),V_split) # dropout is NOT mentioned in the paper
        # attention: [n_batch, n_head, n_Q, n_K]
        # x_raw: [n_batch, n_head, n_Q, d_head]

        # Reshape x
        x_rsh1 = x_raw.permute(0,2,1,3).contiguous()
        # x_rsh1: [n_batch, n_Q, n_head, d_head]
        x_rsh2 = x_rsh1.view(n_batch,-1,self.d_feat)
        # x_rsh2: [n_batch, n_Q, d_feat]

        # Linear
        x = self.lin_O(x_rsh2)
        # x: [n_batch, n_Q, d_feat]
        out = {'Q_feat':Q_feat,'K_feat':K_feat,'V_feat':V_feat,
               'Q_split':Q_split,'K_split':K_split,'V_split':V_split,
               'scores':scores,'attention':attention,
               'x_raw':x_raw,'x_rsh1':x_rsh1,'x_rsh2':x_rsh2,'x':x}
        return out
```

* 9-11 : 각각의 벡터를 신경망에 넣는다
* 17-19 : 그리고 이 벡터를 조각조각 내준다.

```python
# Self-Attention Layer
n_batch = 128
n_src   = 32
d_feat  = 200
n_head  = 5
src = torch.rand(n_batch,n_src,d_feat)
self_attention = MultiHeadedAttention(
    d_feat=d_feat,n_head=n_head,actv=F.relu,USE_BIAS=True,dropout_p=0.1,device=device)
out = self_attention.forward(src,src,src,mask=None)

Q_feat,K_feat,V_feat = out['Q_feat'],out['K_feat'],out['V_feat']
Q_split,K_split,V_split = out['Q_split'],out['K_split'],out['V_split']
scores,attention = out['scores'],out['attention']
x_raw,x_rsh1,x_rsh2,x = out['x_raw'],out['x_rsh1'],out['x_rsh2'],out['x']

# Print out shapes
def sh(_x): return str(_x.shape)[11:-1] 
print ("Input src:\t%s  \t= [n_batch, n_src, d_feat]"%(sh(src)))
print ()
print ("Q_feat:   \t%s  \t= [n_batch, n_src, d_feat]"%(sh(Q_feat)))
print ("K_feat:   \t%s  \t= [n_batch, n_src, d_feat]"%(sh(K_feat)))
print ("V_feat:   \t%s  \t= [n_batch, n_src, d_feat]"%(sh(V_feat)))
print ()
print ("Q_split:  \t%s  \t= [n_batch, n_head, n_src, d_head]"%(sh(Q_split)))
print ("K_split:  \t%s  \t= [n_batch, n_head, n_src, d_head]"%(sh(K_split)))
print ("V_split:  \t%s  \t= [n_batch, n_head, n_src, d_head]"%(sh(V_split)))
print ()
print ("scores:   \t%s  \t= [n_batch, n_head, n_src, n_src]"%(sh(scores)))
print ("attention:\t%s  \t= [n_batch, n_head, n_src, n_src]"%(sh(attention)))
print ()
print ("x_raw:    \t%s  \t= [n_batch, n_head, n_src, d_head]"%(sh(x_raw)))
print ("x_rsh1:   \t%s  \t= [n_batch, n_src, n_head, d_head]"%(sh(x_rsh1)))
print ("x_rsh2:   \t%s  \t= [n_batch, n_src, d_feat]"%(sh(x_rsh2)))
print ()
print ("Output x: \t%s  \t= [n_batch, n_src, d_feat]"%(sh(x)))
```

```text
Input src:	[128, 32, 200]  	= [n_batch, n_src, d_feat]

Q_feat:   	[128, 32, 200]  	= [n_batch, n_src, d_feat]
K_feat:   	[128, 32, 200]  	= [n_batch, n_src, d_feat]
V_feat:   	[128, 32, 200]  	= [n_batch, n_src, d_feat]

Q_split:  	[128, 5, 32, 40]  	= [n_batch, n_head, n_src, d_head]
K_split:  	[128, 5, 32, 40]  	= [n_batch, n_head, n_src, d_head]
V_split:  	[128, 5, 32, 40]  	= [n_batch, n_head, n_src, d_head]

scores:   	[128, 5, 32, 32]  	= [n_batch, n_head, n_src, n_src]
attention:	[128, 5, 32, 32]  	= [n_batch, n_head, n_src, n_src]

x_raw:    	[128, 5, 32, 40]  	= [n_batch, n_head, n_src, d_head]
x_rsh1:   	[128, 32, 5, 40]  	= [n_batch, n_src, n_head, d_head]
x_rsh2:   	[128, 32, 200]  	= [n_batch, n_src, d_feat]

Output x: 	[128, 32, 200]  	= [n_batch, n_src, d_feat]
```

