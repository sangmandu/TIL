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



