---
description: '210831'
---

# DAY 9 : Loss Function

![](https://lh4.googleusercontent.com/MoM3Nj2A5wlwljwnlDYCyh4c7x7FSQZu5q1Rmijb6IVDZ7WP2rVxnGqBFk3xMenpLtn_GRSLpTiMvjPd6astZK8Re1iFcEQpkZ1A1regJM-bAZFjlRx4j1mz_kduArV2A10SWd57=s0)

여러 Loss 함수들을 실험해보았다. F1은 기대했던 것과 달리 성능이 매우 안좋았고, Symmetric이 성능이 좋았다. 같이 로스 함수 실험하던 조원도 똑같은 결과가 나왔다. 다만, efficientnet-b4 에 한해서만 좋았고, b2나 b3 그리고 다른 모델에서는 focal이 성능이 더 잘나왔다. 그래서 focal 함수를 사용하기로 결정했다.

