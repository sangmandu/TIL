---
description: '210911'
---

# Attention Is All You Need

## Abstract

유망한 transduction 모델들은 인코더와 디코더를 포함하는 복잡한 반복구조나 합성곱 네트워크에 기반을 두고있다. 그 중 가장 좋은 성능을 내는 모델은 attention 메커니즘을 가지고 있는 인코더와 디코더로 이루어져있다. Transformer는 기존의 기반을 두지 않고 오로지 attention 메커니즘에만 기반을 두는 간단한 네트워크 모델이다. Transformer는 기존 모델보다 질적으로 더 우월하고 병렬작업이 훨씬 좋으며 학습하는데 적은 시간이 소요된다.

* transduction
  * 한국어로 변역하기에는 변환, 전환, 전이, 전도 등의 의미로 사용되어 번역하기 쉬운 단어는 아니다. transduce란 무언가를 다른 형태로 변환하는 것을 의미한다. 여기서는 주어진 특정 예제\(학습 데이터\)를 이용해 다른 특정 예제\(평가 데이터\)를 예측하는 것으로 이해할 수 있다.
* attention 메커니즘
  * RNN과 LSTM에 대한 이해가 있으면 더 쉽게 이해할 수 있다. RNN에서 Sequence를 계속 입력받다보면 과거 정보가 점점 희미해지게 되는데, 이 때 LSTM이 고안되었다. 기존의 RNN보다는 개선되었지만 완벽히 Long Term Dependency를 해결하지 못했고 이를 개선하기 위한 메커니즘이 attention 이다. 특정 time step에서 인코더의 전체 입력 문장을 다시 한번 참고하기 위한 방법이다.



## 1 Introduction

RNN, LSTM, GRU는 기계 번역이나 언어 모델링과 같은 transduction 문제와 시퀀스 모델링에 대해 가장 잘 해결할 수 있는 최신 모델들이다.  Recurrent 모델들은 input과 output의 위치에 따라 계산을 하고 각 위치에서 input과 이전 hidden state $$ h_{t-1} $$를 가지고 현재의 hidden state $$ h_t $$를 만들어낸다. 근데, 이러한 과정은 시퀀스의 길이가 길어지면 길어질수록 메모리 문제가 발생해서 배치 사이즈도 점점 작아지게 되고 병렬화도 할 수 없게되는 문제점이 있다. 최근 연구들은 이러한 문제에 대해 내부 인자를 조작하고 조건부 계산을 하면서 계산 효율을 높였고 그러면서 점점 모델의 성능도 증가했지만 본질적인 문제를 해결하지는 못했다.

Attention mecahnisms은 input과 output의 거리에 대한 의존도를 고려하지 않는다라는 점에서 sequence modeling과 transduction models에 다양한 task들에 있어 설득력있는 모델이되었다.

논문에서 제시하는 Transformer는 의도적으로 recurrence를 회피한다. 그대신 전체적으로 attention mechanism에 의존해서 input과 output 사이에 있는 전반적인 의존성을 해소한다. 그래서 Transformer는 병렬화에 더 적합하고 번역 기술에 있어서 조금의 학습으로도 더 좋은 품질의 결과물을 줄 수 있다.





















