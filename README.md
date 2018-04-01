# adversarial-discriminative-domain-adaptation-for-hashing
An implementation of "[adversarial discriminative domain adaptation](https://arxiv.org/abs/1702.05464)". The original work targets classification task, but this project implements it for hashing purpose. The classification loss is replaced with pairwise similarity loss

## How to run
Make sure pytorch, and the latest version of [ml_toolkit](https://github.com/MarkusZhang/ml_toolkit) are installed
Then run `run.py`

## Test results
Training on MNIST(mini), MNIST_M(10% mini), Testing on MNIST_M

- Before domain adaptation
  - Precision@2 = 39.83%
  - MAP@2 = 42.84%
  
- After domain adaptation
  - Precision@2 = 68.00%
  - MAP@2 = 70.36%
