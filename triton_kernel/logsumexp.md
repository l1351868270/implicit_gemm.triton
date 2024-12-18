
# forward
$X \in {R}^{M \times N}$

关于行计算 $\text{logsumexp(X)}$

$\text{logsumexp}(x)_i = log \sum_{j} e^{x_{ij}}$

# backward
## 求导
通用求导公式

$\frac{\partial e^x}{\partial x} = e^x$

$\frac{\partial log(x)}{\partial x} = \frac{1}{x}$

$\text{logsumexp}(x)_i$ 求导

$\frac{\partial {logsumexp}(x)_k}{\partial x_{ij}}$

$= \frac {1}{\sum_{l} e^{x_{kl}}} . \frac{\partial \sum_{l} e^{x_{kl}}}{\partial x_{ij}}$

### 分两种情况
情况一: $i \neq k$ 

$\frac{\partial {logsumexp}(x)_k}{\partial x_{ij}}$ = 0

情况二: $i = k$ 

$\frac{\partial {logsumexp}(x)_k}{\partial x_{ij}}$

$= \frac {1}{\sum_{l} e^{x_{kl}}} . \frac{\partial \sum_{l} e^{x_{kl}}}{\partial x_{ij}}$

$= \frac {1}{\sum_{l} e^{x_{il}}} . \frac{\partial \sum_{l} e^{x_{il}}}{\partial x_{ij}}$

$= \frac {1}{\sum_{l} e^{x_{il}}} .  x_{ij}$

$= \frac {x_{ij}}{\sum_{l} e^{x_{il}}}$

## 链式法则

$\frac{\partial f(logsumexp(X))}{\partial x_{ij}}$

$=\sum_{k}(\frac{\partial f(logsumexp(X))}{\partial logsumexp(x)_k} . \frac{\partial logsumexp(x)_k}{\partial x_{ij}})$

$=\frac{\partial f(logsumexp(X))}{\partial logsumexp(x)_i} . \frac{\partial logsumexp(x)_i}{\partial x_{ij}}$

$=\frac{\partial f(logsumexp(X))}{\partial logsumexp(x)_i} . \frac {x_{ij}}{\sum_{l} e^{x_{il}}}$

$=df_i . softmax(x_{ij})$

# 参考文献
https://pytorch.org/docs/stable/generated/torch.logsumexp.html