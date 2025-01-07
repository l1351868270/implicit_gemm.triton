# forward
$X \in {R}^{M \times N}$

关于行计算 $\text{logsumexp(X)}$

<p>
$logsumexp(x)_i = log \sum_{j} e^{x_{ij}}$
</p>

# backward
## 求导
### 通用求导

$\frac{\partial e^x}{\partial x} = e^x$

$\frac{\partial log(x)}{\partial x} = \frac{1}{x}$

### x求导

<p>
$\frac{\partial {logsumexp}(x)_k}{\partial x_{ij}}$
</p>

$= \frac {1}{\sum_{l} e^{x_{kl}}} . \frac{\partial \sum_{l} e^{x_{kl}}}{\partial x_{ij}}$

#### $i \neq k$ 

<p>
$\frac{\partial {logsumexp}(x)_k}{\partial x_{ij}}$ = 0
</p>

#### $i = k$ 

<p>
$\frac{\partial {logsumexp}(x)_k}{\partial x_{ij}}$
</p>

$= \frac {1}{\sum_{l} e^{x_{kl}}} . \frac{\partial \sum_{l} e^{x_{kl}}}{\partial x_{ij}}$

$= \frac {1}{\sum_{l} e^{x_{il}}} . \frac{\partial \sum_{l} e^{x_{il}}}{\partial x_{ij}}$

$= \frac {1}{\sum_{l} e^{x_{il}}} .  x_{ij}$

$= \frac {x_{ij}}{\sum_{l} e^{x_{il}}}$

## 链式法则

### 元素形式

$\frac{\partial f(logsumexp(X))}{\partial x_{ij}}$

<p>
$=\sum_{k}(\frac{\partial f(logsumexp(X))}{\partial logsumexp(x)_k} . \frac{\partial logsumexp(x)_k}{\partial x_{ij}})$
</p>

<p>
$=\frac{\partial f(logsumexp(X))}{\partial logsumexp(x)_i} . \frac{\partial logsumexp(x)_i}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial f(logsumexp(X))}{\partial logsumexp(x)_i} . \frac {x_{ij}}{\sum_{l} e^{x_{il}}}$
</p>
  
$=df_i . softmax(x_{ij})$

# 参考文献
https://pytorch.org/docs/stable/generated/torch.logsumexp.html