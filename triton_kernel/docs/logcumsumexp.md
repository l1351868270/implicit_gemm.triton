# forward
$X \in {R}^{M \times N}$

关于行计算 $\text{logcumsumexp(X)}$

<p>
$logcumsumexp(x)_{ij} = log (\sum_{j=0}^{i} e^{x_{ij}})$
</p>

# 求导
## 通用求导

$\frac{\partial e^x}{\partial x} = e^x$

$\frac{\partial log(x)}{\partial x} = \frac{1}{x}$

## x求导

<p>
$\frac{\partial {logcumsumexp}(x)_{kl}}{\partial x_{ij}}$
</p>

$= \frac {1}{\sum_{q=0}^{l} e^{x_{kq}}} . \frac{\partial \sum_{q=0}^{l} e^{x_{kq}}}{\partial x_{ij}}$

### $i \neq k$ 

<p>
$\frac{\partial {logcumsumexp}(x)_{kl}}{\partial x_{ij}} = 0$
</p>

### $i = k, l < j$ 

<p>
$\frac{\partial {logcumsumexp}(x)_{kl}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial {logcumsumexp}(x)_{il}}{\partial x_{ij}}$
</p>

$= \frac {1}{\sum_{q=0}^{l} e^{x_{iq}}} . \frac{\partial \sum_{q=0}^{l} e^{x_{iq}}}{\partial x_{ij}}$

$=0$

### $k \neq i, l >= j$

<p>
$\frac{\partial {logcumsumexp}(x)_{kl}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial {logcumsumexp}(x)_{il}}{\partial x_{ij}}$
</p>

$= \frac {1}{\sum_{q=0}^{l} e^{x_{iq}}} . \frac{\partial \sum_{q=0}^{l} e^{x_{iq}}}{\partial x_{ij}}$

$= \frac {1}{\sum_{q=0}^{l} e^{x_{iq}}} . e^{x_{ij}}$

$=softmax(x_{ij})$

## 链式法则

### 元素形式

$\frac{\partial f(logcumsumexp(X))}{\partial x_{ij}}$

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(\frac{\partial f(logcumsumexp(X))}{\partial logcumsumexp(x)_{kl}} . \frac{\partial logcumsumexp(x)_{kl}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(df_{kl} . \frac{\partial logcumsumexp(x)_{kl}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{l=0}^{N-1}(df_{il} . \frac{\partial logcumsumexp(x)_{il}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{l=j}^{N-1}(df_{il} . \frac{\partial logcumsumexp(x)_{il}}{\partial x_{ij}})$
</p>

$=\sum_{l=j}^{N-1}(df_{il} . softmax(x_{ij}))$

# 参考文献
https://pytorch.org/docs/stable/generated/torch.logcumsumexp.html