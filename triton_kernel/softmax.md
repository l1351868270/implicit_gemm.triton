# forward
$X \in {R}^{M \times N}$

关于行计算 $\text{softmax(X)}$

$softmax(x_{ij}) = \frac{e^{x_{ij}}}{\sum_{k=0}^{N} e^{x_{ik}}}$

# 求导
## 通用求导

$\frac{\partial e^x}{\partial x} = e^x$

$\frac {\partial \frac{ f(x)}{ g(x)}} {\partial x} = \frac{\frac {\partial f(x)}{\partial x} . g(x) \ - f(x) . \frac {\partial g(x)}{\partial x}} {g(x)^2}$

## $x$求导


$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}} =  \frac{ \frac{\partial e^{x_{kl}}}{\partial {x_{ij}}} . \sum_{p=0}^{N} e^{x_{kp}} -  e^{x_{kl}} . \frac{\partial \sum_{p=0}^{N} e^{x_{kp}}}{\partial {x_{ij}}}} {(\sum_{p=0}^{N} e^{x_{kp}})^2}$


### $i \neq k$ 

$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}} = 0$

###  $i = k$, $l \neq j$

$\frac{\partial e^{x_{kl}}}{\partial x_{ij}} = \frac{\partial e^{x_{il}}}{\partial x_{ij}} = 0$


${\frac{\partial \sum_{p=0}^{N} e^{x_{kp}}}{\partial {x_{ij}}}} = {\frac{\partial \sum_{p=0}^{N} e^{x_{ip}}}{\partial {x_{ij}}}} = {\frac{\partial  e^{x_{ij}}}{\partial {x_{ij}}}} = e^{x_{ij}}$

$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}}$

$= \frac{\partial \text{softmax}(x_{il})}{\partial x_{ij}}$

$= \frac{-e^{x_{il}}.e^{x_{ij}}}{(\sum_{p=0}^{N} e^{x_{ip}}) ^ 2}$

$= -{softmax}(x_{il}) . {softmax}(x_{ij})$

###  $i = k$, $l = j$

$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}}$

$= \frac{\partial \text{softmax}(x_{ij})}{\partial x_{ij}}$

$= \frac{e^{x_{ij}}.\sum_{p=0}^{N} e^{x_{ip}}-e^{x_{ij}}.e^{x_{ij}}}{(\sum_{p=0}^{N} e^{x_{ip}}) ^ 2}$

$= {softmax}(x_{ij})-{softmax}(x_{ij})^2$

## 链式法则

### 元素形式

$\frac{\partial f(softmax(X))}{\partial x_{ij}}$

$=\sum_{p=0}^{M-1} \sum_{q=0}^{N-1} \frac{\partial f(softmax(X))}{\partial softmax(x_{pq})} . \frac{\partial softmax(x_{pq})}{\partial x_{ij}}$

$=\sum_{p=0}^{M-1} \sum_{q=0}^{N-1} df_{pq} . \frac{\partial softmax(x_{pq})}{\partial x_{ij}}$

$=\sum_{q=0}^{N-1} df_{iq} . \frac{\partial softmax(x_{iq})}{\partial x_{ij}}$

$=df_{ij} . (softmax(x_{ij}) - softmax(x_{ij})^2) + \sum_{q \neq j} df_{iq}.(-softmax(x_{iq}) . softmax(x_{ij}))$

$=softmax(x_{ij}) . df_{ij} - softmax(x_{ij}) . \sum_{q=0}^{N-1} df_{iq} . softmax(x_{iq})$

$=softmax(x_{ij})(df_{ij} - \sum softmax(x_{iq}).df_{iq})$

### 行形式
所以 $\frac{\partial f(softmax(X))}{\partial x_{ij}}$ 只与第i行有关, 记 $x_{i:}$ 为第 $i$ 行的任意一个元素

$\frac{\partial f(softmax(X))}{\partial x_{i:}}$

$=softmax(x_{i:})(df_{i:} - \sum_{q} softmax(x_{iq}).df_{iq})$

### 矩阵形式

$\frac{\partial f(softmax(X))}{\partial X}$

$=softmax(X)(df_{X} - sum(softmax(X).df_{X}, dim=0,keepdim=True))$
