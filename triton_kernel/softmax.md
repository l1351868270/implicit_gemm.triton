# forward
$X \in {R}^{M \times N}$

关于列计算 $\text{softmax(X)}$

$$\text{softmax}(x_{ij}) = \frac{e^{x_{ij}}}{\sum_{k=0}^{N} e^{x_{ik}}}$$

# backward
## 求导
通用求导公式

$$\frac{\partial e^x}{\partial x} = e^x$$

$$\frac{\partial f(x)}{\partial g(x)} = \frac{\frac {\partial f(x)}{\partial x} . g(x) \ - f(x) . \frac {\partial g(x)}{\partial x}} {g(x)^2}$$

$softmax$代入求导


$$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}} =  \frac{ \frac{\partial e^{x_{kl}}}{\partial {x_{ij}}} . \sum_{p=0}^{N} e^{x_{kp}} -  e^{x_{kl}} . \frac{\partial \sum_{p=0}^{N} e^{x_{kp}}}{\partial {x_{ij}}}} {(\sum_{p=0}^{N} e^{x_{kp}})^2}$$

### 分三种情况
情况一: $i \neq k$ 由于我们是关于列求 $softmax$ ,所以 $\text{softmax}(x_{kl})$ 不是关于 $x_{ij}$ 的函数所以 

$$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}} = 0$$

情况二:  $i = k$, $l \neq j$ 此时 $e^{x_{kl}}$ 不是关于 ${x_{ij}}$ 的函数,所以 

$$\frac{\partial e^{x_{kl}}}{\partial x_{ij}} = \frac{\partial e^{x_{il}}}{\partial x_{ij}} = 0$$

同理 

$${\frac{\partial \sum_{p=0}^{N} e^{x_{kp}}}{\partial {x_{ij}}}} = {\frac{\partial \sum_{p=0}^{N} e^{x_{ip}}}{\partial {x_{ij}}}} = {\frac{\partial  e^{x_{ij}}}{\partial {x_{ij}}}} = e^{x_{ij}}$$

所以

$$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}} = \frac{\partial \text{softmax}(x_{il})}{\partial x_{ij}} = \frac{-e^{x_{il}}.e^{x_{ij}}}{(\sum_{p=0}^{N} e^{x_{ip}}) ^ 2} = -{softmax}(x_{il}) . {softmax}(x_{ij})$$

情况三:  $i = k$, $l = j$ 此时

$\frac{\partial \text{softmax}(x_{kl})}{\partial x_{ij}}$

$= \frac{\partial \text{softmax}(x_{ij})}{\partial x_{ij}}$

$= \frac{e^{x_{ij}}.\sum_{p=0}^{N} e^{x_{ip}}-e^{x_{ij}}.e^{x_{ij}}}{(\sum_{p=0}^{N} e^{x_{ip}}) ^ 2}$

$= {softmax}(x_{ij})-{softmax}(x_{ij})^2$

