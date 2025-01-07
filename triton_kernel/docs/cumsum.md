# forward
$X \in {R}^{M \times N}$

关于行计算 ${cumsum(X)}$

$cumsum(x)_{ij} = \sum_{l=0}^{j}{x_{il}}$

# backward
## 求导

$\frac {\partial cumsum(x)_{kl}}{x_{ij}} = \frac {\partial \sum_{q=0}^{l}{x_{kq}}} {x_{ij}}$

### 分三种情况
情况一: $k \neq i$

$\frac {\partial cumsum(x)_{kl}}{x_{ij}} = 0 $

情况二: $k \neq i, l < j$

$\frac {\partial cumsum(x)_{kl}}{x_{ij}} $

$=\frac {\partial cumsum(x)_{il}}{x_{ij}} $

$= 0 $

情况三: $k \neq i, l >= j$

$\frac {\partial cumsum(x)_{kl}}{x_{ij}} $

$=\frac {\partial cumsum(x)_{il}}{x_{ij}} $

$= 1 $

## 链式法则

$\frac{\partial f(cumsum(X))}{\partial x_{ij}}$

$=\sum_{p=0}^{M-1} \sum_{q=0}^{N-1} \frac{\partial f(cumsum(X))}{\partial cumsum(x_{pq})} . \frac{\partial cumsum(x_{pq})}{\partial x_{ij}}$

$=\sum_{p=0}^{M-1} \sum_{q=0}^{N-1} df_{pq} . \frac{\partial cumsum(x_{pq})}{\partial x_{ij}}$

$=\sum_{q=0}^{N-1} df_{iq} . \frac{\partial cumsum(x_{iq})}{\partial x_{ij}}$

$=\sum_{q=j}^{N-1} df_{iq} . \frac{\partial cumsum(x_{iq})}{\partial x_{ij}}$

$=\sum_{q=j}^{N-1} df_{iq}$

# 参考文献
# https://pytorch.org/docs/stable/generated/torch.cumsum.html