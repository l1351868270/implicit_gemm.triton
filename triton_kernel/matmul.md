# forward
$C \in {R}^{M \times N}, A \in {R}^{M \times K}, B \in {R}^{K \times N}$

计算 $C = AB$

$c_{pq} = \sum_{k=0}^{K-1} a_{pk}b_{kq}$

# 求导
通用求导公式

$\frac{\partial ax}{\partial x} = a$

$c_{pq}$代入求导

$\frac{\partial \sum_{k=0}^{K-1} a_{pk}b_{kq}}{\partial b_{ij}}$

### 分三种情况
情况一:  $j \neq q$ , 由于 $c_{pq}$ 只和 $B$ 的第 $q$ 列有关，所以

$\frac{\partial c_{pq}}{\partial b_{ij}} = 0$

情况二: $j = q$ ,

$\frac{\partial c_{pq}}{\partial b_{ij}}$

$\frac{\partial c_{pj}}{\partial b_{ij}} = a_{pi}$

## 链式法则
$\frac{\partial f(AB)}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} \sum_{q=0}^{N-1} \frac{\partial f(AB)}{\partial c_{pq}} . \frac{\partial c_{pq}}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} \frac{\partial f(AB)}{\partial c_{pj}} . \frac{\partial c_{pj}}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} \frac{\partial f(AB)}{\partial c_{pj}} . a_{pi}$

所以

$\frac{\partial f(AB)}{\partial B} = A^T . \frac{\partial f(AB)}{\partial C}$ 

同理

$\frac{\partial f(AB)}{\partial a_{ij}}$

$=\sum_{p=0}^{N-1} \sum_{q=0}^{N-1} \frac{\partial f(AB)}{\partial c_{pq}} . \frac{\partial c_{pq}}{\partial a_{ij}}$

$=\sum_{q=0}^{N-1} \frac{\partial f(AB)}{\partial c_{iq}} . \frac{\partial c_{iq}}{\partial a_{ij}}$

$=\sum_{p=0}^{N-1} \frac{\partial f(AB)}{\partial c_{iq}} . b_{jq}$

所以

$\frac{\partial f(AB)}{\partial A} = \frac{\partial f(AB)}{\partial C}.B^T$