# forward
$C \in {R}^{M \times N}, A \in {R}^{M \times K}, B \in {R}^{K \times N}$

$C = AB$

$c_{pq} = \sum_{k=0}^{K-1} a_{pk}b_{kq}$

# 求导
## 通用求导

$\frac{\partial ax}{\partial x} = a$

## A求导
###  $i \neq p$ 

$\frac{\partial c_{pq}}{\partial a_{ij}} = 0$

### $i = p$

$\frac{\partial c_{pq}}{\partial a_{ij}}=\frac{\partial c_{iq}}{\partial a_{ij}} = b_{jq}$

## B求导
###  $j \neq q$ 

$\frac{\partial c_{pq}}{\partial b_{ij}} = 0$

### $j = q$

$\frac{\partial c_{pq}}{\partial b_{ij}}=\frac{\partial c_{pj}}{\partial b_{ij}} = a_{pi}$

# 链式法则

## A链式法则

### 元素形式

$\frac{\partial f(AB)}{\partial a_{ij}}$

$=\sum_{p=0}^{N-1} \sum_{q=0}^{N-1} df_{pq} . \frac{\partial c_{pq}}{\partial a_{ij}}$

$=\sum_{q=0}^{N-1} df_{iq} . \frac{\partial c_{iq}}{\partial a_{ij}}$

$=\sum_{p=0}^{N-1} df_{iq} . b_{jq}$

### 矩阵形式

$\frac{\partial f(AB)}{\partial A} = \frac{\partial f(AB)}{\partial C}.B^T$

## B链式法则

### 元素形式

$\frac{\partial f(AB)}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} \sum_{q=0}^{N-1} \frac{\partial f(AB)}{\partial c_{pq}} . \frac{\partial c_{pq}}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} \sum_{q=0}^{N-1} df_{pq} . \frac{\partial c_{pq}}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} df_{pj} . \frac{\partial c_{pj}}{\partial b_{ij}}$

$=\sum_{p=0}^{N-1} df_{pj} . a_{pi}$

### 矩阵形式

$\frac{\partial f(AB)}{\partial B} = A^T . \frac{\partial f(AB)}{\partial C}$ 