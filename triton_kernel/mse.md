
# forward

$A,B \in {R}^{M \times N}$

$MSE = \frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} (A_{ij} - B_{ij})^2$

# 求导
## 通用求导

$\frac{\partial (x - y)^2}{\partial x} = 2(x-y)$

$\frac{\partial (y - x)^2}{\partial y} = 2(y-x)$

## A求导

$\frac{\partial MSE(A,B)}{\partial A_{ij}}$

$=\frac{\partial \frac{1}{M \times N} \sum_{p=0}^{M-1} \sum_{l=0}^{N-1} (A_{kl} - B_{kl})^2}{\partial A_{ij}}$

$=\frac{\partial \frac{1}{M \times N} (A_{ij} - B_{ij})^2}{\partial A_{ij}}$

$=\frac{1}{M \times N}2(A_{ij} - B_{ij})$

## B求导

$\frac{\partial MSE(A, B)}{\partial B_{ij}}$

$=\frac{\partial \frac{1}{M \times N} \sum_{p=0}^{M-1} \sum_{l=0}^{N-1} (A_{kl} - B_{kl})^2}{\partial B_{ij}}$

$=\frac{\partial \frac{1}{M \times N} (A_{ij} - B_{ij})^2}{\partial B_{ij}}$

$=\frac{1}{M \times N}2(B_{ij}-A_{ij})$

# 链式法则

## A链式法则

### 元素形式

$\frac{\partial f(MSE(A, B))}{\partial A_{ij}}$

$=\frac{\partial f(MSE(A, B))}{\partial MSE(A, B)} . \frac{\partial MSE(A, B)}{\partial A_{ij}}$

$=df.\frac{1}{M \times N}2(A_{ij} - B_{ij})$

### 矩阵形式

$=2.df.\frac{1}{M \times N}(A - B)$


## B链式法则

### 元素形式

$\frac{\partial f(MSE(A, B))}{\partial B_{ij}}$

$=\frac{\partial f(MSE(A, B))}{\partial MSE(A, B)} . \frac{\partial MSE(A, B)}{\partial B_{ij}}$

$=df.\frac{1}{M \times N}2(B_{ij}-A_{ij})$

### 矩阵形式

$=2.df.\frac{1}{M \times N}(B - A)$

