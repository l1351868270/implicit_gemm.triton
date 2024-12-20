
# forward

$input \in {R}^{M \times infs}$

$weight \in {R}^{outfs \times infs}$

$bias \in {R}^{outfs}$

$linear \in {R}^{M \times outfs}$

$linear = input@weight^T + bias$

$linear_{ij} = \sum_{k=0}^{infs-1}input_{ik}.weight_{jk} + bias_{i}$

# 求导
## 通用求导

$\frac{\partial ax}{\partial x} = a$

## input求导

$\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk} + bias_{p}}{\partial input_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk}}{\partial input_{ij}}$

### $p \neq i$

$\frac{\partial linear(input)_{pq}}{\partial input_{ij}} = 0$

### $p = i$

$\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$

$=\frac{\partial linear(input)_{iq}}{\partial input_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{ik}weight_{qk}}{\partial input_{ij}}$

$=\frac{\partial input_{ij}weight_{qj}}{\partial input_{ij}}$

$=weight_{qj}$

## weight求导

$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk} + bias_{p}}{\partial weight_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk}}{\partial weight_{ij}}$

### $q \neq i$

$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}} = 0$

### $q = i$

$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$

$=\frac{\partial linear(weight)_{pi}}{\partial weight_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{ik}}{\partial weight_{ij}}$

$=\frac{\partial input_{pj}weight_{ij}}{\partial weight_{ij}}$

$=input_{pj}$

## bias求导

$\frac{\partial linear(bias)_{p}}{\partial bias_{i}}$

$=\frac{\partial bias_{p}}{\partial bias_{i}}$

### $p \neq i$

$\frac{\partial linear(bias)_{p}}{\partial bias_{i}}=\frac{\partial bias_{p}}{\partial bias_{i}}=0$

### $p = i$

$\frac{\partial linear(bias)_{p}}{\partial bias_{i}}=\frac{\partial bias_{p}}{\partial bias_{i}}=1$

# 链式法则

## input链式法则

### 元素形式

$\frac{\partial f(linear(input)_{kl})}{\partial input_{ij}}$

$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(input)_{kl})}{\partial linear(input)_{pq}}.\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$

$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$

$=\sum_{q=0}^{outfs-1}df_{iq}.\frac{\partial linear(input)_{iq}}{\partial input_{ij}}$

$=\sum_{q=0}^{outfs-1}df_{iq}.weight_{qj}$

矩阵形式

$\frac{\partial f(input)}{\partial input}$

$=df@weight$

## weight链式法则

$\frac{\partial f(linear(weight)_{kl})}{\partial weight_{ij}}$

$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(weight)_{kl})}{\partial linear(weight)_{pq}}.\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$

$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$

$=\sum_{p=0}^{M-1}df_{pi}.\frac{\partial linear(weight)_{pi}}{\partial weight_{ij}}$

$=\sum_{p=0}^{M-1}df_{pi}.input_{pj}$

矩阵形式

$\frac{\partial f(weight)}{\partial weight}$

$=df^{T}@intput$

## bias链式法则

$\frac{\partial f(linear(bias)_{kl})}{\partial bias_{i}}$

$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(bias)_{kl})}{\partial linear(bias)_{pq}}.\frac{\partial linear(bias)_{p}}{\partial bias_{i}}$

$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(bias)_{p}}{\partial bias_{i}}$

$=\sum_{q=0}^{outfs-1}df_{iq}.\frac{\partial linear(bias)_{i}}{\partial bias_{i}}$

$=\sum_{q=0}^{outfs-1}df_{iq}$

矩阵形式

$\frac{\partial f(bias)}{\partial bias} = \sum_{q=0}^{outfs-1} df_{:q}$

# 链式法则二

$A \in {R}^{M \times K}$

$B \in {R}^{B \times N}$

$C = AB$

矩阵乘法链式法则(matmul.md)

$\frac{\partial f(A)}{\partial A} = \frac{\partial f(C)}{\partial C}@B^T$

$\frac{\partial f(B)}{\partial B} = A^T @ \frac{\partial f(C)}{\partial C}$ 

$A = input$

$B = weight^T$

$\frac{\partial f(input)}{\partial input} = \frac{\partial f(linear)}{\partial linear}@(weight^T)^T = \frac{\partial f(linear)}{\partial linear}@weight$

$\frac{\partial f(weight^T)}{\partial weight^T} = A^T @ \frac{\partial f(linear)}{\partial C}$ 

$\frac{\partial f(weight)}{\partial weight} = (\frac{\partial f(linear)}{\partial C}) ^ T @ A$ 

# 参考文献
[pytorch linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
