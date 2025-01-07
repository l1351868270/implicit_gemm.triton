
# forward

$input \in {R}^{M \times infs}$

$weight \in {R}^{outfs \times infs}$

$bias \in {R}^{outfs}$

$linear \in {R}^{M \times outfs}$

$linear = input@weight^T + bias$

$linear_{ij} = \sum_{k=0}^{infs-1}input_{ik}.weight_{jk} + bias_{j}$

# 求导
## 通用求导

$\frac{\partial ax}{\partial x} = a$

## input求导

<p>
$\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk} + bias_{k}}{\partial input_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk}}{\partial input_{ij}}$

### $p \neq i$

<p>
$\frac{\partial linear(input)_{pq}}{\partial input_{ij}} = 0$
</p>

### $p = i$

<p>
$\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

<p>
$=\frac{\partial linear(input)_{iq}}{\partial input_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{ik}weight_{qk}}{\partial input_{ij}}$

$=\frac{\partial input_{ij}weight_{qj}}{\partial input_{ij}}$

$=weight_{qj}$

## weight求导

<p>
$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk} + bias_{k}}{\partial weight_{ij}}$

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{qk}}{\partial weight_{ij}}$

### $q \neq i$

<p>
$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}} = 0$
</p>

### $q = i$

<p>
$\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

<p>
$=\frac{\partial linear(weight)_{pi}}{\partial weight_{ij}}$
</p>

$=\frac{\partial \sum_{k=0}^{K-1} input_{pk}weight_{ik}}{\partial weight_{ij}}$

$=\frac{\partial input_{pj}weight_{ij}}{\partial weight_{ij}}$

$=input_{pj}$

## bias求导

<p>
$\frac{\partial linear(bias)_{q}}{\partial bias_{j}}$
</p>

$=\frac{\partial bias_{q}}{\partial bias_{j}}$

### $q \neq j$

<p>
$\frac{\partial linear(bias)_{q}}{\partial bias_{j}}=\frac{\partial bias_{q}}{\partial bias_{j}}=0$
</p>

### $q = j$

<p>
$\frac{\partial linear(bias)_{q}}{\partial bias_{j}}=\frac{\partial bias_{j}}{\partial bias_{i}}=1$
</p>

# 链式法则

## input链式法则

### 元素形式

<p>
$\frac{\partial f(linear(input)_{kl})}{\partial input_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(input)_{kl})}{\partial linear(input)_{pq}}.\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(input)_{pq}}{\partial input_{ij}}$
</p>

<p>
$=\sum_{q=0}^{outfs-1}df_{iq}.\frac{\partial linear(input)_{iq}}{\partial input_{ij}}$
</p>

$=\sum_{q=0}^{outfs-1}df_{iq}.weight_{qj}$

### 矩阵形式

$\frac{\partial f(input)}{\partial input}$

$=df@weight$

## weight链式法则

### 元素形式

<p>
$\frac{\partial f(linear(weight)_{kl})}{\partial weight_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(weight)_{kl})}{\partial linear(weight)_{pq}}.\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(weight)_{pq}}{\partial weight_{ij}}$
</p>

<p>
$=\sum_{p=0}^{M-1}df_{pi}.\frac{\partial linear(weight)_{pi}}{\partial weight_{ij}}$
</p>

$=\sum_{p=0}^{M-1}df_{pi}.input_{pj}$

### 矩阵形式

$\frac{\partial f(weight)}{\partial weight}$

$=df^{T}@intput$

## bias链式法则

### 元素形式

<p>
$\frac{\partial f(linear(bias)_{kl})}{\partial bias_{j}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}\frac{\partial f(linear(bias)_{kl})}{\partial linear(bias)_{pq}}.\frac{\partial linear(bias)_{q}}{\partial bias_{j}}$
</p>

<p>
$=\sum_{p=0}^{M-1}\sum_{q=0}^{outfs-1}df_{pq}.\frac{\partial linear(bias)_{q}}{\partial bias_{j}}$
</p>

<p>
$=\sum_{p=0}^{M-1}df_{pj}.\frac{\partial linear(bias)_{j}}{\partial bias_{j}}$
</p>

$=\sum_{p=0}^{M-1}df_{pj}$

### 矩阵形式

$\frac{\partial f(bias)}{\partial bias} = \sum_{p=0}^{M-1} df_{p,j \in (outfs-1)}$

# 链式法则二

$A \in {R}^{M \times K}$

$B \in {R}^{K \times N}$

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
