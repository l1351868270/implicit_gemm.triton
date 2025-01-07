# forward
$X \in {R}^{M \times N}$

关于行计算 ${rmsnorm(X)}$

<p>
  $rms(x)_i=\sqrt{\frac{1}{N}\sum_{j=0}^{N-1}x_{ij}^2}$
</p>

<p>
$rrms(x)_i = \frac{1}{rms(x)_i} = \frac{1}{\sqrt{\frac{1}{N}\sum_{j=0}^{N-1}x_{ij}^2}}$
</p>

<p>
$rmsnorm(x)_{ij} = \frac{x_{ij}}{rms(x)_i}.γ_{j} = rrms(x)_i.x_{ij}.γ_{j}$
</p>

$u = \frac{1}{N}\sum_{j=0}^{N-1}{x_{ij}^2}$

$rms(x)_i = \sqrt{u}$

<p>
$rmsnorm(x)_{ij} = \frac {x_{ij}}{\sqrt{u}}.γ_{j}$
</p>

# 求导

## 通用求导

$\frac{\partial x^n}{\partial x} = nx^{n-1}$

$\frac{\partial \sqrt{x}}{\partial x} = \frac{1}{2}x^{-\frac{1}{2}} = \frac{1}{2\sqrt{x}}$

$\frac{\partial x^2}{\partial x} = 2x$

$\frac{\partial \frac{1}{x}}{\partial x} = -\frac{1}{x^2}$

$\frac {\partial \frac{ f(x)}{ g(x)}} {\partial x} = \frac{\frac {\partial f(x)}{\partial x} . g(x) \ - f(x) . \frac {\partial g(x)}{\partial x}} {g(x)^2}$

## x求导

## x求导一

### $rms(x)_i$ 求导

<p>
$\frac {\partial rms(x)_{k}}{\partial x_{ij}}$
</p>

####  $k \neq i$

<p>
$\frac {\partial rms(x)_{k}}{\partial x_{ij}} = 0$
</p>

#### $k = i$

<p>
$\frac {\partial rms(x)_{k}}{\partial x_{ij}}$
</p>

<p>
$=\frac {\partial rms(x)_{i}}{\partial x_{ij}}$
</p>

<p>
$=\frac{1}{2\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}.\frac {\partial \frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}{\partial x_{ij}}$
</p>

<p>
$=\frac{1}{2\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}.\frac {2x_{ij}}{N}$
</p>

<p>
$=\frac{x_{ij}}{N\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}$
</p>

<p>
$=\frac{x_{ij}}{N.rms(x)_i}$
</p>

### ${rmsnorm(X)}$ 求导

#### $k \neq i$

<p>
$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}} = 0$
</p>

#### $k = i, l \neq j$

<p>
$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\frac {\partial x_{kl}}{\partial x_{ij}}.rms(x)_k-x_{kl}.\frac {\partial rms(x)_k}{\partial x_{ij}}}{(rms(x)_k)^2}.γ_{l}$
</p>

<p>
$=\frac{\frac {\partial x_{il}}{\partial x_{ij}}.rms(x)_i-x_{il}.\frac {\partial rms(x)_i}{\partial x_{ij}}}{(rms(x)_i)^2}.γ_{l}$
</p>

<p>
$=\frac{-x_{il}.\frac {\partial rms(x)_i}{\partial x_{ij}}}{(rms(x)_i)^2}.γ_{l}$
</p>

<p>
$=\frac{-x_{il}.x_{ij}}{N.(rms(x)_i)^3}.γ_{l}$
</p>

#### $k = i, l = j$

<p>
$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\frac {\partial x_{kl}}{\partial x_{ij}}.rms(x)_k-x_{kl}.\frac {\partial rms(x)_k}{\partial x_{ij}}}{(rms(x)_k)^2}.γ_{l}$
</p>

<p>
$=\frac{\frac {\partial x_{ij}}{\partial x_{ij}}.rms(x)_i-x_{ij}.\frac {\partial rms(x)_i}{\partial x_{ij}}}{(rms(x)_i)^2}.γ_{j}$
</p>

<p>
$=(\frac {1}{rms(x)_i} + \frac{-x_{ij}.x_{ij}}{N.(rms(x)_i)^3}).γ_{j}$
</p>

## x求导二

### $rrms(x)_i$ 求导

<p>
$\frac {\partial rrms(x)_{k}}{\partial x_{ij}}$
</p>

$=\frac {\partial \frac{1}{\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}}{\partial x_{ij}}$

$=\frac {\partial (\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}}{\partial x_{ij}}$

$=-\frac{1}{2}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}} . \frac{2}{N}$

$=-\frac{x_{ij}}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}$


### ${rmsnorm(X)}$ 求导

#### $k \neq i$

<p>
$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}} = 0$
</p>

#### $k = i, l \neq j$

<p>
$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial rrms(x)_k.x_{kl}.γ_{l}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial rrms(x)_i.x_{il}.γ_{l}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial rrms(x)_i}{\partial x_{ij}}.{x_{il}.γ_{l}}+rrms(x)_i.\frac {x_{il}}{\partial x_{ij}}$
</p>

<p>
$=-\frac{x_{ij}}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.x_{il}.γ_{l}$
</p>

#### $k = i, l \neq j$

<p>
$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial rrms(x)_k.x_{kl}.γ_{l}}{\partial x_{ij}}$
</p>

<p>
$=\frac{\partial rrms(x)_i.x_{il}.γ_{l}}{\partial x_{ij}}$
<p>

<p>
$=\frac{\partial rrms(x)_i}{\partial x_{ij}}.{x_{il}.γ_{l}}+rrms(x)_i.\frac {x_{il}}{\partial x_{ij}}$
</p>

<p>
$=-\frac{x_{ij}^2}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.γ_{j} + (\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}.γ_{j}$
</p>

## γ求导

<p>
$\frac {\partial rmsnorm(γ)_{kl}}{\partial γ_{j}}$
</p>

### $l \neq j$

<p>
$\frac {\partial rmsnorm(γ)_{kl}}{\partial γ_{j}} = 0$
</p>

### $l = j$

<p>
$\frac {\partial rmsnorm(γ)_{kl}}{\partial γ_{j}}$
</p>

<p>
$=\frac {\partial rmsnorm(γ)_{kj}}{\partial γ_{j}}$
</p>

<p>
$=(\frac{1}{N}\sum_{q=0}^{N-1}x_{kq}^2)^{-\frac{1}{2}}.x_{kj}$
</p>

# 链式法则

## x链式法则一

### 元素形式

<p>
$\frac{\partial f(rmsnorm(X))}{\partial x_{ij}}$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(\frac{\partial f(rmsnorm(X))}{\partial rmsnorm(x)_{kl}} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(df_{kl} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{l=0}^{N-1}(df_{il} . \frac{\partial rmsnorm(x)_{il}}{\partial x_{ij}})$
</p>

<p>
$=df_{ij} .(\frac {1}{rms(x)_i} + \frac{-x_{ij}.x_{ij}}{N.(rms(x)_i)^3}).γ_{j} + \sum_{l \neq j} df_{il} . \frac{-x_{il}.x_{ij}}{N.(rms(x)_i)^3}.γ_{l}$
</p>

<p>
$=df_{ij}.\frac {1}{rms(x)_i}.γ_{j} - \sum_{j = 0} ^ {N-1} df_{il} . \frac{x_{il}.x_{ij}}{N.(rms(x)_i)^3}.γ_{l}$
</p>

<p>
$=df_{ij}.\frac {1}{rms(x)_i}.γ_{j} - \sum_{j = 0} ^ {N-1} df_{il}.\frac{x_{ij}}{N.(rms(x)_i)^2}.rmsnorm(x_{il})$
</p>

<p>
$=df_{ij}.\frac {1}{rms(x)_i}.γ_{j} - \frac{x_{ij}}{N.(rms(x)_i)^2}.\sum_{j = 0} ^ {N-1} df_{il}.rmsnorm(x_{il})$
</p>

<p>
$=df_{ij}.{rrms(x)_i}.γ_{j} - \frac{x_{ij}.(rrms(x)_i)^2}{N}.\sum_{j = 0} ^ {N-1} df_{il}.rmsnorm(x_{il})$
</p>

## x链式法则二

### 元素形式

<p>
$\frac{\partial f(rmsnorm(X))}{\partial x_{ij}}$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(\frac{\partial f(rmsnorm(X))}{\partial rmsnorm(x)_{kl}} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(df_{kl} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$
</p>

<p>
$=\sum_{l=0}^{N-1}(df_{il} . \frac{\partial rmsnorm(x)_{il}}{\partial x_{ij}})$
</p>

<p>
$=df_{ij}.[-\frac{x_{ij}^2}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.γ_{j} + (\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}.γ_{j}] + \sum_{l \neq j}[-\frac{x_{ij}}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.x_{il}.γ_{l}]$
</p>

<p>
$=df_{ij}.(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}.γ_{j} - \frac{x_{ij}}{N}\sum_{l}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.df_{il}.x_{il}.γ_{l} $
</p>

<p>
$=df_{ij}.rrms(x)_i.γ_{j} - \frac{x_{ij}}{N}\sum_{l}rrsm(x)_i^{3}.df_{il}.x_{il}.γ_{l} $
</p>

### 矩阵形式

$\frac{\partial f(rmsnorm(X))}{\partial X}$

$=df.rrms(X).γ-\frac{X}{N}.sum(rrms(X).df.X.γ, dim=-1, keepdim=True)$

## γ链式法则

### 元素形式

<p>
$\frac{\partial f(rmsnorm(γ))}{\partial γ_{j}}$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(\frac{\partial f(rmsnorm(γ))}{\partial rmsnorm(γ)_{kl}} . \frac{\partial rmsnorm(γ)_{kl}}{\partial γ_{j}})$
</p>

<p>
$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}df_{kl} . \frac{\partial rmsnorm(γ)_{kl}}{\partial γ_{j}}$
</p>


<p>
$=\sum_{k=0}^{M-1}df_{kj} . \frac{\partial rmsnorm(γ)_{kj}}{\partial γ_{j}}$
</p>


<p>
$=\sum_{k=0}^{M-1}df_{kj} . (\frac{1}{N}\sum_{q=0}^{N-1}x_{kq}^2)^{-\frac{1}{2}}.x_{kj}$
</p>


# 参考
https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

https://arxiv.org/pdf/1910.07467



