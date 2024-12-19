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
$rmsnorm(x)_{ij} = \frac{x_{ij}}{rms(x)_i}.γ_{ij} = rrms(x)_i.x_{ij}.γ_{ij}$
</p>

$u = \frac{1}{N}\sum_{j=0}^{N-1}{x_{ij}^2}$

$rms(x)_i = \sqrt{u}$

<p>
$rmsnorm(x)_{ij} = \frac {x_{ij}}{\sqrt{u}}.γ_{ij}$
</p>

# backward
## 求导一
通用求导公式

$\frac{\partial x^n}{\partial x} = nx^{n-1}$

$\frac{\partial \sqrt{x}}{\partial x} = \frac{1}{2}x^{-\frac{1}{2}} = \frac{1}{2\sqrt{x}}$

$\frac{\partial x^2}{\partial x} = 2x$

$\frac{\partial \frac{1}{x}}{\partial x} = -\frac{1}{x^2}$

$\frac {\partial \frac{ f(x)}{ g(x)}} {\partial x} = \frac{\frac {\partial f(x)}{\partial x} . g(x) \ - f(x) . \frac {\partial g(x)}{\partial x}} {g(x)^2}$

$rms(x)_i$ 求导
<p>
$\frac {\partial rms(x)_{k}}{\partial x_{ij}}$
</p>
情况一: $k \neq i$
<p>
$\frac {\partial rms(x)_{k}}{\partial x_{ij}} = 0$
</p>
情况二: $k = i$

$\frac {\partial rms(x)_{k}}{\partial x_{ij}}$

$=\frac {\partial rms(x)_{i}}{\partial x_{ij}}$

$=\frac{1}{2\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}.\frac {\partial \frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}{\partial x_{ij}}$

$=\frac{1}{2\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}.\frac {2x_{ij}}{N}$

$=\frac{x_{ij}}{N\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}$

$=\frac{x_{ij}}{Nrms(x)_i}$


${rmsnorm(X)}$ 求导

分三种情况:
情况一: $k \neq i$

$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}} = 0$

情况二: $k = i, l \neq j$

$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$

$ = \frac{\frac {\partial x_{kl}}{\partial x_{ij}}.rms(x)_k-x_{kl}.\frac {\partial rms(x)_k}{\partial x_{ij}}}{(rms(x)_k)^2}.γ_{kl}$

$ = \frac{\frac {\partial x_{il}}{\partial x_{ij}}.rms(x)_i-x_{il}.\frac {\partial rms(x)_i}{\partial x_{ij}}}{(rms(x)_i)^2}.γ_{il}$

$ = \frac{-x_{il}.\frac {\partial rms(x)_i}{\partial x_{ij}}}{(rms(x)_i)^2}.γ_{il}$

$ = \frac{-x_{il}.x_{ij}}{N.(rms(x)_i)^3}.γ_{il}$

情况三: $k = i, l = j$

$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$

$ = \frac{\frac {\partial x_{kl}}{\partial x_{ij}}.rms(x)_k-x_{kl}.\frac {\partial rms(x)_k}{\partial x_{ij}}}{(rms(x)_k)^2}.γ_{kl}$

$ = \frac{\frac {\partial x_{ij}}{\partial x_{ij}}.rms(x)_i-x_{ij}.\frac {\partial rms(x)_i}{\partial x_{ij}}}{(rms(x)_i)^2}.γ_{ij}$


$ = (\frac {1}{rms(x)_i} + \frac{-x_{ij}.x_{ij}}{N.(rms(x)_i)^3}).γ_{ij}$
# 求导二

$rrms(x)_i$ 求导

$\frac {\partial rrms(x)_{k}}{\partial x_{ij}}$

$= \frac {\partial \frac{1}{\sqrt{\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2}}}{\partial x_{ij}}$

$=\frac {\partial (\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}}{\partial x_{ij}}$

$=-\frac{1}{2}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}} . \frac{2}{N}$

$=-\frac{x_{ij}}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}$


${rmsnorm(X)}$ 求导

分三种情况:
情况一: $k \neq i$

$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}} = 0$

情况二: $k = i, l \neq j$

$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$

$ = \frac{\partial rrms(x)_k.x_{kl}.γ_{kl}}{\partial x_{ij}}$

$ = \frac{\partial rrms(x)_i.x_{il}.γ_{il}}{\partial x_{ij}}$

$ = \frac{\partial rrms(x)_i}{\partial x_{ij}}.{x_{il}.γ_{il}}+rrms(x)_i.\frac {x_{il}}{\partial x_{ij}}$

$=-\frac{x_{ij}}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.x_{il}.γ_{il}$

情况三: $k = i, l \neq j$

$\frac {\partial rmsnorm(x)_{kl}}{\partial x_{ij}}$

$ = \frac{\partial rrms(x)_k.x_{kl}.γ_{kl}}{\partial x_{ij}}$

$ = \frac{\partial rrms(x)_i.x_{il}.γ_{il}}{\partial x_{ij}}$

$ = \frac{\partial rrms(x)_i}{\partial x_{ij}}.{x_{il}.γ_{il}}+rrms(x)_i.\frac {x_{il}}{\partial x_{ij}}$

$=-\frac{x_{ij}^2}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.γ_{ij} + (\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}.γ_{ij}$

## 链式法则一

$\frac{\partial f(rmsnorm(X))}{\partial x_{ij}}$

$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(\frac{\partial f(rmsnorm(X))}{\partial rmsnorm(x)_{kl}} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$

$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(df_{kl} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$

$=\sum_{l=0}^{N-1}(df_{il} . \frac{\partial rmsnorm(x)_{il}}{\partial x_{ij}})$

$ = df_{ij} .(\frac {1}{rms(x)_i} + \frac{-x_{ij}.x_{ij}}{N.(rms(x)_i)^3}).γ_{ij} + \sum_{l \neq j} df_{il} . \frac{-x_{il}.x_{ij}}{N.(rms(x)_i)^3}.γ_{il}$

$ = df_{ij}.\frac {1}{rms(x)_i}.γ_{ij} - \sum_{j = 0} ^ {N-1} df_{il} . \frac{x_{il}.x_{ij}}{N.(rms(x)_i)^3}.γ_{il}$

$ = df_{ij}.\frac {1}{rms(x)_i}.γ_{ij} - \sum_{j = 0} ^ {N-1} df_{il}.\frac{x_{ij}}{N.(rms(x)_i)^2}.rmsnorm(x_{il})$

$ = df_{ij}.\frac {1}{rms(x)_i}.γ_{ij} - \frac{x_{ij}}{N.(rms(x)_i)^2}.\sum_{j = 0} ^ {N-1} df_{il}.rmsnorm(x_{il})$

$ = df_{ij}.{rrms(x)_i}.γ_{ij} - \frac{x_{ij}.(rrms(x)_i)^2}{N}.\sum_{j = 0} ^ {N-1} df_{il}.rmsnorm(x_{il})$

## 链式法则二

$\frac{\partial f(rmsnorm(X))}{\partial x_{ij}}$

$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(\frac{\partial f(rmsnorm(X))}{\partial rmsnorm(x)_{kl}} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$

$=\sum_{k=0}^{M-1}\sum_{l=0}^{N-1}(df_{kl} . \frac{\partial rmsnorm(x)_{kl}}{\partial x_{ij}})$

$=\sum_{l=0}^{N-1}(df_{il} . \frac{\partial rmsnorm(x)_{il}}{\partial x_{ij}})$

$=df_{ij}.[-\frac{x_{ij}^2}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.γ_{ij} + (\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}.γ_{ij}] + \sum_{l \neq j}[-\frac{x_{ij}}{N}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.x_{il}.γ_{il}]$

$=df_{ij}.(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{1}{2}}.γ_{ij} - \frac{x_{ij}}{N}\sum_{l}(\frac{1}{N}\sum_{q=0}^{N-1}x_{iq}^2)^{-\frac{3}{2}}.df_{il}.x_{il}.γ_{il} $


# 参考
https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

https://arxiv.org/pdf/1910.07467



