# Spiderweb central configurations

In this example, we will prove the existence and local uniqueness of a [central configuration](https://en.wikipedia.org/wiki/Central_configuration) in the ``N``-body problem.

More precisely, we will look at configurations given by ``N = n \times \ell+1`` masses located at the intersection points of ``\ell`` concurrent equidistributed half-lines with ``n`` circles and a central mass ``m_0``. The ``\ell`` masses on the ``i``-th circle are equal to a positive constant ``m_i`` and we allow the particular case ``m_0 = 0``. These central configurations are called *spiderweb central configurations*.[^1]

[^1]: O. Hénot and C. Rousseau, [Spiderweb central configurations](https://doi.org/10.1007/s12346-019-00330-y), *Qualitative Theory of Dynamical Systems*, **18** (2019), 1135–1160.

The ``N``-body problem consists in describing the positions ``\mathbf{r}_1(t),\dots,\mathbf{r}_N(t)`` of ``N`` masses ``m_1,\dots,m_N`` interacting through Newton's gravitational law:

```math
m_i \frac{d^2}{dt^2} \mathbf{r}_i
=
-\sum_{j\neq i} \frac{G m_i m_j(\mathbf{r}_i - \mathbf{r}_j)}{| \mathbf{r}_i - \mathbf{r}_j |^3}
=
-\frac{\partial}{\partial \mathbf{r}_i} U (\mathbf{r}),
\qquad
U (\mathbf{r})
:=
-\sum_{i<j} \frac{G m_i m_j}{|\mathbf{r}_i - \mathbf{r}_j|},
```

for ``i = 1,\dots,N``, with ``\mathbf{r} \in \{ (\mathbf{r}_1,\dots,\mathbf{r}_N) \in \mathbb{R}^{3N} \, : \, \mathbf{r}_i \neq \mathbf{r}_j , \, i \neq j\}``, where ``G`` denotes the gravitational constant.

In the following, we fix the centre of mass at the origin and scale ``G = 1``. Moreover, due to the symmetries of a spiderweb central configuration, it is sufficient to consider the accelerations of the ``n`` bodies on the positive horizontal axis, and the numbers ``r_1, \dots, r_n`` also denote the positions of the masses on this semi-axis.

The configuration of ``N`` bodies is *central* at some time ``t^*`` if ``\frac{d^2}{dt^2}\mathbf{r}(t^*) = \lambda \mathbf{r}(t^*)`` for some common ``\lambda``. It is easy to see that ``\lambda`` is a strictly negative value given by ``\lambda = U(\mathbf{r})/I(\mathbf{r}) < 0`` where ``I := \sum_{i = 1}^N m_i |\mathbf{r}_i(\mathbf{r})|^2`` is the moment of inertia. Essentially, the value of ``\lambda`` scales the system and can be chosen arbitrarily.

Then, the original system of ODEs reduces to the following system of equations in ``\mathbb{R}^n``:

```math
\lambda r_i
=
-\sum_{k=1}^{\ell-1} \frac{m_i}{2^{3/2}r_i^2(1 - \cos(\theta_k))^{1/2}} -\frac{m_0}{r_i^2} - \sum_{\begin{smallmatrix}j=1\\j\neq i \end{smallmatrix}}^n \sum_{k=0}^{\ell-1} \frac{m_j(r_i - r_j \cos(\theta_k))}{(r_i^2 + r_j^2 - 2 r_i r_j \cos(\theta_k))^{3/2}},
```

for ``i = 1, \dots, n``, with ``\theta_k := \frac{2\pi k}{\ell}`` and ``r = (r_1,\dots,r_n) \in \mathbb{R}^n``.

Thus, a spiderweb central configuration is a zero of the mapping ``F := (F_1, \dots, F_n) : \mathbb{R}^n \to \mathbb{R}^n`` given by

```math
F_i(r) :=
\lambda r_i + \frac{m_0}{r_i^2} + \frac{m_i}{2^{3/2}r_i^2}\zeta_{\ell} + \sum_{\begin{smallmatrix}j = 1 \\ j \neq i \end{smallmatrix}}^{n} \sum_{k=0}^{\ell-1} \frac{m_j(r_i - r_j \cos(\theta_k))}{(r_i^2 + r_j^2 - 2 r_i r_j \cos(\theta_k))^{3/2}} , \qquad i = 1,\, \dots,\, n,
```

where ``\zeta_{\ell} := \sum_{k=1}^{\ell-1} (1 - \cos(\theta_k))^{-1/2}``.

The Jacobian matrix is given by

```math
\frac{\partial}{\partial r_j} F_i(r) =
\begin{cases}
\displaystyle \lambda - \frac{2m_0}{r_i^3} - \frac{m_i}{r_i^3\sqrt{2}}\zeta_{\ell}
-\sum_{\begin{smallmatrix}j' = 1 \\ j' \neq i\end{smallmatrix}}^{n} \frac{m_{j'}}{2}\sum_{k=0}^{\ell-1}
\frac{4r_i^2 + r_{j'}^2 - 8 r_i r_{j'} \cos(\theta_k) + 3 r_{j'}^2 \cos(2\theta_k)}{(r_i^2 + r_{j'}^2 - 2 r_i r_{j'} \cos(\theta_k))^{5/2}}, & j = i,\\
\displaystyle -\frac{m_j}{2} \sum_{k=0}^{\ell-1}
\frac{-4(r_i^2 + r_j^2) \cos(\theta_k) + r_i r_j (7 + \cos(2\theta_k))}{(r_i^2 + r_j^2 - 2 r_i r_j \cos(\theta_k))^{5/2}}, & j \neq i.
\end{cases}
```

Consider the fixed-point operator ``T : \mathbb{R}^n \to \mathbb{R}^n`` defined by

```math
T(x) := x - A F(x),
```

where ``A : \mathbb{R}^n \to \mathbb{R}^n`` is the injective operator corresponding to a numerical approximation of ``DF(x_0)^{-1}`` for some numerical zero ``x_0 \in \mathbb{R}^n`` of ``F``.

Let ``R > 0``. According to the [first-order Radii Polynomial Theorem](@ref first_order_RPT), we need to estimate ``|T(x_0) - x_0|_\infty`` and ``\sup_{x \in \text{cl}( B_R(x_0) )} |DT(x)|_\infty`` which can be readily computed with interval arithmetic.

We can now write our computer-assisted proof:

```@example
using RadiiPolynomial

function F(x, m₀, m, λ, l)
    T = eltype(x)
    n = length(x)
    π2l⁻¹ = 2convert(T, π)/l
    F_ = Sequence(ParameterSpace()^n, Vector{T}(undef, n))
    for i ∈ 1:n
        F_[i] = λ*x[i] + m₀/x[i]^2
        for k ∈ 1:l-1
            θₖ = k*π2l⁻¹
            F_[i] += m[i]/(2x[i]^2 * sqrt(2 - 2cos(θₖ)))
        end
        for j ∈ 1:n
            if i ≠ j
                for k ∈ 0:l-1
                    θₖ = k*π2l⁻¹
                    F_[i] += m[j]*(x[i] - x[j]*cos(θₖ))/sqrt(x[i]^2 + x[j]^2 - 2x[i]*x[j]*cos(θₖ))^3
                end
            end
        end
    end
    return F_
end

function DF(x, m₀, m, λ, l)
    T = eltype(x)
    n = length(x)
    π2l⁻¹ = 2convert(T, π)/l
    DF_ = LinearOperator(ParameterSpace()^n, ParameterSpace()^n, zeros(T, n, n))
    for j ∈ 1:n, i ∈ 1:n
        if i == j
            DF_[i,i] += λ - 2m₀/x[i]^3
            for k ∈ 1:l-1
                θₖ = k*π2l⁻¹
                DF_[i,i] -= m[i]/(x[i]^3 * sqrt(2 - 2cos(θₖ)))
            end
        else
            for k ∈ 0:l-1
                θₖ = k*π2l⁻¹
                DF_[i,i] -= m[j]*(4x[i]^2 + x[j]^2 - 8x[i]*x[j]*cos(θₖ) + 3x[j]^2*cos(2θₖ))/(2sqrt(x[i]^2 + x[j]^2 - 2x[i]*x[j]*cos(θₖ))^5)
                DF_[i,j] -= m[j]*(-4(x[i]^2 + x[j]^2)*cos(θₖ) + x[i]*x[j]*(7 + cos(2θₖ)))/(2sqrt(x[i]^2 + x[j]^2 - 2x[i]*x[j]*cos(θₖ))^5)
            end
        end
    end
    return DF_
end

n = 18 # number of circles
l = 100 # number of masses per circle

# numerical solution

m₀ = 0.0 # central mass
m = fill(1/l, n) # vector of masses
λ = -1.0

x₀ = Sequence(ParameterSpace()^n, float.(1:n))
x₀, success = newton(x -> (F(x, m₀, m, λ, l), DF(x, m₀, m, λ, l)), x₀;
    tol = 1e-12, maxiter = 50)

# proof

m₀_interval = Interval(0.0)
m_interval = fill(inv(Interval(l)), n)
λ_interval = Interval(-1.0)

R = 1e-12

x₀_interval = Interval.(x₀)
x₀R_interval = Interval.(inf.(x₀_interval .- R), sup.(x₀_interval .+ R))
F_interval = F(x₀_interval, m₀_interval, m_interval, λ_interval, l)
DF_interval = DF(x₀R_interval, m₀_interval, m_interval, λ_interval, l)
A = inv(mid.(DF_interval))

Y = norm(A * F_interval, Inf)
Z₁ = opnorm(A * DF_interval - I, Inf)
showfull(interval_of_existence(Y, Z₁, R))
```

The following figure[^2] shows the numerical approximation of the proven spiderweb central configuration.

[^2]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

![](../../assets/spiderweb.svg)
