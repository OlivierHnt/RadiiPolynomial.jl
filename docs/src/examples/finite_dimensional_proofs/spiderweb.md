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
-\sum_{i<j} \frac{G m_i m_j}{|\mathbf{r}_i-\mathbf{r}_j|},
```

for ``i = 1,\dots,N``, with ``\mathbf{r} \in \{ (\mathbf{r}_1,\dots,\mathbf{r}_N) \in \R^{3N} \, : \, \mathbf{r}_i \neq \mathbf{r}_j , \,  i \neq j\}``, where ``G`` denotes the gravitational constant.

In the following, we fix the centre of mass at the origin and scale ``G = 1``. Moreover, due to the symmetries of a spiderweb central configuration, it is sufficient to consider the accelerations of the ``n`` bodies on the positive horizontal axis, and the numbers ``r_1, \dots, r_n`` also denote the positions of the masses on this semi-axis.

The configuration of ``N`` bodies is *central* at some time ``t^*`` if ``\frac{d^2}{dt^2}\mathbf{r}(t^*) = \lambda \mathbf{r}(t^*)`` for some common ``\lambda``. It is easy to see that ``\lambda`` is a strictly negative value given by ``\lambda = U(\mathbf{r})/I(\mathbf{r}) < 0`` where ``I := \sum_{i = 1}^N m_i |\mathbf{r}_i(\mathbf{r})|^2`` is the moment of inertia. Essentially, the value of ``\lambda`` scales the system and can be chosen arbitrarily.

Then, the original system of ODEs reduces to the following system of equations in ``\mathbb{R}^n``:

```math
\lambda r_i
=
-\sum_{k=1}^{\ell-1} \frac{m_i }{2^{3/2}r_i^2( 1 - \cos \theta_k )^{1/2}} -\frac{ m_0 }{r_i^2} - \sum_{\begin{smallmatrix}j=1\\j\neq i \end{smallmatrix}}^n \sum_{k=0}^{\ell-1} \frac{m_j( r_i - r_j \cos \theta_k )}{( r_i^2 + r_j^2 - 2 r_i r_j \cos \theta_k )^{3/2}},
```

for ``i = 1, \dots, n``, with ``\theta_k := \frac{2\pi k}{\ell}`` and ``r = (r_1,\dots,r_n) \in \{ r \in \R^n \, : \, 0 < r_1 < \ldots < r_n\}``.

Thus, a spiderweb central configuration is a zero of the mapping ``F := (F_1, \dots, F_n) : \mathbb{R}^n \to \mathbb{R}^n`` given by

```math
F_i(r) :=
\lambda \, r_i + \frac{ m_i }{2^{3/2}r_i^2}\zeta_{\ell} + \frac{ m_0}{r_i^2} + \sum_{\begin{smallmatrix}j=1\\j\neq i \end{smallmatrix}}^{n} \sum_{k=0}^{\ell-1} \frac{m_j( r_i - r_j \cos \theta_k )}{( r_i^2 + r_j^2 - 2 r_i r_j \cos \theta_k )^{3/2}} , \qquad i =1,\, \dots,\, n,
```

where ``\zeta_{\ell} := \sum_{k=1}^{\ell-1} (1-\cos \theta_k)^{-1/2}``.

The Jacobian matrix is given by

```math
\frac{\partial}{\partial r_j} F_i(r) =
\begin{cases}
\displaystyle \lambda - \frac{ m_i }{r_i^3\sqrt{2}}\zeta_{\ell} - \frac{ 2m_0 }{r_i^3}
-\sum_{\begin{smallmatrix}j=1\\j\neq i \end{smallmatrix}}^{n} \frac{m_j}{2}\sum_{k=0}^{\ell-1}
\frac{4r_i^2+r_j^2-8 r_i r_j \cos \theta_k + 3 r_j^2 \cos 2\theta_k}{( r_i^2 + r_j^2 - 2 r_i r_j \cos \theta_k )^{5/2}},& j=i,\\
\displaystyle -\frac{m_j}{2} \sum_{k=0}^{\ell-1}
\frac{-4(r_i^2+r_j^2)\cos \theta_k + r_i r_j (7+ \cos 2\theta_k)}{( r_i^2 + r_j^2 - 2 r_i r_j \cos \theta_k)^{5/2}}, & j\neq i.
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

function F_DF!(F::Vector{T}, DF::Matrix{T}, x::Vector{T}, xR::Vector{T}, m₀::T, m::Vector{T}, λ::T, n::Int, l::Int) where {T}
    length(F) == size(DF, 1) == size(DF, 2) == length(x) == length(xR) == length(m) || return throw(DimensionMismatch)
    π2_l = 2convert(T, π)/l
    sqrt2 = sqrt(convert(T, 2))
    DF .= Matrix{T}(I*λ, n, n)
    @inbounds for j ∈ axes(DF, 2)
        xⱼ² = x[j]*x[j]
        xRⱼ² = xR[j]*xR[j]
        @inbounds for i ∈ axes(DF, 1)
            xᵢ², xᵢxⱼ = x[i]*x[i], x[i]*x[j]
            xRᵢ², xRᵢxRⱼ = xR[i]*xR[i], xR[i]*xR[j]
            xRᵢ³ = xRᵢ²*xR[i]
            if j == 1
                F[i] = λ*x[i] + m₀/xᵢ²
            end
            if i == j
                DF[i,j] -= 2m₀/xRᵢ³
            end
            @inbounds for k ∈ 0:l-1
                θ = k*π2_l
                cosθ = cos(θ)
                cos2θ = cos(2θ)
                if i ≠ j
                    d = sqrt(xᵢ² + xⱼ² - 2xᵢxⱼ*cosθ)
                    dᵣ = sqrt(xRᵢ² + xRⱼ² - 2xRᵢxRⱼ*cosθ)
                    dᵣ⁵2 = 2dᵣ*dᵣ*dᵣ*dᵣ*dᵣ
                    F[i] += m[j]*(x[i] - x[j]*cosθ)/(d*d*d)
                    DF[i,i] -= m[j]*(4xRᵢ² + xRⱼ² - 8xRᵢxRⱼ*cosθ + 3xRⱼ²*cos2θ)/dᵣ⁵2
                    DF[i,j] -= m[j]*(-4(xRᵢ² + xRⱼ²)*cosθ + xRᵢxRⱼ*(7 + cos2θ))/dᵣ⁵2
                elseif i == j && k > 0
                    ζ = sqrt2*sqrt(1-cosθ)
                    F[i] += m[i]/(2xᵢ²*ζ)
                    DF[i,j] -= m[i]/(xRᵢ³*ζ)
                end
            end
        end
    end
    return F, DF
end

n = 18 # number of circles
l = 100 # number of masses per circle

# numerical solution

m₀ = 0.0 # central mass
m = fill(1/l, n) # vector of masses
λ = -1.0

x₀ = float.(1:n)
F = Vector{Float64}(undef, n)
DF = Matrix{Float64}(undef, n, n)
newton!((F, DF, x) -> F_DF!(F, DF, x, x, m₀, m, λ, n, l),
    x₀, F, DF;
    tol = 1e-12, maxiter = 50, verbose = false)
sort!(x₀)

# proof

m₀_interval = Interval(0.0)
m_interval = fill(@interval(1/l), n)
λ_interval = Interval(-1.0)

R = 1e-12

x₀_interval = Interval.(x₀)
x₀R_interval = Interval.(inf.(x₀_interval .- R), sup.(x₀_interval .+ R))
F_interval = Vector{Interval{Float64}}(undef, n)
DF_interval = Matrix{Interval{Float64}}(undef, n, n)
F_DF!(F_interval, DF_interval, x₀_interval, x₀R_interval, m₀_interval, m_interval, λ_interval, n, l)
A = inv(mid.(DF_interval))

Y = norm(Interval.(mag.(A * F_interval)), Inf)
Z₁ = opnorm(Interval.(mag.(A * DF_interval - I)), Inf)
showfull(interval_of_existence(Y, Z₁, R))
```

The following figure shows the numerical approximation of the spiderweb central configuration.

![](../../assets/spiderweb.svg)
