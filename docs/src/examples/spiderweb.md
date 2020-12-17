# Spiderweb central configurations

In this example we will prove the existence and local uniqueness of a class of relative equilibria called *central configurations* in the ``N``-body problem.

More precisely, we will look at configurations given by ``N = n \times \ell+1`` masses located at the intersection points of ``\ell`` concurrent equidistributed half-lines with ``n`` circles and a central mass ``m_0``. The ``\ell`` masses on the ``i``-th circle are equal to a positive constant ``m_i`` and we allow the particular case ``m_0 = 0``. These central configurations are called *spiderweb central configurations*. For more details, the interested reader can refer to [Spiderweb Central Configurations](https://doi.org/10.1007/s12346-019-00330-y).

The ``N``-body problem consists in describing the positions ``\mathbf{r}_1(t),\dots,\mathbf{r}_N(t)`` of ``N`` masses ``m_1,\dots,m_N`` interacting through Newton's gravitational law:

```math
m_i \frac{d^2}{dt^2} \mathbf{r}_i
=
-\sum_{j\neq i} \frac{G m_i m_j(\mathbf{r}_i - \mathbf{r}_j)}{| \mathbf{r}_i - \mathbf{r}_j |^3}
=
-\frac{\partial}{\partial \mathbf{r}_i} U (\mathbf{r})
=
\mathbf{F}_i(\mathbf{r}),
\qquad
U (\mathbf{r}) = -\sum_{i<j} \frac{G m_i m_j}{|\mathbf{r}_i-\mathbf{r}_j|},
```

for ``i = 1,\dots,N``, with ``\mathbf{r} \in \{ (\mathbf{r}_1,\dots,\mathbf{r}_N) \in \R^{3N} \, : \, \mathbf{r}_i \neq \mathbf{r}_j , \,  i \neq j\}``, where ``G`` denotes the gravitational constant.

In the following, we fix the centre of mass at the origin and scale ``G = 1``. Moreover, due to the symmetries of a spiderweb central configuration, it is sufficient to consider the accelerations of the ``n`` bodies on the positive horizontal axis, and the numbers ``r_1,\dots,r_n`` also denote the positions of the masses on this semi-axis.

The configuration of ``N`` bodies is *central* at some time ``t^*`` if ``\frac{d^2}{dt^2}\mathbf{r}(t^*) = \lambda \mathbf{r}(t^*)`` for some common ``\lambda``, where ``\mathbf{r} \in \{ (\mathbf{r}_1,\dots,\mathbf{r}_N) \in \R^{3N} \, : \, \mathbf{r}_i \neq \mathbf{r}_j , \,  i \neq j\}``. It is easy to see that ``\lambda`` is a strictly negative value given by ``\lambda = U(\mathbf{r})/I(\mathbf{r}) < 0`` where ``I = \sum_{i = 1}^N m_i |\mathbf{r}_i(\mathbf{r})|^2`` is the moment of inertia. Essentially, the value of ``\lambda`` scales the system and can be chosen arbitrarily.

Then, the original system of ODEs in ``\mathbb{R}^{n\ell + 1}`` reduces to the following system of equations in ``\mathbb{R}^n``:

```math
\lambda r_i
=
-\sum_{k=1}^{\ell-1} \frac{m_i }{2^{3/2}r_i^2( 1 - \cos \theta_k )^{1/2}} -\frac{ m_0 }{r_i^2} - \sum_{\begin{smallmatrix}j=1\\j\neq i \end{smallmatrix}}^n \sum_{k=0}^{\ell-1} \frac{m_j( r_i - r_j \cos \theta_k )}{( r_i^2 + r_j^2 - 2 r_i r_j \cos \theta_k )^{3/2}},
```

for ``i=1,\dots,n``, with ``\theta_k \doteqdot \frac{2\pi k}{\ell}`` and ``r = (r_1,\dots,r_n) \in \{ r \in \R^n \, : \, 0 < r_1 < \ldots < r_n\}``.

Thus, a spiderweb central configuration is a zero of the map ``F = (F_1, \dots, F_n) : \mathbb{R}^n \to \mathbb{R}^n`` given by

```math
F_i(r) \doteqdot
\lambda \, r_i + \frac{ m_i }{2^{3/2}r_i^2}\zeta_{\ell} + \frac{ m_0}{r_i^2} + \sum_{\begin{smallmatrix}j=1\\j\neq i \end{smallmatrix}}^{n} \sum_{k=0}^{\ell-1} \frac{m_j( r_i - r_j \cos \theta_k )}{( r_i^2 + r_j^2 - 2 r_i r_j \cos \theta_k )^{3/2}} , \qquad i =1,\, \dots,\, n,
```

where ``\zeta_{\ell} \doteqdot \sum_{k=1}^{\ell-1} (1-\cos \theta_k)^{-1/2}``.

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

We can now write our computer-assisted proof:

```Julia
using RadiiPolynomial, IntervalArithmetic, LinearAlgebra

function newton_spiderweb(x₀::Vector, F_DF::Function; tol::Real, maxiter::Int)
    F, DF = F_DF(x₀)
    if norm(F, Inf) ≤ tol
        printstyled("0-th iteration → Newton's method succeeded!\n"; color = :green)
        return x₀
    end
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        sort!(x)
        F, DF = F_DF(x)
        if norm(F, Inf) ≤ tol
            printstyled(i, "-th iteration → Newton's method succeeded!\n"; color = :green)
            return x
        end
    end
    printstyled("Newton's method failed!\n"; color = :red)
    return x
end

function F_DF(x::Vector{T}, xᵣ::Vector{T}, m₀::T, m::Vector{T}, λ::T, n::Int, l::Int) where {T}
    @assert length(x) == length(xᵣ)
    π2_l = 2convert(T, π)/l
    sqrt2 = sqrt(convert(T, 2))
    F = λ*x
    DF = Matrix{T}(I*λ, n, n)
    indices = eachindex(x)
    l_range = 0:l-1
    @inbounds for j ∈ indices
        xⱼ² = x[j]*x[j]
        xᵣⱼ² = xᵣ[j]*xᵣ[j]
        @inbounds for i ∈ indices
            xᵢ², xᵢxⱼ = x[i]*x[i], x[i]*x[j]
            xᵣᵢ², xᵣᵢxᵣⱼ = xᵣ[i]*xᵣ[i], xᵣ[i]*xᵣ[j]
            xᵣᵢ³ = xᵣᵢ²*xᵣ[i]
            if j == 1
                F[i] += m₀/xᵢ²
            end
            if i == j
                DF[i,j] += -2m₀/xᵣᵢ³
            end
            @inbounds for k ∈ l_range
                θ = k*π2_l
                cosθ = cos(θ)
                cos2θ = cos(2θ)
                if i ≠ j
                    d = sqrt(xᵢ² + xⱼ² - 2xᵢxⱼ*cosθ)
                    dᵣ = sqrt(xᵣᵢ² + xᵣⱼ² - 2xᵣᵢxᵣⱼ*cosθ)
                    dᵣ⁵ = dᵣ*dᵣ*dᵣ*dᵣ*dᵣ
                    F[i] += m[j]*(x[i] - x[j]*cosθ)/(d*d*d)
                    DF[i,i] += -m[j]*(4xᵣᵢ² + xᵣⱼ² - 8xᵣᵢxᵣⱼ*cosθ + 3xᵣⱼ²*cos2θ)/(2dᵣ⁵)
                    DF[i,j] += -m[j]*(-4(xᵣᵢ² + xᵣⱼ²)*cosθ + xᵣᵢxᵣⱼ*(7 + cos2θ))/(2dᵣ⁵)
                elseif i == j && k > 0
                    ζ = sqrt2*sqrt(1-cosθ)
                    F[i] += m[i]/(2xᵢ²*ζ)
                    DF[i,j] += -m[i]/(xᵣᵢ³*ζ)
                end
            end
        end
    end
    return F, DF
end

n = 18 # number of circles
l = 100 # number of masses per circle

m₀ = 0.0 # central mass
m = ones(n)/l # vector of masses
λ = -1.0

# numerical solution

x₀ = newton_spiderweb(float.(1:n), x -> F_DF(x, x, m₀, m, λ, n, l); tol = 1e-12, maxiter = 50)

# proof

m₀_interval = @interval(0.0)
m_interval = [@interval(1/l) for _ ∈ 1:n]
λ_interval = @interval(-1.0)

x₀_interval = [@interval(xᵢ) for xᵢ ∈ x₀]

r₀ = 1e-5
xᵣ_interval = [xᵢ ± r₀ for xᵢ ∈ x₀]

F, DF = F_DF(x₀_interval, xᵣ_interval, m₀_interval, m_interval, λ_interval, n, l)
A = inv(mid.(DF))

pb = ZeroFindingProblemFiniteDimension(F, DF, A, nothing, r₀)

roots_radii_polynomial(Y(pb), Z₁(pb), r₀)
```
