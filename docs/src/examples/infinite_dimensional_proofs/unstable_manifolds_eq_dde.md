# Unstable manifolds of equilibria for delay differential equations (DDE)

In this example, we will rigorously compute the unstable manifolds of the equilibria for the cubic Ikeda equation

```math
\frac{d}{dt} u(t) = f(u(t), u(t-\tau)) := u(t-\tau) - u(t-\tau)^3.
```

The linearization at some equilibrium ``c \in \mathbb{R}`` yields

```math
\frac{d}{dt} v(t) = (1 - 3c^2) v(t-\tau).
```

The right-hand side of the above equation is an infinite dimensional endomorphism acting on ``C([-\tau, 0], \mathbb{R})``. Its compactness guarantees that the spectrum is comprised of eigenvalues accumulating at ``0``; in particular, there are finitely many eigenvalues whose real parts are strictly positive. As a matter of fact, an eigenvector ``\xi \in C([-\tau, 0], \mathbb{C})`` associated to an eigenvalue ``\lambda \in \mathbb{C}`` is given by ``\xi(s) = e^{s \lambda} \xi(0)``, for all ``s \in [-\tau, 0]`` and ``\xi(0) \neq 0``, such that

```math
\Psi(\lambda) := \lambda - (1 - 3c^2) e^{-\tau \lambda} = 0.
```

For the cubic Ikeda equation, the equilibria are ``0``, ``1`` or ``-1``. For the equilibrium ``c = 0``, there is a unique real unstable eigenvalue. While for the equilibria ``c = 1`` and ``c = -1``, there are two complex conjugate unstable eigenvalues.

Let ``\lambda_1, \dots, \lambda_d`` be the unstable eigenvalues and ``\xi_1, \dots, \xi_d`` their respective eigenvector. Denote by ``\Lambda : \mathbb{C}^d \to \mathbb{C}^d`` the diagonal matrix such that ``\Lambda_{i,i} := \lambda_i``; also, denote by ``\Xi : \mathbb{C}^d \to C([-\tau, 0], \mathbb{C})`` the matrix whose ``i``-th column is the eigenvector ``\xi_i``.

Let ``\nu > 0`` and ``X := (\ell^1_\nu, *)`` where

```math
\ell^1_\nu := \left\{ \{ x_\alpha \}_{\alpha_1 + \ldots + \alpha_d \geq 0} \in \mathbb{C}^{(\mathbb{N} \cup \{0\})^d} \, : \, | x |_{\ell^1_\nu} := \sum_{\alpha_1 + \ldots + \alpha_d \geq 0} |x_\alpha| \nu^{\alpha_1 + \ldots + \alpha_d} < +\infty \right\}
```

and ``* : \ell^1_\nu \times \ell^1_\nu \to \ell^1_\nu`` is the Cauchy product given by

```math
x * y := \left\{ \sum_{\beta_1 + \ldots + \beta_d \geq 0}^\alpha x_{\alpha - \beta} y_\beta \right\}_{\alpha_1 + \ldots + \alpha_d \geq 0}, \qquad \text{for all } x, y \in \ell^1_\nu.
```

For any sequence ``x \in X``, the series ``\sum_{\alpha_1 + \ldots + \alpha_d \geq 0} x_\alpha \sigma^\alpha`` defines an analytic function in ``C^\omega(\mathbb{D}_\nu^d, \mathbb{C})`` where ``\mathbb{D}_\nu := \{ z \in \mathbb{C} \, : \, |z| \leq \nu \}``; while the Cauchy product ``*`` corresponds to the product of analytic functions in sequence space.

The Banach algebra ``X`` is a suitable space to look for a parameterization of the unstable manifold. Indeed, it is a standard result from DDE theory that analytic vector fields yield analytic unstable manifolds of equilibria. In the context of this example, it holds that the unstable manifold is parameterized by an analytic function ``P : \mathbb{C}^d \to C([-\tau, 0], \mathbb{C})`` satisfying ``\frac{d}{ds} P(\sigma)(s) = [DP(\sigma) \Lambda \sigma](s)`` for all ``s \in [-\tau, 0]`` and ``\sigma \in \{ \sigma \in \mathbb{D}_\nu^d \, : \, \Lambda \sigma \in \mathbb{D}_\nu^d\}`` along with ``[DP(\sigma) \Lambda \sigma](0) = f([P (\sigma)](0), [P(\sigma)](-\tau))``.[^1]

[^1]: O. Hénot, J.-P. Lessard and J. D. Mireles James, [Parameterization of unstable manifolds for DDEs: formal series solutions and validated error bounds](https://doi.org/10.1007/s10884-021-10002-8), *Journal of Dynamics and Differential Equations* (2021).

In terms of the Taylor coefficients, the previous equalities yield

```math
P(\sigma)(s) = \sum_{\alpha_1 + \ldots + \alpha_d \geq 0} \tilde{x}_\alpha e^{s (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} \sigma^\alpha
```

where ``\tilde{x} \in X`` is given component-wise by

```math
\tilde{x}_\alpha :=
\begin{cases}
c, & \alpha_1 = \ldots = \alpha_d = 0,\\
\xi_1, & \alpha_1 = 1, \alpha_2 = \ldots = \alpha_d = 0,\\
\vdots\\
\xi_d, & \alpha_d = 1, \alpha_1 = \ldots = \alpha_{d-1} = 0,\\
\Psi(\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)^{-1} \left(-e^{-\tau (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} [\tilde{x} * \tilde{x} * \tilde{x}]_{\tilde{x}_\alpha = 0}\right)_\alpha, & \alpha_1 + \ldots + \alpha_d \geq 2.
\end{cases}
```

For all ``x \in X``, consider the projection operators

```math
\begin{aligned}
(\pi^n x)_\alpha &:= \begin{cases} x_\alpha, & \alpha_1 + \ldots + \alpha_d \leq n, \\ 0, & \alpha_1 + \ldots + \alpha_d > n, \end{cases}\\
\pi^{\infty(n)} x &:= x - \pi^n x.
\end{aligned}
```

Given that ``\pi^n \tilde{x}`` is a finite sequence of known Taylor coefficients, it follows that the remaining coefficients are a fixed-point of the mapping ``T : \pi^{\infty(n)} X \to \pi^{\infty(n)} X`` given component-wise by

```math
( T(h) )_\alpha :=
\begin{cases}
0, & \alpha_1 + \ldots + \alpha_d \leq n,\\
\Psi(\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)^{-1} \left( -e^{-\tau (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} [(\pi^n \tilde{x} +h)*(\pi^n \tilde{x} +h)*(\pi^n \tilde{x} +h)]_{h_\alpha = 0} \right)_\alpha, & \alpha_1 + \ldots + \alpha_d > n.
\end{cases}
```

Let ``R > 0``. Since ``T \in C^1(\pi^{\infty(n)} X, \pi^{\infty(n)} X)`` we may use the [first-order Radii Polynomial Theorem](@ref first_order_RPT) for which we use the estimates

```math
\begin{aligned}
|T(0)|_X &\leq \max_{\mu \in \{\Re(\lambda_1), \dots, \Re(\lambda_d)\}} \frac{1}{(n+1)\mu - |1 - 3c^2| e^{-τ (n+1)\mu}} |\pi^{\infty(n)} (\pi^n \tilde{y}*\pi^n \tilde{y}*\pi^n \tilde{y})|_X,\\
\sup_{h \in \text{cl}( B_R(0) )} |DT(h)|_{\mathscr{B}(X, X)} &\leq \max_{\mu \in \{\Re(\lambda_1), \dots, \Re(\lambda_d)\}} \frac{3}{(n+1)\mu - |1 - 3c^2| e^{-τ (n+1)\mu}} (|\pi^n \tilde{y}|_X + R)^2,
\end{aligned}
```

where ``\tilde{y} := \left\{ \tilde{x}_\alpha e^{-\tau (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} \right\}_{\alpha_1 + \ldots + \alpha_d \geq 0}``.

We can now write our computer-assisted proof:

```julia
using RadiiPolynomial

Ψ(λ, c, τ) = λ - (1 - 3c^2) * exp(-τ*λ)

DλΨ(λ, c, τ) = 1 + τ * (1 - 3c^2) * exp(-τ*λ)

# 1D unstable manifold of 0

let τ = @interval(1.59), c = Interval(0.0)
    λ_approx, _ = newton(λ -> (Ψ(λ, 0.0, 1.59), DλΨ(λ, 0.0, 1.59)), 0.47208; verbose = false)
    R_eigval = 1e-14
    Y_eigval = abs(Ψ(Interval(λ_approx), c, τ))
    Z₁_eigval = abs(1 - DλΨ(λ_approx, mid(c), mid(τ)) \ DλΨ(λ_approx ± R_eigval, c, τ))
    ie_eigval = interval_of_existence(Y_eigval, Z₁_eigval, R_eigval)
    λ = λ_approx ± inf(ie_eigval)

    n = 150
    x̃ = Sequence(Taylor(n), zeros(Interval{Float64}, n+1))
    x̃[1] = 1
    ỹ = copy(x̃)
    ỹ[1] *= exp(-τ * λ)
    for α ∈ 2:n
        αλ = α*λ
        ỹ_ = Sequence(Taylor(α), view(ỹ, 0:α))
        x̃[α] = -Ψ(αλ, c, τ) \ (ỹ_ ^̄  3)[α]
        ỹ[α] = x̃[α] * exp(-τ * αλ)
    end

    ν = Interval(5.0)
    X = ℓ¹(GeometricWeight(ν))
    R = 1e-12
    tail_ỹ³ = ỹ ^ 3
    tail_ỹ³[0:n] .= 0
    C = (n+1)*λ - abs(1 - 3c^2) * exp(-τ * (n+1)*λ)
    Y = C \ norm(tail_ỹ³, X)
    Z₁ = C \ (3(norm(ỹ, X) + R)^2)
    ie = interval_of_existence(Y, Z₁, R)
end

# 2D unstable manifold of 1

let τ = @interval(1.59), c = Interval(1.0)
    λ_approx, _ = newton(λ -> (Ψ(λ, 1.0, 1.59), DλΨ(λ, 1.0, 1.59)), 0.32056+1.15780im; verbose = false)
    R_eigval = 1e-14
    Y_eigval = abs(Ψ(Interval(λ_approx), c, τ))
    Z₁_eigval = abs(1 - DλΨ(λ_approx, mid(c), mid(τ)) \ DλΨ(complex(real(λ_approx) ± R_eigval, imag(λ_approx) ± R_eigval), c, τ))
    ie_eigval = interval_of_existence(Y_eigval, Z₁_eigval, R_eigval)
    λ = complex(real(λ_approx) ± inf(ie_eigval), imag(λ_approx) ± inf(ie_eigval))

    n = 25
    x̃ = Sequence(Taylor(n) ⊗ Taylor(n), zeros(Complex{Interval{Float64}}, (n+1)^2))
    x̃[(0,0)] = 1
    x̃[(1,0)] = x̃[(0,1)] = 0.35
    ỹ = copy(x̃)
    ỹ[(1,0)] *= exp(-τ * λ)
    ỹ[(0,1)] *= exp(-τ * conj(λ))
    for α₂ ∈ 0:n, α₁ ∈ 0:n-α₂
        if α₁ + α₂ ≥ 2
            αλ = α₁*λ + α₂*conj(λ)
            ỹ_ = Sequence(Taylor(α₁) ⊗ Taylor(α₂), view(ỹ, (0:α₁, 0:α₂)))
            x̃[(α₁,α₂)] = -Ψ(αλ, c, τ) \ (ỹ_ ^̄  3)[(α₁,α₂)]
            ỹ[(α₁,α₂)] = x̃[(α₁,α₂)] * exp(-τ * αλ)
        end
    end

    X = ℓ¹()
    R = 1e-12
    tail_ỹ³ = ỹ ^ 3
    for α₂ ∈ 0:n, α₁ ∈ 0:n-α₂
        tail_ỹ³[(α₁,α₂)] = 0
    end
    C = (n+1)*real(λ) - abs(1 - 3c^2) * exp(-τ * (n+1)*real(λ))
    Y = C \ norm(tail_ỹ³, X)
    Z₁ = C \ (3(norm(ỹ, X) + R)^2)
    ie = interval_of_existence(Y, Z₁, R)
end
```

The following animation[^2] shows the numerical approximation of the proven unstable manifolds.

[^2]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@raw html
<video width="800" height="400" controls autoplay loop>
  <source src="../../../assets/ikeda.mp4" type="video/mp4">
</video>
```
