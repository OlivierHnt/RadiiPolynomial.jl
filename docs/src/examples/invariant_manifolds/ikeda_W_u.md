# Unstable manifolds of equilibria of the cubic Ikeda equation

In this example, we will rigorously compute the unstable manifolds of the equilibria for the cubic Ikeda equation

```math
\frac{d}{dt} u(t) = f(u(t), u(t-\tau)) \bydef u(t-\tau) - u(t-\tau)^3.
```

The linearization at some equilibrium ``c \in \mathbb{R}`` yields

```math
\frac{d}{dt} v(t) = (1 - 3c^2) v(t-\tau).
```

The right-hand side of the above equation is an infinite dimensional endomorphism acting on ``C([-\tau, 0], \mathbb{R})``. Its compactness guarantees that the spectrum is comprised of eigenvalues accumulating at ``0``; in particular, there are finitely many eigenvalues whose real parts are strictly positive. As a matter of fact, an eigenvector ``\xi \in C([-\tau, 0], \mathbb{C})`` associated with an eigenvalue ``\lambda \in \mathbb{C}`` is given by ``\xi(s) = e^{s \lambda} \xi(0)``, for all ``s \in [-\tau, 0]`` and ``\xi(0) \neq 0``, such that

```math
\Psi(\lambda) \bydef \lambda - (1 - 3c^2) e^{-\tau \lambda} = 0.
```

The characteristic function ``\Psi`` and its derivative with respect to ``\lambda``, denoted ``D\Psi``, may be implemented as follows:

```@example ikeda_W_u
Ψ(λ, c, τ) = λ - (1 - 3c^2) * exp(-τ*λ)

DΨ(λ, c, τ) = 1 + τ * (1 - 3c^2) * exp(-τ*λ)
nothing # hide
```

For the cubic Ikeda equation, the equilibria are ``0``, ``1`` or ``-1``. For the equilibrium ``c = 0``, there is a unique real unstable eigenvalue. While for the equilibria ``c = 1`` and ``c = -1``, there are two complex conjugate unstable eigenvalues.

For the equilibrium ``c = 0``, we may use the first-order Radii Polynomial Theorem to rigorously compute the unstable eigenvalue:

```@example ikeda_W_u
using RadiiPolynomial

λ̄₀, success = newton(λ -> (Ψ(λ, 0.0, 1.59), DΨ(λ, 0.0, 1.59)), 0.5)

R = 1e-14

τ = I"1.59"

Y = abs(Ψ(interval(λ̄₀), interval(0), τ))
Z₁ = abs(1 - interval(DΨ(λ̄₀, 0, mid(τ))) \ DΨ(interval(λ̄₀, R; format = :midpoint), interval(0), τ))
ϵ₀ = inf(interval_of_existence(Y, Z₁, R)[1])
λ₀ = interval(λ̄₀, ϵ₀; format = :midpoint)

setdisplay(:full)

λ₀
```

Similarly, for the equilibria ``c = 1`` and ``c = -1``, we may use the same strategy to compute one of the two complex conjugate unstable eigenvalues:

```@example ikeda_W_u
λ̄₁, success = newton(λ -> (Ψ(λ, 1.0, 1.59), DΨ(λ, 1.0, 1.59)), 0.3+1.0im)

Y = abs(Ψ(interval(λ̄₁), interval(1), τ))
Z₁ = abs(1 - interval(DΨ(λ̄₁, 1, τ)) \ DΨ(interval(λ̄₁, R; format = :midpoint), interval(1), τ))
ϵ₁ = inf(interval_of_existence(Y, Z₁, R)[1])
λ₁ = interval(λ̄₁, ϵ₁; format = :midpoint)

setdisplay(:full)

λ₁
```

Let ``\lambda_1, \dots, \lambda_d`` be the unstable eigenvalues and ``\xi_1, \dots, \xi_d`` the respective eigenvectors. Denote by ``\Lambda : \mathbb{C}^d \to \mathbb{C}^d`` the diagonal matrix such that ``\Lambda_{i,i} \bydef \lambda_i``; also, denote by ``\Xi : \mathbb{C}^d \to C([-\tau, 0], \mathbb{C})`` the matrix whose ``i``-th column is the eigenvector ``\xi_i``.

Let

```math
X \bydef \left\{ \{ x_\alpha \}_{\alpha_1 + \ldots + \alpha_d \ge 0} \in \mathbb{C}^{(\mathbb{N} \cup \{0\})^d} \, : \, \| x \|_X \bydef \sum_{\alpha_1 + \ldots + \alpha_d \ge 0} |x_\alpha| < +\infty \right\}
```

and ``* : X \times X \to X`` be the Cauchy product given by

```math
x * y \bydef \left\{ \sum_{\beta_1 + \ldots + \beta_d \ge 0}^\alpha x_{\alpha - \beta} y_\beta \right\}_{\alpha_1 + \ldots + \alpha_d \ge 0}, \qquad \text{for all } x, y \in X.
```

For any sequence ``x \in X``, the Taylor series ``\sum_{\alpha_1 + \ldots + \alpha_d \ge 0} x_\alpha \sigma^\alpha`` defines an analytic function in ``C^\omega(\mathbb{D}^d, \mathbb{C})`` where ``\mathbb{D} \bydef \{ z \in \mathbb{C} \, : \, |z| \le 1 \}``; while the Cauchy product ``*`` corresponds to the product of Taylor series in sequence space.

The Banach space ``X`` is a suitable space to represent a parameterization of the unstable manifold. Indeed, it is a standard result from DDE theory that analytic vector fields yield analytic unstable manifolds of equilibria. In the context of this example, it holds that the unstable manifold is parameterized by an analytic function ``P : \mathbb{C}^d \to C([-\tau, 0], \mathbb{C})`` satisfying ``\frac{d}{ds} [P(\sigma)](s) = [DP(\sigma) \Lambda \sigma](s)`` along with ``[DP(\sigma) \Lambda \sigma](0) = f([P (\sigma)](0), [P(\sigma)](-\tau))``.[^1]

[^1]: O. Hénot, J.-P. Lessard and J. D. Mireles James, [Parameterization of unstable manifolds for DDEs: formal series solutions and validated error bounds](https://doi.org/10.1007/s10884-021-10002-8), *Journal of Dynamics and Differential Equations*, **34** (2022), 1285-1324.

In terms of the Taylor coefficients, the previous equalities yield

```math
[P(\sigma)](s) = \sum_{\alpha_1 + \ldots + \alpha_d \ge 0} \tx_\alpha e^{s (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} \sigma^\alpha
```

where ``\tx \in X`` is given component-wise by

```math
\tx_\alpha \bydef
\begin{cases}
c, & \alpha_1 = \ldots = \alpha_d = 0,\\
\xi_1, & \alpha_1 = 1, \alpha_2 = \ldots = \alpha_d = 0,\\
\vdots\\
\xi_d, & \alpha_d = 1, \alpha_1 = \ldots = \alpha_{d-1} = 0,\\
\Psi(\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)^{-1} \left(-e^{-\tau (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} [\tx * \tx * \tx]_{\tx_\alpha = 0}\right)_\alpha, & \alpha_1 + \ldots + \alpha_d \ge 2.
\end{cases}
```

Observe that the unstable manifold of the equilibrium ``c = -1`` is the same as the of the equilibrium ``c = 1`` modulo a change of sign. Thus, we shall only study the unstable manifolds of the equilibria ``c = 0`` and ``c = 1``.

For the equilibrium ``c = 0``, we may implement the ``1``-dimensional recurrence relation as follows:

```@example ikeda_W_u
n₀ = 85
x̃₀ = zeros(Interval{Float64}, Taylor(n₀))
x̃₀[1] = interval(5)
ỹ₀ = copy(x̃₀)
ỹ₀[1] *= exp(-τ * λ₀)
for α ∈ 2:n₀
    x̃₀[α] = -Ψ(α*λ₀, interval(0), τ) \ pow_bar(Sequence(Taylor(α), view(ỹ₀, 0:α)), 3)[α]
    ỹ₀[α] = x̃₀[α] * exp(-τ * α*λ₀)
end
```

Similarly, for the equilibrium ``c = 1``, we may implement the ``2``-dimensional recurrence relation as follows:

```@example ikeda_W_u
n₁ = 25
x̃₁ = zeros(Complex{Interval{Float64}}, Taylor(n₁) ⊗ Taylor(n₁))
x̃₁[(0,0)] = interval(1)
x̃₁[(1,0)] = x̃₁[(0,1)] = interval(0.35)
ỹ₁ = copy(x̃₁)
ỹ₁[(1,0)] *= exp(-τ * λ₁)
ỹ₁[(0,1)] *= exp(-τ * conj(λ₁))
for α₂ ∈ 0:n₁, α₁ ∈ 0:n₁-α₂
    if α₁ + α₂ ≥ 2
        x̃₁[(α₁,α₂)] = -Ψ(α₁*λ₁ + α₂*conj(λ₁), interval(1), τ) \ pow_bar(Sequence(Taylor(α₁) ⊗ Taylor(α₂), view(ỹ₁, (0:α₁, 0:α₂))), 3)[(α₁,α₂)]
        ỹ₁[(α₁,α₂)] = x̃₁[(α₁,α₂)] * exp(-τ * (α₁*λ₁ + α₂*conj(λ₁)))
    end
end
```

Consider the truncation operator

```math
(\Pi_n x)_\alpha \bydef
\begin{cases} x_\alpha, & \alpha_1 + \ldots + \alpha_d \le n,\\
0, & \alpha_1 + \ldots + \alpha_d > n,
\end{cases}
\qquad \text{for all } x \in X,
```

as well as the complementary operator ``\Pi_{\infty(n)} \bydef I - \Pi_n``.

Given that ``\Pi_n \tx`` is a finite sequence of known Taylor coefficients, it follows that the remaining coefficients are a fixed-point of the mapping ``T : \Pi_{\infty(n)} X \to \Pi_{\infty(n)} X`` given component-wise by

```math
( T(h) )_\alpha \bydef
\begin{cases}
0, & \alpha_1 + \ldots + \alpha_d \le n,\\
\Psi(\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)^{-1} \left( -e^{-\tau (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} [(\Pi_n \tx +h)*(\Pi_n \tx +h)*(\Pi_n \tx +h)]_{h_\alpha = 0} \right)_\alpha, & \alpha_1 + \ldots + \alpha_d > n.
\end{cases}
```

Let ``R > 0``. Since ``T \in C^1(\Pi_{\infty(n)} X, \Pi_{\infty(n)} X)`` we may use the first-order Radii Polynomial Theorem for which we use the estimates

```math
\begin{aligned}
\|T(0)\|_X &\le \max_{\mu \in \{\Re(\lambda_1), \dots, \Re(\lambda_d)\}} \frac{1}{(n+1)\mu - |1 - 3c^2| e^{-τ (n+1)\mu}} \|\Pi_{\infty(n)} (\Pi_n \tilde{y}*\Pi_n \tilde{y}*\Pi_n \tilde{y})\|_X,\\
\sup_{h \in \text{cl}( B_R(0) )} \|DT(h)\|_{\mathscr{B}(X, X)} &\le \max_{\mu \in \{\Re(\lambda_1), \dots, \Re(\lambda_d)\}} \frac{3}{(n+1)\mu - |1 - 3c^2| e^{-τ (n+1)\mu}} (\|\Pi_n \tilde{y}\|_X + R)^2,
\end{aligned}
```

where ``\tilde{y} \bydef \left\{ \tx_\alpha e^{-\tau (\alpha_1 \lambda_1 + \ldots + \alpha_d \lambda_d)} \right\}_{\alpha_1 + \ldots + \alpha_d \ge 0}``.

The computer-assisted proof for the ``1``-dimensional unstable manifold of ``c = 0`` may be implemented as follows:

```@example ikeda_W_u
X = ℓ¹()

R = 1e-12

tail_ỹ₀³ = ỹ₀ ^ 3
tail_ỹ₀³[0:n₀] .= interval(0)
C₀ = interval(n₀+1) * λ₀ - exp(-τ * interval(n₀+1) * λ₀)

Y = C₀ \ norm(tail_ỹ₀³, X)

Z₁ = C₀ \ (interval(3) * (norm(ỹ₀, X) + R)^2)

# error bound for the Taylor coefficients of order α > 85 of the parameterization on the domain [-1, 1]

setdisplay(:full)

interval_of_existence(Y, Z₁, R)
```

Similarly, the computer-assisted proof for the ``2``-dimensional unstable manifold of ``c = 1`` may be implemented as follows:

```@example ikeda_W_u
tail_ỹ₁³ = ỹ₁ ^ 3
for α₂ ∈ 0:n₁, α₁ ∈ 0:n₁-α₂
    tail_ỹ₁³[(α₁,α₂)] = interval(0)
end
C₁ = interval(n₁+1) * real(λ₁) - interval(2) * exp(-τ * interval(n₁+1) * real(λ₁))

Y = C₁ \ norm(tail_ỹ₁³, X)

Z₁ = C₁ \ (interval(3) * (norm(ỹ₁, X) + R)^2)

# error bound for the Taylor coefficients of order α₁ + α₂ > 25 of the parameterization on the domain 𝔻²

setdisplay(:full)

interval_of_existence(Y, Z₁, R)
```

The following animation[^2] shows:
- the equilibria ``0, \pm 1`` (red markers).
- the numerical approximation of the parameterization of the 1D unstable manifold of ``0``: on the domain of the computer-assisted proof ``[-1, 1]`` (green line) and on a larger domain (black line).
- the numerical approximation of the parameterization of the 2D unstable manifold of ``\pm 1``: on the domain of the computer-assisted proof ``\mathbb{D}^2`` (blue surfaces) and on a larger domain (black wireframes).

[^2]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@raw html
<video width="800" height="400" controls autoplay loop>
  <source src="../ikeda_W_u.mp4" type="video/mp4">
</video>
```
