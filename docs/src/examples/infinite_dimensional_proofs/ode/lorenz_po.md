# Periodic orbit of the Lorenz system

In this example, we will prove the existence of a periodic orbit of the Lorenz system

```math
\frac{d}{dt} u(t) = f(u(t), \sigma, \rho, \beta) \bydef
\begin{pmatrix}
\sigma(u_2(t) - u_1(t))\\
u_1(t)(\rho - u_3(t)) - u_2(t)\\
u_1(t) u_2(t) - \beta u_3(t)
\end{pmatrix}, \qquad \sigma, \rho, \beta \in \mathbb{R}.
```

The vector field ``f`` and its derivative with respect to ``u``, denoted ``D_u f``, may be implemented as follows:

```@example lorenz_po
using RadiiPolynomial

function f!(f, u, σ, ρ, β)
    u₁, u₂, u₃ = eachcomponent(u)
    project!(component(f, 1), σ*(u₂ - u₁))
    project!(component(f, 2), u₁*(ρ - u₃) - u₂)
    project!(component(f, 3), u₁*u₂ - β*u₃)
    return f
end

function Df!(Df, u, σ, ρ, β)
    u₁, u₂, u₃ = eachcomponent(u)
    project!(component(Df, 1, 1), Multiplication(-σ*one(u₁)))
    project!(component(Df, 1, 2), Multiplication(σ*one(u₂)))
    project!(component(Df, 1, 3), Multiplication(zero(u₃)))
    project!(component(Df, 2, 1), Multiplication(ρ-u₃))
    project!(component(Df, 2, 2), Multiplication(-one(u₂)))
    project!(component(Df, 2, 3), Multiplication(-u₁))
    project!(component(Df, 3, 1), Multiplication(u₂))
    project!(component(Df, 3, 2), Multiplication(u₁))
    project!(component(Df, 3, 3), Multiplication(-β*one(u₃)))
    return Df
end
nothing # hide
```

Let ``\nu > 1``,

```math
\ell^1_{\nu, \mathbb{Z}} \bydef \left\{ u \in \mathbb{C}^\mathbb{Z} \, : \, \|u\|_{\ell^1_{\nu, \mathbb{Z}}} \bydef \sum_{k \in \mathbb{Z}} |u_k| \nu^{|k|} \right\}
```

and ``* : \ell^1_{\nu, \mathbb{Z}} \times \ell^1_{\nu, \mathbb{Z}} \to \ell^1_{\nu, \mathbb{Z}}`` be the discrete convolution given by

```math
u * v \bydef \left\{ \sum_{l \in \mathbb{Z}} u_{k - l} v_l \right\}_{k \in \mathbb{Z}}, \qquad \text{for all } u, v \in \ell^1_{\nu, \mathbb{Z}}.
```

For any sequence ``u \in \ell^1_{\nu, \mathbb{Z}}``, the Fourier series ``\sum_{k \in \mathbb{Z}} u_k e^{i \omega k t}``, for some frequency ``\omega > 0``, defines an analytic ``2\pi\omega^{-1}``-periodic function in ``C^\omega(\mathbb{R}, \mathbb{C})``; while the discrete convolution ``*`` corresponds to the product of Fourier series in sequence space.

The Banach space ``\ell^1_{\nu, \mathbb{Z}}`` is a suitable space to represent each component of a periodic solution of the Lorenz system. Indeed, it is a standard result from ODE theory that analytic vector fields yield analytic solutions.[^1]

[^1]: A. Hungria, J.-P. Lessard and J. D. Mireles James, [Rigorous numerics for analytic solutions of differential equations: the radii polynomial approach](https://doi.org/10.1090/mcom/3046), *Mathematics of Computation*, **85** (2016), 1427-1459.

Define the Banach space ``X \bydef \mathbb{C} \times (\ell^1_{\nu, \mathbb{Z}})^3`` endowed with the norm ``\|x\|_X \bydef |\gamma| + \|u_1\|_{\ell^1_{\nu, \mathbb{Z}}} + \|u_2\|_{\ell^1_{\nu, \mathbb{Z}}} + \|u_3\|_{\ell^1_{\nu, \mathbb{Z}}}`` for all ``x = (\gamma, u_1, u_2, u_3) \in X``. It follows that the sequence of coefficients of a ``2\pi\gamma``-periodic Fourier series solving the Lorenz equations is a zero of the mapping ``F : X \to X`` given by

```math
F(x) \bydef
\begin{pmatrix}
\sum_{j = 1}^3 (\sum_{k = -K}^K (u_j)_k - \xi_j)\eta_j\\
\left\{ \gamma ( f(u, \sigma, \rho, \beta) )_k - i k u_k \right\}_{k \in \mathbb{Z}}
\end{pmatrix}, \qquad \text{for all } x = (\gamma, u_1, u_2, u_3) \in \text{domain}(F),
```

where ``\xi \in \mathbb{R}^3`` is a chosen approximate position of the periodic orbit at ``t = 0`` and ``\eta \in \mathbb{R}^3`` the corresponding approximate tangent vector at ``\xi``. By means of the *phase condition* ``\sum_{j = 1}^3 (\sum_{k = -n}^n (u_j)_k - \xi_j)\eta_j``, the translation invariance of the periodic orbit is removed.

The mapping ``F`` and its Fréchet derivative, denoted ``DF``, may be implemented as follows:

```@example lorenz_po
function F!(F, x, σ, ρ, β, ξ, η)
    γ, u = x[1], component(x, 2)

    F[1] =
        (component(u, 1)(0) - ξ[1]) * η[1] +
        (component(u, 2)(0) - ξ[2]) * η[2] +
        (component(u, 3)(0) - ξ[3]) * η[3]

    project!(component(F, 2), γ * f!(component(F, 2), u, σ, ρ, β) - differentiate(u))

    return F
end

function DF!(DF, x, σ, ρ, β, η)
    γ, u = x[1], component(x, 2)

    DF .= 0

    component(component(DF, 1, 2), 1)[1,:] .= η[1]
    component(component(DF, 1, 2), 2)[1,:] .= η[2]
    component(component(DF, 1, 2), 3)[1,:] .= η[3]

    f!(component(DF, 2, 1), u, σ, ρ, β)

    project!(component(DF, 2, 2), γ * Df!(component(DF, 2, 2), u, σ, ρ, β) - Derivative(1))

    return DF
end
nothing # hide
```

Consider the fixed-point operator ``T : X \to X`` defined by

```math
T(x) \bydef x - A F(x),
```

where ``A : X \to X`` is an injective operator corresponding to an approximation of ``DF(\bar{x})^{-1}`` for some numerical zero ``\bar{x} = (\bar{\gamma}, \bar{u}_1, \bar{u}_2, \bar{u}_3) \in X`` of ``F``.

Given an initial guess, the numerical zero ``\bar{x}`` of ``F`` may be obtained by Newton's method:

```@example lorenz_po
σ, ρ, β = 10.0, 28.0, 8/3

K = 60

x̄ = zeros(ComplexF64, ParameterSpace() × Fourier(K, 1.0)^3)
x̄[1] = 1.5/(2π) # γ, i.e. approximate inverse of the frequency
component(component(x̄, 2), 1)[1:2:5] =
    [-2.9 - 4.3im,
      1.6 - 1.1im,
      0.3 + 0.4im]
component(component(x̄, 2), 2)[1:2:5] =
    [-1.2 - 5.4im,
      3.0 + 0.8im,
     -0.4 + 1.1im]
component(component(x̄, 2), 3)[0:2:4] =
    [ 23,
      3.8 + 4.7im,
     -1.8 + 0.9im]
component(component(x̄, 2), 1)[-5:2:-1] .= conj.(component(component(x̄, 2), 1)[5:-2:1])
component(component(x̄, 2), 2)[-5:2:-1] .= conj.(component(component(x̄, 2), 2)[5:-2:1])
component(component(x̄, 2), 3)[-4:2:0] .= conj.(component(component(x̄, 2), 3)[4:-2:0])

ξ = component(x̄, 2)(0)
η = differentiate(component(x̄, 2))(0)

newton!((F, DF, x) -> (F!(F, x, σ, ρ, β, ξ, η), DF!(DF, x, σ, ρ, β, η)), x̄)

# impose that x̄[1] is real and component(x̄, 2) are the coefficients of a real Fourier series
x̄[1] = real(x̄[1])
for i ∈ 1:3
    component(component(x̄, 2), i)[0] = real(component(component(x̄, 2), i)[0])
    component(component(x̄, 2), i)[-K:-1] .= conj.(component(component(x̄, 2), i)[K:-1:1])
end
```

Let ``R > 0``. Since ``T \in C^2(X, X)`` we may use the [second-order Radii Polynomial Theorem](@ref second_order_RPT) such that we need to estimate ``\|T(\bar{x}) - \bar{x}\|_X``, ``\|DT(\bar{x})\|_{\mathscr{B}(X, X)}`` and ``\sup_{x \in \text{cl}( B_R(\bar{x}) )} \|D^2T(x)\|_{\mathscr{B}(X, \mathscr{B}(X, X))}``.

To this end, consider the truncation operator

```math
(\Pi_K u)_k \bydef
\begin{cases}
u_k, & |k| \le K,\\
0, & |k| > K,
\end{cases}
\qquad \text{for all } u \in \ell^1_{\nu, \mathbb{Z}}.
```

Using the same symbol, this projection extends naturally to ``(\ell^1_{\nu, \mathbb{Z}})^3`` and ``X`` by acting on each component as follows ``\Pi_K u \bydef (\Pi_K u_1, \Pi_K u_2, \Pi_K u_3)``, for all ``u = (u_1, u_2, u_3) \in (\ell^1_{\nu, \mathbb{Z}})^3``, and ``\Pi_K x \bydef (\gamma, \Pi_K u_1, \Pi_K u_2, \Pi_K u_3)``, for all ``x = (\gamma, u_1, u_2, u_3) \in X``. For each of the Banach spaces ``\ell^1_{\nu, \mathbb{Z}}, (\ell^1_{\nu, \mathbb{Z}})^3, X``, we define the complementary operator ``\Pi_{\infty(K)} \bydef I - \Pi_K``.

Thus, denoting ``\bar{u} = (\bar{u}_1, \bar{u}_2, \bar{u}_3)``, we have

```math
\begin{aligned}
\|T(\bar{x}) - \bar{x}\|_X &\le
\|\Pi_K A \Pi_K F(\bar{x})\|_X + \frac{\bar{\gamma}}{n+1} \|\Pi_{\infty(K)} f(\bar{u}, \sigma, \rho, \beta)\|_{(\ell^1_{\nu, \mathbb{Z}})^3},\\
\|DT(\bar{x})\|_{\mathscr{B}(X, X)} &\le
\|\Pi_K A \Pi_K DF(\bar{x}) \Pi_{2K} - \Pi_K\|_{\mathscr{B}(X, X)} + \frac{1}{n+1} \max\Big( \|\Pi_{\infty(K)} f(\bar{u}, \sigma, \rho, \beta)\|_{(\ell^1_{\nu, \mathbb{Z}})^3},\\
&\qquad \bar{\gamma} \max\left(\sigma + \|\rho-\bar{u}_3\|_{\ell^1_{\nu, \mathbb{Z}}} + \|\bar{u}_2\|_{\ell^1_{\nu, \mathbb{Z}}}, \sigma + 1 + \|\bar{u}_1\|_{\ell^1_{\nu, \mathbb{Z}}}, \|\bar{u}_1\|_{\ell^1_{\nu, \mathbb{Z}}} + \beta\right) \Big),\\
\sup_{x \in \text{cl}( B_R(\bar{x}) )} \|D^2T(x)\|_{\mathscr{B}(X, \mathscr{B}(X, X))} &\le
\left(\|\Pi_K A \Pi_K\|_{\mathscr{B}(X, X)} + \frac{1}{n+1}\right) \max\Big( 2 (\bar{\gamma} + R),\\
&\qquad \max\left(\sigma + \|\rho-\bar{u}_3\|_{\ell^1_{\nu, \mathbb{Z}}} + \|\bar{u}_2\|_{\ell^1_{\nu, \mathbb{Z}}} + 2R, \sigma + 1 + \|\bar{u}_1\|_{\ell^1_{\nu, \mathbb{Z}}} + R, \|\bar{u}_1\|_{\ell^1_{\nu, \mathbb{Z}}} + R + \beta\right) \Big).
\end{aligned}
```

The computer-assisted proof may be implemented as follows:

```@example lorenz_po
ν = interval(1.05)
X_F = ℓ¹(GeometricWeight(ν))
X_F³ = NormedCartesianSpace(X_F, ℓ¹())
X = NormedCartesianSpace((ℓ¹(), X_F³), ℓ¹())
R = 1e-10

σ_interval, ρ_interval, β_interval = interval(10), interval(28), interval(8)/interval(3)

x̄_interval = Sequence(ParameterSpace() × Fourier(K, interval(1))^3, interval.(coefficients(x̄)))
γ̄_interval = real(x̄_interval[1])
ū_interval = component(x̄_interval, 2)

ξ_interval = interval.(ξ)
η_interval = interval.(η)

F_interval = zeros(eltype(x̄_interval), ParameterSpace() × Fourier(2K, interval(1))^3)
F!(F_interval, x̄_interval, σ_interval, ρ_interval, β_interval, ξ_interval, η_interval)

tail_γ̄f_interval = copy(component(F_interval, 2))
for i ∈ 1:3
    component(tail_γ̄f_interval, i)[-K:K] .= interval(0)
end

DF_interval = zeros(eltype(x̄_interval), space(F_interval), space(x̄_interval))
DF!(DF_interval, x̄_interval, σ_interval, ρ_interval, β_interval, η_interval)

A = interval.(inv(mid.(project(DF_interval, space(x̄_interval), space(x̄_interval)))))
bound_tail_A = inv(interval(K+1))

# computation of the bounds

Y = norm(A * F_interval, X) + bound_tail_A * norm(tail_γ̄f_interval, X_F³)

opnorm_Df = max(σ_interval + norm(ρ_interval-component(ū_interval, 3), X_F) + norm(component(ū_interval, 2), X_F),
                σ_interval + 1 + norm(component(ū_interval, 1), X_F),
                norm(component(ū_interval, 1), X_F) + β_interval)

Z₁ = opnorm(A * DF_interval - UniformScaling(interval(1)), X) +
    bound_tail_A * max(norm(tail_γ̄f_interval / γ̄_interval, X_F³), γ̄_interval * opnorm_Df)

Z₂ = (opnorm(A, X) + bound_tail_A) * max(2 * (γ̄_interval + R),
    max(σ_interval + norm(ρ_interval - component(ū_interval, 3), X_F) + R + norm(component(ū_interval, 2), X_F) + R,
        σ_interval + 1 + norm(component(ū_interval, 1), X_F) + R,
        norm(component(ū_interval, 1), X_F) + R + β_interval))

setdisplay(:full)

interval_of_existence(Y, Z₁, Z₂, R)
```

The following animation[^2] shows the numerical approximation of the proven periodic orbit (blue line) and the equilibria (red markers).

[^2]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@raw html
<video width="800" height="400" controls autoplay loop>
  <source src="../lorenz_po.mp4" type="video/mp4">
</video>
```
