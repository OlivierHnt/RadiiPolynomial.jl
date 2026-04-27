```@contents
Pages = ["lorenz_po.md"]
Depth = 3
```

# Lorenz system

In this example, we prove the existence of a periodic orbit of the Lorenz system

```math
\frac{\mathrm{d}}{\mathrm{d}t} u(t) = f(u(t)) \bydef
\begin{pmatrix}
\sigma(u_2(t) - u_1(t))\\
u_1(t)(\rho - u_3(t)) - u_2(t)\\
u_1(t) u_2(t) - \beta u_3(t)
\end{pmatrix}, \qquad \sigma, \rho, \beta \in \mathbb{R}.
```

### Step 1: Problem definition

We start by casting the problem of finding a periodic orbit into a corresponding zero-finding problem posed on an infinite-dimensional Banach space.

The vector field ``f`` and its derivative are implemented as follows:
```@example lorenz_po
using RadiiPolynomial

function f(u, params)
    σ, ρ, β = params
    u₁, u₂, u₃ = u
    return [σ*(u₂ - u₁)
            u₁*(ρ - u₃) - u₂
            u₁*u₂ - β*u₃]
end

function Df(u, params)
    σ, ρ, β = params
    u₁, u₂, u₃ = u
    return [-σ*one(u₁)    σ*one(u₂)    zero(u₃)
            ρ-u₃          -one(u₂)     -u₁
            u₂            u₁           -β*one(u₃)]
end
nothing # hide
```

Given ``\nu \ge 1``, we consider the Banach space modelling Fourier coefficients of ``2\pi``-periodic functions:

```math
X_{F, \nu} \bydef \left\{ u(t) = \sum_{k \in \mathbb{Z}} u_k e^{i t k} \, : \, \| u \|_{X_{F, \nu}} \bydef \sum_{k \in \mathbb{Z}} |u_k| \nu^{|k|} < \infty \right\}.
```

This space is naturally equipped with the discrete convolution ``* : X_{F, \nu} \times X_{F, \nu} \to X_{F, \nu}`` given component-wise by, for any ``u, w \in X_{F, \nu}``,

```math
(u * w)_k \bydef \sum_{l \in \mathbb{Z}} u_{k - l} w_l, \qquad k \in \mathbb{Z},
```

so that ``u(t) w(t) = (u * w)(t)``.

The Banach space ``X_{F, \nu}`` is a suitable space to represent each component of a periodic solution of the Lorenz system. Indeed, it is a standard result from ODE theory that analytic vector fields yield analytic solutions.[^1]

[^1]: A. Hungria, J.-P. Lessard and J. D. Mireles James, [Rigorous numerics for analytic solutions of differential equations: the radii polynomial approach](https://doi.org/10.1090/mcom/3046), *Mathematics of Computation*, **85** (2016), 1427-1459.

Define the Banach space

```math
X \bydef (X_{F, \nu})^3 \times \mathbb{C},
```

endowed with the norm

```math
\|x\|_X \bydef \sum_{j=1}^3 \|u^{(j)}\|_{X_{F, \nu}} + |\tau|, \qquad \text{for all } x = (u^{(1)}, u^{(2)}, u^{(3)}, \tau) \in X.
```

After rescaling time ``s \mapsto t(s) = \tau s`` to normalize the period to ``2\pi``, we seek a zero of the mapping ``F : X \to X`` given by

```math
F(x) \bydef
\begin{pmatrix}
\displaystyle \frac{\mathrm{d}}{\mathrm{d}s} u  - \tau f(u) \\
\Phi_\xi (u) \\
\end{pmatrix}, \qquad \text{for all } x = (u^{(1)}, u^{(2)}, u^{(3)}, \tau) \in \text{domain}(F),
```

where ``t \mapsto \xi(t)`` is a fixed approximatation of the derivative of the periodic solution, and ``\Phi_\xi : X_{F, \nu}^3 \to \mathbb{C}`` is the *phase condition* given by, for any ``u \in X_{F, \nu}``,

```math
\Phi_\xi (u) \bydef \sum_{j = 1}^3 \sum_{k \in \mathbb{Z}} (\xi^{(j)}_k)^\dagger u_k^{(j)},
```

which removes the time translation invariance of the periodic solution.
Here, the superscript ``\dagger`` denotes the complex conjugate.

The mapping ``F`` and its Fréchet derivative are implemented as follows:

```@example lorenz_po
function F(x, params, ξ_)
    u, τ = block(x)
    ξ = block(ξ_)
    return [Derivative(1) .* block(u) - τ[1] * f(block(u), params),
            [adjoint(ξ[1]) adjoint(ξ[2]) adjoint(ξ[3])] * [block(u, j) for j = 1:3]]
end

function DF(x, params, ξ_)
    u, τ = block(x)
    ξ = block(ξ_)
    M = Matrix{Any}(undef, 2, 2)

    L = [Derivative(1)  0*I            0*I
         0*I            Derivative(1)  0*I
         0*I            0*I            Derivative(1)]

    M[1,1] = L - τ[1] * Multiplication.(Df(block(u), params))

    M[1,2] = [LinearOperator.(-f(block(u), params));;]

    M[2,1] = [adjoint(ξ[1]) adjoint(ξ[2]) adjoint(ξ[3])]

    M[2,2] = [LinearOperator(0);;]

    return M
end
nothing # hide
```

Consider the fixed-point operator ``T : X \to X`` defined by

```math
T(x) \bydef x - A F(x),
```

where ``A : X \to X`` is an operator corresponding to an approximation of ``DF(\bx)^{-1}``, for some numerical zero ``\bx = (\bar{u}^{(1)}, \bar{u}^{(2)}, \bar{u}^{(3)}, \bar{\tau}) \in X`` of ``F``.

Consider the truncation operator is defined as

```math
(\Pi_{\le K} u)_k \bydef
\begin{cases}
u_k, & |k| \le K,\\
0, & |k| > K,
\end{cases}
\qquad \text{for all } u \in X_{F, \nu}.
```

Using the same symbol, this projection extends naturally to ``X`` by acting on each component as follows ``\Pi_{\le K} x \bydef (\Pi_{\le K} u^{(1)}, \Pi_{\le K} u^{(2)}, \Pi_{\le K} u^{(3)}, \tau)``, for all ``x = (u^{(1)}, u^{(2)}, u^{(3)}, \tau) \in X``. In all cases, we define the tail operator ``\Pi_{> K} \bydef I - \Pi_{\le K}``.

### Step 2: Approximate zero (floating-point arithmetic)

We numerically compute an approximate zero by performing a finite-dimensional truncation of the problem and iterating Newton's method.

Given an initial guess, the approximate zero is obtained by running Newton's method on the truncated problem, namely ``\Pi_{\le K} \circ F \circ \Pi_{\le K}``.

```@example lorenz_po
σ, ρ, β = 10.0, 28.0, 8/3

K = 40

u_guess = zeros(ComplexF64, Fourier(K, 1.0)^3)
block(u_guess, 1)[1:2:5] =
    [-2.9 - 4.3im,
      1.6 - 1.1im,
      0.3 + 0.4im]
block(u_guess, 2)[1:2:5] =
    [-1.2 - 5.4im,
      3.0 + 0.8im,
     -0.4 + 1.1im]
block(u_guess, 3)[0:2:4] =
    [ 23,
      3.8 + 4.7im,
     -1.8 + 0.9im]
block(u_guess, 1)[-5:2:-1] .= conj.(block(u_guess, 1)[5:-2:1])
block(u_guess, 2)[-5:2:-1] .= conj.(block(u_guess, 2)[5:-2:1])
block(u_guess, 3)[-4:2:0]  .= conj.(block(u_guess, 3)[4:-2:0])

ξ = differentiate(u_guess)

τ_guess = 1.5/(2π) # approximate inverse of the frequency
nothing # hide
```

For an input `x_guess` representing an element of ``\Pi_{\le K} X``, the `newton` function will automatically perform the truncation ``\Pi_{\le K}``:

```@example lorenz_po
x_guess = Sequence(Fourier(K, 1.0)^3 × ScalarSpace()^1, [coefficients(u_guess) ; τ_guess])

x_bar, newton_success = newton(x -> (F(x, (σ, ρ, β), ξ), DF(x, (σ, ρ, β), ξ)), x_guess; verbose = true)
nothing # hide
```

### Step 3: Approximate inverse (floating-point arithmetic)

We proceed to construct the approximate inverse ``A \approx DF(\bar{x})^{-1}`` at the numerical approximation `x_bar`. Because ``A`` is used within the rigorous estimates, we construct it using standard floating-point arithmetic and immediately convert it to interval arithmetic for subsequent bounded steps.

```@example lorenz_po
σ_interval, ρ_interval, β_interval = interval(10), interval(28), interval(8)/interval(3)
params_interval = (σ_interval, ρ_interval, β_interval)

ξ_interval = interval(ξ)

conjugacy_symmetry!(x_bar) # impose real-valued Fourier series
x_bar_interval = interval(x_bar)

F_interval = F(x_bar_interval, params_interval, ξ_interval)
DF_interval = DF(x_bar_interval, params_interval, ξ_interval)

Π = Projection(space(x_bar_interval))
A_K_interval = interval(inv(mid(Π * DF_interval * Π)))

A_tail_11 = (interval(I) - Projection(space(x_bar_interval)[1])) * Integral(1)
A_tail_12 = interval(zeros(ScalarSpace()^1,  Fourier(0, 1.)^3))
A_tail_21 = interval(zeros(Fourier(0, 1.)^3, ScalarSpace()^1))
A_tail_22 = interval(zeros(ScalarSpace()^1,  ScalarSpace()^1))
A_tail = [A_tail_11 A_tail_12
          A_tail_21 A_tail_22]

A = block(A_K_interval) + A_tail
nothing # hide
```

### Step 4: Estimating the bounds (interval arithmetic)

Let ``R > 0``. Since ``T \in C^2(X, X)`` we may use the second-order Radii Polynomial Theorem so that we need to estimate

```math
\begin{aligned}
Y &\ge \|T(\bar{x}) - \bar{x}\|_{X},\\
Z_1 &\ge \|DT(\bar{x})\|_{\mathscr{L}(X, X)},\\
Z_2 &\ge \sup_{x \in B(\bar{x}, R)} \|D^2 T(x)\|_{\mathscr{BL}(X, X)}.
\end{aligned}
```

After some work, we find

```math
\begin{aligned}
Y &= \|\Pi_{\le 2K} A \Pi_{\le 2K} F(\bar{x})\|_{X_{T, \nu}}, \\
Z_1 &= \|\Pi_{\le 2K+1} - \Pi_{\le 3K+1} A \Pi_{\le 3K+1} DF(\bar{x}) \Pi_{\le 2K+1}\|_{\mathscr{L}(X, X)}, \\
Z_2 &= \max\left( \|\Pi_{\le K} A \Pi_{\le K}\|_{\mathscr{L}(X_{T, \nu}, X_{T, \nu})}, \frac{1}{K+1}\right) \max\Big( 2 (\bar{\gamma} + R), \\
&\qquad \max\left(\sigma + \|\rho-\bar{u}^{(3)}\|_{X_{F, \nu}} + \|\bar{u}^{(2)}\|_{X_{F, \nu}} + 2R, \sigma + 1 + \|\bar{u}^{(1)}\|_{X_{F, \nu}} + R, \|\bar{u}^{(1)}\|_{X_{F, \nu}} + R + \beta\right) \Big).
\end{aligned}
```

Since ``Z_2`` depends on ``R``, we make the heuristic choice ``R = 10Y``.

The computer-assisted proof leading to the a posteriori rigorous error estimate on `x_bar` is then completed by evaluating the formulas algebraically with interval arithmetic:

```@example lorenz_po
ν = interval(1)
X_F = Ell1(GeometricWeight(ν))
X_F³ = NormedCartesianSpace(X_F, Ell1())
X = NormedCartesianSpace((X_F³, Ell1()), Ell1())

#- Y bound
Π_2K = Projection(Fourier(2K, interval(1))^3 × ScalarSpace()^1)

Y = norm(A * Π_2K * F_interval, X)

#- Z₁ bound
Π_2Kp1 = Projection(Fourier(2K+1, interval(1))^3 × ScalarSpace()^1)

Z₁ = opnorm(Π_2Kp1 - A * (DF_interval * Π_2Kp1), X)

#- Z₂ bound
R = exact(10 * sup(Y))
u₁_bar_interval, u₂_bar_interval, u₃_bar_interval = block(block(x_bar_interval, 1))
τ_bar_interval = block(x_bar_interval, 2)[1]

Z₂ = max(opnorm(A_K_interval, X), inv(interval(K+1))) *
    max(exact(2) * (abs(τ_bar_interval) + R),
        max(σ_interval + norm(ρ_interval - u₃_bar_interval, X_F) + R + norm(u₂_bar_interval, X_F) + R,
            σ_interval + 1 + norm(u₁_bar_interval, X_F) + R,
            norm(u₁_bar_interval, X_F) + R + β_interval))

#

ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R; verbose = true)
nothing # hide
```

```@example lorenz_po
inf(ie) # smallest error
```

The following figure[^2] shows the numerical approximation of the proven periodic orbit (blue line) and the equilibria (red markers).

[^2]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@example lorenz_po
using GLMakie

fig = Figure()
ax = Axis3(fig[1,1], aspect = :data, azimuth = 0.9π, elevation = 0.25)
lines!(ax, [Point3f(real(block(x_bar, 1)(t))) for t = LinRange(-π, π, 501)];
    color = :blue, label = L"\bar{u}(t)")
meshscatter!(ax, [Point3f(0, 0, 0), Point3f(-sqrt(β*(ρ-1)), -sqrt(β*(ρ-1)), ρ), Point3f(sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ)];
    color = :red, markersize = 0.5, label = "Equilibria")
axislegend(ax; position = :lt)
fig
```
