```@contents
Pages = ["logistic_ivp.md"]
Depth = 3
```

# Logistic equation

In this example, we prove the existence of a solution of the logistic equation

```math
\begin{cases}
\displaystyle \frac{\mathrm{d}}{\mathrm{d}t} u(t) = u(t)(1 - u(t)), & t \in [-2, 2],\\
u(0) = 1/2.
\end{cases}
```

### Step 1: Problem definition

We start by casting the initial value problem into a corresponding zero-finding problem posed on an infinite-dimensional Banach space.

Given ``\nu \ge 1``, we consider the Banach space modelling Taylor coefficients of analytic functions on ``[-\nu, \nu]``:

```math
X_{T, \nu} \bydef \left\{ u(t) = \sum_{k \ge 0} u_k t^k \, : \, \| u \|_{X_{T, \nu}} \bydef \sum_{k \ge 0} |u_k| \nu^k < \infty \right\}.
```

This space is naturally equipped with the Cauchy product ``* : X_{T, \nu} \times X_{T, \nu} \to X_{T, \nu}`` given component-wise by, for any ``u, w \in X_{T, \nu}``,

```math
(u * w)_k \bydef \sum_{l = 0}^k u_{k - l} w_l, \qquad k \ge 0,
```

so that ``u(t) w(t) = (u * w)(t)``.

The Banach space ``X_{T, \nu}`` is a suitable space to represent a solution of the logistic equation, since it is known that analytic vector fields yield analytic solutions.[^1]

[^1]: A. Hungria, J.-P. Lessard and J. D. Mireles James, [Rigorous numerics for analytic solutions of differential equations: the radii polynomial approach](https://doi.org/10.1090/mcom/3046), *Mathematics of Computation*, **85** (2016), 1427-1459.

It follows that the sequence of coefficients of a Taylor series solving the initial value problem is a zero of the mapping ``F : X_{T, \nu} \to X_{T, \nu}`` given by

```math
[F(u)](t) \bydef u - \frac{1}{2} - \int_0^t u(s)(1 - u(s)) \, \mathrm{d}s.
```

The mapping ``F`` and its Fréchet derivative are implemented as follows:

```@example logistic_ivp
using RadiiPolynomial

F(u) = u - exact(0.5) - Integral(1) * (u*(exact(1) - u))

DF(u) = exact(I) - Integral(1) * Multiplication(exact(1) - exact(2) * u)
nothing # hide
```

Consider the fixed-point operator ``T : X_{T, \nu} \to X_{T, \nu}`` defined by

```math
T(u) \bydef u - A F(u),
```

where ``A : X_{T, \nu} \to X_{T, \nu}`` is an operator corresponding to an approximation of ``DF(\bar{u})^{-1}``, for some approximate zero ``\bar{u} \in X_{T, \nu}`` of ``F``.

### Step 2: Approximate zero (floating-point arithmetic)

We numerically compute an approximate zero by performing a finite-dimensional truncation of the problem and iterating Newton's method.

Consider the truncation operator ``\Pi_{\le K} : X_{T, \nu} \to X_{T, \nu}`` given component-wise by

```math
(\Pi_{\le K} u)_k \bydef
\begin{cases}
u_k, & k \le K,\\
0, & k > K,
\end{cases} \qquad \text{for all } u \in X_{T, \nu},
```

as well as the tail operator ``\Pi_{> K} \bydef I - \Pi_{\le K}``.

Given an initial guess, the approximate zero is obtained by running Newton's method on the truncated problem, namely ``\Pi_{\le K} \circ F \circ \Pi_{\le K}``. For an input `u_guess` representing an element of ``\Pi_{\le K} X_{T, \nu}``, the `newton` function will automatically perform the truncation ``\Pi_{\le K}``:

```@example logistic_ivp
K = 27

u_guess = zeros(Taylor(K))

u_bar, newton_success = newton(u -> (F(u), DF(u)), u_guess; verbose = true)
nothing # hide
```

### Step 3: Approximate inverse (floating-point arithmetic)

We proceed to construct the approximate inverse ``A \approx DF(\bar{u})^{-1}`` at the numerical approximation `u_bar`.

```@example logistic_ivp
Π = Projection(Taylor(K))
A_K_interval = interval(inv(Π * DF(u_bar) * Π))

A_tail = interval(I) - interval(Π)

A_interval = A_K_interval + A_tail
nothing # hide
```

### Step 4: Estimating the bounds (interval arithmetic)

Let ``R > 0``. Since ``T \in C^2(X_{T, \nu}, X_{T, \nu})`` we use the second-order Radii Polynomial Theorem so that we need to estimate

```math
\begin{aligned}
Y &\ge \|T(\bar{u}) - \bar{u}\|_{X_{T, \nu}}, \\
Z_1 &\ge \|DT(\bar{u})\|_{\mathscr{L}(X_{T, \nu}, X_{T, \nu})}, \\
Z_2 &\ge \sup_{u \in B(\bar{u}, R)} \|D^2 T(u)\|_{\mathscr{BL}(X_{T, \nu}, X_{T, \nu})}.
\end{aligned}
```

After some work, we find

```math
\begin{aligned}
Y &= \|\Pi_{\le 2K+1} A \Pi_{\le 2K+1} F(\bar{u})\|_{X_{T, \nu}}, \\
Z_1 &= \|\Pi_{\le K+1} - \Pi_{\le 2K+1} A \Pi_{\le 2K+1} DF(\bar{u}) \Pi_{\le K+1}\|_{\mathscr{L}(X_{T, \nu}, X_{T, \nu})}, \\
Z_2 &= 2 \nu \max\big( \|\Pi_{\le K} A \Pi_{\le K}\|_{\mathscr{L}(X_{T, \nu}, X_{T, \nu})}, 1\big).
\end{aligned}
```

In particular, since ``Z_2`` is independent of ``R``, we may freely set ``R = \infty``.

The computer-assisted proof leading to the a posteriori rigorous error estimate on `u_bar` is then completed by evaluating the formulas algebraically with interval arithmetic:

```@example logistic_ivp
u_bar_interval = interval(u_bar)

ν = interval(2)
X_T = Ell1(GeometricWeight(ν))

#- Y bound

Y = norm(A_interval * F(u_bar_interval), X_T)

#- Z₁ bound
Π_Kp1 = Projection(Taylor(K+1))

Z₁ = opnorm(interval(Π_Kp1) - A_interval * DF(u_bar_interval) * Π_Kp1, X_T)

#- Z₂ bound
R = Inf

Z₂ = max(opnorm(A_K_interval, X_T), interval(1)) * ν * interval(2)

# verify the contraction

ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R; verbose = true)
nothing # hide
```

```@example logistic_ivp
inf(ie) # smallest error
```

The following figure[^2] shows the numerical approximation of the proven solution in the interval ``[-2, 2]`` along with the theoretical solution ``t \mapsto (1 + e^{-t})^{-1}``.

[^2]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@example logistic_ivp
using GLMakie

fig = Figure()
ax = Axis(fig[1,1], xticks = -5:5)
lines!(ax, [Point2f(t, 1/(1+exp(-t))) for t = LinRange(-5, 5, 501)];
    color = :black, label = L"1/(1+e^{-t})")
lines!(ax, [Point2f(t, u_bar(t)) for t = LinRange(-2, 2, 501)];
    color = :blue, label = L"\bar{u}(t)")
scatter!(ax, Point2f(0, u_bar(0));
    color = :red)
axislegend(ax; position = :lt)
fig
```
