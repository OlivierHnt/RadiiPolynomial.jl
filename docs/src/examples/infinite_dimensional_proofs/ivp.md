# Initial value problem for Ordinary Differential Equations (ODE)

In this example, we will prove the existence of a solution of the initial value problem

```math
\begin{cases}
\displaystyle \frac{d}{dt} u(t) = f(u(t)) := u(t)^2 - u(t),\\
u(0) = 1/2.
\end{cases}
```

Let ``\nu > 0`` and ``X := (\ell^1_\nu, *)`` where

```math
\ell^1_\nu := \left\{ \{ x_\alpha \}_{\alpha \geq 0} \in \mathbb{R}^{\mathbb{N} \cup \{0\}} \, : \, | x |_{\ell^1_\nu} := \sum_{\alpha \geq 0} |x_\alpha| \nu^\alpha < +\infty \right\}
```

and ``* : \ell^1_\nu \times \ell^1_\nu \to \ell^1_\nu`` is the Cauchy product given by

```math
x * y := \left\{ \sum_{\beta = 0}^\alpha x_{\alpha - \beta} y_\beta \right\}_{\alpha \geq 0}, \qquad \text{for all } x, y \in \ell^1_\nu.
```

The Banach algebra ``X`` is a suitable space to look for a solution of the initial value problem. Indeed, it is a standard result from ODE theory that analytic vector fields yield analytic solutions. For any sequence ``x \in X``, the series ``\sum_{\alpha \geq 0} x_\alpha t^\alpha`` defines an analytic function in ``C^\omega([-\nu, \nu], \mathbb{R})``; while the Cauchy product ``*`` corresponds to the product of analytic functions in sequence space.[^1]

[^1]: A. Hungria, J.-P. Lessard and J. D. Mireles James, [Rigorous numerics for analytic solutions of differential equations: the radii polynomial approach](https://doi.org/10.1090/mcom/3046), *Mathematics of Computation*, **85** (2016), 1427-1459.

It follows that the sequence of coefficients of a Taylor series solving the initial value problem is a zero of the mapping ``F : X \to X`` given component-wise by

```math
( F(x) )_\alpha :=
\begin{cases}
x_0 - 1/2, & \alpha = 0,\\
\alpha x_\alpha - (x*x - x)_{\alpha-1}, & \alpha \geq 1.
\end{cases}
```

Consider the fixed-point operator ``T : X \to X`` defined by

```math
T(x) := x - A F(x),
```

where ``A : X \to X`` is the injective operator corresponding to a numerical approximation of ``DF(x_0)^{-1}`` for some numerical zero ``x_0 \in X`` of ``F``.

Let ``R > 0``. According to the Radii Polynomial Theorem, we need to estimate ``|T(x_0) - x_0|_X``, ``|DT(x_0)|_{\mathscr{B}(X, X)}`` and ``\sup_{y \in \text{cl}( B_R(x_0) )} |D^2T(y)|_{\mathscr{B}(X^2, X)}``.

To this end, for all ``x \in X``, consider the projection operators

```math
\begin{aligned}
(\pi^n x)_\alpha &:= \begin{cases} x_\alpha, & \alpha \leq n, \\ 0, & \alpha > n, \end{cases}\\
\pi^{\infty(n)} x &:= x - \pi^n x.
\end{aligned}
```

Thus, for all ``x_0 \in X`` and ``R > 0``, we have

```math
\begin{aligned}
|T(x_0) - x_0|_X &\leq |\pi^n A \pi^n F(x_0)|_X + \frac{1}{n+1} |\pi^{\infty(n)} F(x_0)|_X,\\
|DT(x_0)|_{\mathscr{B}(X, X)} &\leq |\pi^n A \pi^n DF(x_0) \pi^n - I|_{\mathscr{B}(X, X)} + \frac{\nu}{n+1} |2x_0 - 1|_X,\\
\sup_{y \in \text{cl}( B_R(x_0) )} |D^2T(y)|_{\mathscr{B}(X^2, X)} &\leq 2 \nu \left( |\pi^n A \pi^n|_{\mathscr{B}(X, X)} + \frac{1}{n+1} \right).
\end{aligned}
```

In particular, from the latter estimate, we may freely choose ``R = \infty``.

We can now write our computer-assisted proof:

```@example
using RadiiPolynomial

function F(x::Sequence{Taylor})
    f = x^2 - x
    F_ = Sequence(Taylor(order(f)+1), Vector{eltype(x)}(undef, length(f)+1))
    F_[0] = x[0] - 0.5
    F_[1:end] .= Derivative(1) * x - f
    return F_
end

function DF(x::Sequence{Taylor}, domain::Taylor, codomain::Taylor, ::Type{CoefType}) where {CoefType}
    DF_ = LinearOperator(domain, codomain, zeros(CoefType, dimension(codomain), dimension(domain)))
    DF_[0,0] = one(CoefType)
    DF_[1:end,:] .=
        project(Derivative(1), domain, Taylor(order(codomain)-1), CoefType) .-
        project(Multiplication(2x - 1), domain, Taylor(order(codomain)-1), CoefType)    
    return DF_
end

# numerical solution

n = 27
x₀ = Sequence(Taylor(n), zeros(n+1))
x₀, success = newton(x -> (project(F(x), space(x)), DF(x, space(x), space(x), eltype(x))),
    x₀;
    verbose = false)

# proof

x₀_interval = Interval.(x₀)
F_interval = F(x₀_interval)
tail_F_interval = copy(F_interval)
tail_F_interval[0:n] .= Interval(0.0)
DF_interval = DF(x₀_interval, space(x₀_interval), space(x₀_interval), eltype(x₀_interval))
A = inv(mid.(DF_interval))
bound_tail_A = inv(Interval(n+1))

ν = Interval(1.0)
metric = Weightedℓ¹Norm(GeometricWeights(ν))
R = Inf

Y = norm(A * F_interval, metric) + bound_tail_A * norm(tail_F_interval, metric)
Z₁ = opnorm(A * DF_interval - I, metric) + bound_tail_A * ν * norm(2x₀_interval - 1, metric)
Z₂ = 2ν * (opnorm(A, metric) + bound_tail_A)
showfull(interval_of_existence(Y, Z₁, Z₂, R))
```
