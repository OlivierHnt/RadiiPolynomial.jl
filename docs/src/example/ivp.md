# Initial value problems

## Scalar ODE

In this example, we will rigorously compute the solution of the ODE:

```math
\begin{cases}
\frac{d}{dt}
u(t)
= f(u(t))
=
u(t)^2,\\
u(0) = 1.
\end{cases}
```

The first and second derivative of ``f`` are given by

```math
\begin{aligned}
Df(u)
&=
2u,\\
|D^2 f|(u)
&=
2.
\end{aligned}
```

A typical code to compute the truncation error looks like this:

```@example
using IntervalArithmetic, RadiiPolynomial

f(u) = u^2

f(u, i::Int) = (u^2)[i]

Df(u) = 2u

abs_D²f(u) = 2

N = 200

solution_interval = ivp_ODE(@interval(1.0);
    f = f, order = N)

ν = @interval(0.8)
r₀ = 1e-3

∞f_interval = f(solution_interval)
∞f_interval[0:N-1] .= 0
Df_interval = Df(solution_interval)
abs_D²f_interval = abs_D²f(norm(solution_interval, ν) + r₀)

∞C∞ = @interval(1/(N+1))

pb = TailProblem(solution_interval, r₀, ν, ∞f_interval, Df_interval, abs_D²f_interval, ∞C∞)

truncation_error = roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)
```

Observe that the solution is given by the Taylor series ``u(t) = \sum_{i=0}^{+\infty} t^i`` which imposes ``\nu < 1``. This leads to numerical instability for large values of ``N`` (e.g. ``N = 300``).

Another strategy is to rescale the ODE via ``t \mapsto s(t) \doteqdot Lt`` for some ``L>0`` such that the sequence is in ``\ell^1``; that is, we consider ``w(s) \doteqdot u(s/L)`` and solve

```math
\begin{cases}
\frac{d}{dt}
w(s)
= L f(w(s)),\\
w(0) = u(0).
\end{cases}
```

Thus,

```@example
using IntervalArithmetic, RadiiPolynomial

f(u) = u^2

f(u, i::Int) = (u^2)[i]

Df(u) = 2u

abs_D²f(u) = 2

N = 300

solution_interval = ivp_ODE(@interval(1.0);
    f = (u, i) -> @interval(0.8)*f(u, i), order = N)

ν = @interval(1.0)
r₀ = 1e-3

∞f_interval = @interval(0.8)*f(solution_interval)
∞f_interval[0:N-1] .= 0
Df_interval = @interval(0.8)*Df(solution_interval)
abs_D²f_interval = @interval(0.8)*abs_D²f(norm(solution_interval, ν) + r₀)

∞C∞ = @interval(1/(N+1))

pb = TailProblem(solution_interval, r₀, ν, ∞f_interval, Df_interval, abs_D²f_interval, ∞C∞)

truncation_error = roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)
```

However, the smallest valid truncation error has worsened considerably.





## System of ODEs

In this example, we will rigorously compute the solution of the system of ODEs:

```math
\begin{cases}
\begin{pmatrix}
u_1(t)\\
u_2(t)
\end{pmatrix}
= f(u(t))
=
\begin{pmatrix}
-u_2(t)\\
u_1(t)
\end{pmatrix},\\
u(0) = (1,0).
\end{cases}
```

The first and second derivative of ``f`` are given by

```math
\begin{aligned}
Df(u)
&=
\begin{pmatrix}
0 & -1\\
1 & 0
\end{pmatrix},\\
|D^2 f|(u)
&=
\begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{pmatrix}
\end{aligned}
```

This leads to:

```@example
using IntervalArithmetic, RadiiPolynomial

f(u) =
    [-u[2]
      u[1]]

f(u, i::Int) =
   [-u[2][i]
     u[1][i]]

Df(u) =
    [zero(u[1]) -one(u[2])
     one(u[1])  zero(u[2])]

abs_D²f(u) =
    [0 0 0 0
     0 0 0 0]

N = 10

solution_interval = ivp_ODE([@interval(1.0), @interval(0.0)];
    f = f, order = N)

ν = @interval(1.0)
r₀ = Inf

∞f_interval = f(solution_interval)
∞f_interval[1][0:N-1] .= 0
∞f_interval[2][0:N-1] .= 0
Df_interval = Df(solution_interval)
abs_D²f_interval = abs_D²f(norm(solution_interval, ν) + r₀)

∞C∞ = @interval(1/(N+1))

pb = TailProblem(solution_interval, r₀, ν, ∞f_interval, Df_interval, abs_D²f_interval, ∞C∞)

truncation_error = roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)
```

Observe that the solution is given by ``u(t) = (\cos(t), \sin(t))`` such that the truncation error is ``1/(N+1)!`` which is slightly smaller than our truncation error.
