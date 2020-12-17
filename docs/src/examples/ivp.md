# Initial value problems

Consider the initial value problem

```math
\begin{cases}
\frac{d}{dt}
u(t)
= f(u(t)),\\
u(0) = \phi.
\end{cases}
```

If ``u`` is analytic, consider its sequence of Taylor coefficients ``\{u_\alpha\}_{\alpha = 0}^{+\infty}``. The initial value problem yields an explicit recursive formula for each Taylor coefficients. If one were to compute the first ``n`` Taylor coefficients with interval arithmetic, then we can formulate a [`TailProblem`](@ref) as the tail can then be seen as the fixed point of ``G : \pi^{\infty(n)} (\ell^1_{\mathbb{N},\nu})^d \to \pi^{\infty(n)} (\ell^1_{\mathbb{N},\nu})^d`` defined by

```math
G(h) \doteqdot \pi^{\infty(n)} D^{-1} \pi^{\infty(n)} f(\{u_\alpha\}_{\alpha = 0}^n + h)
```

where

```math
(D^{-1} h)_{\alpha} \doteqdot
\begin{cases}
\alpha^{-1} h_{\alpha-1}, & \alpha \geq 1,\\
0, & \alpha = 0.
\end{cases}
```

More generally, we can write a [`ZeroFindingProblemCategory1`](@ref) by considering ``F : (\ell^1_{\mathbb{N},\nu})^d \to (\ell^1_{\mathbb{N},\nu})^d`` defined by

```math
F(h) \doteqdot (E_0 h, -(D h)_0, -(D h)_1, \dots) + (-\phi, f_0(h), f_1(h), \dots)
```

where ``E_0`` is the evaluation functional at ``0``.




## Scalar ODE with finite time blow-up

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
D^2 |f|(u)
&=
2.
\end{aligned}
```

A typical code to compute the truncation error looks like this:

```Julia
using RadiiPolynomial, IntervalArithmetic

f(u) = u^2

f(u, i::Int) = (u^2)[i]

Df(u) = 2u

D²_abs_f(u) = 2

N = 200

solution_interval = ivp_ODE(@interval(1.0); f = f, order = N)

ν = @interval(0.8)
r₀ = 1e-3

∞f_interval = f(solution_interval)
∞f_interval[0:N-1] .= 0
Df_interval = Df(solution_interval)
D²_abs_f_interval = D²_abs_f(norm(solution_interval, ν) + r₀)

∞L∞_bound = @interval(1/(N+1))

pb = TailProblem(∞f_interval, Df_interval, D²_abs_f_interval, ∞L∞_bound, ν, r₀)

prove(pb)
```

Observe that the solution is given by ``u(t) = 1/(1-t)`` such that ``u(t) = \sum_{i=0}^{+\infty} t^i`` for ``|t| \leq \nu < 1``.

Note that having no decay in the coefficients leads to drastic loss of accuracy with interval arithmetic and FFT for large values of ``N``.

It might be useful to use the rescaling ``t \mapsto s(t) \doteqdot \nu t`` for some ``\nu > 0`` such that the sequence of Taylor coefficients is in ``\ell^1``; that is, we consider ``w(s) \doteqdot u(s/L)`` which solves

```math
\begin{cases}
\frac{d}{ds}
w(s)
= L f(w(s)),\\
w(0) = u(0).
\end{cases}
```

The above gives us rigorous enclosures for each coefficients and the truncation error. If a general ``C^0`` bound is enough, then we can speed things up by not computing rigorously the first ``N`` Taylor coefficients. The proof is now formulated as a zero finding problem and can be written as follow:

```Julia
using RadiiPolynomial, IntervalArithmetic

f(u) = u^2

f(u, i::Int) = (u^2)[i]

Df(u) = 2u

D²_abs_f(u) = 2

function ᴺF_∞f(x::Sequence{Taylor}, val, f)
    f_ = f(x)
    F = Sequence(Taylor(order(f_)+1),
        [x[0] - val
         coefficients(f_ - differentiate(x))])
    ᴺF = project(F, space(x))
    F[0:order(x)] .= 0
    return ᴺF, F
end

function ᴺDF_ᴺAᴺ_Df(x::Sequence{Taylor}, Df)
    Df_ = Df(x)
    _Df_ = Operator(space(x), derivative_range(space(x)), Df_)
    ᴺDF = Operator(space(x), space(x),
        [transpose( coefficients( Functional(space(x), Evaluation(0.0)) ) )
         coefficients( _Df_ -̄ Derivative() )])
    ᴺAᴺ = Operator(space(x), space(x), [@interval(DⱼFᵢ) for DⱼFᵢ ∈ inv(mid.(ᴺDF.coefficients))])
    return ᴺDF, ᴺAᴺ, Df_
end

N = 200

solution = ivp_ODE(1.0; f = f, order = N)

solution_interval = Sequence(space(solution), [@interval(solᵢ) for solᵢ ∈ solution])

ν = @interval(0.8)
r₀ = 1e-3

ᴺF_interval, ∞f_interval = ᴺF_∞f(solution_interval, 1.0, f)
ᴺDF_interval, ᴺAᴺ, Df_interval = ᴺDF_ᴺAᴺ_Df(solution_interval, Df)
D²_abs_f_interval = D²_abs_f(norm(solution_interval, ν) + r₀)

ᴺL∞_bound = 0
∞L⁻¹∞_bound = @interval(1/(N+1))

pb = ZeroFindingProblemCategory1(
    ∞f_interval,
    Df_interval,
    D²_abs_f_interval,
    ᴺF_interval,
    ᴺDF_interval,
    ᴺAᴺ,
    ᴺL∞_bound,
    ∞L⁻¹∞_bound,
    ν,
    r₀)

prove(pb)
```




## System of ODEs

In this example, we will rigorously compute the solution of the system of ODEs:

```math
\begin{cases}
\frac{d}{dt}
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
D^2 |f|(u)
&=
\begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{pmatrix}
\end{aligned}
```

This leads to:

```Julia
using RadiiPolynomial, IntervalArithmetic

f(u) =
    [-u[2]
      u[1]]

f(u, i::Int) =
   [-u[2][i]
     u[1][i]]

Df(u) =
    [zero(u[1]) -one(u[2])
     one(u[1])  zero(u[2])]

D²_abs_f(u) =
    [0 0 0 0
     0 0 0 0]

N = 10

solution_interval = ivp_ODE([@interval(1.0), @interval(0.0)]; f = f, order = N)

ν = @interval(1.0)
r₀ = Inf

∞f_interval = f(solution_interval)
∞f_interval[1][0:N-1] .= 0
∞f_interval[2][0:N-1] .= 0
Df_interval = Df(solution_interval)
D²_abs_f_interval = D²_abs_f(norm(solution_interval, ν) + r₀)

∞L∞_bound = @interval(1/(N+1))

pb = TailProblem(∞f_interval, Df_interval, D²_abs_f_interval, ∞L∞_bound, ν, r₀)

prove(pb)
```

Observe that the solution is given by ``u(t) = (\cos(t), \sin(t))`` such that the theoretical truncation error is ``1/(N+1)! \in [2.50521 \times 10^{-8}, 2.50522 \times 10^{-8}]`` which is slightly smaller than our truncation error.

Once again, we can formulate a zero finding problem:

```Julia
using RadiiPolynomial, IntervalArithmetic, LinearAlgebra

f(u) =
    [-u[2]
      u[1]]

f(u, i::Int) =
   [-u[2][i]
     u[1][i]]

Df(u) =
    [zero(u[1]) -one(u[2])
     one(u[1])  zero(u[2])]

D²_abs_f(u) =
    [0 0 0 0
     0 0 0 0]

function ᴺF_∞f(x::AbstractVector{<:Sequence{Taylor}}, val, f)
    f_ = f(x)
    F = [Sequence(Taylor(order(f_[i])+1),
            [x[i][0] - val[i]
             coefficients(f_[i] - differentiate(x[i]))]) for i ∈ eachindex(x)]
    ᴺF = project.(F, space.(x))
    for i ∈ eachindex(x)
        F[i][0:order(x[i])] .= 0
    end
    return ᴺF, F
end

function ᴺDF_ᴺAᴺ_Df(x::AbstractVector{<:Sequence{Taylor}}, Df)
    Df_ = Df(x)
    _Df_ = Operator.(space.(x), derivative_range.(space.(x)), Df_)
    ᴺDF = [i == j ?
        Operator(space(x[j]), space(x[i]),
            [transpose(coefficients(Functional(space(x[j]), Evaluation(0.0))))
             coefficients( _Df_[i,j] -̄ Derivative() )]) :
        Operator(space(x[j]), space(x[i]),
            [zeros(1, length(space(x[j])))
             coefficients( _Df_[i,j] )])
        for i ∈ eachindex(x), j ∈ eachindex(x)]    
    ᴺAᴺ = [@interval(ᴺAᴺᵢⱼ) for ᴺAᴺᵢⱼ ∈ inv(mid.(mapreduce(j -> mapreduce(i -> coefficients(ᴺDF[i,j]), vcat, eachindex(x)), hcat, eachindex(x))))]
    ᴺAᴺ_ = [Operator(space(x[j]), space(x[i]), ᴺAᴺ[1+(i-1)*length(space(x[i])):i*length(space(x[i])),1+(j-1)*length(space(x[j])):j*length(space(x[j]))]) for i ∈ eachindex(x), j ∈ eachindex(x)]
    return ᴺDF, ᴺAᴺ_, Df_
end

N = 10

solution = ivp_ODE([1.0, 0.0]; f = f, order = N)

solution_interval =
    [Sequence(space(solution[1]), [@interval(solᵢ) for solᵢ ∈ solution[1]])
     Sequence(space(solution[2]), [@interval(solᵢ) for solᵢ ∈ solution[2]])]

ν = @interval(1.0)
r₀ = Inf

ᴺF_interval, ∞f_interval = ᴺF_∞f(solution_interval, [1.0, 0.0], f)
ᴺDF_interval, ᴺAᴺ, Df_interval = ᴺDF_ᴺAᴺ_Df(solution_interval, Df)
D²_abs_f_interval = D²_abs_f(norm.(solution_interval, ν) .+ r₀)

ᴺL∞_bound = 0
∞L⁻¹∞_bound = @interval(1/(N+1))

pb = ZeroFindingProblemCategory1(
    ∞f_interval,
    Df_interval,
    D²_abs_f_interval,
    ᴺF_interval,
    ᴺDF_interval,
    ᴺAᴺ,
    ᴺL∞_bound,
    ∞L⁻¹∞_bound,
    ν,
    r₀)

prove(pb)
```
