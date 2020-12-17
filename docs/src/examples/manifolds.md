# Manifolds

## Stable/unstable manifolds of equilibria in ODEs

In this example, we will rigorously compute the stable/unstable manifolds for the equilibria of the Lorenz equations:

```math
\frac{d}{dt}
\begin{pmatrix}
u_1(t)\\
u_2(t)\\
u_3(t)
\end{pmatrix}
= f(u(t))
=
\begin{pmatrix}
\sigma(u_2(t) - u_1(t))\\
u_1(t)(\rho - u_3(t)) - u_2(t)\\
u_1(t) u_2(t) - \beta u_3(t)
\end{pmatrix}.
```

The equilibria are

```math
\begin{aligned}
c_0 &= (0, 0, 0),\\
c_+ &= (\sqrt{\beta(\rho - 1)}, \sqrt{\beta(\rho - 1)}, \rho - 1),\\
c_- &= (-\sqrt{\beta(\rho - 1)}, -\sqrt{\beta(\rho - 1)}, \rho - 1).
\end{aligned}
```

The first and second derivative of ``f`` are given by

```math
\begin{aligned}
Df(u)
&=
\begin{pmatrix}
-\sigma  & \sigma & 0     \\
\rho-u_3 & -1     & -u_1  \\
u_2      & u_1    & -\beta
\end{pmatrix},\\
D^2 |f|(u)
&=
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 1 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}.
\end{aligned}
```

The last ingredient needed is ``\hat{f}`` which is obtained by keeping only the nonlinear terms of ``f``, that is

```math
\hat{f}(u)
=
\begin{pmatrix}
0\\
-u_1 u_3\\
u_1 u_2
\end{pmatrix}.
```

We can now proceed to compute the manifolds and give a rigorous a posteriori error bound

```Julia
using RadiiPolynomial, LinearAlgebra

function f(u, p)
    σ, ρ, β = p
    return [σ*(u[2] - u[1])
            u[1]*(ρ - u[3]) - u[2]
            u[1]*u[2] - β*u[3]]
end

function Df(u, p)
    σ, ρ, β = p
    return [-σ     σ    0
            ρ-u[3] -1   -u[1]
            u[2]   u[1] -β]
end

function D²_abs_f(u, p)
    σ, ρ, β = p
    return [0 0 0 0 0 0 0 0 0
            0 0 1 0 0 0 1 0 0
            0 1 0 1 0 0 0 0 0]
end

function f̂(u, p, α)
    σ, ρ, β = p
    return [0
            -(u[1]*u[3])[α]
            (u[1]*u[2])[α]]
end

## Parameters

σ, ρ, β = -2.2, 1.33, 8/3

## Manifolds c₀

c₀ = [0.0, 0.0, 0.0]
Df₀ = Df(c₀, [σ, ρ, β])
Λ₀, Ξ₀ = eigen(Df₀)

λ₀_stable = Λ₀[1]
ξ₀_stable = Ξ₀[:,1]
P₀_stable = manifold_ODE_equilibrium(c₀, real(ξ₀_stable), real(λ₀_stable);
    Df = Df₀, f̂ = (u, α) -> f̂(u, [σ, ρ, β], α), order = 20)

λ₀_unstable = Λ₀[2:3]
ξ₀_unstable = Ξ₀[:,2:3]
P₀_unstable = manifold_ODE_equilibrium(complex(c₀), ξ₀_unstable, λ₀_unstable;
    Df = Df₀, f̂ = (u, α) -> f̂(u, [σ, ρ, β], α), order = (20, 20))

## Manifolds c₊

c₊ = [sqrt(β*(ρ - 1)), sqrt(β*(ρ - 1)), ρ - 1]
Df₊ = Df(c₊, [σ, ρ, β])
Λ₊, Ξ₊ = eigen(Df₊)

λ₊_unstable = Λ₊[3]
ξ₊_unstable = Ξ₊[:,3]
P₊_unstable = manifold_ODE_equilibrium(c₊, real(ξ₊_unstable), real(λ₊_unstable);
    Df = Df₊, f̂ = (u, α) -> f̂(u, [σ, ρ, β], α), order = 20)

λ₊_stable = Λ₊[1:2]
ξ₊_stable = Ξ₊[:,1:2]
P₊_stable = manifold_ODE_equilibrium(complex(c₊), ξ₊_stable, λ₊_stable;
    Df = Df₊, f̂ = (u, α) -> f̂(u, [σ, ρ, β], α), order = (20, 20))

## Manifolds c₋

c₋ = [-sqrt(β*(ρ - 1)), -sqrt(β*(ρ - 1)), ρ - 1]
Df₋ = Df(c₋, [σ, ρ, β])
Λ₋, Ξ₋ = eigen(Df₋)

λ₋_unstable = Λ₋[3]
ξ₋_unstable = Ξ₋[:,3]
P₋_unstable = manifold_ODE_equilibrium(c₋, real(ξ₋_unstable), real(λ₋_unstable);
    Df = Df₋, f̂ = (u, α) -> f̂(u, [σ, ρ, β], α), order = 20)

λ₋_stable = Λ₋[1:2]
ξ₋_stable = Ξ₋[:,1:2]
P₋_stable = manifold_ODE_equilibrium(complex(c₋), ξ₋_stable, λ₋_stable;
    Df = Df₋, f̂ = (u, α) -> f̂(u, [σ, ρ, β], α), order = (20, 20))
```





## Unstable manifolds of equilibria in DDEs

In this example, we will rigorously compute the unstable manifolds for the equilibria of the Ikeda equation:

```math
\frac{d}{dt}
u(t) = f(u(t), u(t-\tau)) = u(t-\tau) - u(t-\tau)^3.
```

The equilibria are

```math
\begin{aligned}
c_0 &= 0,\\
c_1 &= 1,\\
c_{-1} &= -1.
\end{aligned}
```

The first and second derivative of ``f`` are given by

```math
\begin{aligned}
Df(u, v) &= (0, 1-3v^2),\\
D^2 |f|(u, v) &= \begin{pmatrix} 0 & 0\\ 0 & 6v \end{pmatrix}.
\end{aligned}
```

It follows that the characteristic equation reads

```math
\Psi(\lambda) = (1-3v^2)e^{-\lambda \tau} - \lambda.
```

Lastly,

```math
\hat{f}(u, v) = -v^3.
```

We can now proceed to compute the manifolds and give a rigorous a posteriori error bound

```Julia
using RadiiPolynomial, LinearAlgebra

f(u, v) = v - v^3

Df(u, v) = [0, 1-3v^3]

D²_abs_f(u, v) =
    [0 0
     0 6v]

Ψ(v, τ, λ) = (1-3v^3)*exp(-λ*τ) - λ

f̂(u, v, α) = -(v^3)[α]

## Delay

τ = 1.59

## Manifolds c₀

c₀ = 0.0
λ₀ = newton(0.47208, λ -> (λ - exp(-λ*τ), 1 + τ*exp(-λ*τ)))
ξ₀ = 7.5
P₀ = manifold_DDE_equilibrium(c₀, ξ₀, λ₀;
    Ψ = λ -> Ψ(c₀, τ, λ), f̂ = (u, αλ, α) -> exp(-αλ*τ)*f̂(u, u, α), order = 20)

## Manifolds c₁

c₁ = complex(1.0)
λ₁₊ = newton(0.32056+1.15780im, λ -> (λ + 2exp(-λ*τ), 1 - 2*τ*exp(-λ*τ)))
λ₁ = [λ₁₊, conj(λ₁₊)]
ξ₁ = complex([0.9, 0.9])
P₁ = manifold_DDE_equilibrium(c₁, ξ₁, λ₁;
    Ψ = λ -> Ψ(c₁, τ, λ), f̂ = (u, αλ, α) -> exp(-αλ*τ)*f̂(u, u, α), order = (20, 20))
```
