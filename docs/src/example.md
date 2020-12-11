# Example

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
|D^2 f|(u)
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

```@example
using LinearAlgebra, RadiiPolynomial

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

function abs_D²f(u, p)
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
Λ₀, Ξ₀ = eigen(Df(c₀, [σ, ρ, β]))

λ₀_stable = Λ₀[1]
ξ₀_stable = Ξ₀[:,1]
P₀_stable = manifold_ODE_equilibrium(c₀, real(ξ₀_stable), real(λ₀_stable);
    f̂ = f̂, Df = Df, p = [σ, ρ, β], order = 20)

λ₀_unstable = Λ₀[2:3]
ξ₀_unstable = Ξ₀[:,2:3]
P₀_unstable = manifold_ODE_equilibrium(complex(c₀), ξ₀_unstable, λ₀_unstable;
    f̂ = f̂, Df = Df, p = [σ, ρ, β], order = (20, 20))

## Manifolds c₊

c₊ = [sqrt(β*(ρ - 1)), sqrt(β*(ρ - 1)), ρ - 1]
Λ₊, Ξ₊ = eigen(Df(c₊, [σ, ρ, β]))

λ₊_unstable = Λ₊[3]
ξ₊_unstable = Ξ₊[:,3]
P₊_unstable = manifold_ODE_equilibrium(c₊, real(ξ₊_unstable), real(λ₊_unstable);
    f̂ = f̂, Df = Df, p = [σ, ρ, β], order = 20)

λ₊_stable = Λ₊[1:2]
ξ₊_stable = Ξ₊[:,1:2]
P₊_stable = manifold_ODE_equilibrium(complex(c₊), ξ₊_stable, λ₊_stable;
    f̂ = f̂, Df = Df, p = [σ, ρ, β], order = (20, 20))

## Manifolds c₋

c₋ = [-sqrt(β*(ρ - 1)), -sqrt(β*(ρ - 1)), ρ - 1]
Λ₋, Ξ₋ = eigen(Df(c₋, [σ, ρ, β]))

λ₋_unstable = Λ₋[3]
ξ₋_unstable = Ξ₋[:,3]
P₋_unstable = manifold_ODE_equilibrium(c₋, real(ξ₋_unstable), real(λ₋_unstable);
    f̂ = f̂, Df = Df, p = [σ, ρ, β], order = 20)

λ₋_stable = Λ₋[1:2]
ξ₋_stable = Ξ₋[:,1:2]
P₋_stable = manifold_ODE_equilibrium(complex(c₋), ξ₋_stable, λ₋_stable;
    f̂ = f̂, Df = Df, p = [σ, ρ, β], order = (20, 20))
```
