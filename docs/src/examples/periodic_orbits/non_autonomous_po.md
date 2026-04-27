```@contents
Pages = ["non_autonomous_po.md"]
Depth = 3
```

# Non-autonomous ODE

In this example, we prove the existence of a ``2\pi``-periodic solution to the non-autonomous ODE

```math
u''(t) + \beta_1 u'(t) + \beta_2 u(t) - u(t)^2 = \beta_3 \cos(t),
```

where ``\beta_1 = 1/10``, ``\beta_2 = 4``, ``\beta_3 = 1``.

### Step 1: Problem definition

```@example non_autonomous_po
using RadiiPolynomial

struct LOp{T} <: AbstractLinearOperator
    β₁ :: T
    β₂ :: T
end
RadiiPolynomial.domain(::LOp, codom::Fourier) = codom
RadiiPolynomial.codomain(::LOp, dom::Fourier) = dom
function RadiiPolynomial.getcoefficient(L::LOp, (codom, k)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer})
    x = inv(-exact(k)^2 + exact(im*k)*L.β₁ + L.β₂)
    return ifelse(k == j, x, zero(x))
end
```

```@example non_autonomous_po
F(u, β, c) = u + LOp(β[1], β[2]) * (-u^2 - β[3] * c)

DF(u, β) = exact(I) + LOp(β[1], β[2]) * Multiplication(-exact(2) * u)
```

### Step 2: Approximate zero (floating-point arithmetic)

```@example non_autonomous_po
β_approx = [0.1, 4.0, 1.0]

c = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])

K = 10

u_init = zeros(ComplexF64, Fourier(K, 1.0))

u_bar, success_newton = newton(u -> (F(u, β_approx, c), DF(u, β_approx)), u_init; verbose = true)
conjugacy_symmetry!(u_bar) # impose real-valued solution
nothing # hide
```

```@example non_autonomous_po
using GLMakie
lines(LinRange(-π, π, 101), t -> real(u_bar(t)))
```

### Step 3: Approximate inverse (floating-point arithmetic)

```@example non_autonomous_po
Π = interval( Projection(Fourier(K, 1.0)) )

A_K = interval(inv(mid.(Π * DF(u_bar, β_approx) * Π)))

A = A_K + (interval(I) - Π)
nothing # hide
```

### Step 4: Estimating the bounds (interval arithmetic)

```@example non_autonomous_po
ν = interval(1.1) # does not have to be exactly 11/10
X = Ell1(GeometricWeight(ν))

β_interval = [I"0.1", I"4.0", I"1.0"]
c_interval = interval(c)
u_bar_interval = interval(u_bar)

#- Y bound

Y = norm(A * F(u_bar_interval, β_interval, c_interval), X)

#- Z₁ bound
@assert K ≥ sup(sqrt(β_interval[2]))
Π_2Kp1 = interval( Projection(Fourier(2K+1, 1.0)) )

Z₁ = opnorm(Π_2Kp1 - A * DF(u_bar_interval, β_interval) * Π_2Kp1, X)

#- Z₂ bound
R = Inf

Z₂ = exact(2) * max(opnorm(A_K, X), interval(1))

#

ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, Inf; verbose = true)
nothing # hide
```

```@example non_autonomous_po
inf(ie) # smallest error
```
