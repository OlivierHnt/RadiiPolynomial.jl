```@contents
Pages = ["nonlinear_diffusion.md"]
Depth = 3
```

# Nonlinear diffusion

In this example, we prove the existence of a steady-state of a reaction-diffusion equation with nonlinear diffusion

```math
\begin{cases}
\partial_t u = \Delta \Phi(u) + R(u), & x \in (0, 1), \\
\displaystyle \frac{\partial u}{\partial n} = 0, & x \in \{0, 1\},
\end{cases}
```

with

```math
\begin{aligned}
\Phi(u) &= u^2, \\
R(u) &= u - u^2 + g(x), \\
g(x) &= \frac{1}{2} + 3 \cos(\pi x) + 2 \cos(2 \pi x) - \cos(3 \pi x) + 6 \cos(4 \pi x).
\end{aligned}
```

See the reference[^1] for more details.

[^1]: M. Breden, [Computer-assisted proofs for some nonlinear diffusion problems](https://doi.org/10.1016/j.cnsns.2022.106292), *Communications in Nonlinear Science and Numerical Simulation*, **109** (2022), 106292.

### Step 1: Problem definition

```@example nonlinear_diffusion
Φ(u) = u^2
DΦ(u) = exact(2) * u

R(u, g) = u - u^2 + g
DR(u) = exact(1) - exact(2) * u

F(u, g) = Laplacian() * Φ(u) + R(u, g)
DF(u) = Laplacian() * Multiplication(DΦ(u)) + Multiplication(DR(u))
nothing # hide
```

### Step 2: Approximate zero (floating-point arithmetic)

```@example nonlinear_diffusion
using RadiiPolynomial

g = Sequence(evensym(Fourier(4, π)), [1/2, 3/2, 1, -1/2, 3])

K = 20

u_init = Sequence(evensym(Fourier(K, π)),
    [1.362741344081890 ; 0.052107816731015 ; 0.008200891820525 ; -0.002635040629728 ; 0.007018830076427 ; zeros(K-4)])

u_bar, newton_success = newton(u -> (F(u, g), DF(u)), u_init; verbose = true)
nothing # hide
```

```@example nonlinear_diffusion
using GLMakie
lines(LinRange(0, 1, 201), t -> real(u_bar(t)))
```

### Step 3: Approximate inverse (floating-point arithmetic)

```@example nonlinear_diffusion
struct PseudoInverseLaplacian <: AbstractLinearOperator end
RadiiPolynomial.domain(::PseudoInverseLaplacian, s::SymmetricSpace{<:Fourier}) = s
RadiiPolynomial.codomain(::PseudoInverseLaplacian, s::SymmetricSpace{<:Fourier}) = s
RadiiPolynomial.getcoefficient(::PseudoInverseLaplacian, (codom, i)::Tuple{SymmetricSpace{<:Fourier},Integer}, (dom, j)::Tuple{SymmetricSpace{<:Fourier},Integer}) =
    (i == j) & !(i == j == 0) ? inv(- (frequency(dom) * exact(i))^2) : zero(frequency(dom))
```

```@example nonlinear_diffusion
Π = Projection(evensym(Fourier(K, π)))
Π_2K = Projection(evensym(Fourier(2K, π)))
A_finite = interval(Π * inv(Π_2K * DF(u_bar) * Π_2K) * Π)

M_cos = DΦ(u_bar)
M_fou = Projection(Fourier(K, π)) * DΦ(u_bar)
M_fou_inv = inv(M_fou)
M_cos_inv_interval = interval(Π * M_fou_inv)

A_tail = (Multiplication(M_cos_inv_interval) - interval(Π) * Multiplication(M_cos_inv_interval) * interval(Π)) * PseudoInverseLaplacian()

A = A_finite + A_tail
nothing # hide
```

### Step 4: Estimating the bounds (interval arithmetic)

```@example nonlinear_diffusion
g_interval = interval(g)
u_bar_interval = interval(u_bar)

#- Y bound

Y = norm(A * F(u_bar_interval, g_interval), 1)

#- Z₁ bound

Z₁_finite = opnorm(interval(Π_2K) - A * DF(u_bar_interval) * interval(Π_2K), 1)

Z₁_tail = norm(exact(1) - M_cos_inv_interval * DΦ(u_bar_interval)) +
    norm(M_cos_inv_interval, 1) / (exact(K+1)^2 * interval(π)^2) * norm(DR(u_bar_interval), 1)

Z₁ = max(Z₁_finite, Z₁_tail)

#- Z₂ bound
opnorm_A_Delta = max(opnorm(A * Laplacian() * interval(Π), 1), norm(M_cos_inv_interval, 1))

Z₂ = exact(4) * opnorm_A_Delta

#

ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, Inf; verbose = true)
nothing # hide
```

```@example nonlinear_diffusion
inf(ie) # smallest error
```
