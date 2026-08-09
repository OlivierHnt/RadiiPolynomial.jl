```@contents
Pages = ["cahn_hilliard.md"]
Depth = 3
```

# Cahn-Hilliard

In this example, we prove the existence of a steady-state of the Cahn-Hilliard equation

```math
\partial_t u = -\Delta \left( \frac{1}{\beta} \Delta u + u - u^3 \right), \qquad x \in \mathbb{T},
```

where ``\mathbb{T}`` is the circle of circumference ``2``. Integrating the steady-state equation twice yields

```math
\Delta u + \beta (u - u^3 - c) = 0,
```

where the constant of integration ``c`` plays the role of the chemical potential, and ``\beta > 0`` is the inverse squared interface width. We look for an even solution, that is a cosine series, which amounts to Neumann boundary conditions on the half-domain.

### Step 1: Problem definition

The map ``F`` and its derivative ``DF`` are implemented as follows:

```@example cahn_hilliard
using RadiiPolynomial

F(u, c, β) = Laplacian() * u + β * (u - u^3 - c)

DF(u, c, β) = Laplacian() + Multiplication(β * (exact(1) - exact(3) * u^2))
nothing # hide
```

The approximate inverse constructed in Step 3 uses the Laplacian's pseudo-inverse on the tail, which we implement as a custom `AbstractDiagonalOperator` by giving its diagonal entries (the ``0``-th mode is mapped to zero since ``\Delta`` is not invertible on constants):

```@example cahn_hilliard
struct PseudoInverseLaplacian <: RadiiPolynomial.AbstractDiagonalOperator end

RadiiPolynomial.getcoefficient(::PseudoInverseLaplacian, (codom, i)::Tuple{SymmetricSpace{<:Fourier},Integer}, (dom, j)::Tuple{SymmetricSpace{<:Fourier},Integer}) =
    (i == j) & !(i == j == 0) ? inv(-(frequency(dom) * exact(i))^2) : zero(frequency(dom))
nothing # hide
```

### Step 2: Approximate zero (floating-point arithmetic)

We fix the parameters and run Newton's method in the space of even Fourier sequences of order ``K``, obtained with `evensym`. The starting profile ``\bar{u} \approx 0.9 \cos(\pi x)`` selects a large-amplitude state whose two phases ``u \approx \pm 1`` are separated by interfaces; this is a different solution from the small-amplitude one continued in the [Parameter continuation for the Cahn-Hilliard equation](@ref) example, which proves a family over a whole region of the ``(c, \beta)`` plane containing the parameters below.

```@example cahn_hilliard
K = 32
c, β = 0.0, 40.15625

u_init = Sequence(evensym(Fourier(K, π)), [c ; 0.45 ; zeros(K-1)])

u_bar, success_newton = newton(u -> (F(u, c, β), DF(u, c, β)), u_init)
nothing # hide
```

The Fourier coefficients decay to the level of machine precision, which confirms that the order ``K`` is large enough:

```@example cahn_hilliard
abs(u_bar[K-1])
```

Since ``c = 0``, the map ``F`` is odd in ``u``, so ``-\bar{u}`` is a solution as well and the even-numbered coefficients of ``\bar{u}`` vanish identically. Note that this is a property of the solution, not an assumption: the proof below is carried out in the full space of even Fourier sequences.

### Step 3: Approximate inverse (floating-point arithmetic)

On the finite part we invert the projected derivative; on the tail we use ``\Delta^{-1}``, which is the leading order behaviour of ``DF(\bar{u})^{-1}`` as the mode index grows. Writing ``\Pi`` for the projection onto the modes ``|k| \leq K``, the operator

```math
A \bydef A_\text{finite} \Pi + \Delta^{-1} (I - \Pi)
```

is our approximate inverse of ``DF(\bar{u})``.

```@example cahn_hilliard
Π = Projection(space(u_bar))

A_finite = inv(Π * DF(u_bar, c, β) * Π)
nothing # hide
```

### Step 4: Bounds estimation (interval arithmetic)

From here on every quantity is enclosed with interval arithmetic. We use the ``\ell^1`` norm on the Fourier coefficients, for which [`norm`](@ref) and [`opnorm`](@ref) account for the multiplicity of each orbit of the symmetry.

```@example cahn_hilliard
using LinearAlgebra

c_i, β_i = interval(c), interval(β)
u_i = interval(u_bar)
A_finite_i = interval(A_finite)
Π_i = interval(Π)

A = A_finite_i + PseudoInverseLaplacian() * (interval(I) - Π_i)
nothing # hide
```

The bound ``Y \geq \| A F(\bar{u}) \|`` is computed directly, since ``F(\bar{u})`` has finitely many nonzero modes:

```@example cahn_hilliard
Y = norm(A * F(u_i, c_i, β_i), Ell1())
```

For ``Z_1 \geq \| I - A DF(\bar{u}) \|`` we split the modes. Since ``\bar{u}^2`` has modes up to ``2K``, the operator ``I - A DF(\bar{u})`` is captured exactly by its truncation to the modes ``|k| \leq 3K+1``. Beyond that truncation, ``\Pi h = 0`` and ``\Pi DF(\bar{u}) h = 0``, so

```math
(I - A DF(\bar{u})) h = - \Delta^{-1} (I - \Pi) \beta (1 - 3 \bar{u}^2) h,
```

whose image only involves modes ``|k| \geq K+2``. The ``\ell^1`` operator norm being a maximum over the columns, the two contributions combine with a maximum:

```@example cahn_hilliard
Π_3K = interval(Projection(evensym(Fourier(3K+1, π))))

Z₁_finite = opnorm(Π_3K - A * (DF(u_i, c_i, β_i) * Π_3K), Ell1())

Z₁_tail = β_i * (interval(1) + interval(3) * norm(u_i, Ell1())^2) / (interval(π) * interval(K+2))^2

Z₁ = max(Z₁_finite, Z₁_tail)
```

For ``Z_2`` we use ``DF(w) - DF(\bar{u}) = -3 \beta (w + \bar{u})(w - \bar{u})`` as a multiplication operator, so that ``\| A (DF(w) - DF(\bar{u})) \| \leq 3 \beta \| A \| (2 \| \bar{u} \| + R) \| w - \bar{u} \|`` for all ``w`` in the ball of radius ``R`` around ``\bar{u}``. The operator ``A`` is block diagonal with respect to ``\Pi``, so its norm is the largest of the two blocks, the tail block being bounded by ``\pi^{-2}``:

```@example cahn_hilliard
R = exact(10sup(Y))

normA = max(opnorm(A_finite_i, Ell1()), interval(1)/interval(π)^2)

Z₂ = exact(3) * β_i * normA * (exact(2) * norm(u_i, Ell1()) + R)
```

The computer-assisted proof is completed by the Radii Polynomial Theorem:

```@example cahn_hilliard
ie, success_contraction = interval_of_existence(Y, Z₁, Z₂, R; verbose = true)
nothing # hide
```

```@example cahn_hilliard
inf(ie) # smallest error
```

There exists a steady-state within a distance `inf(ie)` of ``\bar{u}`` in the ``\ell^1`` norm.

```@example cahn_hilliard
using CairoMakie

fig = Figure()
ax = Axis(fig[1,1]; xlabel = L"x", ylabel = L"\bar{u}(x)")
lines!(ax, [Point2f(x, real(u_bar(x))) for x = LinRange(-1, 1, 501)]; color = :blue)
fig
```
