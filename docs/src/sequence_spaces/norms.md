```@setup norms
using RadiiPolynomial
```

# Norms

The spaces introduced in the section [Vector spaces](@ref vector_spaces) are not normed a priori. Indeed, to satisfy the Radii Polynomial Theorem, it is useful to tune the Banach space by adjusting its norm on the fly.

RadiiPolynomial defines norms which are known to turn a [`VectorSpace`](@ref) into a Banach space or a Banach algebra.

All norms mentioned below are a subtype of the abstract type [`Norm`](@ref).

```julia
Norm
├─ CartesianPowerNorm
├─ CartesianProductNorm
├─ Weightedℓ¹Norm
├─ ℓᵖNorm
└─ 𝐻ˢNorm
```

```@docs
Norm
```

## ``\ell^p`` norm

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}`` and ``p \in [1, +\infty) \cup \{ +\infty \}``. The ``\ell^p`` space is defined as

```math
\ell^p := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^p} :=
\begin{cases}
\left( \sum_{\alpha \in \mathscr{I}} | a_\alpha | ^ p \right)^{1/p}, & 1 \leq p < +\infty,\\
\sup_{\alpha \in \mathscr{I}} | a_\alpha |, & p = +\infty.
\end{cases} \right\}.
```

The [`ℓᵖNorm`](@ref) (`\ell<TAB>\^p<TAB>Norm`) wraps such a ``p``.

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
norm(a, ℓᵖNorm(Inf))
```

```@docs
ℓᵖNorm
```

## Weighted ``\ell^1`` norm

The [`Weightedℓ¹Norm`](@ref) (`Weighted\ell<TAB>\^1<TAB>Norm`) represents the norm for a weighted ``\ell^1`` space.

```@docs
Weightedℓ¹Norm
```

### Geometric weights

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``.

The geometric weights of rate ``\nu > 0`` are the numbers ``\nu^{|\alpha|}`` for all ``\alpha \in \mathscr{I}``. The corresponding weighted ``\ell^1`` space is defined as

```math
\ell^1_\nu := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^1_\nu} := \sum_{\alpha \in \mathscr{I}} |a_\alpha| \nu^{|\alpha|} \right\}.
```

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
norm(a, Weightedℓ¹Norm(GeometricWeights(2.0)))
```

### Algebraic weights

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``.

The algebraic weights of rate ``s \geq 0`` are the numbers ``(1 + |\alpha|)^s`` for all ``\alpha \in \mathscr{I}``. The corresponding weighted ``\ell^1`` space is defined as

```math
\ell^1_s := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^1_s} := \sum_{\alpha \in \mathscr{I}} |a_\alpha| (1 + |\alpha|)^s \right\}.
```

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
norm(a, Weightedℓ¹Norm(AlgebraicWeights(2.0)))
```

## ``H^s`` norm

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}`` and ``s \in [1, +\infty)``. The Sobolev space ``H^s`` is defined as

```math
H^s := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{H^s} := \left( \sum_{\alpha \in \mathscr{I}} | a_\alpha | \left( 1 + \sum_{i=1}^d | \alpha_i |^2 \right)^s \right)^{1/2} \right\}.
```

The [`𝐻ˢNorm`](@ref) (`\itH<TAB>\^s<TAB>Norm`) wraps such a ``s``.

```@repl norms
a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
norm(a, 𝐻ˢNorm(2.0))
```

```@docs
𝐻ˢNorm
```

## Cartesian norms

One may use:
- [`CartesianPowerNorm`](@ref) to use the same norm for each space constituting a [`CartesianSpace`](@ref).
- [`CartesianProductNorm`](@ref) to use a different norm for each space constituting a [`CartesianSpace`](@ref).

```@repl norms
a = Sequence(Taylor(1)^2 × Chebyshev(1)^2, [1, 2, 3, 4, 5, 6, 7, 8])
inner_norm = CartesianProductNorm((Weightedℓ¹Norm(GeometricWeights(2.0)), Weightedℓ¹Norm(AlgebraicWeights(3.0))), ℓᵖNorm(Inf))
norm(c, CartesianPowerNorm(inner_norm, ℓᵖNorm(1)))
```

```@docs
CartesianPowerNorm
CartesianProductNorm
```
