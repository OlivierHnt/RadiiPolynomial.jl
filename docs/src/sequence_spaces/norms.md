```@setup norms
using RadiiPolynomial
```

# Norms

The spaces introduced in the section [Vector spaces](@ref vector_spaces) are not normed a priori. Indeed, to satisfy the Radii Polynomial Theorem, it is useful to tune the Banach space on the fly.

All Banach spaces mentioned below are a subtype of the abstract type [`BanachSpace`](@ref).

```julia
BanachSpace
├─ NormedCartesianSpace
├─ Weightedℓ¹
├─ ℓ¹
├─ ℓ∞
└─ Hˢ
```

```@docs
BanachSpace
```

## ``\ell^1`` and ``\ell^\infty``

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``.

The ``\ell^1`` space is defined as

```math
\ell^1 := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^1} := \sum_{\alpha \in \mathscr{I}} | a_\alpha | \right\}.
```

and the ``\ell^\infty`` space is defined as

```math
\ell^\infty := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^\infty} := \sup_{\alpha \in \mathscr{I}} | a_\alpha | \right\}.
```

These Banach spaces are representing by the structures [`ℓ¹`](@ref) (`\ell<TAB>\^1<TAB>`) and [`ℓ∞`](@ref) (`\ell<TAB>\infty<TAB>`).

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
norm(a, ℓ¹())
norm(a, ℓ∞())
```

```@docs
ℓ¹
ℓ∞
```

## Weighted ``\ell^1``

The [`Weightedℓ¹`](@ref) (`Weighted\ell<TAB>\^1<TAB>`) represents a weighted ``\ell^1`` space.

```@docs
Weightedℓ¹
```

### Geometric weights

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``.

The geometric weights of rate ``\nu > 0`` are the numbers ``\nu^{|\alpha|}`` for all ``\alpha \in \mathscr{I}``. The corresponding weighted ``\ell^1`` space is defined as

```math
\ell^1_\nu := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^1_\nu} := \sum_{\alpha \in \mathscr{I}} |a_\alpha| \nu^{|\alpha|} \right\}.
```

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
norm(a, Weightedℓ¹(GeometricWeights(2.0)))
```

Note that such a Banach space may also yield a Banach algebra for a suitable choice of the rate ``\nu``. Namely,
- for [`Taylor`](@ref), ``\ell^1_\nu`` is a Banach algebra for all ``\nu > 0`` .
- for [`Fourier`](@ref) and [`Chebyshev`](@ref), ``\ell^1_\nu`` is a Banach algebra for all ``\nu \geq 1``.

```@repl norms
a = Sequence(Taylor(2) ⊗ Fourier(2, 1.) ⊗ Chebyshev(2), ones(3*5*3));
b = Sequence(Taylor(1) ⊗ Fourier(3, 1.) ⊗ Chebyshev(0), ones(2*7*1));
X = Weightedℓ¹((GeometricWeights(0.2), GeometricWeights(1.2), GeometricWeights(2.0)));
norm(a*b, X)
norm(a, X) * norm(b, X)
```

### Algebraic weights

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``.

The algebraic weights of rate ``s \geq 0`` are the numbers ``(1 + |\alpha|)^s`` for all ``\alpha \in \mathscr{I}``. The corresponding weighted ``\ell^1`` space is defined as

```math
\ell^1_s := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{\ell^1_s} := \sum_{\alpha \in \mathscr{I}} |a_\alpha| (1 + |\alpha|)^s \right\}.
```

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
norm(a, Weightedℓ¹(AlgebraicWeights(2.0)))
```

Note that such a Banach space may also yield a Banach algebra.

```@repl norms
a = Sequence(Taylor(2) ⊗ Fourier(2, 1.) ⊗ Chebyshev(2), ones(3*5*3));
b = Sequence(Taylor(1) ⊗ Fourier(3, 1.) ⊗ Chebyshev(0), ones(2*7*1));
X = Weightedℓ¹((AlgebraicWeights(0.2), AlgebraicWeights(1.2), AlgebraicWeights(2.0)));
norm(a*b, X)
norm(a, X) * norm(b, X)
```

## ``H^s``

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}`` and ``s \in [1, +\infty)``. The Sobolev space ``H^s`` is defined as

```math
H^s := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, +\infty > | a |_{H^s} := \left( \sum_{\alpha \in \mathscr{I}} | a_\alpha | \left( 1 + \sum_{i=1}^d | \alpha_i |^2 \right)^s \right)^{1/2} \right\}.
```

The [`Hˢ`](@ref) (`\itH<TAB>\^s<TAB>`) wraps such a ``s``.

```@repl norms
a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5])
norm(a, Hˢ(2.0))
```

```@docs
Hˢ
```

## Normed cartesian space

For the norm of a [`CartesianSpace`](@ref), one may use a [`NormedCartesianSpace`](@ref) to either:
- use the same [`BanachSpace`](@ref) for each space.
- use a different [`BanachSpace`](@ref) for each space.

```@repl norms
a = Sequence(Taylor(1)^2 × Chebyshev(1)^2, [1, 2, 3, 4, 5, 6, 7, 8])
inner = NormedCartesianSpace((Weightedℓ¹(GeometricWeights(2.0)), Weightedℓ¹(AlgebraicWeights(3.0))), ℓ∞())
norm(a, NormedCartesianSpace(inner, ℓ¹()))
```

```@docs
NormedCartesianSpace
```
