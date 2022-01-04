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
\ell^1 := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^1} := \sum_{\alpha \in \mathscr{I}} | a_\alpha | < +\infty \right\}.
```

and the ``\ell^\infty`` space is defined as

```math
\ell^\infty := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^\infty} := \sup_{\alpha \in \mathscr{I}} | a_\alpha | < +\infty \right\}.
```

These Banach spaces are representing by the structures [`ℓ¹`](@ref) (`\ell<TAB>\^1<TAB>`) and [`ℓ∞`](@ref) (`\ell<TAB>\infty<TAB>`).

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0]);
norm(a, ℓ¹())
norm(a, ℓ∞())
```

```@docs
ℓ¹
ℓ∞
```

## Weighted ``\ell^1``

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``.

The weighted ``\ell^1`` space is defined as

```math
\ell^1_w := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^1_\nu} := \sum_{\alpha \in \mathscr{I}} |a_\alpha| w_1(\alpha_1) \cdots w_d(\alpha_d) < +\infty \right\},
```

where, for ``i = 1, \dots, d``, either:
- ``w_i`` is a sequence of geometric weights of rate ``\nu_i > 0`` given by ``w_i(\alpha_i) := \nu_i^{|\alpha_i|}``;
- ``w_i`` is a sequence of algebraic weights of rate ``s_i \geq 0`` given by ``w_i(\alpha_i) := (1 + |\alpha_i|)^{s_i}``.

[`Weightedℓ¹`](@ref) (`Weighted\ell<TAB>\^1<TAB>`) represents a weighted ``\ell^1`` space by wrapping `GeometricWeights` and `AlgebraicWeights`.

```@repl norms
a = Sequence(Taylor(2), [1.0, 2.0, 3.0]);
norm(a, Weightedℓ¹(GeometricWeights(2.0)))
norm(a, Weightedℓ¹(AlgebraicWeights(2.0)))
b = Sequence(Taylor(2) ⊗ Fourier(2, 1.) ⊗ Chebyshev(2), ones(3*5*3));
norm(b, Weightedℓ¹((GeometricWeights(0.2), GeometricWeights(1.2), AlgebraicWeights(0.9))))
```

```@docs
Weightedℓ¹
```

## ``H^s``

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}`` and ``s \in [1, +\infty)``. The Sobolev space ``H^s`` is defined as

```math
H^s := \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{H^s} := \left( \sum_{\alpha \in \mathscr{I}} | a_\alpha | \left( 1 + | \alpha_1 |^2 + \dots + | \alpha_d |^2 \right)^s \right)^{1/2} < +\infty \right\}.
```

[`Hˢ`](@ref) (`H\^s<TAB>`) represents the Sobolev space ``H^s`` by wrapping the exponent ``s``.

```@repl norms
a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]);
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
a = Sequence(Taylor(1)^2 × Chebyshev(1)^2, [1, 2, 3, 4, 5, 6, 7, 8]);
inner = NormedCartesianSpace((Weightedℓ¹(GeometricWeights(2.0)), Weightedℓ¹(AlgebraicWeights(3.0))), ℓ∞());
norm(a, NormedCartesianSpace(inner, ℓ¹()))
```

```@docs
NormedCartesianSpace
```
