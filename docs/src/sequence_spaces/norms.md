```@setup norms
using RadiiPolynomial
```

# Norms

The choice of the Banach space to apply the Radii Polynomial Theorem (cf. Section [Radii polynomial approach](@ref radii_polynomial_approach)) is integral to the success of the computer-assisted proof. In practice, it is useful to tune the Banach space on the fly to adjust the norm estimates.

Accordingly, the spaces introduced in Section [Vector spaces](@ref vector_spaces) are not normed a priori. The norm of a [`Sequence`](@ref) or a [`LinearOperator`](@ref) is obtained via the functions `norm` and `opnorm` respectively; in both cases, one must specify a [`BanachSpace`](@ref).

```julia
BanachSpace
├─ NormedCartesianSpace
├─ ℓ¹
├─ ℓ²
└─ ℓ∞
```

```@docs
BanachSpace
```

## ``\ell^1``, ``\ell^2`` and ``\ell^\infty``

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``. Consider the weighted ``\ell^1, \ell^2, \ell^\infty`` spaces defined by

```math
\begin{aligned}
\ell^1_w &:= \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^1_w} := \sum_{\alpha \in \mathscr{I}} |a_\alpha| w(\alpha) < +\infty \right\}, \\
\ell^2_w &:= \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^2_w} := \sqrt{\sum_{\alpha \in \mathscr{I}} |a_\alpha|^2 w(\alpha)} < +\infty \right\}, \\
\ell^\infty_w &:= \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^\infty_w} := \sup_{\alpha \in \mathscr{I}} | a_\alpha | w(\alpha) < +\infty \right\},
\end{aligned}
```

where ``w : \mathscr{I} \to (0, +\infty)`` is a weight function.

The Banach spaces [`ℓ¹`](@ref) (`\ell<TAB>\^1<TAB>`), [`ℓ²`](@ref) (`\ell<TAB>\^2<TAB>`) and [`ℓ∞`](@ref) (`\ell<TAB>\infty<TAB>`) wraps one or multiple [`Weight`](@ref).

```julia
Weight
├─ AlgebraicWeight
├─ BesselWeight
├─ GeometricWeight
└─ IdentityWeight
```

Given a set of indices ``\mathscr{I}^\prime \subset \mathbb{Z}``:

- an [`AlgebraicWeight`](@ref) of rate ``s \geq 0`` is defined by ``w(\alpha) := (1 + |\alpha|)^s`` for all ``\alpha \in \mathscr{I}^\prime``.

- a [`BesselWeight`](@ref) of rate ``s \geq 0`` is defined by ``w(\alpha) := (1 + |\alpha|)^s`` for all ``\alpha \in \mathscr{I}^\prime``. This weight is specific to [`ℓ²`](@ref) and [`Fourier`](@ref) as it describes the [Sobolev space](https://en.wikipedia.org/wiki/Sobolev_space) ``H^s``.

- a [`GeometricWeight`](@ref) of rate ``\nu > 0`` is defined by ``w(\alpha) := \nu^{|\alpha|}`` for all ``\alpha \in \mathscr{I}^\prime``.

- an [`IdentityWeight`](@ref) is defined by ``w(\alpha) := 1`` for all ``\alpha \in \mathscr{I}^\prime``. This is the default weight for [`ℓ¹`](@ref), [`ℓ²`](@ref) and [`ℓ∞`](@ref).

```@repl norms
a = Sequence(Taylor(2), [1.0, 1.0, 1.0]); # 1 + x + x^2
norm(a, ℓ¹(AlgebraicWeight(1.0)))
b = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]); # cos(x)
norm(b, ℓ²(BesselWeight(2.0)))
c = Sequence(Chebyshev(2), [1.0, 0.5, 0.5]); # 1 + 2(x/2 + (2x^2 - 1)/2)
norm(c, ℓ∞()) # ℓ∞() == ℓ∞(IdentityWeight())
```

In the context of a ``d``-dimensional [`TensorSpace`](@ref), one prescribes weights ``w_1, \dots, w_d`` for each dimension. The weight is defined by ``w(\alpha) = w_1(\alpha_1) \times \dots \times w_d(\alpha_d)`` for all ``\alpha = (\alpha_1, \dots, \alpha_d) \in \mathscr{I}^{\prime\prime}`` where ``\mathscr{I}^{\prime\prime} \subset \mathbb{Z}^d`` is the appropriate set of indices.

```@repl norms
a = Sequence(Taylor(2) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(2), ones(3*5*3));
norm(a, ℓ¹((AlgebraicWeight(1.0), GeometricWeight(2.0), IdentityWeight())))
```

However, the ``d``-dimensional version of [`BesselWeight`](@ref) is defined by ``w(\alpha) := (1 + |\alpha_1| + \dots + |\alpha_d|)^s`` for all ``\alpha = (\alpha_1, \dots, \alpha_d) \in \mathbb{Z}^d``. Only one [`BesselWeight`](@ref) is required for every [`Fourier`](@ref) space composing the [`TensorSpace`](@ref).

```@repl norms
a = Sequence(Fourier(2, 1.0) ⊗ Fourier(3, 1.0), ones(5*7));
norm(a, ℓ²(BesselWeight(2.0)))
```

```@docs
ℓ¹
ℓ²
ℓ∞
Weight
IdentityWeight
GeometricWeight
AlgebraicWeight
BesselWeight
```

## Normed cartesian space

For the norm of a [`CartesianSpace`](@ref), one may use a [`NormedCartesianSpace`](@ref) to either:
- use the same [`BanachSpace`](@ref) for each space.
- use a different [`BanachSpace`](@ref) for each space.

```@repl norms
a = Sequence(Taylor(1)^2, [1.0, 2.0, 3.0, 4.0]);
norm(a, NormedCartesianSpace(ℓ¹(), ℓ∞()))
norm(a, NormedCartesianSpace((ℓ¹(), ℓ²()), ℓ∞()))
```

```@docs
NormedCartesianSpace
```
