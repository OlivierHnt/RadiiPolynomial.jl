```@setup norms
using RadiiPolynomial
```

# Norms

The choice of the Banach space to apply the Radii Polynomial Theorem (cf. Section [Radii polynomial approach](@ref radii_polynomial_approach)) is integral to the success of the computer-assisted proof. In practice, it is useful to tune the Banach space on the fly to adjust the norm estimates.

Accordingly, the spaces introduced in Section [Vector spaces](@ref vector_spaces) are not normed a priori. The norm of a [`Sequence`](@ref) or a [`LinearOperator`](@ref) is obtained via the functions [`norm`](@ref) and [`opnorm`](@ref) respectively; in both cases, one must specify a [`BanachSpace`](@ref).

```
BanachSpace
├─ NormedCartesianSpace
├─ Ell1
├─ Ell2
└─ EllInf
```

## ``\ell^1``, ``\ell^2`` and ``\ell^\infty``

Let ``\mathscr{I}`` be a set of indices such that ``\mathscr{I} \subset \mathbb{Z}^d`` for some ``d \in \mathbb{N}``. Consider the weighted ``\ell^1, \ell^2, \ell^\infty`` spaces (cf. [``\ell^p`` spaces](https://en.wikipedia.org/wiki/Sequence_space#ℓp_spaces)) defined by

```math
\begin{aligned}
\ell^1_w &\bydef \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^1_w} \bydef \sum_{\alpha \in \mathscr{I}} |a_\alpha| w(\alpha) < \infty \right\}, \\
\ell^2_w &\bydef \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^2_w} \bydef \sqrt{\sum_{\alpha \in \mathscr{I}} |a_\alpha|^2 w(\alpha)} < \infty \right\}, \\
\ell^\infty_w &\bydef \left\{ a \in \mathbb{C}^\mathscr{I} \, : \, | a |_{\ell^\infty_w} \bydef \sup_{\alpha \in \mathscr{I}} | a_\alpha | w(\alpha) < \infty \right\},
\end{aligned}
```

where ``w : \mathscr{I} \to (0, \infty)`` is a weight function.

The Banach spaces [`Ell1`](@ref), [`Ell2`](@ref) and [`EllInf`](@ref) wraps one or multiple [`Weight`](@ref).

```
Weight
├─ AlgebraicWeight
├─ BesselWeight
├─ GeometricWeight
└─ IdentityWeight
```

Given a set of indices ``\mathscr{I}^\prime \subset \mathbb{Z}``:

- an [`AlgebraicWeight`](@ref) of rate ``s \ge 0`` is defined by ``w(\alpha) \bydef (1 + |\alpha|)^s`` for all ``\alpha \in \mathscr{I}^\prime``.

- a [`BesselWeight`](@ref) of rate ``s \ge 0`` is defined by ``w(\alpha) \bydef (1 + \alpha^2)^s`` for all ``\alpha \in \mathscr{I}^\prime``. This weight is specific to [`Ell2`](@ref) and [`Fourier`](@ref) as it describes the [Sobolev space](https://en.wikipedia.org/wiki/Sobolev_space) ``H^s``.

- a [`GeometricWeight`](@ref) of rate ``\nu > 0`` is defined by ``w(\alpha) \bydef \nu^{|\alpha|}`` for all ``\alpha \in \mathscr{I}^\prime``.

- an [`IdentityWeight`](@ref) is defined by ``w(\alpha) \bydef 1`` for all ``\alpha \in \mathscr{I}^\prime``. This is the default weight for [`Ell1`](@ref), [`Ell2`](@ref) and [`EllInf`](@ref).

```@repl norms
a = Sequence(Taylor(2), [1.0, 1.0, 1.0]); # 1 + x + x^2
norm(a, Ell1(AlgebraicWeight(1.0)))
b = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]); # cos(x)
norm(b, Ell2(BesselWeight(2.0)))
c = Sequence(Chebyshev(2), [1.0, 0.5, 0.5]); # 1 + 2(x/2 + (2x^2 - 1)/2)
norm(c, EllInf()) # EllInf() == EllInf(IdentityWeight())
```

Note that [`ℓ¹`](@ref) (`\ell<tab>\^1<tab>`), [`ℓ²`](@ref) (`\ell<tab>\^2<tab>`) and [`ℓ∞`](@ref) (`\ell<tab>\infty<tab>`) are the respective unicode aliases of [`Ell1`](@ref), [`Ell2`](@ref) and [`EllInf`](@ref).

In the context of a ``d``-dimensional [`TensorSpace`](@ref), one prescribes weights ``w_1, \dots, w_d`` for each dimension. The weight is defined by ``w(\alpha) = w_1(\alpha_1) \times \ldots \times w_d(\alpha_d)`` for all ``\alpha = (\alpha_1, \dots, \alpha_d) \in \mathscr{I}^{\prime\prime}`` where ``\mathscr{I}^{\prime\prime} \subset \mathbb{Z}^d`` is the appropriate set of indices.

```@repl norms
a = ones(Taylor(2) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(2));
norm(a, Ell1((AlgebraicWeight(1.0), GeometricWeight(2.0), IdentityWeight())))
```

However, the ``d``-dimensional version of [`BesselWeight`](@ref) is defined by ``w(\alpha) \bydef (1 + \alpha_1^2 + \ldots + \alpha_d^2)^s`` for all ``\alpha = (\alpha_1, \dots, \alpha_d) \in \mathbb{Z}^d``. Only one [`BesselWeight`](@ref) is required for every [`Fourier`](@ref) space composing the [`TensorSpace`](@ref).

```@repl norms
a = ones(Fourier(2, 1.0) ⊗ Fourier(3, 1.0));
norm(a, Ell2(BesselWeight(2.0)))
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

## API

```@docs
BanachSpace
norm
opnorm
Weight
IdentityWeight
GeometricWeight
geometricweight
AlgebraicWeight
algebraicweight
BesselWeight
Ell1
ℓ¹
Ell2
ℓ²
EllInf
ℓ∞
NormedCartesianSpace
```
