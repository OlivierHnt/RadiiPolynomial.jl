# Radii polynomial

## Radii Polynomial Theorem

Let ``N \in \mathbb{N}``, ``X`` be a normed vector space, ``U`` be an open subset of ``X`` and ``G \in C^m (U,X)`` where ``m \geq N``. Consider ``x_0`` and ``r_*>0`` such that ``\overline{B_{r_*}(x_0)} \subset U``.
Suppose ``Y, Z_1, \dots, Z_{N-1} \geq 0`` and ``Z_N : [0,+\infty) \to [0,+\infty)`` satisfy

```math
\begin{aligned}
\| G(x_0) - x_0 \|_X &\leq Y,\\ \| D^k G(x_0) \|_{B(X^{\otimes k}, X)} &\leq Z_k \quad (k=1,\dots,N-1),\\ \sup_{y \in \overline{B_{r_*}(x_0)}} \| D^N G(y) \|_{B(X^{\otimes N}, X)} &\leq Z_N(r_*).
\end{aligned}
```

Define

```math
p_N(r,r_*) \doteqdot Y - r + \sum_{k=1}^{N-1} \frac{r^k}{k!} Z_k + \frac{r^N}{N!} Z_N(r_*).
```

If there exists ``r_0 \in [0, r_*]`` such that

```math
\begin{aligned} p_N(r_0,r_*) &\leq 0,\\ \frac{r_0^{N-1}}{(N-1)!} Z_N(r_*) &< 1, \end{aligned}
```

then ``G`` has a unique fixed point ``\tilde{x} \in \overline{B_{r_0}(x_0)}``.

!!! note
    For Newton-like operators, the Radii Polynomial Theorem is an instance of the [Newton-Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).

!!! note
    The current state of the package only implements the cases ``N = 1`` and ``N = 2``.

## Finite dimensional problems

If ``X`` is a finite dimensional normed vector space, then every bounds presented above can be readily computed and made rigorous via interval arithmetic (cf. [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)).

As a rule of thumb, we recommend to use:
- ``N = 1`` if the uniqueness is not important or can be achieved analytically;
- ``N = 2`` otherwise.

## Infinite dimensional problems

If ``X`` is a Banach space, then the computation of the bounds depends on each problem. In the context of dynamical systems, the desired solution is usually an element of a function space (e.g. periodic orbits, homoclinic orbits, etc.). Based on their regularity, such function can be represented by an infinite sum over some basis. Equivalently, this function can be seen as an element of a sequence space.

Typically, we consider a sequence space ``\ell^1_{\mathbb{S}_1 \times \dots \times \mathbb{S}_d,\nu}``, where ``\mathbb{S}_i = \mathbb{N}, \mathbb{Z}`` for ``i = 1,\dots,d``, defined as

```math
\ell^1_{\mathbb{S}_1 \times \dots \times \mathbb{S}_d,\nu} \doteqdot
\{ a \, : \, \sum_{\alpha} | a_\alpha | \nu^α < +\infty \textnormal{ where } \alpha \in \mathbb{S}_1 \times \dots \times \mathbb{S}_d\}.
```

Therefore, to perform such computer-assisted proofs the [`RadiiPolynomial.jl`](https://github.com/OlivierHnt/RadiiPolynomial.jl) package handles sequence spaces with their elements and other related objects such as linear functionals and linear operators.
