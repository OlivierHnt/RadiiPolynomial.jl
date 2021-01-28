# Radii polynomial

## [Radii Polynomial Theorem](@id radii_polynomial_thm)

Let ``N \in \mathbb{N}``, ``X`` be a normed vector space, ``U`` be an open subset of ``X`` and ``G \in C^m (U,X)`` where ``m \geq N``. Consider ``x_0`` and ``r_0>0`` such that ``\overline{B_{r_0}(x_0)} \subset U``.
Suppose ``Y, Z_1, \dots, Z_{N-1} \geq 0`` and ``Z_N : [0,+\infty) \to [0,+\infty)`` satisfy

```math
\begin{aligned}
\| G(x_0) - x_0 \|_X &\leq Y,\\ \| D^k G(x_0) \|_{B(X^{\otimes k}, X)} &\leq Z_k \quad (k=1,\dots,N-1),\\ \sup_{y \in \overline{B_{r_0}(x_0)}} \| D^N G(y) \|_{B(X^{\otimes N}, X)} &\leq Z_N(r_0).
\end{aligned}
```

Define

```math
p_N(r,r_0) \doteqdot Y - r + \sum_{k=1}^{N-1} \frac{r^k}{k!} Z_k + \frac{r^N}{N!} Z_N(r_0).
```

If there exists ``r_1 \in [0, r_0]`` such that

```math
\begin{aligned} p_N(r_1,r_0) &\leq 0,\\ \frac{r_1^{N-1}}{(N-1)!} Z_N(r_0) &< 1, \end{aligned}
```

then ``G`` has a unique fixed point ``\tilde{x} \in \overline{B_{r_1}(x_0)}``.

!!! note
    For Newton-like operators, the Radii Polynomial Theorem is an instance of the [Newton-Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).

!!! note
    While the above theorem is given for arbitrary ``N``, we focus on the cases ``N = 1`` and ``N = 2``.



## Finite dimensional problems

If ``X`` is a finite dimensional normed vector space, then every bounds presented above can be readily computed and made rigorous via interval arithmetic (cf. [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)).

For the proof, you can create:
- a [`FixedPointProblemFiniteDimension`](@ref) when the desired solution is the fixed point of a contracting map ``G``;
- a [`ZeroFindingProblemFiniteDimension`](@ref) when the desired solution is the zero of a map ``F``. The Radii Polynomial Theorem is then applied to a Newton-like operator ``G \doteqdot Id - A F`` where ``A`` is a user-defined injective linear operator.

Once the finite dimensional problem has been set, one can call the [`Y`](@ref), [`Z₁`](@ref), [`Z₂`](@ref) functions to obtain the corresponding bound. Then, the roots of the radii polynomial can be computed via [`roots_radii_polynomial`](@ref).

As a rule of thumb, it is recommended to use the [`Radii Polynomial Theorem`](@ref radii_polynomial_thm) with ``N = 1`` if the uniqueness is not important.




## Infinite dimensional problems

If ``X`` is a Banach space, then the computation of the bounds is more delicate. In the context of dynamical systems, the desired solution is usually an element of a function space (e.g. periodic orbits, homoclinic orbits, etc.). Based on their regularity, such function can be represented by an infinite sum over some basis. Equivalently, this function can be seen as an element of a sequence space.

We define a sequence space ``\ell^1_{\mathbb{S}_1 \times \dots \times \mathbb{S}_d,\nu}``, where ``\mathbb{S}_i = \mathbb{N} \cup \{0\}, \mathbb{Z}`` for ``i = 1,\dots,d``, by

```math
\ell^1_{\mathbb{S}_1 \times \dots \times \mathbb{S}_d,\nu} \doteqdot
\left\{ \{a_\alpha\}_{\alpha \in \mathbb{S}_1 \times \dots \times \mathbb{S}_d} \, : \, \sum_{\alpha \in \mathbb{S}_1 \times \dots \times \mathbb{S}_d} | a_\alpha | \nu^α < +\infty \right\}.
```

We define the truncation operator

```math
(\pi^n h)_\alpha \doteqdot
\begin{cases}
h_\alpha, & \|\alpha\|_\infty \leq n,\\
0, & \|\alpha\|_\infty > n,
\end{cases}, \qquad
\pi^{\infty(n)} \doteqdot Id - \pi^n.
```

For the proof, you can create:
- a [`TailProblem`](@ref) when the desired solution is the fixed point of a map ``G = \pi^{\infty(n)} L \pi^{\infty(n)} f \pi^{\infty(n)}``. Theoretical bound of ``\pi^{\infty(n)} L \pi^{\infty(n)}`` must be provided.
- a [`ZeroFindingProblemCategory1`](@ref) when the desired solution is the zero of a map ``F \doteqdot L + f`` where ``L`` is an invertible linear operator. The Radii Polynomial Theorem is then applied to a Newton-like operator ``G \doteqdot Id - A F`` with ``A \doteqdot \pi^n A \pi^n + \pi^{\infty(n)} L^{-1} \pi^{\infty(n)}`` where ``\pi^n A \pi^n`` is a user-defined injective linear operator. Theoretical bounds of ``\pi^{\infty(n)} L^{-1} \pi^{\infty(n)}`` and ``\pi^n L \pi^{\infty(k)}`` for some ``k \geq n`` must be provided.
