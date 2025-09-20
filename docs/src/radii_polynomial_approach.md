# State of the art

The study of complex [dynamical systems](https://en.wikipedia.org/wiki/Dynamical_system) requires numerical computations to access the dynamics. While numerical methods provide accurate approximations, they often come at the cost of rounding, discretization errors and the surrender of an a posteriori error bound between the approximation and the exact solution of the original problem.

Computer-assisted proofs aim to **validate numerical simulations** and **derive mathematical theorems**, thereby binding computational results with topological, geometric and qualitative methods of nonlinear theory.

[RadiiPolynomial.jl](https://github.com/OlivierHnt/RadiiPolynomial.jl) is a Julia package to conduct the computational steps of a type of computer-assisted proofs referred to as the *radii polynomial approach* (see the next section below).

## [Radii polynomial approach](@id radii_polynomial_approach)

Given a problem in dynamical systems (e.g. existence of an invariant set, stability analysis, etc.), one approach of computer-assisted proofs consists in representing the desired solution ``\tx`` as an isolated fixed-point in a Banach space ``X``. The assistance of the computer is used to verify that the corresponding fixed-point operator ``T`` abides by the [Banach Fixed-Point Theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) in a vicinity of a numerical approximation ``\bx``. Note that a particular case of this procedure is the well-known [Newton-Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).

We refer to this strategy as the *radii polynomial approach* since, in practice, we prove the contraction of ``T`` in balls whose radii are determined by the roots of a polynomial. For the sake of completeness, we report the fundamental principles in the following theorem.

```@raw html
<div class="theorem" text="Radii Polynomial Theorem">
```
Let ``X`` be a Banach space, ``U`` an open subset of ``X``, ``T \in C^1(U, X)`` an operator, ``\bx \in U`` and ``R > 0`` such that ``\text{cl}( B_R(\bx) ) \subset U``.
- (First-order) Suppose ``Y, Z_1 \ge 0`` satisfy
```math
\begin{aligned}
\|T(\bx) - \bx\|_X &\le Y,\\
\sup_{x \in \text{cl}( B_R(\bx) )} \|DT(x)\|_{\mathscr{B}(X, X)} &\le Z_1,
\end{aligned}
```
and define the *radii polynomial* by ``p(r) \bydef Y + (Z_1 - 1) r``.
If there exists a *radius* ``\bar{r} \in [0, R]`` such that ``p(\bar{r}) \le 0`` and ``Z_1 < 1``, then ``T`` has a unique fixed-point ``\tx \in \text{cl}( B_{\bar{r}} (\bx) )``.
- (Second-order) Suppose ``Y, Z_1, Z_2 \ge 0`` satisfy
```math
\begin{aligned}
\|T(\bx) - \bx\|_X &\le Y,\\
\|DT(\bx)\|_{\mathscr{B}(X, X)} &\le Z_1,\\
\|DT(x) - DT(\bx)\|_{\mathscr{B}(X, X)} &\le Z_2 \|x - \bx\|_X, \qquad \text{for all } x \in \text{cl}( B_R(\bx) ),
\end{aligned}
```
and define the *radii polynomial* by ``p(r) \bydef Y + (Z_1 - 1) r + \frac{Z_2}{2} r^2``.
If there exists a *radius* ``\bar{r} \in [0, R]`` such that ``p(\bar{r}) \le 0`` and ``Z_1 + Z_2 \bar{r} < 1``, then ``T`` has a unique fixed-point ``\tx \in \text{cl}( B_{\bar{r}} (\bx) )``.
```@raw html
</div>
<br>
```

The set of all possible radii is called the *interval of existence*. Its minimum gives the sharpest computed a posteriori error bound on ``\bx``. On the other hand, its maximum represents the largest computed radius of the ball, centred at ``\bx``, within which the solution is unique.

```@docs
interval_of_existence
```

## Further readings

A mini-lecture series on validated numerics can be found [here](https://olivierhnt.github.io/Computer-assisted-proofs-in-nonlinear-analysis/).
