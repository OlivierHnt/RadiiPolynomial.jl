# State of the art

The study of complex [dynamical systems](https://en.wikipedia.org/wiki/Dynamical_system) requires numerical computations to access the dynamics. While numerical methods provide accurate approximations, they often come at the cost of rounding, discretization errors and the surrender of an a posteriori error bound between the approximation and the exact solution of the original problem.

A posteriori validation methods are computer-assisted proof techniques used to
- **rigorously validate numerical simulations**, and
- **translate computational results into proven mathematical theorems**.

[RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) is a software library, shipped as an open-source Julia package, that provides a set of abstractions for implementing the so-called *radii polynomial approach* described below.

## [Radii polynomial approach](@id radii_polynomial_approach)

Given a problem in dynamical systems (e.g. existence of an invariant set, stability analysis, etc.), one approach of a posteriori validation consists in representing the desired solution ``\tx`` as an isolated fixed-point in a Banach space ``X``. The assistance of the computer is used to verify that the corresponding fixed-point operator ``T`` abides by the [Banach Fixed-Point Theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) in a vicinity of a numerical approximation ``\bx``.

We refer to this strategy as the *radii polynomial approach* since the contraction of ``T`` is established in a closed ball whose radius is determined by the roots of a polynomial. This is the content of the following theorem.

```@raw html
<div class="theorem" text="Radii Polynomial Theorem">
```
Let ``X`` be a Banach space, ``U`` an open subset of ``X``, ``T \in C^1(U, X)`` an operator, ``\bx \in U`` and ``R \in [0, \infty) \cup \{\infty\}`` such that ``B(\bx, R) \subset U``.
- (First-order) Suppose there are positive constants ``Y, Z_1 = Z_1(R)`` satisfying
```math
\begin{aligned}
\|T(\bx) - \bx\|_X &\le Y,\\
\sup_{x \in B(\bx, R)} \|DT(x)\|_{\mathscr{L}(X, X)} &\le Z_1,
\end{aligned}
```
and define the *radii polynomial* by ``p(r) \bydef Y + (Z_1 - 1) r``.
If there exists a *radius* ``r \in [0, R]`` such that ``p(r) \le 0`` and ``Z_1 < 1``, then ``T`` has a unique fixed-point ``\tx \in B( \bx, r) ``.
- (Second-order) Suppose there are positive constant ``Y, Z_1, Z_2 = Z_2(R)`` satisfying
```math
\begin{aligned}
\|T(\bx) - \bx\|_X &\le Y,\\
\|DT(\bx)\|_{\mathscr{L}(X, X)} &\le Z_1,\\
\|DT(x) - DT(\bx)\|_{\mathscr{L}(X, X)} &\le Z_2 \|x - \bx\|_X, \qquad \text{for all } x \in B(\bx, R),
\end{aligned}
```
and define the *radii polynomial* by ``p(r) \bydef Y + (Z_1 - 1) r + \frac{Z_2}{2} r^2``.
If there exists a *radius* ``r \in [0, R]`` such that ``p(r) \le 0`` and ``Z_1 + Z_2 r < 1``, then ``T`` has a unique fixed-point ``\tx \in B( \bx, r) ``.
```@raw html
</div>
<br>
```

## Bird's-eye view of the validation procedure

A robust approach is to first phrase the problem as finding the zero of an operator ``F`` and prove, using the above Radii Polynomial Theorem, the local contraction near ``\bx`` of the fixed-point operator

```math
T(x) = x - AF(x),
```

where ``A`` is an approximation of the inverse of ``DF(\bx)``.

The mathematical objects involved in the Radii Polynomial Theorem map to data structures within the RadiiPolynomial library.
The validation procedure goes as follows:

- **Step 1: Problem definition**. Choose the Banach space ``X`` and the zero-finding problem ``F(x) = 0``.
    In the library: The underlying domain ``X`` on which the problem is posed is defined by combining a formal topological basis via a [vector space](manual/vector_spaces.md) (e.g., Fourier or Taylor expansions) with corresponding norms and weights using a [Banach space](manual/norms.md).

- **Step 2: Approximate zero ``\bx`` of ``F``**. Compute (with floating-point arithmetic) the approximate zero ``\bx`` of ``F`` using Newton's method to refine an initial guess that came out of some numerical method.
    In the library: The approximate zero ``\bx \in X`` and the finite truncation of ``F(\bx)`` are represented as [sequence](manual/sequences.md) data types. The finite truncation of ``DF(\bx)`` is represented as a [linear operator](manual/linear_operators.md) structure.
    Moreover, to implement ``F`` and ``DF``, a suite of mathematical [special operators](manual/special_operators.md) (such as derivative, integration, evaluation, etc.) is available. The truncation of the Banach space ``X`` is materialized by the [projection](manual/special_operators.md) operator.

- **Step 3: Approximate inverse ``A`` of ``DF(\bx)``**. Construct (with floating-point arithmetic) the approximate inverse ``A`` of ``DF(\bx)`` exploiting the structure of ``DF(\bx)``.

- **Step 4: Bounds estimation**. Compute (with interval arithmetic) the bounds. If ``F`` is quadratic or lower order, then ``R = \infty``, otherwise we make the heuristic choice ``R = 10^k Y`` for some integer ``k \ge 1``.
    The set of all possible radii is called the *interval of existence*. Its minimum gives the sharpest computed a posteriori error bound on ``\bx``. On the other hand, its maximum represents the largest computed radius of the ball, centred at ``\bx``, within which the solution is unique.

```@docs
interval_of_existence
```
