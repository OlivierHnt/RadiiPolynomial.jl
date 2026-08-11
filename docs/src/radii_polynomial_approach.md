# The radii polynomial approach

The study of [dynamical systems](https://en.wikipedia.org/wiki/Dynamical_system) requires numerical computations to access the dynamics.
While numerical methods provide accurate approximations, they often come at the cost of rounding, discretization errors and the surrender of an a posteriori error bound between the approximation and the exact solution of the original problem.

A posteriori validation methods are computer-assisted proof techniques used to
- **rigorously validate numerical simulations**, and
- **translate computational results into proven mathematical theorems**.

[RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) is a software library, shipped as an open-source Julia package, that provides a set of abstractions for implementing the so-called *radii polynomial approach* described below.

## [Motivation: A complete proof in a few lines](@id motivation)

Before any sequence space, here is the whole method on a scalar equation.
We prove that ``F(u) = u^3 - 2`` has a root near ``1.3``, by showing that the quasi-Newton operator ``T(u) \bydef u - A F(u)`` is a local contraction:

```@example intro
using RadiiPolynomial

F(u)  = u^3 - exact(2)                                # Step 1: formulation
DF(u) = exact(3) * u^2

u_bar, converged = newton(u -> (F(u), DF(u)), 1.0)    # Step 2: approximation
A = inv(DF(u_bar))

R   = 0.1                                             # Step 3: bounds
u_R = interval(u_bar, R; format = :midpoint)          # the ball B(ū, R), exactly
Y   = abs(interval(A) * F(interval(u_bar)))
Z₁  = abs(interval(1) - interval(A) * DF(u_R))

ie, proved = interval_of_existence(Y, Z₁, R)          # Step 4: conclusion
```

`proved == true`, so there is a genuine root within `inf(ie)` of `u_bar`, and it is the only one within `sup(ie)`:

```@example intro
inf(ie), sup(ie)
```

Everything below is this same argument, carried out in Banach spaces where the unknown can be modelled by sequences rather than a number.
A complete walkthrough of this example is given in [A first proof](@ref).

## [Radii polynomial approach](@id radii_polynomial_approach)

Given a problem in dynamical systems (e.g. existence of an invariant set, stability analysis, etc.), one approach of a posteriori validation consists in representing the desired solution ``\exact`` as an isolated fixed point in a Banach space ``X``.
The assistance of the computer is used to verify that the corresponding fixed-point operator ``T`` abides by the [Banach Fixed-Point Theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) in a vicinity of a numerical approximation ``\num``.

We refer to this strategy as the *radii polynomial approach* since the contraction of ``T`` is established in a closed ball whose radius is determined by the roots of a polynomial.
This is the content of the following theorem.

!!! theorem "Radii Polynomial Theorem"
    Let ``X`` be a Banach space, ``U`` an open subset of ``X``, ``T \in C^1(U, X)`` an operator, ``\num \in U`` and ``R \in [0, \infty) \cup \{\infty\}`` such that ``B(\num, R) \subset U``.

    - **(First-order)** Suppose there are positive constants ``Y, Z_1 = Z_1(R)`` satisfying

      ```math
      \begin{aligned}
      \|T(\num) - \num\|_X &\le Y, \\
      \sup_{u \in B(\num, R)} \|DT(u)\|_{\mathscr{L}(X, X)} &\le Z_1,
      \end{aligned}
      ```

      and define the *radii polynomial* by ``p(r) \bydef Y + r (Z_1 - 1)``.

    - **(Second-order)** Suppose there are positive constants ``Y, Z_1, Z_2 = Z_2(R)`` satisfying

      ```math
      \begin{aligned}
      \|T(\num) - \num\|_X &\le Y, \\
      \|DT(\num)\|_{\mathscr{L}(X, X)} &\le Z_1, \\
      \|DT(u) - DT(\num)\|_{\mathscr{L}(X, X)} &\le Z_2 \|u - \num\|_X, \qquad \text{for all } u \in B(\num, R),
      \end{aligned}
      ```

      and define the *radii polynomial* by ``p(r) \bydef Y + r (Z_1 - 1) + \frac{r^2}{2} Z_2``.

    In either case, if there exists a *radius* ``r \in [0, R]`` such that ``p(r) \le 0`` and ``p'(r) < 0``, then ``T`` has a unique fixed point ``\exact \in B(\num, r)``.

The set of admissible radii is called the *interval of existence*.
Its **minimum is the sharpest computed error bound** on ``\num`` and its **maximum is the largest radius of the ball in which the solution is unique**.

## The pipeline

Typically, the a posteriori validation has the following five stages.

| Step | | Arithmetic | What you produce |
|:--|:--|:--|:--|
| **0** | Oracle | anything at all | a numerical picture of the solution |
| **1** | Formulation | pen and paper | ``X`` and ``F`` |
| **2** | Approximation | floating point | ``\num`` and ``A \approx DF(\num)^{-1}`` |
| **3** | Bounds | interval arithmetic | ``Y``, ``Z_1``, ``Z_2``, and ``R`` |
| **4** | Conclusion | interval arithmetic | a radius ``r``, and a theorem |

**Step 0 is an input to the pipeline, not its first action.**
The oracle may be as heuristic, as borrowed, since no part of the proof depends on where it came from, only on the numbers it produced.
In particular, the *arithmetic changes as you descend*: nothing rigorous is required at the top, and validated numerics enters when the contraction is checked.

### Step 1: Formulation

Choose the Banach space ``X`` and the fixed-point problem ``T(u) = u``, so that the solutions you are after are the isolated fixed points of ``T``.
**This is the step that decides whether the proof can close at all**: a poor choice cannot be repaired by better numerics later.

In the library, ``X`` is built by combining a formal basis (a [vector space](manual/vector_spaces.md) such as `Taylor`, `Fourier` or `Chebyshev`) with a norm and weight, forming a [Banach space](manual/norms.md).

Choosing ``T`` is part of the same decision.
A robust strategy, which we follow henceforth, is to consider a quasi-Newton operator ``T(u) = u - AF(u)``, where the solution is an isolated zero of ``F``.
This is effective when ``F`` has a computable Jacobian, whose structure can be exploited so that a good approximate inverse ``A`` of ``DF(\num)`` can be constructed.
This is nothing but an instance the [Newton--Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).

Of course, the injectivity of ``A`` is crucial for the isolated zeros to be one-to-one with the fixed points of ``T``.
Generally, however, ``A`` is built such that this property is a direct consequence of the local contraction.

### Step 2: Approximation

Translate the oracle in the approximate zero ``\num`` of ``F`` with [`newton`](@ref), then build the approximate inverse ``A`` of ``DF(\num)``.

The approximate zero and the finite truncation of ``F(\num)`` are [sequences](manual/sequences.md); the truncation of ``DF(\num)`` is a [linear operator](manual/linear_operators.md).
To implement ``F`` and ``DF`` there is a suite of [special operators](manual/special_operators.md) (derivative, integral, evaluation, multiplication) and the truncation of ``X`` is materialized by [`Projection`](@ref).

### Step 3: Bounds

Everything from here is enclosed with interval arithmetic.
Derive and evaluate ``Y`` and ``Z_1(R)``, or ``Y``, ``Z_1`` and ``Z_2(R)``.

If ``F`` is quadratic or lower order then ``DF(u) - DF(\num)`` is linear in ``u - \num``, so ``Z_2`` does not depend on ``R`` and you may take ``R = \infty``.
Otherwise a standard heuristic is ``R = 10^k Y`` for a small integer ``k \ge 1``.

#### First-order or second-order?

The first-order theorem needs only ``Y`` and ``Z_1(R)``.
It is a natural choice in finite dimensions, where `interval(x̄, R; format = :midpoint)` *encloses* the ball for any choice of ``p``-norm and interval arithmetic performs the supremum over it for you (as in the simple proof we started with).

Generally, the second-order form costs one extra bound (and more algebra to derive the second-order derivative) and buys a larger uniqueness radius.

### Step 4: Conclusion

Feed the bounds to [`interval_of_existence`](@ref) and read off what was proved.
What the radius *means* depends on the space chosen in Step 1.
For now, it suffices to say that if ``X`` models Taylor, Fourier or Chebyshev coefficients, then a radius ``r`` also gives a bound on the ``C^0``-norm.

## API

```@meta
CollapsedDocStrings = true
```

```@docs
newton
newton!
interval_of_existence
set_of_radii
```

```@docs
ConvergenceCriterion
ResidualTolCriterion
ResidualCriterion
StepCriterion
CombinedCriterion
```
