# Radii Polynomial Theorem

Let ``X`` be a Banach space, ``U`` an open subset of ``X``, ``T \in C^k(U, X)``, ``x_0 \in U`` and ``R \geq 0``. Suppose there are constants ``Y, Z_1, \dots, Z_k \geq 0`` satisfying

```math
\begin{aligned}
Y &\geq |T(x_0) - x_0|_X,\\
Z_i &\geq |D^i T(x_0)|_{\mathscr{B}(X^i, X)}, \qquad i = 1, \dots, k-1,\\
Z_k &\geq \sup_{y \in \text{cl}( B_R(x_0) )} |D^k T(y)|_{\mathscr{B}(X^k, X)},
\end{aligned}
```

and define the *radii polynomial* by

```math
p(r) := Y - r + \sum_{i = 1}^k \frac{Z_i}{i!} r^i.
```

The Radii Polynomial Theorem states that if ``p(r_0) \leq 0`` for some ``r_0 \in [0, R]``, then ``T`` satisfies the [Banach Fixed Point Theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem) in the closed ball ``\text{cl}( B_{r_0} (x_0) )``; the set of all possible radii is called the *interval of existence*.

In practice, ``p`` is linear or quadratic and the interval of existence consists in a segment of the positive real line.

The infimum of the interval of existence gives the sharpest computed a posteriori error bound on ``x_0``. The supremum of the interval of existence represents the largest computed radius of the ball centred at ``x_0`` within which the solution is unique.

The `interval_of_existence` method returns the `Interval` such that ``p`` is negative.

```@docs
interval_of_existence
```
