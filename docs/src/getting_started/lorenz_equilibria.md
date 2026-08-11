```@contents
Pages = ["lorenz_equilibria.md"]
Depth = 4
```

# Equilibria of the Lorenz system

We prove the existence of the two nontrivial equilibria of the Lorenz system

```math
\begin{aligned}
\dot{u}^{(1)} &= \sigma (u^{(2)} - u^{(1)}), \\
\dot{u}^{(2)} &= u^{(1)}(\rho - u^{(3)}) - u^{(2)}, \\
\dot{u}^{(3)} &= u^{(1)} u^{(2)} - \beta u^{(3)},
\end{aligned}
\qquad (\sigma, \rho, \beta) = (10, 28, 8/3).
```

The unknowns are a point of ``\mathbb{R}^3`` rather than a number, which is the simplest possible setting in which to meet the library's **cartesian spaces**.

!!! about "About this proof"
    **Proves** the two nontrivial equilibria ``C^\pm`` of the Lorenz system.

    **Compares** the [Radii Polynomial Theorem](@ref radii_polynomial_approach) in its first- and second-order form: three bounds, ``Y``, ``Z_1`` and ``Z_2``.

    **Involves** an unknown with components: [`ScalarSpace`](@ref), `^`, [`component`](@ref) and the inner/outer split of [`NormedCartesianSpace`](@ref).

    **Assumes** [A first proof](@ref).

### Step 1: Formulation

The equilibria are the zeros of the vector field itself, ``F(u) = 0`` with ``u = (u^{(1)}, u^{(2)}, u^{(3)}) \in \mathbb{R}^3``.
We take ``X = \mathbb{R}^3`` with the norm ``\|u\|_X = \max\left(|u^{(1)}|, |u^{(2)}|, |u^{(3)}| \right)``.

In the library, ``\mathbb{R}^3`` is constructed via `ScalarSpace()^3`: a [`CartesianPower`](@ref) of the space of plain numbers [`ScalarSpace`](@ref).
A [`Sequence`](@ref) in that space just wraps a 3-coefficient vector and a [`LinearOperator`](@ref) acting on them wraps a 3-by-3 matrix.

```@example lorenz_equilibria
using RadiiPolynomial, LinearAlgebra

function F(u, params)
    σ, ρ, β = params
    u1, u2, u3 = u[1], u[2], u[3]
    return Sequence(space(u), [σ*(u2 - u1), u1*(ρ - u3) - u2, u1*u2 - β*u3])
end

function DF(u, params)
    σ, ρ, β = params
    u1, u2, u3 = u[1], u[2], u[3]
    return LinearOperator(space(u), space(u),
        [        -σ            σ       zero(u1)
             ρ - u3      -one(u1)            -u1
                 u2            u1            -β ])
end
nothing # hide
```

The norm is prescribed separately, and the maximum of the components absolute value is embodied by ``\ell^\infty``.

```@example lorenz_equilibria
X = ScalarSpace()^3
X_norm = EllInf()
nothing # hide
```

### Step 2: Approximation (floating-point arithmetic)

#### The approximate zero

```@example lorenz_equilibria
params = (10.0, 28.0, 8/3)

u_bar, converged = newton(u -> (F(u, params), DF(u, params)), Sequence(X, [8.0, 8.0, 26.0]))
```

The components are reachable with [`component`](@ref), which is how you address a block of a cartesian unknown:

```@example lorenz_equilibria
component(u_bar, 3)
```

#### The approximate inverse

``DF(\num)`` is a ``3 \times 3`` matrix, so `A` is simply an approximate matrix inverse.

```@example lorenz_equilibria
A = inv(DF(u_bar, params))
nothing # hide
```

### Step 3: Bounds (interval arithmetic)

``\beta = 8/3`` is **not** a representable floating-point number, so the parameters must be enclosed rather than converted.
`interval(8)/interval(3)` produces the tightest interval containing ``8/3``; writing `interval(8/3)` instead would enclose the *double* nearest ``8/3``, which is a different real number.

```@example lorenz_equilibria
params_i = (interval(10), interval(28), interval(8)/interval(3))

u_i = interval(u_bar)
A_i = interval(A)

Y = norm(A_i * F(u_i, params_i), X_norm)
```

#### A first-order proof

The first-order Radii Polynomial Theorem needs a bound on ``\|I - A DF(u)\|`` over an entire ball.
As in [A first proof](@ref), interval arithmetic supplies it by making every component an interval of radius ``R`` and evaluate `DF` there.

```@example lorenz_equilibria
R = 1.0
u_R = Sequence(X, interval.(coefficients(u_bar), R; format = :midpoint))

Z₁_R = opnorm(I - A_i * DF(u_R, params_i), X_norm)

ie_first, proved_first = interval_of_existence(Y, Z₁_R, R)
inf(ie_first), sup(ie_first), proved_first
```

That is already a complete proof, and it needed only ``F`` and ``DF``.
But the chosen value of ``R``, which bounds the uniqueness radius, must be small for ``A`` to accuratly approximate uniformly ``DF(u)^{-1}`` for all ``u \in B(\num, R)``.
**The first-order theorem is not ideal to certify a large uniqueness radius.**

#### The second-order bound

To do better, we put the limiting constraint ``Z_1 < 1`` at the center of the ball ``\|I - A DF(\num)\| < 1`` and we control how fast ``DF`` varies nearby.

```@example lorenz_equilibria
Z₁ = opnorm(I - A_i * DF(u_i, params_i), X_norm)
```

Writing ``h = u - \num``, every entry of ``DF`` is affine in ``u``, so the difference is exactly

```math
DF(u) - DF(\num) =
\begin{pmatrix}
0 & 0 & 0\\
-h^{(3)} & 0 & -h^{(1)}\\
h^{(2)} & h^{(1)} & 0
\end{pmatrix}.
```

The ``\ell^\infty``-induced norm of a matrix is its largest row sum, here
``\max\left(0, |h^{(3)}| + |h^{(1)}|, |h^{(2)}| + |h^{(1)}|\right) \le 2 \|h\|_X``.
Hence

```math
\|A (DF(u) - DF(\num))\|_{\mathscr{L}(X,X)} \le 2 \|A\|_{\mathscr{L}(X,X)} \|u - \num\|_X,
\qquad Z_2 \bydef 2 \|A\|_{\mathscr{L}(X,X)}.
```

**This ``Z_2`` is independent of ``R``.**
That is to be expected since ``F`` is quadratic, so ``DF`` is affine and ``DF(u) - DF(\num)`` is *linear* in ``u - \num``.
Thus, its Lipschitz constant is global.
We may therefore take ``R = \infty``.

```@example lorenz_equilibria
R  = Inf
Z₂ = exact(2) * opnorm(A_i, X_norm)

ie, proved = interval_of_existence(Y, Z₁, Z₂, R)
inf(ie), sup(ie), proved
```

The uniqueness radius is ``2\times`` that of the first-order proof.

### Step 4: Conclusion

There exists an exact equilibrium of the Lorenz system within `inf(ie)` of `u_bar`, and it is the
only one within `sup(ie)`.

The next table compares the two proofs.

| | error bound `inf(ie)` | uniqueness radius `sup(ie)` | needs |
|:--|:--|:--|:--|
| first-order | ``\approx 2.4 \times 10^{-15}`` | ``1``, same as the ``R`` we chose | ``F``, ``DF`` |
| second-order | ``\approx 3.9 \times 10^{-15}`` | ``\approx 2.19``, comes from the problem itself | ``F``, ``DF``, ``\mathrm{Lip}(DF)`` |
