```@contents
Pages = ["first_validation.md"]
Depth = 4
```

# A first validation

We prove that the equation

```math
u^3 - 2 = 0
```

has a solution in ``\mathbb{R}``.

!!! about "About this example"
    **Proves** that ``u^3 - 2 = 0`` has a solution in ``\mathbb{R}``.

    **Uses** the [Radii Polynomial Theorem](@ref radii_polynomial_approach) in its first-order form: two bounds, ``Y`` and ``Z_1``.

### Step 1: Formulation

We study the zeros of ``F(u) \bydef u^3 - 2``, and we will look for them as fixed points of

```math
G(u) \bydef u - A F(u),
```

where ``A`` is a number approximating ``1/DF(\num)``.
A fixed point of ``G`` is a zero of ``F`` precisely when ``A \ne 0``, which we get for free here.

```@example first_validation
using RadiiPolynomial

F(u)  = u^3 - exact(2)
DF(u) = exact(3) * u^2
nothing # hide
```

`exact(2)` says that the literal `2` is to be considered an exact value.
The last experiment at the bottom of this page shows what happens if it is omitted.

### Step 2: Approximation (floating-point arithmetic)

Nothing here needs to be rigorous, this step only has to produce sufficiently accurate numbers:
- the quality of the approximation ``\num`` is measured by ``Y``, and
- the quality of the approximation ``A`` is measured by ``Z_1``.

#### The approximate zero

```@example first_validation
u_bar, converged = newton(u -> (F(u), DF(u)), 1.0)
```

[`newton`](@ref) takes a single function returning both ``F`` and ``DF``, and returns the refined value together with a flag.
**The flag is meant to be read**: `converged == false` means the value may not be sufficiently good numerically for the validation to succeed.

#### The approximate inverse

```@example first_validation
A = inv(DF(u_bar))
```

### Step 3: Bounds (interval arithmetic)

From here everything requires interval arithmetic.
We need the bounds

```math
|A F(\num)| \le Y, \qquad \sup_{u \in B(\num, R)} |1 - A DF(u)| \le Z_1.
```

The second is a supremum over a whole ball.
Interval arithmetic can do this computation readily since `interval(u_bar, R; format = :midpoint)` *is* the ball ``B(\num, R)``, so evaluating `DF` on it returns an enclosure of ``DF`` over every point of that ball at once.

```@example first_validation
Y  = abs(interval(A) * F(interval(u_bar)))

R   = 2sup(Y)
u_R = interval(u_bar, R; format = :midpoint)   # the ball B(ū, R)
Z₁  = abs(interval(1) - interval(A) * DF(u_R))

Y, Z₁
```

### Step 4: Conclusion

We now check that ``G`` contracts on a ball:

```@example first_validation
ie, proved = interval_of_existence(Y, Z₁, R)
```

`proved` is `true`, so the Radii Polynomial Theorem applies: there is a real root of ``u^3 - 2 = 0`` within `inf(ie)` of `u_bar`, and it is the only one within `sup(ie)`.

```@example first_validation
inf(ie), sup(ie)
```

The proven bound is at the scale of machine epsilon.

## Two things to try

!!! details "Make the ball too small"
    ``r`` has to fit inside ``[0, R]``.
    Ask for a ball far smaller than machine epsilon and the theorem correctly declines:

    ```@example first_validation
    R = 1e-20
    ie, proved = interval_of_existence(Y, Z₁, R)
    ```

    The returned interval is empty and `proved == false`.
    **A failure to verify the Radii Polynomial Theorem is not a proof that no solution exists.**

!!! details "Forget `exact`"
    Replace `exact(2)` by a bare `2` and compare the guarantee flags:

    ```@example first_validation
    Y_sloppy = abs(interval(A) * (interval(u_bar)^3 - 2))
    isguaranteed(Y), isguaranteed(Y_sloppy)
    ```

    The value is the same, but the second carries an `NG` flag: IntervalArithmetic can no longer certify that the computation was performed soundly since an operation mixed intervals and non-interval operands.
    `interval_of_existence` propagates that flag, so a validation built on it would report `proved == true` while `isguaranteed(ie) == false`.
    **Both must hold.**
