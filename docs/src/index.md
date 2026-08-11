# RadiiPolynomial.jl

**RadiiPolynomial validates a numerical solution to an equation** by computing rigorously a radius ``r`` such that a true solution exists within a closed ball (and is the only one there) centered at the approximation and of radius ``r``.

Here is a complete computer-assisted proof for the initial value problem

```math
\begin{cases}
\displaystyle \dot{u}(t) = u(t)(1 - u(t)), & t \in [-2, 2],\\
u(0) = 1/2.
\end{cases}
```

It proves that a genuine analytic solution lies within `inf(ie)` of a polynomial approximation.

```@example index
using RadiiPolynomial, LinearAlgebra

# IVP as a zero-finding problem
F(u)  = u - exact(0.5) - Integral(1) * (u * (exact(1) - u))
DF(u) = exact(I) - Integral(1) * Multiplication(exact(1) - exact(2) * u)

# approximate solution, with floating-point arithmetic
K = 27
u_bar, converged = newton(u -> (F(u), DF(u)), zeros(Taylor(K)))

Π = Projection(Taylor(K))
A_finite = inv(Π * DF(u_bar) * Π)

# bounds, with interval arithmetic
ν   = interval(2)
X   = Ell1(GeometricWeight(ν))
u_i = interval(u_bar)
A_i = interval(A_finite) + (interval(I) - interval(Π))
Y   = norm(A_i * F(u_i), X)
Z₁  = opnorm(interval(Projection(Taylor(K+1))) - A_i * DF(u_i) * Projection(Taylor(K+1)), X)
Z₂  = max(opnorm(A_i * Π, X), interval(1)) * ν * interval(2)

# check the hypotheses of the Radii Polynomial Theorem
ie, proved = interval_of_existence(Y, Z₁, Z₂, Inf)
inf(ie), proved
```

That `proved == true` is what distinguishes this from a strandard numerical ODE solver.
A complete walkthrough of this example is given in [The logistic equation](@ref).

## Where to go

```@raw html
<div class="boxintro-container">
```

**[Understand the method](radii_polynomial_approach.md)** -- The Radii Polynomial Theorem, and the five steps every proof on this site follows.
Start here if the method is new to you.

**[Get started](getting_started/first_proof.md)** -- Four short pages, from a cube root to an infinite-dimensional problem.
Look here if you have never done a computer-assisted proof.

**[See what can be validated](examples/index_examples.md)** -- Worked proofs of steady states, periodic orbits, branches through folds and whole regions of parameter space.
Look here for a problem shaped like yours.

**[Look something up](manual/vector_spaces.md)** -- The reference for spaces, sequences, operators and norms.
Look here when you need a particular tool.

```@raw html
</div>
```

## Installation

RadiiPolynomial requires Julia v1.10 or above.

```julia-repl
julia> using Pkg

julia> Pkg.add("RadiiPolynomial")
```

## Open source and citation

RadiiPolynomial is released under the [MIT license](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/LICENSE.md)
and developed at [github.com/OlivierHnt/RadiiPolynomial.jl](https://github.com/OlivierHnt/RadiiPolynomial.jl).

If you use it in a publication, in research or in teaching, please cite it using the BibTeX
template [CITATION.bib](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/CITATION.bib)
([more information](https://doi.org/10.5281/zenodo.5705258)).

```@docs
RadiiPolynomial
```
