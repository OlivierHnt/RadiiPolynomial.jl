<h1 align="center">
RadiiPolynomial

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.5705258-_?colorA=363a4f&colorB=f5a97f)](https://zenodo.org/doi/10.5281/zenodo.5705258)
[![Build Status](https://github.com/OlivierHnt/RadiiPolynomial.jl/workflows/CI/badge.svg)](https://github.com/OlivierHnt/RadiiPolynomial.jl/actions/workflows/ci.yml)
</h1>

**RadiiPolynomial.jl** is a Julia package for computer-assisted proofs in dynamical systems: it validates a numerical approximation by computing rigorously a radius $r$ such that a exact solution of the equation exists, and is the only one, within a distance $r$ of the approximation.

Built on top of [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl), RadiiPolynomial provides the sequences, operators and norms needed to carry out this argument.

Applications to dynamical systems include:

⚡ Fixed points, steady states, and periodic orbits

⚡ Invariant manifolds and connecting orbits

⚡ Bifurcation diagrams and parameter-dependent solution branches

### 🔎 A complete proof

Here is a computer-assisted proof for the initial value problem

```math
\begin{cases}
\dot{u}(t) = u(t)(1 - u(t)), & t \in [-2, 2],\\
u(0) = 1/2.
\end{cases}
```

```julia
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

```
(2.232051998824705e-6, true)
```

That `proved == true` is what distinguishes this from a standard numerical ODE solver: it is now a theorem that a genuine analytic solution of the initial value problem lies within `inf(ie)` of the polynomial `u_bar`, and that it is the only one within `sup(ie)`. A complete walkthrough of this proof is given in [The logistic equation](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/getting_started/logistic_equation).

### 🧭 Documentation

**[The radii polynomial approach](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/radii_polynomial_approach)** -- The Radii Polynomial Theorem, and the five steps every proof follows. Start here if the method is new to you.

**[Getting started](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/getting_started/first_proof)** -- Four short pages, from a cube root to an infinite-dimensional problem. Look here if you have never done a computer-assisted proof.

**[Examples](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/examples/index_examples)** -- Worked proofs of steady states, periodic orbits, branches through folds and whole regions of parameter space. Look here for a problem shaped like yours.

**[Manual](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/manual/vector_spaces)** -- The reference for spaces, sequences, operators and norms. Look here when you need a particular tool.

### 📦 Installation

RadiiPolynomial requires Julia v1.10 or above ([julialang.org/downloads](https://julialang.org/downloads)).

```julia
julia> using Pkg

julia> Pkg.add("RadiiPolynomial")
```

### 📚 Citation

If you use the RadiiPolynomial library in your publication, research, teaching, or other activities, please use the BibTeX template [CITATION.bib](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/CITATION.bib) ([more information](https://doi.org/10.5281/zenodo.5705258)).

### ⚖️ License

RadiiPolynomial is released under the [MIT license](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/LICENSE.md).
