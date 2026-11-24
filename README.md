<h1 align="center">
RadiiPolynomial

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.5705258-_?colorA=363a4f&colorB=f5a97f)](https://zenodo.org/doi/10.5281/zenodo.5705258)
[![Build Status](https://github.com/OlivierHnt/RadiiPolynomial.jl/workflows/CI/badge.svg)](https://github.com/OlivierHnt/RadiiPolynomial.jl/actions/workflows/ci.yml)
</h1>

**RadiiPolynomial.jl** is a Julia package for computer-assisted proofs in dynamical systems.
Given a numerical approximation of a solution, it computes a rigorous radius $r$ such that an exact solution exists, and is unique, within distance $r$ of the approximation.

The package provides the sequence spaces, operators and norms the validation argument requires.
It also relies on [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl) to control rounding errors.

Typical applications include:

⚡ Steady states, periodic orbits

⚡ Invariant manifolds, connecting orbits

⚡ Bifurcation, branches of solutions

### 📦 Installation

RadiiPolynomial requires Julia v1.10 or above ([julialang.org/downloads](https://julialang.org/downloads)).

```julia
julia> using Pkg

julia> Pkg.add("RadiiPolynomial")
```

### 🔎 A complete validation

Here is a computer-assisted proof of a solution to the initial value problem

```math
\begin{cases}
\dot{u}(t) = u(t)(1 - u(t)), & t \in [-2, 2],\\
u(0) = 1/2.
\end{cases}
```

```julia
using RadiiPolynomial, LinearAlgebra

# IVP as a zero-finding problem
F(u)  = u - exact(0.5) - Integral() * (u * (exact(1) - u))
DF(u) = exact(I) - Integral() * Multiplication(exact(1) - exact(2) * u)

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

This proves that an analytic solution of the initial value problem lies within `inf(ie)` of the polynomial `u_bar`, and that it is the only solution within `sup(ie)`.
The example is explained step by step in [The logistic equation](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/getting_started/logistic_equation).

### 🧭 Documentation

- [The radii polynomial approach](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/radii_polynomial_approach): the theorem and the steps of a validation.
- [Getting started](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/getting_started/first_validation): four short pages, from a cube root to an infinite-dimensional problem.
- [Examples](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/examples/index_examples): steady states, periodic orbits, branches through folds, regions of parameter space.
- [Manual](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable/manual/vector_spaces): reference for spaces, sequences, operators and norms.


### 📚 Citation

If you use the RadiiPolynomial library in your publication, research, teaching, or other activities, please cite it using [CITATION.bib](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/CITATION.bib).

### ⚖️ License

RadiiPolynomial is released under the [MIT license](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/LICENSE.md).
