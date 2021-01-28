# RadiiPolynomial

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://OlivierHnt.github.io/RadiiPolynomial.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://OlivierHnt.github.io/RadiiPolynomial.jl/dev)
[![Build Status](https://github.com/OlivierHnt/RadiiPolynomial.jl/workflows/CI/badge.svg)](https://github.com/OlivierHnt/RadiiPolynomial.jl/actions)
[![Coverage](https://codecov.io/gh/OlivierHnt/RadiiPolynomial.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/OlivierHnt/RadiiPolynomial.jl)

The solution of many problems in dynamical systems can be seen as the fixed point of an operator. The *radii polynomial approach* is a strategy in computer-assisted proofs to obtain a rigorous a posteriori error bound on a numerical approximation of the fixed point. More precisely, the Radii Polynomial Theorem gives us sufficient conditions for the operator to be a contraction in a closed ball centred at the numerical approximation; this implies the existence and uniqueness of the desired fixed point within this ball.

For Newton-like operators, the Radii Polynomial Theorem is an instance of the [Newton-Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).

The *radii polynomial approach* is a *functional analytic approach* as opposed to a *topological approach* (e.g. [CAPD](http://capd.ii.uj.edu.pl/index.php) library).

## Installation

```julia
julia> ]
pkg> add RadiiPolynomial
```

## Dependencies

* [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl).
