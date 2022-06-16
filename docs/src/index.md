## Presentation

The solution of many problems in dynamical systems can be seen as the fixed-point of an operator. In computer-assisted proofs, the Radii Polynomial Theorem gives us closed ball(s), centred at a numerical approximation of the fixed-point, within which the operator satisfies the [Banach Fixed-Point Theorem](https://en.wikipedia.org/wiki/Banach_fixed-point_theorem).[^1]

[^1]: For Newton-like operators, the Radii Polynomial Theorem is an instance of the [Newton-Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).

Hence, the desired solution is the unique fixed-point within the ball(s) whose radius yields an a posteriori error bound on the numerical approximation.

[RadiiPolynomial.jl](https://github.com/OlivierHnt/RadiiPolynomial.jl) is a Julia package to conduct the computational steps of the Radii Polynomial Theorem. For the entailed rigorous floating-point computations, the RadiiPolynomial software relies on [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)[^2].

[^2]: L. Benet and D. P. Sanders, [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl).

When the solution lies in a Banach space involving function spaces, the standard approach is to interpret the function spaces as sequence spaces. Thus, RadiiPolynomial is concerned with the latter (cf. [ApproxFun.jl](https://github.com/JuliaApproximation/ApproxFun.jl) for a Julia package to approximate functions).

## Installing

The RadiiPolynomial software requires to [install Julia](https://julialang.org/downloads/) (v1.6 or above).

Then, start Julia and execute the following command in the REPL:

```julia
using Pkg; Pkg.add("RadiiPolynomial")
```

## Citing

If you use the RadiiPolynomial software in your publication, research, teaching, or other activities, please use the following BibTeX template (cf. [CITATION.bib](https://github.com/OlivierHnt/RadiiPolynomial.jl/blob/main/CITATION.bib)):

```bibtex
@software{RadiiPolynomial.jl,
  author = {Olivier Hénot},
  title  = {RadiiPolynomial.jl},
  url    = {https://github.com/OlivierHnt/RadiiPolynomial.jl},
  year   = {},
  doi    = {}
}
```

The empty fields `year` and `doi` should correspond with the cited version of the RadiiPolynomial software. For instance, if you wish to cite the software as a whole: `year = {2021}` and `doi = {10.5281/zenodo.5705258}`.

You may refer to [10.5281/zenodo.5705258](https://doi.org/10.5281/zenodo.5705258) for more informations.
