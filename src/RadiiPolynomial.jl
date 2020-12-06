module RadiiPolynomial

    using IntervalArithmetic, LinearAlgebra

##

include("spaces/spaces.jl")
    export SequenceSpace, UnivariateSpace, Taylor, Fourier, Chebyshev, TensorSpace,
           order, ⊗, ∪̄, multiplication_range_space, pow_range_space, derivation_range_space,
           integration_range_space, isindexof

## sequences

include("sequences/sequences.jl")
    export Sequence, project, shift, rescale, rescale!, banach_algebra_rounding!, norm

include("sequences/fft.jl")
    export size_fft, fft, ifft!

include("sequences/arithmetic.jl")
    export +̄, -̄, *̄, ^̄

include("sequences/calculus.jl")
    export differentiate, integrate

include("sequences/evaluate.jl")
    export evaluate

include("sequences/broadcast.jl")

## functionals

include("functionals/functionals.jl")
    export Functional, opnorm, Evaluation

include("functionals/arithmetic.jl")

include("functionals/broadcast.jl")

## operators

include("operators/operators.jl")
    export Operator, Derivative, Integral, Shift, Rescale

include("operators/arithmetic.jl")

##

# include("spaces/cartesian.jl")
#    export CartesianSpace, ×
# include("sequences/cartesian.jl")
# include("functionals/cartesian.jl")
# include("operators/cartesian.jl")

include("manifolds.jl")
    export manifold_ODE_equilibrium, manifold_DDE_equilibrium

include("rpa.jl")
    export roots_radii_polynomial

end
