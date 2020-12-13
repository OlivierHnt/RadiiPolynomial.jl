module RadiiPolynomial

    using IntervalArithmetic, LinearAlgebra

##

include("spaces/spaces.jl")
    export SequenceSpace, UnivariateSpace, Taylor, Fourier, Chebyshev, TensorSpace,
           order, ⊗, ∪̄, multiplication_range, pow_range, derivative_range,
           integral_range, isindexof

## sequences

include("sequences/sequences.jl")
    export Sequence, space, coefficients, project, shift, rescale, rescale!, banach_algebra_rounding!, norm

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
    export Functional, domain, opnorm, Evaluation

include("functionals/arithmetic.jl")

include("functionals/broadcast.jl")

## operators

include("operators/operators.jl")
    export Operator, Derivative, Integral, Shift, Rescale

include("operators/arithmetic.jl")

include("operators/broadcast.jl")

##

# include("spaces/cartesian.jl")
#    export CartesianSpace, ×
# include("sequences/cartesian.jl")
# include("functionals/cartesian.jl")
# include("operators/cartesian.jl")

include("ivp.jl")
    export ivp_ODE

include("manifolds.jl")
    export manifold_ODE_equilibrium, manifold_DDE_equilibrium

include("rpa.jl")
    export roots_radii_polynomial, rpa_finite_dimension, rpa_finite_dimension_newton, newton

end
