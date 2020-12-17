module RadiiPolynomial

    using IntervalArithmetic, LinearAlgebra

## hack _setindex from Base for type stability of high dimensional arrays
@inline Base._setindex(v, i::Int, args...) =
    ntuple(dim -> ifelse(i == dim, v, args[dim]), length(args))

##

include("spaces/spaces.jl")
    export SequenceSpace, UnivariateSpace, Taylor, Fourier, Chebyshev, TensorSpace,
           order, frequency, ⊗, ∪̄, multiplication_range, derivative_range,
           integral_range, isindexof

include("spaces/cartesian.jl")
    export CartesianSpace, ×

## sequences

include("sequences/sequences.jl")
    export Sequence, space, coefficients, project, shift, rescale, rescale!, norm

include("sequences/fft.jl")
    export fft_size, fft, ifft!

include("sequences/arithmetic.jl")
    export +̄, -̄, *̄, ^̄

include("sequences/calculus.jl")
    export differentiate, integrate

include("sequences/evaluate.jl")
    export evaluate

include("sequences/broadcast.jl")

include("sequences/cartesian.jl")
    export eachcomponent

## functionals

include("functionals/functionals.jl")
    export Functional, domain, opnorm, Evaluation

include("functionals/arithmetic.jl")

include("functionals/broadcast.jl")

include("functionals/cartesian.jl")

## operators

include("operators/operators.jl")
    export Operator, codomain, AbstractOperator, Derivative, Integral, Shift, Rescale

include("operators/arithmetic.jl")

include("operators/broadcast.jl")

include("operators/cartesian.jl")

##

include("linear_algebra.jl")

include("ivp.jl")
    export ivp_ODE

include("manifolds.jl")
    export manifold_ODE_equilibrium, manifold_DDE_equilibrium

include("rpa.jl")
    export roots_radii_polynomial, FixedPointProblemFiniteDimension,
           ZeroFindingProblemFiniteDimension, TailProblem, ZeroFindingProblemCategory1,
           Y, Z₁, Z₂, prove, newton

end
