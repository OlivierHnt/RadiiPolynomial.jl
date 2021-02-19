module RadiiPolynomial

    using IntervalArithmetic, LinearAlgebra, Printf

## hack _setindex from Base for type stability of high dimensional arrays
@inline Base._setindex(v, i::Int, args...) =
    ntuple(dim -> ifelse(i == dim, v, args[dim]), length(args))

## hack intersect from Base for performance (match intersect behaviour on CartesianIndices)
@inline Base.intersect(a::Base.Iterators.ProductIterator, b::Base.Iterators.ProductIterator) =
    Base.Iterators.ProductIterator(intersect.(a.iterators, b.iterators))

## from TaylorSeries.jl to show superscript and subscript

const subscript_digits = ["₀","₁","₂","₃","₄","₅","₆","₇","₈","₉"]
const superscript_digits = ["⁰","¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹"]
subscriptify(n::Int) = join([subscript_digits[i+1] for i ∈ reverse(digits(n))])
superscriptify(n::Int) = join([superscript_digits[i+1] for i ∈ reverse(digits(n))])

## Sequence spaces

include("sequence_spaces/spaces.jl")
    export VectorSpace, ParameterSpace, SequenceSpace, UnivariateSpace,
           Taylor, Fourier, Chebyshev, TensorSpace, CartesianSpace,
           CartesianPowerSpace, CartesianProductSpace, ×, ⊗, order, frequency,
           spaces, nb_cartesian_product

    export dimension, dimensions, startindex, endindex, allindices, isindexof

#

include("sequence_spaces/sequence.jl")
    export Sequence, space
include("sequence_spaces/operator.jl")
    export Operator, domain, codomain

    export coefficients, eachcomponent, eachcol, eachrow, component

#

include("sequence_spaces/arithmetic/space.jl")
    export addition_range, addition_bar_range, convolution_range, convolution_bar_range
include("sequence_spaces/arithmetic/sequence.jl")
include("sequence_spaces/arithmetic/operator.jl")
include("sequence_spaces/arithmetic/action.jl")
include("sequence_spaces/arithmetic/convolution.jl")
    export geometric_decay
include("sequence_spaces/arithmetic/fft.jl")
    export fft_length, fft_size, dfs_dimension, dfs_dimensions, dfs, idfs!

    export +̄, -̄, *̄, ^̄

#

include("sequence_spaces/special_operators/projection.jl")
    export project
include("sequence_spaces/special_operators/derivative.jl")
    export Derivative, derivative_range, differentiate
include("sequence_spaces/special_operators/integral.jl")
    export Integral, integral_range, integrate
include("sequence_spaces/special_operators/evaluation.jl")
    export Evaluation, evaluate
include("sequence_spaces/special_operators/scale.jl")
    export Scale, scale, scale!
include("sequence_spaces/special_operators/shift.jl")
    export Shift, shift, shift!

#

include("sequence_spaces/norm.jl")
    export norm, opnorm, norm_weighted_ℓ¹, opnorm_weighted_ℓ¹

#

include("sequence_spaces/linear_algebra.jl")
    export eigvals, eigvecs, eigen

## Radii polynomial

include("rpa/roots.jl")
    export roots_radii_polynomial
include("rpa/finite.jl")
    export FixedPointProblemFiniteDimension, ZeroFindingProblemFiniteDimension
include("rpa/infinite.jl")
    export TailProblem, ZeroFindingProblemCategory1
include("rpa/newton.jl")
    export newton

    export Y, Z₁, Z₂, prove

## Applications

include("ivp.jl")
    export ivp_ODE

include("manifolds.jl")
    export manifold_ODE_equilibrium, manifold_DDE_equilibrium

end
