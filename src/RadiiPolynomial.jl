module RadiiPolynomial

    using Printf, Reexport
    @reexport using IntervalArithmetic, LinearAlgebra, SparseArrays

include("utilities.jl")

# Sequence spaces

include("sequence_spaces/vector_spaces.jl")
    export VectorSpace, ParameterSpace, SequenceSpace, BaseSpace,
        TensorSpace, ⊗, TensorIndices, Taylor, Fourier, Chebyshev,
        CartesianSpace, CartesianPower, CartesianProduct, ×

    export order, frequency, space, spaces, dimension, dimensions, indices,
        nb_cartesian_product

#

include("sequence_spaces/sequence.jl")
    export Sequence
include("sequence_spaces/linear_operator.jl")
    export LinearOperator, domain, codomain
include("sequence_spaces/broadcast.jl")

    export coefficients, eachcol, eachrow, eachcomponent, component

#

include("sequence_spaces/norm.jl")
    export BanachSpace, ℓ¹, ℓ², ℓ∞, Weight, IdentityWeight, GeometricWeight,
        geometricweight, AlgebraicWeight, algebraicweight, BesselWeight, rate,
        NormedCartesianSpace

#

include("sequence_spaces/linear_algebra.jl")

#

include("sequence_spaces/arithmetic/add_conv_image.jl")
include("sequence_spaces/arithmetic/sequence.jl")
include("sequence_spaces/arithmetic/linear_operator.jl")
include("sequence_spaces/arithmetic/action.jl")
    export image, +̄, -̄, add!, radd!, ladd!, sub!, rsub!, lsub!
include("sequence_spaces/arithmetic/convolution.jl")
    export banach_rounding_order, banach_rounding!, banach_rounding_mul,
        banach_rounding_pow, *̄, banach_rounding_mul_bar, ^̄, banach_rounding_pow_bar
include("sequence_spaces/arithmetic/fft.jl")
    export fft_size, fft, fft!, ifft!, rifft!

#

include("sequence_spaces/special_operators/projection.jl")
    export project, project!
include("sequence_spaces/special_operators/multiplication.jl")
    export Multiplication
include("sequence_spaces/special_operators/calculus.jl")
    export Derivative, differentiate, differentiate!, Integral, integrate, integrate!
include("sequence_spaces/special_operators/evaluation.jl")
    export Evaluation, evaluate, evaluate!
include("sequence_spaces/special_operators/scale.jl")
    export Scale, scale, scale!
include("sequence_spaces/special_operators/shift.jl")
    export Shift, shift, shift!

# Radii polynomial

include("rpa/interval_existence.jl")
    export interval_of_existence, C¹Condition, C²Condition
include("rpa/newton.jl")
    export newton, newton!

end
