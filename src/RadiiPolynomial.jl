"""
    RadiiPolynomial

Library for computer-assisted proofs in dynamical systems.

Learn more: https://olivierhnt.github.io/RadiiPolynomial.jl
"""
module RadiiPolynomial

using Printf, Reexport
import LinearAlgebra, SparseArrays
@reexport using IntervalArithmetic





# Sequence spaces

import LinearAlgebra: ×
include("sequence_spaces/vector_spaces.jl")
    export VectorSpace, EmptySpace, ParameterSpace, SequenceSpace, BaseSpace,
        TensorSpace, ⊗, TensorIndices, Taylor, Fourier, Chebyshev, CosFourier,
        SinFourier, CartesianSpace, CartesianPower, CartesianProduct, ×,
        order, frequency, space, spaces, dimension, dimensions,
        indices, nspaces
include("sequence_spaces/symmetry.jl")
    export LinearIdx, PhaseVal, GroupElement, Group, SymmetricSpace, symmetry, desymmetrize, evensym, oddsym
include("sequence_spaces/banach_spaces.jl")
    export Weight, IdentityWeight, GeometricWeight, AlgebraicWeight, BesselWeight, rate,
        BanachSpace, Ell1, ℓ¹, Ell2, ℓ², EllInf, ℓ∞, NormedCartesianSpace, weight



include("sequence_spaces/sequences/sequence.jl")
    export AbstractSequence, Sequence, coefficients, getcoefficient, eachcomponent, component,
        conjugacy_symmetry!, geometricweight, algebraicweight, polish!
include("sequence_spaces/sequences/infinite_sequence.jl")
    export InfiniteSequence, sequence_norm, sequence_error, banachspace
#- operations
include("sequence_spaces/sequences/fft.jl")
    export fft_size, fft, fft!, ifft, rifft, ifft!, rifft!
import LinearAlgebra: rmul!, lmul!, rdiv!, ldiv!
include("sequence_spaces/sequences/arithmetic.jl")
    export codomain, add!, radd!, ladd!, sub!, rsub!, lsub!, rmul!, lmul!,
        rdiv!, ldiv!
include("sequence_spaces/sequences/convolution.jl")
    export set_conv_algorithm, mul_bar, pow_bar
include("sequence_spaces/sequences/elementary.jl")



import LinearAlgebra: UniformScaling, I
include("sequence_spaces/linear_operators/linear_operator.jl")
    export AbstractLinearOperator, LinearOperator, domain, eachcol, eachrow, transpose, adjoint,
        Add, Negate, ComposedOperator, UniformScalingOperator
include("sequence_spaces/linear_operators/banded_linear_operator.jl")
    export BandedLinearOperator, UniformScaling, I
include("sequence_spaces/linear_operators/projection.jl")
    export Projection, project, project!, tail, tail!
#- operations
import LinearAlgebra: mul!
include("sequence_spaces/linear_operators/action.jl")
    export mul!
include("sequence_spaces/linear_operators/arithmetic.jl")
include("sequence_spaces/linear_operators/special_operators/multiplication.jl")
    export Multiplication, sequence
include("sequence_spaces/linear_operators/special_operators/calculus.jl")
    export Derivative, differentiate, differentiate!,
        Integral, integrate, integrate!,
        Laplacian, laplacian, laplacian!
include("sequence_spaces/linear_operators/special_operators/evaluation.jl")
    export Evaluation, evaluate, evaluate!, value
include("sequence_spaces/linear_operators/special_operators/scale.jl")
    export Scale, scale, scale!
include("sequence_spaces/linear_operators/special_operators/shift.jl")
    export Shift, shift, shift!



import LinearAlgebra: norm, opnorm
include("sequence_spaces/norm.jl")
    export norm, opnorm
include("sequence_spaces/broadcast.jl")



include("utilities.jl")

# Radii polynomial approach

include("rpa/interval_existence.jl")
    export interval_of_existence
include("rpa/newton.jl")
    export newton, newton!

end
