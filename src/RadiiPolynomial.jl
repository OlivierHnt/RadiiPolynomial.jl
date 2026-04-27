"""
    RadiiPolynomial

Library for computer-assisted proofs in dynamical systems.

Learn more: https://olivierhnt.github.io/RadiiPolynomial.jl
"""
module RadiiPolynomial

using Printf, Reexport
import LinearAlgebra, StaticArrays
@reexport using IntervalArithmetic





# Sequence spaces

import LinearAlgebra: ×
include("sequence_spaces/vector_spaces.jl")
    export VectorSpace, EmptySpace, ScalarSpace, SequenceSpace, BaseSpace,
        TensorSpace, ⊗, TensorIndices, Taylor, Fourier, Chebyshev,
        CartesianSpace, CartesianPower, CartesianProduct, ×,
        order, frequency, space, spaces, dimension, dimensions,
        indices, nspaces
include("sequence_spaces/symmetry.jl")
    export IndexAction, CoefAction, GroupElement, Group, elements, SymmetricSpace, symmetry, desymmetrize,
        evensym, oddsym, d4sym
include("sequence_spaces/banach_spaces.jl")
    export Weight, IdentityWeight, GeometricWeight, AlgebraicWeight, BesselWeight, rate,
        BanachSpace, Ell1, ℓ¹, Ell2, ℓ², EllInf, ℓ∞, NormedCartesianSpace, weight





# Sequences

include("sequence_spaces/sequences/sequence.jl")
    export AbstractSequence, Sequence, coefficients, eachblock, block,
        conjugacy_symmetry!, geometricweight, algebraicweight, polish!
include("sequence_spaces/sequences/infinite_sequence.jl")
    export InfiniteSequence, sequence_norm, sequence_error, banachspace
#- operations
include("sequence_spaces/sequences/fft.jl")
    export fft_size, to_grid, to_grid!, to_seq, to_seq!
import LinearAlgebra: rmul!, lmul!, rdiv!, ldiv!
include("sequence_spaces/sequences/arithmetic.jl")
    export codomain, add!, radd!, ladd!, sub!, rsub!, lsub!, rmul!, lmul!,
        rdiv!, ldiv!
include("sequence_spaces/sequences/convolution.jl")
    export set_conv_algorithm, mul_bar, pow_bar
include("sequence_spaces/sequences/elementary.jl")





# Linear operators

import LinearAlgebra: UniformScaling, I
include("sequence_spaces/linear_operators/linear_operator.jl")
    export AbstractLinearOperator, AbstractDiagonalOperator, LinearOperator, domain, eachcol, eachrow, transpose, adjoint,
        Add, Negate, ComposedOperator, UniformScalingOperator
    export UniformScaling, I
include("sequence_spaces/linear_operators/projection.jl")
    export Projection, project, project!
#- operations
import LinearAlgebra: mul!
include("sequence_spaces/linear_operators/action.jl")
    export mul!
include("sequence_spaces/linear_operators/arithmetic.jl")
include("sequence_spaces/linear_operators/special_operators/special_operators.jl")





# Norms

import LinearAlgebra: norm, opnorm
include("sequence_spaces/norm.jl")
    export norm, opnorm





# Utilities

include("utilities.jl")





# Radii polynomial approach

include("rpa/interval_existence.jl")
    export interval_of_existence
include("rpa/newton.jl")
    export newton, newton!





# Deprecated functions

function ParameterSpace()
    Base.depwarn("`ParameterSpace()` is deprecated and will be removed in a future version, use `ScalarSpace()` instead", :ParameterSpace; force=true)
    return ScalarSpace()
end
function CosFourier(s)
    Base.depwarn("`CosFourier(s)` is deprecated and will be removed in a future version, use `evensym(s)` instead", :CosFourier; force=true)
    return evensym(s)
end
function CosFourier(K, freq)
    Base.depwarn("`CosFourier(K, freq)` is deprecated and will be removed in a future version, use `evensym(Fourier(K, freq))` instead", :CosFourier; force=true)
    return evensym(Fourier(K, freq))
end
function SinFourier(s)
    Base.depwarn("`SinFourier(s)` is deprecated and will be removed in a future version, use `oddsym(s)` instead", :SinFourier; force=true)
    return oddsym(s)
end
function SinFourier(K, freq)
    Base.depwarn("`SinFourier(K, freq)` is deprecated and will be removed in a future version, use `oddsym(Fourier(K, freq))` instead", :SinFourier; force=true)
    return oddsym(Fourier(K, freq))
end
function eachcomponent(a)
    Base.depwarn("`eachcomponent(a)` is deprecated and will be removed in a future version, use `eachblock(a)` instead", :eachcomponent; force=true)
    return eachblock(a)
end
function component(A, i, j)
    Base.depwarn("`component(A, i, j)` is deprecated and will be removed in a future version, use `block(A, i, j)` instead", :component; force=true)
    return block(A, i, j)
end
function component(A, i)
    Base.depwarn("`component(A, i)` is deprecated and will be removed in a future version, use `block(A, i)` instead", :component; force=true)
    return block(A, i)
end

    export ParameterSpace, CosFourier, SinFourier, eachcomponent, component

end
