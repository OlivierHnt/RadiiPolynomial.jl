using Test
using RadiiPolynomial

@testset verbose = true "RadiiPolynomial" begin
    include("sequence_spaces/vector_spaces.jl")
    include("sequence_spaces/symmetry.jl")
    include("sequence_spaces/banach_spaces.jl")

    include("sequence_spaces/sequences/sequence.jl")
    include("sequence_spaces/sequences/infinite_sequence.jl")
    include("sequence_spaces/sequences/fft.jl")
    include("sequence_spaces/sequences/arithmetic.jl")
    include("sequence_spaces/sequences/convolution.jl")
    include("sequence_spaces/sequences/refinement.jl")
    include("sequence_spaces/sequences/elementary.jl")

    include("sequence_spaces/linear_operators/linear_operator.jl")
    include("sequence_spaces/linear_operators/projection.jl")
    include("sequence_spaces/linear_operators/action.jl")
    include("sequence_spaces/linear_operators/arithmetic.jl")
    include("sequence_spaces/linear_operators/special_operators/multiplication.jl")
    include("sequence_spaces/linear_operators/special_operators/calculus/derivative.jl")
    include("sequence_spaces/linear_operators/special_operators/calculus/integral.jl")
    include("sequence_spaces/linear_operators/special_operators/calculus/laplacian.jl")
    include("sequence_spaces/linear_operators/special_operators/evaluation.jl")
    include("sequence_spaces/linear_operators/special_operators/scale.jl")
    include("sequence_spaces/linear_operators/special_operators/shift.jl")

    include("sequence_spaces/norm.jl")

    include("utilities.jl")

    include("rpa/interval_existence.jl")
    include("rpa/newton.jl")
    include("rpa/proofs.jl")
end
