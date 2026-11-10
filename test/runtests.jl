using Test
using RadiiPolynomial

include("helpers.jl")

# `Pkg.test(test_args = ["scale", "shift"])` runs only the files whose path contains one of the arguments
const TEST_FILES = filter(file -> isempty(ARGS) || any(arg -> occursin(arg, file), ARGS), [
    "sequence_spaces/vector_spaces.jl",
    "sequence_spaces/symmetry.jl",
    "sequence_spaces/banach_spaces.jl",
    "sequence_spaces/sequences/sequence.jl",
    "sequence_spaces/sequences/infinite_sequence.jl",
    "sequence_spaces/sequences/fft.jl",
    "sequence_spaces/sequences/arithmetic.jl",
    "sequence_spaces/sequences/convolution.jl",
    "sequence_spaces/sequences/refinement.jl",
    "sequence_spaces/sequences/elementary.jl",
    "sequence_spaces/sequences/nonlinearity_soundness.jl",
    "sequence_spaces/linear_operators/linear_operator.jl",
    "sequence_spaces/linear_operators/projection.jl",
    "sequence_spaces/linear_operators/action.jl",
    "sequence_spaces/linear_operators/arithmetic.jl",
    "sequence_spaces/linear_operators/special_operators/multiplication.jl",
    "sequence_spaces/linear_operators/special_operators/calculus/derivative.jl",
    "sequence_spaces/linear_operators/special_operators/calculus/integral.jl",
    "sequence_spaces/linear_operators/special_operators/calculus/laplacian.jl",
    "sequence_spaces/linear_operators/special_operators/evaluation.jl",
    "sequence_spaces/linear_operators/special_operators/scale.jl",
    "sequence_spaces/linear_operators/special_operators/shift.jl",
    "sequence_spaces/norm.jl",
    "utilities.jl",
    "rpa/interval_existence.jl",
    "rpa/newton.jl",
    "rpa/proofs.jl",
])

@testset verbose = true "RadiiPolynomial" begin
    foreach(include, TEST_FILES)
end
