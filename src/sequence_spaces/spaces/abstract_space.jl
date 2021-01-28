"""
    VectorSpace

Abstract type for all vector spaces.
"""
abstract type VectorSpace end

"""
    SingleSpace <: VectorSpace

Abstract type for all vector spaces which are not a `CartesianSpace`.
"""
abstract type SingleSpace <: VectorSpace end
