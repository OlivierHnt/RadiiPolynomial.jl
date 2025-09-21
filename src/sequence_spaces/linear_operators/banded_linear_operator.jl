"""
    BandedLinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix,U} <: AbstractLinearOperator

Linear operator with a finite part and an infinite banded part.

Fields:
- `finite_operator :: LinearOperator{T,S,R}`
- `banded_operator :: U`
"""
struct BandedLinearOperator{T<:VectorSpace,S<:VectorSpace,R<:AbstractMatrix,U} <: AbstractLinearOperator
    finite_operator :: LinearOperator{T,S,R}
    banded_operator :: U
end

finite_operator(A::BandedLinearOperator) = A.finite_operator

banded_operator(A::BandedLinearOperator) = A.banded_operator

domain(A::BandedLinearOperator) = domain(finite_operator(A)) # needed for general methods

codomain(A::BandedLinearOperator) = codomain(finite_operator(A)) # needed for general methods

coefficients(A::BandedLinearOperator) = coefficients(finite_operator(A)) # needed for general methods

# utilities

# note: only look at the type of the finite part
Base.eltype(A::BandedLinearOperator) = eltype(finite_operator(A))
Base.eltype(::Type{<:BandedLinearOperator{<:VectorSpace,<:VectorSpace,T}}) where {T<:AbstractMatrix} = eltype(T)

#

Base.:*(A::BandedLinearOperator, b::AbstractSequence) = (A * Projection(space(b))) * b

codomain(A::BandedLinearOperator, s::VectorSpace) = codomain(finite_operator(A), s) ∪ _codomain(banded_operator(A), s)
_codomain(A, s) = codomain(A, s)
function _codomain(A::Union{Vector,Matrix}, s)
    v = CartesianProduct(ParameterSpace())
    for i ∈ 1:size(A, 1)
        w = _codomain(A[i,1], s[1])
        for j ∈ 2:size(A, 2)
            w = w ∪ _codomain(A[i,j], s[j])
        end
        v = CartesianProduct(spaces(v)..., w)
    end
    nspaces(v) == 2 && return v[2]
    return v[2:nspaces(v)]
end
function _codomain(A::Vector, s::ParameterSpace)
    v = ParameterSpace()
    for i ∈ 1:length(A)
        w = _codomain(A[i], s)
        v = v × w
    end
    nspaces(v) == 2 && return v[2]
    return v[2:nspaces(v)]
end

function _project!(C::LinearOperator, A::BandedLinearOperator)
    coefficients(C) .= zero(eltype(C))
    _band_project!(C, banded_operator(A))
    view(C, codomain(C) ∩ codomain(A), domain(C) ∩ domain(A)) .= view(finite_operator(A), codomain(C) ∩ codomain(A), domain(C) ∩ domain(A))
    return C
end

function _band_project!(C::LinearOperator, A::Matrix)
    for j ∈ axes(A, 2), i ∈ axes(A, 1)
       _band_project!(component(C, i, j), A[i,j])
    end
    return C
end
function _band_project!(C::LinearOperator{<:VectorSpace,ParameterSpace}, A::Matrix)
    for j ∈ axes(A, 2)
       _band_project!(component(C, j), A[j])
    end
    return C
end
function _band_project!(C::LinearOperator, A::Vector)
    for i ∈ axes(A, 1)
       _band_project!(component(C, i), A[i])
    end
    return C
end
_band_project!(C::LinearOperator, A::Union{AbstractLinearOperator,UniformScaling,ComposedFunction}) = project!(C, A)
