"""
    Laplacian <: AbstractLinearOperator

Generic Laplacian operator.
"""
struct Laplacian <: AbstractLinearOperator end

# Tensor space

function domain(Δ::Laplacian, s::TensorSpace)
    s_out = map(sᵢ -> domain(Δ, sᵢ), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return codomain(+, s, TensorSpace(s_out))
end

function codomain(Δ::Laplacian, s::TensorSpace)
    s_out = map(sᵢ -> codomain(Δ, sᵢ), spaces(s))
    return codomain(+, s, TensorSpace(s_out))
end

_coeftype(::Laplacian, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    _coeftype(Derivative(ntuple(i -> 2, Val(N))), s, T)

function getcoefficient(::Laplacian, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T}
    val = zero(T)
    for k ∈ 1:N
        term = one(T)
        for l ∈ 1:N
            @inbounds term *= getcoefficient(Derivative(ifelse(l==k, 2, 0)), (codom[l], i[l]), (dom[l], j[l]), T)
        end
        val += term
    end
    return val
end

# Base space

domain(::Laplacian, s::BaseSpace) = domain(Derivative(2), s)

codomain(::Laplacian, s::BaseSpace) = codomain(Derivative(2), s)

_coeftype(::Laplacian, s::BaseSpace, ::Type{T}) where {T} = real(_coeftype(Derivative(2), s, T))

getcoefficient(::Laplacian, (codom, i)::Tuple{BaseSpace,Integer}, (dom, j)::Tuple{BaseSpace,Integer}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Derivative(2), (codom, i), (dom, j), T)

# Symmetric space

domain(::Laplacian, s::SymmetricSpace{<:BaseSpace}) = domain(Derivative(2), s)

codomain(::Laplacian, s::SymmetricSpace{<:BaseSpace}) = codomain(Derivative(2), s)

_coeftype(::Laplacian, s::SymmetricSpace{<:BaseSpace}, ::Type{T}) where {T} = real(_coeftype(Derivative(2), s, T))



# action

Base.:*(::Laplacian, a::AbstractSequence) = laplacian(a)

function laplacian(a::Sequence)
    Δ = Laplacian()
    space_a = space(a)
    new_space = codomain(Δ, space_a)
    CoefType = _coeftype(Δ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, Δ, a)
    return c
end

function laplacian!(c::Sequence, a::Sequence)
    Δ = Laplacian()
    space_c = space(c)
    new_space = codomain(Δ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, Laplacian(a) has space $new_space"))
    _apply!(c, Δ, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace{<:NTuple{N,BaseSpace}}}, ::Laplacian, a) where {N}
    _apply!(c, Derivative(ntuple(j -> ifelse(j == 1, 2, 0), Val(N))), a)
    c_ = similar(c)
    for i ∈ 2:N
        radd!(c, _apply!(c_, Derivative(ntuple(j -> ifelse(j == i, 2, 0), Val(N))), a))
    end
    return c
end

# Base space

_apply!(c::Sequence{<:BaseSpace}, ::Laplacian, a) = _apply!(c, Derivative(2), a)
