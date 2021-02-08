"""
    Sequence{T<:VectorSpace,S<:AbstractVector}

Compactly supported sequence of the given space.

Fields:
- `space :: T`
- `coefficients :: S`
"""
struct Sequence{T<:VectorSpace,S<:AbstractVector}
    space :: T
    coefficients :: S
    function Sequence{T,S}(space::T, coefficients::S) where {T<:VectorSpace,S<:AbstractVector}
        @assert dimension(space) == length(coefficients)
        return new{T,S}(space, coefficients)
    end
end

Sequence(space::T, coefficients::S) where {T<:VectorSpace,S<:AbstractVector} =
    Sequence{T,S}(space, coefficients)

## space, coefficients, order, frequency

space(a::Sequence) = a.space

coefficients(a::Sequence) = a.coefficients

order(a::Sequence) = order(a.space)
order(a::Sequence, i::Int) = order(a.space, i)

frequency(a::Sequence) = frequency(a.space)
frequency(a::Sequence, i::Int) = frequency(a.space, i)

## utilities

Base.firstindex(a::Sequence) = startindex(a.space)

Base.lastindex(a::Sequence) = endindex(a.space)

Base.eachindex(a::Sequence) = allindices(a.space)

Base.length(a::Sequence) = length(a.coefficients)

Base.size(a::Sequence) = size(a.coefficients) # necessary for broadcasting

Base.iterate(a::Sequence) = iterate(a.coefficients)
Base.iterate(a::Sequence, i) = iterate(a.coefficients, i)

Base.eltype(a::Sequence) = eltype(a.coefficients)
Base.eltype(::Type{Sequence{T,S}}) where {T,S} = eltype(S)

## getindex, view, setindex!

for f ∈ (:getindex, :view)
    @eval Base.@propagate_inbounds function Base.$f(a::Sequence, α)
        @boundscheck(isindexof(α, a.space) || throw(BoundsError(a.space, α)))
        return $f(a.coefficients, _findindex(α, a.space))
    end
end

Base.@propagate_inbounds function Base.setindex!(a::Sequence, x, α)
    @boundscheck(isindexof(α, a.space) || throw(BoundsError(a.space, α)))
    return setindex!(a.coefficients, x, _findindex(α, a.space))
end

## ==, iszero, isapprox

Base.:(==)(a::Sequence, b::Sequence) =
    a.space == b.space && a.coefficients == b.coefficients

Base.iszero(a::Sequence) = iszero(a.coefficients)

Base.isapprox(a::Sequence, b::Sequence; kwargs...) =
    a.space == b.space && isapprox(a.coefficients, b.coefficients; kwargs...)

## copy, similiar

Base.copy(a::Sequence) = Sequence(a.space, copy(a.coefficients))

Base.similar(a::Sequence) = Sequence(a.space, similar(a.coefficients))

## zero

Base.zero(a::Sequence) = Sequence(a.space, zero.(a.coefficients))

## float, complex, real, imag, conj, conj!

for f ∈ (:float, :complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::Sequence) = Sequence(a.space, $f(a.coefficients))
end

## promotion

Base.convert(::Type{Sequence{T₁,S₁}}, a::Sequence{T₂,S₂}) where {T₁,S₁,T₂,S₂} =
    Sequence{T₁,S₁}(convert(T₁, a.space), convert(S₁, a.coefficients))

Base.promote_rule(::Type{Sequence{T₁,S₁}}, ::Type{Sequence{T₂,S₂}}) where {T₁,S₁,T₂,S₂} =
    Sequence{promote_type(T₁,T₂),promote_type(S₁,S₂)}

## broadcasting

struct SequenceStyle <: Base.Broadcast.BroadcastStyle end

Base.Broadcast.broadcastable(a::Sequence) = a

Base.Broadcast.BroadcastStyle(::Type{<:Sequence}) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{0}, ::SequenceStyle) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::SequenceStyle, ::Base.Broadcast.AbstractArrayStyle{0}) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::Base.Broadcast.AbstractArrayStyle{1}, ::SequenceStyle) = SequenceStyle()
Base.Broadcast.BroadcastStyle(::SequenceStyle, ::Base.Broadcast.AbstractArrayStyle{1}) = SequenceStyle()

function Base.similar(bc::Base.Broadcast.Broadcasted{SequenceStyle}, ::Type{ElType}) where {ElType}
    space = _find_space(bc)
    return Sequence(space, similar(Vector{ElType}, dimension(space)))
end

_find_space(bc::Base.Broadcast.Broadcasted) = _find_space(bc.args)
_find_space(args::Tuple) = _find_space(_find_space(args[1]), _find_space(Base.tail(args)))
_find_space(a::Sequence) = a.space
_find_space(::Tuple{}) = nothing
_find_space(::Any) = nothing
_find_space(space::VectorSpace, ::Nothing) = space
_find_space(::Nothing, space::VectorSpace) = space
_find_space(::Nothing, ::Nothing) = nothing
function _find_space(s₁::T, s₂::S) where {T<:VectorSpace,S<:VectorSpace}
    @assert s₁ == s₂
    return convert(promote_type(T, S), s₁)
end

@inline function Base.copyto!(dest::Sequence, src::Sequence)
    dest.space == src.space || return throw(ArgumentError)
    copyto!(dest.coefficients, src.coefficients)
    return dest
end

@inline function Base.copyto!(dest::Sequence, bc::Base.Broadcast.Broadcasted)
    bc.f === identity && bc.args isa Tuple{Sequence} && return copyto!(dest, bc.args[1])
    bc′ = Base.Broadcast.preprocess(dest, bc)
    @inbounds @simd for i ∈ eachindex(bc′)
        dest.coefficients[i] = bc′[i]
    end
    return dest
end

Base.Broadcast._broadcast_getindex(a::Sequence, i::Int) =
    Base.Broadcast._broadcast_getindex(a.coefficients, i)

# to allow a[...] .= f.(...)
Base.@propagate_inbounds Base.Broadcast.dotview(a::Sequence, α) = view(a, α)

## SEQUENCE SPACE

# selectdim

Base.@propagate_inbounds function Base.selectdim(a::Sequence{TensorSpace{T}}, dim::Int, i::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    @boundscheck(!isindexof(i, a.space.spaces[dim]) && throw(BoundsError(a.space.spaces[dim], i)))
    A = reshape(a.coefficients, dimensions(a.space))
    A_ = selectdim(A, dim, _findindex(i, a.space.spaces[dim]))
    space = TensorSpace((a.space.spaces[1:dim-1]..., a.space.spaces[dim+1:N]...))
    return Sequence(space, vec(A_))
end
Base.@propagate_inbounds function Base.selectdim(a::Sequence{TensorSpace{T}}, dim::Int, i::Int) where {T<:NTuple{2,UnivariateSpace}}
    @boundscheck(!isindexof(i, a.space.spaces[dim]) && throw(BoundsError(a.space.spaces[dim], i)))
    A = reshape(a.coefficients, dimensions(a.space))
    A_ = selectdim(A, dim, _findindex(i, a.space.spaces[dim]))
    space = dim == 1 ? a.space.spaces[2] : a.space.spaces[dim-1]
    return Sequence(space, vec(A_))
end

# permutedims

Base.permutedims(a::Sequence{<:TensorSpace}, σ::AbstractVector{Int}) =
    Sequence(a.space.spaces[σ], vec(permutedims(reshape(a.coefficients, dimensions(a.space)), σ)))

# one

function Base.one(a::Sequence{<:SequenceSpace})
    b = zero(a)
    @inbounds b[_constant_index(a.space)] = one(eltype(a))
    return b
end

# show

# Base.show(io::IO, a::Sequence) = print(io, pretty_string(a.space) * ":\n\n" * pretty_string(a))
#
# function pretty_string(a::Sequence{T}) where {T<:SequenceSpace}
#     indices = allindices(a.space)
#     @inbounds strout = string(" (", a[indices[1]], ") ") * string_basis_symbol(T, indices[1])
#     if dimension(a.space) ≤ 10
#         @inbounds for i ∈ 2:dimension(a.space)
#             strout *= "\n" * string(" (", a[indices[i]], ") ") * string_basis_symbol(T, indices[i])
#         end
#     else
#         @inbounds for i ∈ 2:10
#             strout *= "\n" * string(" (", a[indices[i]], ") ") * string_basis_symbol(T, indices[i])
#         end
#         strout *= "\n ⋮"
#         @inbounds for i ∈ max(12,dimension(a.space)-9):dimension(a.space)
#             strout *= "\n" * string(" (", a[indices[i]], ") ") * string_basis_symbol(T, indices[i])
#         end
#     end
#     return strout
# end
#
# string_basis_symbol(::Type{<:UnivariateSpace}, i::Int) = i < 0 ? string("𝜙₋", subscriptify(-i)) : string("𝜙", subscriptify(i))
#
# function string_basis_symbol(::Type{<:TensorSpace}, α::NTuple{N,Int}) where {N}
#     @inbounds s = α[1] < 0 ? string("𝜙⁽¹⁾₋", subscriptify(-α[1])) : string("𝜙⁽¹⁾", subscriptify(α[1]))
#     @inbounds for i ∈ 2:N
#         s *= " ⊗ "
#         s *= α[i] < 0 ? string("𝜙⁽", superscriptify(i), "⁾₋", subscriptify(-α[i])) : string("𝜙⁽", superscriptify(i), "⁾", subscriptify(α[i]))
#     end
#     return s
# end

## CARTESIAN SPACE

function Sequence(space::CartesianProductSpace, a::AbstractVector{T}) where {T<:Sequence}
    length(space.spaces) == length(a) || return throw(DimensionMismatch)
    c = Sequence(space, Vector{eltype(T)}(undef, dimension(space)))
    foreach((cᵢ, aᵢ) -> cᵢ.coefficients .= project(aᵢ, cᵢ.space).coefficients, eachcomponent(c), a)
    return c
end

Sequence(a::AbstractVector{<:Sequence}) = sequence(space(a), coefficients(a))

space(a::AbstractVector{<:Sequence}) = CartesianProductSpace(ntuple(i -> space(a[i]), length(a)))

function coefficients(a::AbstractVector{T}) where {T<:Sequence}
    v = Vector{eltype(T)}(undef, sum(length, a))
    len_ = 0
    @inbounds for aᵢ ∈ a
        len = length(aᵢ)
        view(v, 1+len_:len_+len) .= aᵢ.coefficients
        len_ += len
    end
    return v
end

#

eachcomponent(a::Sequence{<:CartesianSpace}) =
    (@inbounds(component(a, i)) for i ∈ Base.OneTo(nb_cartesian_product(a.space)))

# tools for component

_skip_component(s::CartesianSpace, ::Colon) = 0
_skip_component(s::CartesianPowerSpace, i::Int) = i == 1 ? 0 : (i-1)*dimension(s.space)
_skip_component(s::CartesianPowerSpace, u::UnitRange) = first(u) == 1 ? 0 : (first(u)-1)*dimension(s.space)
_skip_component(s::CartesianProductSpace, i::Int) = i == 1 ? 0 : mapreduce(j -> dimension(s.spaces[j]), +, 1:i-1)
_skip_component(s::CartesianProductSpace, u::UnitRange) = first(u) == 1 ? 0 : mapreduce(j -> dimension(s.spaces[j]), +, 1:first(u)-1)

#

Base.@propagate_inbounds function component(a::Sequence{<:CartesianSpace}, i)
    @boundscheck(isindexof(i, a.space) || throw(BoundsError(a.space, i)))
    space = a.space[i]
    skip = _skip_component(a.space, i)
    return Sequence(space, view(a.coefficients, 1+skip:dimension(space)+skip))
end
