abstract type SequenceSpace end

abstract type UnivariateSpace <: SequenceSpace end

struct Taylor <: UnivariateSpace
    # symmetry :: Symbol
    order :: Int
end

struct Fourier{T} <: UnivariateSpace
    # symmetry :: Symbol
    order :: Int
    frequency :: T
end

struct Chebyshev <: UnivariateSpace
    # symmetry :: Symbol
    order :: Int
end

struct TensorSpace{T<:NTuple{N,UnivariateSpace} where {N}} <: SequenceSpace
    spaces :: T
end

# TensorSpace(spaces::UnivariateSpace...) = TensorSpace(spaces)

⊗(s₁::UnivariateSpace, s₂::UnivariateSpace) = TensorSpace((s₁, s₂))
⊗(s₁::UnivariateSpace, s₂::TensorSpace) = TensorSpace((s₁, s₂.spaces...))
⊗(s₁::TensorSpace, s₂::UnivariateSpace) = TensorSpace((s₁.spaces..., s₂))
⊗(s₁::TensorSpace, s₂::TensorSpace) = TensorSpace((s₁.spaces..., s₂.spaces...))
⊗(s₁::TensorSpace{Tuple{T}}, s₂::TensorSpace{Tuple{}}) where {T<:UnivariateSpace} = s₁.spaces[1]
⊗(s₁::TensorSpace{Tuple{}}, s₂::TensorSpace{Tuple{T}}) where {T<:UnivariateSpace} = s₂.spaces[1]

## getindex

Base.@propagate_inbounds Base.getindex(s::TensorSpace, c::Colon) = TensorSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractRange) = TensorSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::Vector{Int}) = TensorSpace(getindex(s.spaces, u))

## order

order(s::Taylor) = s.order
order(s::Fourier) = s.order
order(s::Chebyshev) = s.order
order(s::TensorSpace) = map(order, s.spaces)

##

multiplication_range_space(s₁::Taylor, s₂::Taylor) = Taylor(s₁.order + s₂.order)
function multiplication_range_space(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    @assert s₁.frequency == s₂.frequency
    NewType = promote_type(T, S)
    return Fourier{NewType}(s₁.order + s₂.order, convert(NewType, s₁.frequency))
end
multiplication_range_space(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(s₁.order + s₂.order)
multiplication_range_space(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(multiplication_range_space, s₁.spaces, s₂.spaces))

pow_range_space(s::SequenceSpace, n::Int) = mapreduce(i -> s, multiplication_range_space, 1:n)

derivation_range_space(s::Taylor) = s.order == 0 ? Taylor(0) : Taylor(s.order-1)
derivation_range_space(s::Fourier) = s
derivation_range_space(s::Chebyshev) = s.order == 0 ? Chebyshev(0) : Chebyshev(s.order-1)
derivation_range_space(s::TensorSpace) = TensorSpace(map(derivation_range_space, s.spaces))

integration_range_space(s::Taylor) = Taylor(s.order+1)
integration_range_space(s::Fourier) = s
integration_range_space(s::Chebyshev) = Chebyshev(s.order+1)
integration_range_space(s::TensorSpace) = TensorSpace(map(integration_range_space, s.spaces))

##

Base.:∩(s₁::SequenceSpace, s₂::SequenceSpace) = throw(MethodError)
Base.:∩(s₁::Taylor, s₂::Taylor) = s₁.order ≤ s₂.order ? s₁ : s₂
function Base.:∩(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    @assert s₁.frequency == s₂.frequency
    NewType = promote_type(T, S)
    s₁.order ≤ s₂.order && return convert(Fourier{NewType}, s₁)
    return convert(Fourier{NewType}, s₂)
end
Base.:∩(s₁::Chebyshev, s₂::Chebyshev) = s₁.order ≤ s₂.order ? s₁ : s₂
Base.:∩(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(∩, s₁.spaces, s₂.spaces))

Base.:∪(s₁::SequenceSpace, s₂::SequenceSpace) = throw(MethodError)
Base.:∪(s₁::Taylor, s₂::Taylor) = s₁.order ≤ s₂.order ? s₂ : s₁
function Base.:∪(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    @assert s₁.frequency == s₂.frequency
    NewType = promote_type(T, S)
    s₁.order ≤ s₂.order && return convert(Fourier{NewType}, s₂)
    return convert(Fourier{NewType}, s₁)
end
Base.:∪(s₁::Chebyshev, s₂::Chebyshev) = s₁.order ≤ s₂.order ? s₂ : s₁
Base.:∪(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(∪, s₁.spaces, s₂.spaces))

# ∪̄ differs from ∩ when there are symmetries
∪̄(s₁::SequenceSpace, s₂::SequenceSpace) = throw(MethodError)
∪̄(s₁::Taylor, s₂::Taylor) = s₁.order ≤ s₂.order ? s₁ : s₂
function ∪̄(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    @assert s₁.frequency == s₂.frequency
    NewType = promote_type(T, S)
    s₁.order ≤ s₂.order && return convert(Fourier{NewType}, s₁)
    return convert(Fourier{NewType}, s₂)
end
∪̄(s₁::Chebyshev, s₂::Chebyshev) = s₁.order ≤ s₂.order ? s₁ : s₂
∪̄(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(∪̄, s₁.spaces, s₂.spaces))

Base.issubset(s₁::SequenceSpace, s₂::SequenceSpace) = false
Base.issubset(s₁::Taylor, s₂::Taylor) = s₁.order ≤ s₂.order
Base.issubset(s₁::Fourier, s₂::Fourier) = s₁.frequency == s₂.frequency && s₁.order ≤ s₂.order
Base.issubset(s₁::Chebyshev, s₂::Chebyshev) = s₁.order ≤ s₂.order
Base.issubset(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(s₁.spaces, s₂.spaces))

Base.:(==)(s₁::SequenceSpace, s₂::SequenceSpace) = false
Base.:(==)(s₁::Taylor, s₂::Taylor) = s₁.order == s₂.order
Base.:(==)(s₁::Fourier, s₂::Fourier) = s₁.order == s₂.order && s₁.frequency == s₂.frequency
Base.:(==)(s₁::Chebyshev, s₂::Chebyshev) = s₁.order == s₂.order
Base.:(==)(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(s₁.spaces, s₂.spaces))

## characterization of spaces

Base.length(s::Taylor) = s.order + 1
Base.length(s::Fourier) = 2s.order + 1
Base.length(s::Chebyshev) = s.order + 1
Base.length(s::TensorSpace) = mapreduce(length, *, s.spaces)

Base.size(s::UnivariateSpace) = tuple(length(s))
Base.size(s::TensorSpace) = map(length, s.spaces)
Base.size(s::TensorSpace, i::Int) = length(s.spaces[i])

# Base.ndims(s::UnivariateSpace) = 1
# Base.ndims(::Type{T}) where {T<:UnivariateSpace} = 1
# Base.ndims(s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} = N
# Base.ndims(::Type{TensorSpace{T}}) where {N,T<:NTuple{N,UnivariateSpace}} = N

Base.firstindex(s::Taylor) = 0
Base.firstindex(s::Fourier) = -s.order
Base.firstindex(s::Chebyshev) = 0
Base.firstindex(s::TensorSpace) = map(firstindex, s.spaces)

Base.lastindex(s::Taylor) = s.order
Base.lastindex(s::Fourier) = s.order
Base.lastindex(s::Chebyshev) = s.order
Base.lastindex(s::TensorSpace) = map(lastindex, s.spaces)

Base.eachindex(s::Taylor) = firstindex(s):lastindex(s)
Base.eachindex(s::Fourier) = firstindex(s):lastindex(s)
Base.eachindex(s::Chebyshev) = firstindex(s):lastindex(s)
Base.eachindex(s::TensorSpace) = vec(collect(Iterators.product(map(eachindex, s.spaces)...))) # MultiIndices(axes(s))

Base.axes(s::UnivariateSpace) = tuple(eachindex(s))
Base.axes(s::TensorSpace) = map(eachindex, s.spaces)
Base.axes(s::TensorSpace, i::Int) = eachindex(s.spaces[i])

## used e.g. Sequence + Number

_constant_index(s::UnivariateSpace) = 0
_constant_index(s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} = ntuple(i -> 0, N)

##

isindexof(c::Colon, space::SequenceSpace) = true # fallback

isindexof(i::Int, space::Taylor) = 0 ≤ i ≤ space.order
isindexof(i::Int, space::Fourier) = -space.order ≤ i ≤ space.order
isindexof(i::Int, space::Chebyshev) = 0 ≤ i ≤ space.order
isindexof(α::NTuple{N,Int}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(t -> isindexof(t[1], t[2]), zip(α, space.spaces))

isindexof(u::AbstractRange, space::Taylor) = 0 ≤ minimum(u) && maximum(u) ≤ space.order
isindexof(u::AbstractRange, space::Fourier) = -space.order ≤ minimum(u) && maximum(u) ≤ space.order
isindexof(u::AbstractRange, space::Chebyshev) = 0 ≤ minimum(u) && maximum(u) ≤ space.order
isindexof(u::Vector{NTuple{N,Int}}, space::TensorSpace) where {N} = all(α -> isindexof(α, space), u)
isindexof(u::NTuple{N,Any}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(t -> isindexof(t[1], t[2]), zip(u, space.spaces))

## internal functions to retrieve linear index

# type for lazy multidimensional indexing. Much slower than generating the indices at once

# struct MultiIndices{T}
#     indices :: T
# end
#
# Base.@propagate_inbounds function Base.getindex(inds::MultiIndices, i::Int)
#     @boundscheck(i ≤ length(inds) && throw(BoundsError(inds, i)))
#     return Base._ind2sub(inds.indices, i)
# end
#
# Base.length(inds::MultiIndices) = mapreduce(length, *, inds.indices)
#
# Base.iterate(inds::MultiIndices) = inds[1], 2
#
# function Base.iterate(inds::MultiIndices, state::Int)
#     state ≤ length(inds) && return inds[state], state + 1
#     return nothing
# end
#
#     export MultiIndices

#

_findindex(c::Colon, space::SequenceSpace) = c # fallback

_findindex(i::Int, space::Taylor) = i + 1
_findindex(i::Int, space::Fourier) = i + space.order + 1
_findindex(i::Int, space::Chebyshev) = i + 1
@generated function _findindex(α::NTuple{N,Int}, space::TensorSpace) where {N}
    # follows column major convention
    idx = :(_findindex(α[1], space[1]))
    n = 1
    for i ∈ 2:N
        n = :(length(space[$i-1]) * $n)
        idx = :($n * (_findindex(α[$i], space[$i]) - 1) + $idx)
    end
    return idx
end

_findindex(u::AbstractRange, space::Taylor) = u .+ 1
_findindex(u::AbstractRange, space::Fourier) = u .+ (space.order + 1)
_findindex(u::AbstractRange, space::Chebyshev) = u .+ 1
function _findindex(u::Vector{NTuple{N,Int}}, space::TensorSpace) where {N}
    v = Vector{Int}(undef, length(u))
    @inbounds for (i,αᵢ) ∈ enumerate(u)
        v[i] = _findindex(αᵢ, space)
    end
    return v
end
_findindex(u::Tuple, space::TensorSpace) = map(_findindex, u, space.spaces)

## promotion

Base.convert(::Type{Fourier{T}}, space::Fourier{S}) where {T,S} =
    Fourier{T}(space.order, convert(T, space.frequency))

Base.convert(::Type{TensorSpace{T}}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N,T<:Tuple{Vararg{UnivariateSpace,N}}} =
    TensorSpace{T}(ntuple(i -> convert(T.parameters[i], space.spaces[i]), N))

Base.promote_rule(::Type{Fourier{T}}, ::Type{Fourier{S}}) where {T,S} =
    Fourier{promote_type(T, S)}

Base.promote_rule(::Type{TensorSpace{T}}, ::Type{TensorSpace{S}}) where {N,T<:Tuple{Vararg{UnivariateSpace,N}},S<:Tuple{Vararg{UnivariateSpace,N}}} =
    TensorSpace{Tuple{map(promote_type, T.parameters, S.parameters)...}}

## show

Base.show(io::IO, space::TensorSpace) = print(io, pretty_string(space))

pretty_string(space::UnivariateSpace) = string(space)

pretty_string(space::TensorSpace{Tuple{}}) = string(Tuple{}())

function pretty_string(space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N}
    @inbounds s = pretty_string(space.spaces[1])
    @inbounds for i ∈ 2:N
        s *= " ⨂ "
        s *= pretty_string(space.spaces[i])
    end
    return s
end
