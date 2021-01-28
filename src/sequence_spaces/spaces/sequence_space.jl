"""
    SequenceSpace <: SingleSpace

Abstract type for sequence spaces.
"""
abstract type SequenceSpace <: SingleSpace end

"""
    UnivariateSpace <: SequenceSpace

Abstract type for univariate sequence spaces.
"""
abstract type UnivariateSpace <: SequenceSpace end

"""
    Taylor <: UnivariateSpace

Taylor sequence space whose elements are Taylor sequences of a prescribed order.

Fields:
- `order :: Int`
"""
struct Taylor <: UnivariateSpace
    order :: Int
end

"""
    Fourier{T} <: UnivariateSpace

Fourier sequence space whose elements are Fourier sequences of a prescribed order
and frequency.

Fields:
- `order :: Int`
- `frequency :: T`
"""
struct Fourier{T} <: UnivariateSpace
    order :: Int
    frequency :: T
end

"""
    Chebyshev <: UnivariateSpace

Chebyshev sequence space whose elements are Chebyshev sequences of a prescribed order.

Fields:
- `order :: Int`
"""
struct Chebyshev <: UnivariateSpace
    order :: Int
end

"""
    TensorSpace{T<:NTuple{N,UnivariateSpace} where {N}} <: SequenceSpace

Multivariate [`SequenceSpace`](@ref) resulting from the tensor product of some
[`UnivariateSpace`](@ref).

Fields:
- `spaces :: T`
"""
struct TensorSpace{T<:NTuple{N,UnivariateSpace} where {N}} <: SequenceSpace
    spaces :: T
end

TensorSpace(spaces::UnivariateSpace...) = TensorSpace(spaces)

⊗(s₁::UnivariateSpace, s₂::UnivariateSpace) = TensorSpace((s₁, s₂))
⊗(s₁::UnivariateSpace, s₂::TensorSpace) = TensorSpace((s₁, s₂.spaces...))
⊗(s₁::TensorSpace, s₂::UnivariateSpace) = TensorSpace((s₁.spaces..., s₂))
⊗(s₁::TensorSpace, s₂::TensorSpace) = TensorSpace((s₁.spaces..., s₂.spaces...))

## getindex

Base.@propagate_inbounds Base.getindex(s::TensorSpace, c::Colon) = TensorSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractRange) = TensorSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractVector{Int}) = TensorSpace(getindex(s.spaces, u))

Base.front(s::TensorSpace) = TensorSpace(Base.front(s.spaces))
Base.tail(s::TensorSpace) = TensorSpace(Base.tail(s.spaces))

## order

order(s::Taylor) = s.order
order(s::Fourier) = s.order
order(s::Chebyshev) = s.order
order(s::TensorSpace) = map(order, s.spaces)
order(s::TensorSpace, i::Int) = order(s.spaces[i])

## frequency

frequency(s::Fourier) = s.frequency
frequency(s::TensorSpace) = map(frequency, s.spaces)
frequency(s::TensorSpace, i::Int) = frequency(s.spaces[i])

##

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

dimension(s::Taylor) = s.order + 1
dimension(s::Fourier) = 2s.order + 1
dimension(s::Chebyshev) = s.order + 1
dimension(s::TensorSpace) = mapreduce(dimension, *, s.spaces)

dimensions(s::TensorSpace) = map(dimension, s.spaces)
dimensions(s::TensorSpace, i::Int) = dimension(s.spaces[i])

startindex(s::Taylor) = 0
startindex(s::Fourier) = -s.order
startindex(s::Chebyshev) = 0
startindex(s::TensorSpace) = map(startindex, s.spaces)

endindex(s::Taylor) = s.order
endindex(s::Fourier) = s.order
endindex(s::Chebyshev) = s.order
endindex(s::TensorSpace) = map(endindex, s.spaces)

allindices(s::Taylor) = 0:s.order
allindices(s::Fourier) = -s.order:s.order
allindices(s::Chebyshev) = 0:s.order
allindices(s::TensorSpace) = Iterators.product(map(allindices, s.spaces)...) # vec(collect(Iterators.product(map(allindices, s.spaces)...)))

## index for the constant term

_constant_index(s::UnivariateSpace) = 0
_constant_index(s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} = ntuple(i -> 0, N)

##

isindexof(c::Colon, space::SequenceSpace) = true

isindexof(i::Int, space::Taylor) = 0 ≤ i ≤ space.order
isindexof(i::Int, space::Fourier) = -space.order ≤ i ≤ space.order
isindexof(i::Int, space::Chebyshev) = 0 ≤ i ≤ space.order
isindexof(α::NTuple{N,Int}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(t -> isindexof(t[1], t[2]), zip(α, space.spaces))

isindexof(u::AbstractRange, space::Taylor) = 0 ≤ minimum(u) && maximum(u) ≤ space.order
isindexof(u::AbstractRange, space::Fourier) = -space.order ≤ minimum(u) && maximum(u) ≤ space.order
isindexof(u::AbstractRange, space::Chebyshev) = 0 ≤ minimum(u) && maximum(u) ≤ space.order
isindexof(u::AbstractVector{NTuple{N,Int}}, space::TensorSpace) where {N} = all(α -> isindexof(α, space), u)
isindexof(u::NTuple{N,Any}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(t -> isindexof(t[1], t[2]), zip(u, space.spaces))
isindexof(u::Base.Iterators.ProductIterator, space::TensorSpace) = isindexof(u.iterators, space)

## internal functions to retrieve linear index

_findindex(c::Colon, space::SequenceSpace) = c

_findindex(i::Int, space::Taylor) = i + 1
_findindex(i::Int, space::Fourier) = i + space.order + 1
_findindex(i::Int, space::Chebyshev) = i + 1
@generated function _findindex(α::NTuple{N,Int}, space::TensorSpace) where {N}
    # follows column major convention
    idx = :(_findindex(α[1], space[1]))
    n = 1
    for i ∈ 2:N
        n = :(dimension(space[$i-1]) * $n)
        idx = :($n * (_findindex(α[$i], space[$i]) - 1) + $idx)
    end
    return idx
end

_findindex(u::AbstractRange, space::Taylor) = u .+ 1
_findindex(u::AbstractRange, space::Fourier) = u .+ (space.order + 1)
_findindex(u::AbstractRange, space::Chebyshev) = u .+ 1
function _findindex(u::AbstractVector{NTuple{N,Int}}, space::TensorSpace) where {N}
    v = Vector{Int}(undef, length(u))
    @inbounds for (i,αᵢ) ∈ enumerate(u)
        v[i] = _findindex(αᵢ, space)
    end
    return v
end
_findindex(u::Tuple, space::TensorSpace) = map(_findindex, u, space.spaces)
function _findindex(u::Base.Iterators.ProductIterator, space::TensorSpace)
    v = Vector{Int}(undef, length(u))
    @inbounds for (i,αᵢ) ∈ enumerate(u)
        v[i] = _findindex(αᵢ, space)
    end
    return v
end

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

Base.show(io::IO, space::SequenceSpace) = print(io, pretty_string(space))

pretty_string(space::Taylor) = string("Taylor(", space.order, ")")
pretty_string(space::Fourier) = string("Fourier(", space.order, ", ", space.frequency, ")")
pretty_string(space::Chebyshev) = string("Chebyshev(", space.order, ")")

pretty_string(space::TensorSpace) =
    mapreduce(sᵢ -> string(" ⨂ ", pretty_string(sᵢ)), *, Base.tail(space.spaces); init = pretty_string(space.spaces[1]))
pretty_string(space::TensorSpace{Tuple{}}) = string(Tuple{}())
