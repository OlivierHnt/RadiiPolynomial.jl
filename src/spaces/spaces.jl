"""
    UnivariateSpace <: SequenceSpace

Abstract type for sequence spaces.
"""
abstract type SequenceSpace end

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
    # symmetry :: Symbol
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
    # symmetry :: Symbol
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
    # symmetry :: Symbol
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
⊗(s₁::TensorSpace{Tuple{T}}, s₂::TensorSpace{Tuple{}}) where {T<:UnivariateSpace} = s₁.spaces[1]
⊗(s₁::TensorSpace{Tuple{}}, s₂::TensorSpace{Tuple{T}}) where {T<:UnivariateSpace} = s₂.spaces[1]

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

multiplication_range(s₁::Taylor, s₂::Taylor) = Taylor(s₁.order + s₂.order)
function multiplication_range(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    @assert s₁.frequency == s₂.frequency
    NewType = promote_type(T, S)
    return Fourier{NewType}(s₁.order + s₂.order, convert(NewType, s₁.frequency))
end
multiplication_range(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(s₁.order + s₂.order)
multiplication_range(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(multiplication_range, s₁.spaces, s₂.spaces))

function multiplication_range(s::SequenceSpace, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
    n == 0 && return s
    n == 1 && return s
    n == 2 && return multiplication_range(s, s)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        s = multiplication_range(s, s)
    end
    new_s = s
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            s = multiplication_range(s, s)
        end
        new_s = multiplication_range(new_s, s)
    end
    return new_s
end

function derivative_range(s::Taylor, n::Int=1)
    @assert n ≥ 1
    s.order < n && return Taylor(0)
    return Taylor(s.order-n)
end
derivative_range(s::Fourier, n::Int=1) = s
function derivative_range(s::Chebyshev, n::Int=1)
    @assert n ≥ 1
    s.order < n && return Chebyshev(0)
    return Chebyshev(s.order-n)
end
derivative_range(s::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, n::Int) where {N} =
    TensorSpace(tuple(s.spaces[1:dims-1]..., derivative_range(s.spaces[dims], n), s.spaces[dims+1:N]...))

integral_range(s::Taylor, n::Int=1) = Taylor(s.order+n)
integral_range(s::Fourier, n::Int=1) = s
integral_range(s::Chebyshev, n::Int=1) = Chebyshev(s.order+n)
integral_range(s::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, n::Int) where {N} =
    TensorSpace(tuple(s.spaces[1:dims-1]..., integral_range(s.spaces[dims], n), s.spaces[dims+1:N]...))

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
Base.eachindex(s::TensorSpace) = vec(collect(Iterators.product(map(eachindex, s.spaces)...)))

Base.axes(s::UnivariateSpace) = tuple(eachindex(s))
Base.axes(s::TensorSpace) = map(eachindex, s.spaces)
Base.axes(s::TensorSpace, i::Int) = eachindex(s.spaces[i])

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
        n = :(length(space[$i-1]) * $n)
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

pretty_string(space::TensorSpace) =
    mapreduce(sᵢ -> string(" ⨂ ", pretty_string(sᵢ)), *, Base.tail(space.spaces); init = pretty_string(space.spaces[1]))
