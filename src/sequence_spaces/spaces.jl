"""
    VectorSpace

Abstract type for all vector spaces.
"""
abstract type VectorSpace end

Base.:(==)(::VectorSpace, ::VectorSpace) = false
Base.issubset(::VectorSpace, ::VectorSpace) = false
Base.intersect(::VectorSpace, ::VectorSpace) = throw(MethodError)
Base.union(::VectorSpace, ::VectorSpace) = throw(MethodError)





##

"""
    ParameterSpace <: VectorSpace

Space of parameters.
"""
struct ParameterSpace <: VectorSpace end

Base.:(==)(::ParameterSpace, ::ParameterSpace) = true
Base.issubset(::ParameterSpace, ::ParameterSpace) = true
Base.intersect(::ParameterSpace, ::ParameterSpace) = ParameterSpace()
Base.union(::ParameterSpace, ::ParameterSpace) = ParameterSpace()

dimension(::ParameterSpace) = 1
startindex(::ParameterSpace) = 1
endindex(::ParameterSpace) = 1
allindices(::ParameterSpace) = Base.OneTo(1)

isindexof(i::Int, ::ParameterSpace) = i == 1
isindexof(u::AbstractRange, ::ParameterSpace) = 1 == first(u) == last(u)
isindexof(::Colon, ::ParameterSpace) = true
isindexof(u::AbstractVector{Int}, s::ParameterSpace) = all(i -> isindexof(i, s), u)

_findindex(i, ::ParameterSpace) = i

# show

Base.show(io::IO, space::ParameterSpace) = print(io, pretty_string(space))
pretty_string(::ParameterSpace) = "š"





## sequence spaces

"""
    SequenceSpace <: VectorSpace

Abstract type for sequence spaces.
"""
abstract type SequenceSpace <: VectorSpace end

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
    function Taylor(order::Int)
        order < 0 && return throw(DomainError(order, "Taylor is only defined for positive orders"))
        return new(order)
    end
end

order(s::Taylor) = s.order

Base.:(==)(sā::Taylor, sā::Taylor) = sā.order == sā.order
Base.issubset(sā::Taylor, sā::Taylor) = sā.order ā¤ sā.order
Base.intersect(sā::Taylor, sā::Taylor) = Taylor(min(sā.order, sā.order))
Base.union(sā::Taylor, sā::Taylor) = Taylor(max(sā.order, sā.order))

dimension(s::Taylor) = s.order + 1
startindex(s::Taylor) = 0
endindex(s::Taylor) = s.order
allindices(s::Taylor) = 0:s.order

isindexof(i::Int, s::Taylor) = (0 ā¤ i) & (i ā¤ s.order)
isindexof(u::AbstractRange, s::Taylor) = (0 ā¤ first(u)) & (last(u) ā¤ s.order)
isindexof(::Colon, ::Taylor) = true
isindexof(u::AbstractVector{Int}, s::Taylor) = all(i -> isindexof(i, s), u)

_findindex(i::Int, s::Taylor) = i + 1
_findindex(u::AbstractRange, s::Taylor) = u .+ 1
_findindex(c::Colon, ::Taylor) = c
_findindex(u::AbstractVector{Int}, s::Taylor) = map(i -> _findindex(i, s), u)

_constant_index(::Taylor) = 0 # index for the constant term

# show

Base.show(io::IO, s::Taylor) = print(io, pretty_string(s))
pretty_string(s::Taylor) = string("Taylor(", s.order, ")")

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
    function Fourier{T}(order::Int, frequency::T) where {T}
        order < 0 && return throw(DomainError(order, "Fourier is only defined for positive orders"))
        frequency < 0 && return throw(DomainError(frequency, "Fourier is only defined for positive frequencies"))
        return new{T}(order, frequency)
    end
end

Fourier(order::Int, frequency::T) where {T} = Fourier{T}(order, frequency)

order(s::Fourier) = s.order

frequency(s::Fourier) = s.frequency

Base.:(==)(sā::Fourier, sā::Fourier) = (sā.frequency == sā.frequency) & (sā.order == sā.order)
Base.issubset(sā::Fourier, sā::Fourier) = (sā.frequency == sā.frequency) & (sā.order ā¤ sā.order)
function Base.intersect(sā::Fourier{T}, sā::Fourier{S}) where {T,S}
    sā.frequency == sā.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(min(sā.order, sā.order), convert(R, sā.frequency))
end
function Base.union(sā::Fourier{T}, sā::Fourier{S}) where {T,S}
    sā.frequency == sā.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(max(sā.order, sā.order), convert(R, sā.frequency))
end

dimension(s::Fourier) = 2s.order + 1
startindex(s::Fourier) = -s.order
endindex(s::Fourier) = s.order
allindices(s::Fourier) = -s.order:s.order

isindexof(i::Int, s::Fourier) = (-s.order ā¤ i) & (i ā¤ s.order)
isindexof(u::AbstractRange, s::Fourier) = (-s.order ā¤ first(u)) & (last(u) ā¤ s.order)
isindexof(::Colon, ::Fourier) = true
isindexof(u::AbstractVector{Int}, s::Fourier) = all(i -> isindexof(i, s), u)

_findindex(i::Int, s::Fourier) = i + s.order + 1
_findindex(u::AbstractRange, s::Fourier) = u .+ (s.order + 1)
_findindex(c::Colon, ::Fourier) = c
_findindex(u::AbstractVector{Int}, s::Fourier) = map(i -> _findindex(i, s), u)

_constant_index(::Fourier) = 0 # index for the constant term

# promotion

Base.convert(::Type{Fourier{T}}, s::Fourier{S}) where {T,S} =
    Fourier{T}(s.order, convert(T, s.frequency))
Base.promote_rule(::Type{Fourier{T}}, ::Type{Fourier{S}}) where {T,S} =
    Fourier{promote_type(T, S)}

# show

Base.show(io::IO, s::Fourier) = print(io, pretty_string(s))
pretty_string(s::Fourier) = string("Fourier(", s.order, ", ", s.frequency, ")")

"""
    Chebyshev <: UnivariateSpace

Chebyshev sequence space whose elements are Chebyshev sequences of a prescribed order.

Fields:
- `order :: Int`
"""
struct Chebyshev <: UnivariateSpace
    order :: Int
    function Chebyshev(order::Int)
        order < 0 && return throw(DomainError(order, "Chebyshev is only defined for positive orders"))
        return new(order)
    end
end

order(s::Chebyshev) = s.order

Base.:(==)(sā::Chebyshev, sā::Chebyshev) = sā.order == sā.order
Base.issubset(sā::Chebyshev, sā::Chebyshev) = sā.order ā¤ sā.order
Base.intersect(sā::Chebyshev, sā::Chebyshev) = Chebyshev(min(sā.order, sā.order))
Base.union(sā::Chebyshev, sā::Chebyshev) = Chebyshev(max(sā.order, sā.order))

dimension(s::Chebyshev) = s.order + 1
startindex(s::Chebyshev) = 0
endindex(s::Chebyshev) = s.order
allindices(s::Chebyshev) = 0:s.order

isindexof(i::Int, s::Chebyshev) = (0 ā¤ i) & (i ā¤ s.order)
isindexof(u::AbstractRange, s::Chebyshev) = (0 ā¤ first(u)) & (last(u) ā¤ s.order)
isindexof(::Colon, ::Chebyshev) = true
isindexof(u::AbstractVector{Int}, s::Chebyshev) = all(i -> isindexof(i, s), u)

_findindex(i::Int, s::Chebyshev) = i + 1
_findindex(u::AbstractRange, s::Chebyshev) = u .+ 1
_findindex(c::Colon, ::Chebyshev) = c
_findindex(u::AbstractVector{Int}, s::Chebyshev) = map(i -> _findindex(i, s), u)

_constant_index(::Chebyshev) = 0 # index for the constant term

# show

Base.show(io::IO, s::Chebyshev) = print(io, pretty_string(s))
pretty_string(s::Chebyshev) = string("Chebyshev(", s.order, ")")

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

spaces(s::TensorSpace) = s.spaces

ā(sā::SequenceSpace, sā::SequenceSpace) = tensor_space(sā, sā)

tensor_space(sā::UnivariateSpace, sā::UnivariateSpace) = TensorSpace((sā, sā))
tensor_space(sā::TensorSpace, sā::TensorSpace) = TensorSpace((sā.spaces..., sā.spaces...))
tensor_space(sā::TensorSpace, sā::UnivariateSpace) = TensorSpace((sā.spaces..., sā))
tensor_space(sā::UnivariateSpace, sā::TensorSpace) = TensorSpace((sā, sā.spaces...))

Base.@propagate_inbounds Base.getindex(s::TensorSpace, c::Colon) = TensorSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractRange) = TensorSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractVector{Int}) = TensorSpace(getindex(s.spaces, u))

Base.front(s::TensorSpace) = TensorSpace(Base.front(s.spaces))
Base.tail(s::TensorSpace) = TensorSpace(Base.tail(s.spaces))

#

Base.:(==)(sā::TensorSpace{<:NTuple{N,UnivariateSpace}}, sā::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(sā.spaces, sā.spaces))
Base.issubset(sā::TensorSpace{<:NTuple{N,UnivariateSpace}}, sā::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(sā.spaces, sā.spaces))
Base.intersect(sā::TensorSpace{<:NTuple{N,UnivariateSpace}}, sā::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(intersect, sā.spaces, sā.spaces))
Base.union(sā::TensorSpace{<:NTuple{N,UnivariateSpace}}, sā::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(union, sā.spaces, sā.spaces))

dimension(s::TensorSpace) = mapreduce(dimension, *, s.spaces)
dimensions(s::TensorSpace) = map(dimension, s.spaces)
dimensions(s::TensorSpace, i::Int) = dimension(s.spaces[i])
startindex(s::TensorSpace) = map(startindex, s.spaces)
endindex(s::TensorSpace) = map(endindex, s.spaces)
allindices(s::TensorSpace) = Base.Iterators.ProductIterator(map(allindices, s.spaces))

isindexof(Ī±::NTuple{N,Any}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(t -> isindexof(t[1], t[2]), zip(Ī±, s.spaces))
isindexof(u::Base.Iterators.ProductIterator, s::TensorSpace) = isindexof(u.iterators, s)
isindexof(::Colon, ::TensorSpace) = true
isindexof(u::AbstractVector{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(Ī± -> isindexof(Ī±, s), u)

@generated function _findindex(Ī±::NTuple{N,Int}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N}
    # follows column major convention
    idx = :(_findindex(Ī±[1], s.spaces[1]))
    n = 1
    for i ā 2:N
        n = :(dimension(s.spaces[$i-1]) * $n)
        idx = :($n * (_findindex(Ī±[$i], s.spaces[$i]) - 1) + $idx)
    end
    return idx
end
_findindex(u::NTuple{N,Any}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    _findindex(Base.Iterators.ProductIterator(u), s)
function _findindex(u::Base.Iterators.ProductIterator, s::TensorSpace)
    v = Vector{Int}(undef, length(u))
    for (i, uįµ¢) ā enumerate(u)
        v[i] = _findindex(uįµ¢, s)
    end
    return v
end
_findindex(c::Colon, ::TensorSpace) = c
_findindex(u::AbstractVector{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    map(Ī± -> _findindex(Ī±, s), u)

_constant_index(::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} = ntuple(i -> 0, Val(N)) # index for the constant term

# order, frequency

order(s::TensorSpace) = map(order, s.spaces)
order(s::TensorSpace, i::Int) = order(s.spaces[i])

frequency(s::TensorSpace) = map(frequency, s.spaces)
frequency(s::TensorSpace, i::Int) = frequency(s.spaces[i])

# promotion

Base.convert(::Type{TensorSpace{T}}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N,T<:Tuple{Vararg{UnivariateSpace,N}}} =
    TensorSpace{T}(ntuple(i -> convert(T.parameters[i], space.spaces[i]), Val(N)))
Base.promote_rule(::Type{TensorSpace{T}}, ::Type{TensorSpace{S}}) where {N,T<:Tuple{Vararg{UnivariateSpace,N}},S<:Tuple{Vararg{UnivariateSpace,N}}} =
    TensorSpace{Tuple{map(promote_type, T.parameters, S.parameters)...}}

# show

Base.show(io::IO, s::TensorSpace) = print(io, pretty_string(s))
pretty_string(s::TensorSpace) =
    mapreduce(sįµ¢ -> string(" āØ ", pretty_string(sįµ¢)), *, Base.tail(s.spaces); init = pretty_string(s.spaces[1]))
pretty_string(s::TensorSpace{Tuple{}}) = string(Tuple{}())





##

abstract type CartesianSpace <: VectorSpace end

startindex(::CartesianSpace) = 1
endindex(s::CartesianSpace) = dimension(s)
allindices(s::CartesianSpace) = Base.OneTo(endindex(s))

isindexof(i::Int, s::CartesianSpace) = (1 ā¤ i) & (i ā¤ dimension(s))
isindexof(u::AbstractRange, s::CartesianSpace) = (1 ā¤ first(u)) & (last(u) ā¤ dimension(s))
isindexof(::Colon, ::CartesianSpace) = true
isindexof(u::AbstractVector{Int}, s::CartesianSpace) = all(i -> isindexof(i, s), u)

_findindex(i, ::CartesianSpace) = i

"""
    CartesianPowerSpace{T<:VectorSpace} <: CartesianSpace

Cartesian space resulting from `dim` cartesian products of a [`VectorSpace`](@ref).

Fields:
- `space :: T`
- `dim :: Int`
"""
struct CartesianPowerSpace{T<:VectorSpace} <: CartesianSpace
    space :: T
    dim :: Int
    function CartesianPowerSpace{T}(space::T, dim::Int) where {T<:VectorSpace}
        dim < 1 && return throw(DomainError(dim, "CartesianPowerSpace is only defined for strictly positive dimensions"))
        return new{T}(space, dim)
    end
end

CartesianPowerSpace(space::T, dim::Int) where {T<:VectorSpace} =
    CartesianPowerSpace{T}(space, dim)

spaces(s::CartesianPowerSpace) = fill(s.space, s.dim)

nb_cartesian_product(s::CartesianPowerSpace) = s.dim

Base.:^(s::VectorSpace, dim::Int) = CartesianPowerSpace(s, dim)

Base.@propagate_inbounds Base.getindex(s::CartesianPowerSpace, ::Colon) = s
Base.@propagate_inbounds function Base.getindex(s::CartesianPowerSpace, i::Int)
    @boundscheck((1 ā¤ i) & (i ā¤ s.dim) || throw(BoundsError(s, i)))
    return s.space
end
Base.@propagate_inbounds function Base.getindex(s::CartesianPowerSpace, u::AbstractRange)
    @boundscheck((1 ā¤ first(u)) & (last(u) ā¤ s.dim) || throw(BoundsError(s, u)))
    return CartesianPowerSpace(s.space, length(u))
end
Base.@propagate_inbounds function Base.getindex(s::CartesianPowerSpace, u::AbstractVector{Int})
    @boundscheck(all(i -> (1 ā¤ i) & (i ā¤ s.dim), u) || throw(BoundsError(s, u)))
    return CartesianPowerSpace(s.space, length(u))
end

#

Base.:(==)(sā::CartesianPowerSpace, sā::CartesianPowerSpace) =
    (sā.dim == sā.dim) & (sā.space == sā.space)
Base.issubset(sā::CartesianPowerSpace, sā::CartesianPowerSpace) =
    (sā.dim == sā.dim) & issubset(sā.space, sā.space)
function Base.intersect(sā::CartesianPowerSpace, sā::CartesianPowerSpace)
    sā.dim == sā.dim || return throw(DimensionMismatch)
    return CartesianPowerSpace(intersect(sā.space, sā.space), sā.dim)
end
function Base.union(sā::CartesianPowerSpace, sā::CartesianPowerSpace)
    sā.dim == sā.dim || return throw(DimensionMismatch)
    return CartesianPowerSpace(union(sā.space, sā.space), sā.dim)
end

dimension(s::CartesianPowerSpace) = dimension(s.space)*s.dim
dimensions(s::CartesianPowerSpace) = fill(dimension(s.space), s.dim)
function dimensions(s::CartesianPowerSpace, i::Int)
    (1 ā¤ i) & (i ā¤ s.dim) || return throw(BoundsError(s, i))
    return dimension(s.space)
end

# order, frequency

order(s::CartesianPowerSpace) = fill(order(s.space), s.dim)
function order(s::CartesianPowerSpace, i::Int)
    (1 ā¤ i) & (i ā¤ s.dim) || return throw(BoundsError(s, i))
    return order(s.space)
end

frequency(s::CartesianPowerSpace) = fill(frequency(s.space), s.dim)
function frequency(s::CartesianPowerSpace, i::Int)
    (1 ā¤ i) & (i ā¤ s.dim) || return throw(BoundsError(s, i))
    return frequency(s.space)
end

# promotion

Base.convert(::Type{CartesianPowerSpace{T}}, s::CartesianPowerSpace) where {T<:VectorSpace} =
    CartesianPowerSpace{T}(convert(T, s.space), s.dim)
Base.promote_rule(::Type{CartesianPowerSpace{T}}, ::Type{CartesianPowerSpace{S}}) where {T<:VectorSpace,S<:VectorSpace} =
    CartesianPowerSpace{promote_type(T, S)}

# show

Base.show(io::IO, s::CartesianPowerSpace) = print(io, pretty_string(s))
pretty_string(s::CartesianPowerSpace) = powcartesian_pretty_string(s.space) * superscriptify(s.dim)
powcartesian_pretty_string(s::VectorSpace) = pretty_string(s)
powcartesian_pretty_string(s::TensorSpace) = "(" * pretty_string(s) * ")"
powcartesian_pretty_string(s::CartesianSpace) = "(" * pretty_string(s) * ")"

"""
    CartesianProductSpace{T<:NTuple{N,VectorSpace} where {N}} <: CartesianSpace

Cartesian space resulting from `N` cartesian products of some [`VectorSpace`](@ref).

Fields:
- `spaces :: T`
"""
struct CartesianProductSpace{T<:NTuple{N,VectorSpace} where {N}} <: CartesianSpace
    spaces :: T
end

spaces(s::CartesianProductSpace) = s.spaces

nb_cartesian_product(s::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} = N

LinearAlgebra.:Ć(sā::VectorSpace, sā::VectorSpace) = CartesianProductSpace((sā, sā))
LinearAlgebra.:Ć(sā::CartesianProductSpace, sā::CartesianProductSpace) = CartesianProductSpace((sā.spaces..., sā.spaces...))
LinearAlgebra.:Ć(sā::CartesianProductSpace, sā::VectorSpace) = CartesianProductSpace((sā.spaces..., sā))
LinearAlgebra.:Ć(sā::VectorSpace, sā::CartesianProductSpace) = CartesianProductSpace((sā, sā.spaces...))

Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, c::Colon) = CartesianProductSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, u::AbstractRange) = CartesianProductSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, u::AbstractVector{Int}) = CartesianProductSpace(getindex(s.spaces, u))

Base.front(s::CartesianProductSpace) = CartesianProductSpace(Base.front(s.spaces))
Base.tail(s::CartesianProductSpace) = CartesianProductSpace(Base.tail(s.spaces))

#

Base.:(==)(sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}, sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(sā.spaces, sā.spaces))
Base.issubset(sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}, sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(sā.spaces, sā.spaces))
Base.intersect(sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}, sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProductSpace(map(intersect, sā.space, sā.space))
Base.union(sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}, sā::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProductSpace(map(union, sā.space, sā.space))

dimension(s::CartesianProductSpace) = mapreduce(dimension, +, s.spaces)
dimensions(s::CartesianProductSpace) = map(dimension, s.spaces)
dimensions(s::CartesianProductSpace, i::Int) = dimension(s.spaces[i])

# order, frequency

order(s::CartesianProductSpace) = map(order, s.spaces)
order(s::CartesianProductSpace, i::Int) = order(s.spaces[i])

frequency(s::CartesianProductSpace) = map(frequency, s.spaces)
frequency(s::CartesianProductSpace, i::Int) = frequency(s.spaces[i])

# promotion

Base.convert(::Type{CartesianProductSpace{T}}, space::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N,T<:Tuple{Vararg{VectorSpace,N}}} =
    CartesianProductSpace{T}(ntuple(i -> convert(T.parameters[i], space.spaces[i]), N))
Base.promote_rule(::Type{CartesianProductSpace{T}}, ::Type{CartesianProductSpace{S}}) where {N,T<:Tuple{Vararg{VectorSpace,N}},S<:Tuple{Vararg{VectorSpace,N}}} =
    CartesianProductSpace{Tuple{map(promote_type, T.parameters, S.parameters)...}}

# show

Base.show(io::IO, space::CartesianProductSpace) = print(io, pretty_string(space))
pretty_string(space::CartesianProductSpace) =
    mapreduce(sįµ¢ -> " Ć " * prodcartesian_pretty_string(sįµ¢), *, Base.tail(space.spaces); init = prodcartesian_pretty_string(space.spaces[1]))
pretty_string(space::CartesianProductSpace{Tuple{}}) = string(Tuple{}())
prodcartesian_pretty_string(s::VectorSpace) = pretty_string(s)
prodcartesian_pretty_string(s::TensorSpace) = "(" * pretty_string(s) * ")"
prodcartesian_pretty_string(s::CartesianProductSpace) = "(" * pretty_string(s) * ")"
