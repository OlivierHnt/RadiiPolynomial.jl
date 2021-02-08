"""
    VectorSpace

Abstract type for all vector spaces.
"""
abstract type VectorSpace end





##

abstract type CartesianSpace <: VectorSpace end

startindex(::CartesianSpace) = 1
endindex(s::CartesianSpace) = dimension(s)
allindices(s::CartesianSpace) = Base.OneTo(endindex(s))

isindexof(i::Int, s::CartesianSpace) = (1 ≤ i) & (i ≤ nb_cartesian_product(s))
isindexof(u::AbstractRange, s::CartesianSpace) = (1 ≤ first(u)) & (last(u) ≤ nb_cartesian_product(s))
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
end

spaces(s::CartesianPowerSpace) = fill(s.space, s.dim)

nb_cartesian_product(s::CartesianPowerSpace) = s.dim

Base.:^(s::VectorSpace, dim::Int) = cartesian_power_space(s, dim)

function cartesian_power_space(s::VectorSpace, dim::Int)
    dim < 1 && return throw(DomainError(dim, "^ is only defined for strictly positive integers"))
    return CartesianPowerSpace(s, dim) # _cartesian_power_space(s, dim)
end
# _cartesian_power_space(s, dim) = CartesianPowerSpace(s, dim)
# _cartesian_power_space(s::CartesianPowerSpace, dim) = CartesianPowerSpace(s.space, s.dim + dim)

Base.@propagate_inbounds Base.getindex(s::CartesianPowerSpace, ::Colon) = s
Base.@propagate_inbounds function Base.getindex(s::CartesianPowerSpace, i::Int)
    @boundscheck((1 ≤ i) & (i ≤ s.dim) || throw(BoundsError(s, i)))
    return s.space
end
Base.@propagate_inbounds function Base.getindex(s::CartesianPowerSpace, u::AbstractRange)
    @boundscheck((1 ≤ first(u)) & (last(u) ≤ s.dim) || throw(BoundsError(s, u)))
    return CartesianPowerSpace(s.space, length(u))
end
Base.@propagate_inbounds function Base.getindex(s::CartesianPowerSpace, u::AbstractVector{Int})
    @boundscheck(all(i -> (1 ≤ i) & (i ≤ s.dim), u) || throw(BoundsError(s, u)))
    return CartesianPowerSpace(s.space, length(u))
end

#

Base.:(==)(s₁::CartesianPowerSpace, s₂::CartesianPowerSpace) =
    (s₁.dim == s₂.dim) & (s₁.space == s₂.space)
Base.issubset(s₁::CartesianPowerSpace, s₂::CartesianPowerSpace) =
    (s₁.dim == s₂.dim) & issubset(s₁.space, s₂.space)
function Base.intersect(s₁::CartesianPowerSpace, s₂::CartesianPowerSpace)
    s₁.dim == s₂.dim || return throw(DimensionMismatch)
    return CartesianPowerSpace(intersect(s₁.space, s₂.space), s₁.dim)
end
function Base.union(s₁::CartesianPowerSpace, s₂::CartesianPowerSpace)
    s₁.dim == s₂.dim || return throw(DimensionMismatch)
    return CartesianPowerSpace(union(s₁.space, s₂.space), s₁.dim)
end

dimension(s::CartesianPowerSpace) = dimension(s.space)*s.dim
dimensions(s::CartesianPowerSpace) = fill(dimension(s.space), s.dim)
function dimensions(s::CartesianPowerSpace, i::Int)
    (1 ≤ i) & (i ≤ s.dim) || return throw(BoundsError(s, i))
    return dimension(s.space)
end

# order, frequency

order(s::CartesianPowerSpace) = fill(order(s.space), s.dim)
function order(s::CartesianPowerSpace, i::Int)
    (1 ≤ i) & (i ≤ s.dim) || return throw(BoundsError(s, i))
    return order(s.space)
end

frequency(s::CartesianPowerSpace) = fill(frequency(s.space), s.dim)
function frequency(s::CartesianPowerSpace, i::Int)
    (1 ≤ i) & (i ≤ s.dim) || return throw(BoundsError(s, i))
    return frequency(s.space)
end

# promotion

Base.convert(::Type{CartesianPowerSpace{T}}, s::CartesianPowerSpace) where {T<:VectorSpace} =
    CartesianPowerSpace{T}(convert(T, s.space), s.dim)
Base.promote_rule(::Type{CartesianPowerSpace{T}}, ::Type{CartesianPowerSpace{S}}) where {T<:VectorSpace,S<:VectorSpace} =
    CartesianPowerSpace{promote_type(T, S)}

# show

Base.show(io::IO, s::CartesianPowerSpace) = print(io, pretty_string(s))
pretty_string(s::CartesianPowerSpace) = string("(", pretty_string(s.space), ")")*superscriptify(s.dim)

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

LinearAlgebra.:×(s₁::VectorSpace, s₂::VectorSpace) = cartesian_product_space(s₁, s₂)

cartesian_product_space(s₁::VectorSpace, s₂::VectorSpace) = CartesianProductSpace((s₁, s₂))
cartesian_product_space(s₁::CartesianProductSpace, s₂::CartesianProductSpace) = CartesianProductSpace((s₁.spaces..., s₂.spaces...))
cartesian_product_space(s₁::CartesianProductSpace, s₂::VectorSpace) = CartesianProductSpace((s₁.spaces..., s₂))
cartesian_product_space(s₁::VectorSpace, s₂::CartesianProductSpace) = CartesianProductSpace((s₁, s₂.spaces...))

Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, c::Colon) = CartesianProductSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, u::AbstractRange) = CartesianProductSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::CartesianProductSpace, u::AbstractVector{Int}) = CartesianProductSpace(getindex(s.spaces, u))

Base.front(s::CartesianProductSpace) = CartesianProductSpace(Base.front(s.spaces))
Base.tail(s::CartesianProductSpace) = CartesianProductSpace(Base.tail(s.spaces))

#

Base.:(==)(s₁::CartesianProductSpace{<:NTuple{N,VectorSpace}}, s₂::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(s₁.spaces, s₂.spaces))
Base.issubset(s₁::CartesianProductSpace{<:NTuple{N,VectorSpace}}, s₂::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(s₁.spaces, s₂.spaces))
Base.intersect(s₁::CartesianProductSpace{<:NTuple{N,VectorSpace}}, s₂::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProductSpace(map(intersect, s₁.space, s₂.space))
Base.union(s₁::CartesianProductSpace{<:NTuple{N,VectorSpace}}, s₂::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProductSpace(map(union, s₁.space, s₂.space))

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
    mapreduce(sᵢ -> string(" × ", "(", pretty_string(sᵢ), ")"), *, Base.tail(space.spaces); init = "(" * pretty_string(space.spaces[1]) * ")")
pretty_string(space::CartesianProductSpace{Tuple{}}) = string(Tuple{}())





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
pretty_string(::ParameterSpace) = "𝕂"





## sequence spaces

"""
    SequenceSpace <: VectorSpace

Abstract type for sequence spaces.
"""
abstract type SequenceSpace <: VectorSpace end

Base.issubset(s₁::SequenceSpace, s₂::SequenceSpace) = false
Base.:(==)(s₁::SequenceSpace, s₂::SequenceSpace) = false

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

order(s::Taylor) = s.order

Base.:(==)(s₁::Taylor, s₂::Taylor) = s₁.order == s₂.order
Base.issubset(s₁::Taylor, s₂::Taylor) = s₁.order ≤ s₂.order
Base.intersect(s₁::Taylor, s₂::Taylor) = Taylor(min(s₁.order, s₂.order))
Base.union(s₁::Taylor, s₂::Taylor) = Taylor(max(s₁.order, s₂.order))

dimension(s::Taylor) = s.order + 1
startindex(s::Taylor) = 0
endindex(s::Taylor) = s.order
allindices(s::Taylor) = 0:s.order

isindexof(i::Int, s::Taylor) = (0 ≤ i) & (i ≤ s.order)
isindexof(u::AbstractRange, s::Taylor) = (0 ≤ first(u)) & (last(u) ≤ s.order)
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
end

order(s::Fourier) = s.order

frequency(s::Fourier) = s.frequency

Base.:(==)(s₁::Fourier, s₂::Fourier) = (s₁.frequency == s₂.frequency) & (s₁.order == s₂.order)
Base.issubset(s₁::Fourier, s₂::Fourier) = (s₁.frequency == s₂.frequency) & (s₁.order ≤ s₂.order)
function Base.intersect(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(min(s₁.order, s₂.order), convert(R, s₁.frequency))
end
function Base.union(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(max(s₁.order, s₂.order), convert(R, s₁.frequency))
end

dimension(s::Fourier) = 2s.order + 1
startindex(s::Fourier) = -s.order
endindex(s::Fourier) = s.order
allindices(s::Fourier) = -s.order:s.order

isindexof(i::Int, s::Fourier) = (-s.order ≤ i) & (i ≤ s.order)
isindexof(u::AbstractRange, s::Fourier) = (-s.order ≤ first(u)) & (last(u) ≤ s.order)
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
end

order(s::Chebyshev) = s.order

Base.:(==)(s₁::Chebyshev, s₂::Chebyshev) = s₁.order == s₂.order
Base.issubset(s₁::Chebyshev, s₂::Chebyshev) = s₁.order ≤ s₂.order
Base.intersect(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(min(s₁.order, s₂.order))
Base.union(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(max(s₁.order, s₂.order))

dimension(s::Chebyshev) = s.order + 1
startindex(s::Chebyshev) = 0
endindex(s::Chebyshev) = s.order
allindices(s::Chebyshev) = 0:s.order

isindexof(i::Int, s::Chebyshev) = (0 ≤ i) & (i ≤ s.order)
isindexof(u::AbstractRange, s::Chebyshev) = (0 ≤ first(u)) & (last(u) ≤ s.order)
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

⊗(s₁::SequenceSpace, s₂::SequenceSpace) = tensor_space(s₁, s₂)

tensor_space(s₁::UnivariateSpace, s₂::UnivariateSpace) = TensorSpace((s₁, s₂))
tensor_space(s₁::TensorSpace, s₂::TensorSpace) = TensorSpace((s₁.spaces..., s₂.spaces...))
tensor_space(s₁::TensorSpace, s₂::UnivariateSpace) = TensorSpace((s₁.spaces..., s₂))
tensor_space(s₁::UnivariateSpace, s₂::TensorSpace) = TensorSpace((s₁, s₂.spaces...))

Base.@propagate_inbounds Base.getindex(s::TensorSpace, c::Colon) = TensorSpace(getindex(s.spaces, c))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractRange) = TensorSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractVector{Int}) = TensorSpace(getindex(s.spaces, u))

Base.front(s::TensorSpace) = TensorSpace(Base.front(s.spaces))
Base.tail(s::TensorSpace) = TensorSpace(Base.tail(s.spaces))

#

Base.:(==)(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(s -> ==(s[1], s[2]), zip(s₁.spaces, s₂.spaces))
Base.issubset(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(s -> issubset(s[1], s[2]), zip(s₁.spaces, s₂.spaces))
Base.intersect(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(intersect, s₁.spaces, s₂.spaces))
Base.union(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(union, s₁.spaces, s₂.spaces))

dimension(s::TensorSpace) = mapreduce(dimension, *, s.spaces)
dimensions(s::TensorSpace) = map(dimension, s.spaces)
dimensions(s::TensorSpace, i::Int) = dimension(s.spaces[i])
startindex(s::TensorSpace) = map(startindex, s.spaces)
endindex(s::TensorSpace) = map(endindex, s.spaces)
allindices(s::TensorSpace) = Base.Iterators.ProductIterator(map(allindices, s.spaces))

isindexof(α::NTuple{N,Any}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(t -> isindexof(t[1], t[2]), zip(α, s.spaces))
isindexof(u::Base.Iterators.ProductIterator, s::TensorSpace) = isindexof(u.iterators, s)
isindexof(u::AbstractVector{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    all(α -> isindexof(α, s), u)

@generated function _findindex(α::NTuple{N,Int}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N}
    # follows column major convention
    idx = :(_findindex(α[1], s.spaces[1]))
    n = 1
    for i ∈ 2:N
        n = :(dimension(s.spaces[$i-1]) * $n)
        idx = :($n * (_findindex(α[$i], s.spaces[$i]) - 1) + $idx)
    end
    return idx
end
_findindex(u::NTuple{N,Any}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    _findindex(Base.Iterators.ProductIterator(u), s)
function _findindex(u::Base.Iterators.ProductIterator, s::TensorSpace)
    v = Vector{Int}(undef, length(u))
    for (i, uᵢ) ∈ enumerate(u)
        v[i] = _findindex(uᵢ, s)
    end
    return v
end
_findindex(u::AbstractVector{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    map(α -> _findindex(α, s), u)

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
    mapreduce(sᵢ -> string(" ⨂ ", pretty_string(sᵢ)), *, Base.tail(s.spaces); init = pretty_string(s.spaces[1]))
pretty_string(s::TensorSpace{Tuple{}}) = string(Tuple{}())
