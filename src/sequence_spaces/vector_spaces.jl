"""
    VectorSpace

Abstract type for all vector spaces.
"""
abstract type VectorSpace end

Base.:(==)(::VectorSpace, ::VectorSpace) = false
Base.issubset(::VectorSpace, ::VectorSpace) = false
Base.intersect(s₁::VectorSpace, s₂::VectorSpace) = throw(MethodError(intersect, (s₁, s₂)))
Base.union(s₁::VectorSpace, s₂::VectorSpace) = throw(MethodError(union, (s₁, s₂)))

dimension(s::VectorSpace) = length(indices(s))

_checkbounds_indices(::Colon, ::VectorSpace) = true
_checkbounds_indices(α, s::VectorSpace) = __checkbounds_indices(α, s)
_checkbounds_indices(u::AbstractRange, s::VectorSpace) =
    __checkbounds_indices(first(u), s) & __checkbounds_indices(last(u), s)
_checkbounds_indices(u::AbstractVector, s::VectorSpace) =
    all(uᵢ -> __checkbounds_indices(uᵢ, s), u)

__checkbounds_indices(::Colon, ::VectorSpace) = true
__checkbounds_indices(α::Int, s::VectorSpace) = α ∈ indices(s)





# Parameter space

"""
    ParameterSpace <: VectorSpace

Parameter space corresponding to a commutative field.

# Example
```jldoctest
julia> ParameterSpace()
𝕂
```
"""
struct ParameterSpace <: VectorSpace end

Base.:(==)(::ParameterSpace, ::ParameterSpace) = true
Base.issubset(::ParameterSpace, ::ParameterSpace) = true
Base.intersect(::ParameterSpace, ::ParameterSpace) = ParameterSpace()
Base.union(::ParameterSpace, ::ParameterSpace) = ParameterSpace()

dimension(::ParameterSpace) = 1
_firstindex(::ParameterSpace) = 1
_lastindex(::ParameterSpace) = 1
indices(::ParameterSpace) = Base.OneTo(1)

__checkbounds_indices(α::Int, ::ParameterSpace) = isone(α)

_findposition(i, ::ParameterSpace) = i





# Sequence spaces

"""
    SequenceSpace <: VectorSpace

Abstract type for all sequence spaces.
"""
abstract type SequenceSpace <: VectorSpace end

"""
    BaseSpace <: SequenceSpace

Abstract type for all sequence spaces that are not a [`TensorSpace`](@ref) but
can be interlaced to form one.
"""
abstract type BaseSpace <: SequenceSpace end

"""
    TensorSpace{T<:Tuple{Vararg{BaseSpace}}} <: SequenceSpace

Tensor space resulting from the tensor product of some [`BaseSpace`](@ref).

Field:
- `spaces :: T`

Constructors:
- `TensorSpace(::Tuple{Vararg{BaseSpace}})`
- `TensorSpace(spaces::BaseSpace...)`: equivalent to `TensorSpace(spaces)`
- `⊗(s₁::BaseSpace, s₂::BaseSpace)`: equivalent to `TensorSpace((s₁, s₂))`
- `⊗(s₁::TensorSpace, s₂::TensorSpace)`: equivalent to `TensorSpace((s₁.spaces..., s₂.spaces...))`
- `⊗(s₁::TensorSpace, s₂::BaseSpace)`: equivalent to `TensorSpace((s₁.spaces..., s₂))`
- `⊗(s₁::BaseSpace, s₂::TensorSpace)`: equivalent to `TensorSpace((s₁, s₂.spaces...))`

See also: [`⊗`](@ref).

# Examples
```jldoctest
julia> s = TensorSpace(Taylor(1), Fourier(2, 1.0), Chebyshev(3))
Taylor(1) ⊗ Fourier{Float64}(2, 1.0) ⊗ Chebyshev(3)

julia> spaces(s)
(Taylor(1), Fourier{Float64}(2, 1.0), Chebyshev(3))
```
"""
struct TensorSpace{T<:Tuple{Vararg{BaseSpace}}} <: SequenceSpace
    spaces :: T
    TensorSpace{T}(spaces::T) where {T<:Tuple{Vararg{BaseSpace}}} = new{T}(spaces)
    TensorSpace{Tuple{}}(::Tuple{}) = throw(ArgumentError("TensorSpace is only defined for at least one BaseSpace"))
end

TensorSpace(spaces::T) where {T<:Tuple{Vararg{BaseSpace}}} = TensorSpace{T}(spaces)
TensorSpace(spaces::BaseSpace...) = TensorSpace(spaces)

spaces(s::TensorSpace) = s.spaces

nspaces(::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = N

"""
    ⊗(s₁::BaseSpace, s₂::BaseSpace)
    ⊗(s₁::TensorSpace, s₂::TensorSpace)
    ⊗(s₁::TensorSpace, s₂::BaseSpace)
    ⊗(s₁::BaseSpace, s₂::TensorSpace)

Create a [`TensorSpace`](@ref) from the tensor product of some [`SequenceSpace`](@ref).

See also: [`TensorSpace`](@ref).

# Examples
```jldoctest
julia> Taylor(1) ⊗ Fourier(2, 1.0)
Taylor(1) ⊗ Fourier{Float64}(2, 1.0)

julia> Taylor(1) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)
Taylor(1) ⊗ Fourier{Float64}(2, 1.0) ⊗ Chebyshev(3)

julia> Taylor(1) ⊗ (Fourier(2, 1.0) ⊗ Chebyshev(3))
Taylor(1) ⊗ Fourier{Float64}(2, 1.0) ⊗ Chebyshev(3)

julia> (Taylor(1) ⊗ Fourier(2, 1.0)) ⊗ Chebyshev(3)
Taylor(1) ⊗ Fourier{Float64}(2, 1.0) ⊗ Chebyshev(3)
```
"""
⊗(s₁::BaseSpace, s₂::BaseSpace) = TensorSpace((s₁, s₂))
⊗(s₁::TensorSpace, s₂::TensorSpace) = TensorSpace((s₁.spaces..., s₂.spaces...))
⊗(s₁::TensorSpace, s₂::BaseSpace) = TensorSpace((s₁.spaces..., s₂))
⊗(s₁::BaseSpace, s₂::TensorSpace) = TensorSpace((s₁, s₂.spaces...))

Base.@propagate_inbounds Base.getindex(s::TensorSpace, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractRange{Int}) = TensorSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, u::AbstractVector{Int}) = TensorSpace(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::TensorSpace, c::Colon) = TensorSpace(getindex(s.spaces, c))

Base.front(s::TensorSpace) = TensorSpace(Base.front(s.spaces))
Base.tail(s::TensorSpace) = TensorSpace(Base.tail(s.spaces))

#

function Base.:(==)(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    s₁[1] == s₂[1] && return Base.tail(s₁) == Base.tail(s₂)
    return false
end
Base.:(==)(s₁::TensorSpace{<:Tuple{BaseSpace}}, s₂::TensorSpace{<:Tuple{BaseSpace}}) =
    s₁[1] == s₂[1]
function Base.issubset(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    issubset(s₁[1], s₂[1]) && return issubset(Base.tail(s₁), Base.tail(s₂))
    return false
end
Base.issubset(s₁::TensorSpace{<:Tuple{BaseSpace}}, s₂::TensorSpace{<:Tuple{BaseSpace}}) =
    issubset(s₁[1], s₂[1])
Base.intersect(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map(intersect, s₁.spaces, s₂.spaces))
Base.union(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map(union, s₁.spaces, s₂.spaces))

dimension(s::TensorSpace) = mapreduce(dimension, *, s.spaces)
dimension(s::TensorSpace, i::Int) = dimension(s.spaces[i])
dimensions(s::TensorSpace) = map(dimension, s.spaces)
_firstindex(s::TensorSpace) = map(_firstindex, s.spaces)
_lastindex(s::TensorSpace) = map(_lastindex, s.spaces)

"""
    TensorIndices{<:Tuple}

Multidimentional rectangular range of indices for some [`TensorSpace`](@ref).

# Examples
```jldoctest
julia> TensorIndices((0:2, -1:1))
TensorIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}}((0:2, -1:1))

julia> indices(Taylor(2) ⊗ Fourier(1, 1.0))
TensorIndices{Tuple{UnitRange{Int64}, UnitRange{Int64}}}((0:2, -1:1))
```
"""
struct TensorIndices{T<:Tuple}
    indices :: T
end
Base.@propagate_inbounds Base.getindex(a::TensorIndices, i) = getindex(Base.Iterators.ProductIterator(a.indices), i)
Base.length(a::TensorIndices) = length(Base.Iterators.ProductIterator(a.indices))
Base.iterate(a::TensorIndices) = iterate(Base.Iterators.ProductIterator(a.indices))
Base.iterate(a::TensorIndices, state) = iterate(Base.Iterators.ProductIterator(a.indices), state)
Base.issubset(a::TensorIndices, b::TensorIndices) = all(issubset.(a.indices, b.indices))
Base.intersect(a::TensorIndices, b::TensorIndices) = TensorIndices(intersect.(a.indices, b.indices))
Base.union(a::TensorIndices, b::TensorIndices) = TensorIndices(union.(a.indices, b.indices))

indices(s::TensorSpace) = TensorIndices(map(indices, s.spaces))

_checkbounds_indices(α::Tuple, s::TensorSpace) = false
_checkbounds_indices(α::NTuple{N,Any}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds(_checkbounds_indices(α[1], s[1])) & _checkbounds_indices(Base.tail(α), Base.tail(s))
_checkbounds_indices(α::Tuple{Any}, s::TensorSpace{<:Tuple{BaseSpace}}) =
    @inbounds _checkbounds_indices(α[1], s[1])
_checkbounds_indices(u::TensorIndices, s::TensorSpace) = _checkbounds_indices(u.indices, s)
_checkbounds_indices(α::NTuple{N,Colon}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    true
_checkbounds_indices(α::Tuple{Colon}, s::TensorSpace{<:Tuple{BaseSpace}}) = true

_findindex_constant(::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = ntuple(i -> 0, Val(N))

_findposition(α::Tuple{Int}, s::TensorSpace{<:Tuple{BaseSpace}}) =
    @inbounds _findposition(α[1], s.spaces[1])
function _findposition(α::NTuple{N,Int}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    @inbounds idx = _findposition(α[1], s.spaces[1])
    @inbounds n = dimension(s.spaces[1])
    return __findposition(Base.tail(α), Base.tail(s.spaces), idx, n)
end
function __findposition(α, spaces, idx, n)
    @inbounds idx += n * (_findposition(α[1], spaces[1]) - 1)
    @inbounds n *= dimension(spaces[1])
    return __findposition(Base.tail(α), Base.tail(spaces), idx, n)
end
__findposition(α::Tuple{Int}, spaces, idx, n) = @inbounds idx + n * (_findposition(α[1], spaces[1]) - 1)
_findposition(u::NTuple{N,Any}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    _findposition(TensorIndices(map(_colon2indices, u, s.spaces)), s)
_colon2indices(u, s) = u
_colon2indices(::Colon, s) = indices(s)
function _findposition(u::TensorIndices{<:NTuple{N,Any}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    v = Vector{Int}(undef, length(u))
    @inbounds for (i, uᵢ) ∈ enumerate(u)
        v[i] = _findposition(uᵢ, s)
    end
    return v
end
_findposition(u::AbstractVector{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    map(α -> _findposition(α, s), u)
_findposition(::NTuple{N,Colon}, ::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = Colon()
_findposition(c::Colon, ::TensorSpace) = c

# order, frequency

order(s::TensorSpace) = map(order, s.spaces)
order(s::TensorSpace, i::Int) = order(s.spaces[i])

frequency(s::TensorSpace) = map(frequency, s.spaces)
frequency(s::TensorSpace, i::Int) = frequency(s.spaces[i])

# promotion

Base.convert(::Type{T}, s::T) where {T<:TensorSpace} = s
Base.convert(::Type{TensorSpace{T}}, s::TensorSpace) where {T} =
    TensorSpace{T}(convert(T, s.spaces))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:TensorSpace} = T
Base.promote_rule(::Type{TensorSpace{T}}, ::Type{TensorSpace{S}}) where {T,S} =
    TensorSpace{promote_type(T, S)}

#

"""
    Taylor <: BaseSpace

Taylor sequence space whose elements are Taylor sequences of a prescribed order.

Field:
- `order :: Int`

Constructor:
- `Taylor(::Int)`

See also: [`Fourier`](@ref) and [`Chebyshev`](@ref).

# Examples
```jldoctest
julia> s = Taylor(2)
Taylor(2)

julia> order(s)
2
```
"""
struct Taylor <: BaseSpace
    order :: Int
    function Taylor(order::Int)
        order < 0 && return throw(DomainError(order, "Taylor is only defined for positive orders"))
        return new(order)
    end
end

order(s::Taylor) = s.order

Base.:(==)(s₁::Taylor, s₂::Taylor) = s₁.order == s₂.order
Base.issubset(s₁::Taylor, s₂::Taylor) = s₁.order ≤ s₂.order
Base.intersect(s₁::Taylor, s₂::Taylor) = Taylor(min(s₁.order, s₂.order))
Base.union(s₁::Taylor, s₂::Taylor) = Taylor(max(s₁.order, s₂.order))

dimension(s::Taylor) = s.order + 1
_firstindex(::Taylor) = 0
_lastindex(s::Taylor) = s.order
indices(s::Taylor) = 0:s.order

__checkbounds_indices(α::Int, s::Taylor) = 0 ≤ α ≤ order(s)

_findindex_constant(::Taylor) = 0

_findposition(i::Int, ::Taylor) = i + 1
_findposition(u::AbstractRange{Int}, ::Taylor) = u .+ 1
_findposition(u::AbstractVector{Int}, s::Taylor) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::Taylor) = c

#

"""
    Fourier{T<:Real} <: BaseSpace

Fourier sequence space whose elements are Fourier sequences of a prescribed order and frequency.

Fields:
- `order :: Int`
- `frequency :: T`

Constructor:
- `Fourier(::Int, ::Real)`

See also: [`Taylor`](@ref) and [`Chebyshev`](@ref).

# Examples
```jldoctest
julia> s = Fourier(2, 1.0)
Fourier{Float64}(2, 1.0)

julia> order(s)
2

julia> frequency(s)
1.0
```
"""
struct Fourier{T<:Real} <: BaseSpace
    order :: Int
    frequency :: T
    function Fourier{T}(order::Int, frequency::T) where {T<:Real}
        (order < 0) | !(frequency ≥ 0) && return throw(DomainError(order, "Fourier is only defined for positive orders and frequencies"))
        return new{T}(order, frequency)
    end
end

Fourier(order::Int, frequency::T) where {T<:Real} = Fourier{T}(order, frequency)

order(s::Fourier) = s.order

frequency(s::Fourier) = s.frequency

Base.:(==)(s₁::Fourier, s₂::Fourier) = (s₁.frequency == s₂.frequency) & (s₁.order == s₂.order)
Base.issubset(s₁::Fourier, s₂::Fourier) = (s₁.frequency == s₂.frequency) & (s₁.order ≤ s₂.order)
function Base.intersect(s₁::Fourier{T}, s₂::Fourier{S}) where {T<:Real,S<:Real}
    s₁.frequency == s₂.frequency || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $(s₁.frequency), s₂ has frequency $(s₂.frequency)"))
    R = promote_type(T, S)
    return Fourier(min(s₁.order, s₂.order), convert(R, s₁.frequency))
end
function Base.union(s₁::Fourier{T}, s₂::Fourier{S}) where {T<:Real,S<:Real}
    s₁.frequency == s₂.frequency || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $(s₁.frequency), s₂ has frequency $(s₂.frequency)"))
    R = promote_type(T, S)
    return Fourier(max(s₁.order, s₂.order), convert(R, s₁.frequency))
end

dimension(s::Fourier) = 2s.order + 1
_firstindex(s::Fourier) = -s.order
_lastindex(s::Fourier) = s.order
indices(s::Fourier) = -s.order:s.order

__checkbounds_indices(α::Int, s::Fourier) = -order(s) ≤ α ≤ order(s)

_findindex_constant(::Fourier) = 0

_findposition(i::Int, s::Fourier) = i + s.order + 1
_findposition(u::AbstractRange{Int}, s::Fourier) = u .+ (s.order + 1)
_findposition(u::AbstractVector{Int}, s::Fourier) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::Fourier) = c

# promotion

Base.convert(::Type{T}, s::T) where {T<:Fourier} = s
Base.convert(::Type{Fourier{T}}, s::Fourier) where {T<:Real} =
    Fourier{T}(s.order, convert(T, s.frequency))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:Fourier} = T
Base.promote_rule(::Type{Fourier{T}}, ::Type{Fourier{S}}) where {T<:Real,S<:Real} =
    Fourier{promote_type(T, S)}

#

"""
    Chebyshev <: BaseSpace

Chebyshev sequence space whose elements are Chebyshev sequences of a prescribed order.

Field:
- `order :: Int`

Constructor:
- `Chebyshev(::Int)`

See also: [`Taylor`](@ref) and [`Fourier`](@ref).

# Examples
```jldoctest
julia> s = Chebyshev(2)
Chebyshev(2)

julia> order(s)
2
```
"""
struct Chebyshev <: BaseSpace
    order :: Int
    function Chebyshev(order::Int)
        order < 0 && return throw(DomainError(order, "Chebyshev is only defined for positive orders"))
        return new(order)
    end
end

order(s::Chebyshev) = s.order

Base.:(==)(s₁::Chebyshev, s₂::Chebyshev) = s₁.order == s₂.order
Base.issubset(s₁::Chebyshev, s₂::Chebyshev) = s₁.order ≤ s₂.order
Base.intersect(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(min(s₁.order, s₂.order))
Base.union(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(max(s₁.order, s₂.order))

dimension(s::Chebyshev) = s.order + 1
_firstindex(::Chebyshev) = 0
_lastindex(s::Chebyshev) = s.order
indices(s::Chebyshev) = 0:s.order

__checkbounds_indices(α::Int, s::Chebyshev) = 0 ≤ α ≤ order(s)

_findindex_constant(::Chebyshev) = 0

_findposition(i::Int, ::Chebyshev) = i + 1
_findposition(u::AbstractRange{Int}, ::Chebyshev) = u .+ 1
_findposition(u::AbstractVector{Int}, s::Chebyshev) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::Chebyshev) = c





# Cartesian spaces

"""
    CartesianSpace <: VectorSpace

Abstract type for all cartesian spaces.
"""
abstract type CartesianSpace <: VectorSpace end

_firstindex(::CartesianSpace) = 1
_lastindex(s::CartesianSpace) = dimension(s)
indices(s::CartesianSpace) = Base.OneTo(dimension(s))

_findposition(i, ::CartesianSpace) = i

_component_findposition(u::AbstractRange{Int}, s::CartesianSpace) =
    mapreduce(i -> _component_findposition(i, s), union, u)
_component_findposition(u::AbstractVector{Int}, s::CartesianSpace) =
    mapreduce(i -> _component_findposition(i, s), union, u)
_component_findposition(c::Colon, s::CartesianSpace) = c

"""
    CartesianPower{T<:VectorSpace} <: CartesianSpace

Cartesian space resulting from the cartesian product of the same [`VectorSpace`](@ref).

Fields:
- `space :: T`
- `n :: Int`

Constructors:
- `CartesianPower(::VectorSpace, ::Int)`
- `^(::VectorSpace, ::Int)`: equivalent to `CartesianPower(::VectorSpace, ::Int)`

See also: [`^(::VectorSpace, ::Int)`](@ref), [`CartesianProduct`](@ref) and [`×`](@ref).

# Examples
```jldoctest
julia> s = CartesianPower(Taylor(1), 3)
Taylor(1)³

julia> space(s)
Taylor(1)

julia> nspaces(s)
3
```
"""
struct CartesianPower{T<:VectorSpace} <: CartesianSpace
    space :: T
    n :: Int
    function CartesianPower{T}(space::T, n::Int) where {T<:VectorSpace}
        n < 0 && return throw(DomainError(n, "CartesianPower is only defined for positive integers"))
        return new{T}(space, n)
    end
end

CartesianPower(space::T, n::Int) where {T<:VectorSpace} =
    CartesianPower{T}(space, n)

space(s::CartesianPower) = s.space

spaces(s::CartesianPower) = fill(s.space, s.n)

nspaces(s::CartesianPower) = s.n

"""
    ^(s::VectorSpace, n::Int)

Create a [`CartesianPower`](@ref) from `n` cartesian product(s) of `s`.

See also: [`CartesianPower`](@ref), [`CartesianProduct`](@ref), [`×`](@ref).

# Examples
```jldoctest
julia> Taylor(1)^3
Taylor(1)³

julia> (Taylor(1)^3)^2
(Taylor(1)³)²
```
"""
Base.:^(s::VectorSpace, n::Int) = CartesianPower(s, n)

Base.@propagate_inbounds function Base.getindex(s::CartesianPower, i::Int)
    @boundscheck((1 ≤ i) & (i ≤ s.n) || throw(BoundsError(s, i)))
    return s.space
end
Base.@propagate_inbounds function Base.getindex(s::CartesianPower, u::AbstractRange{Int})
    @boundscheck((1 ≤ first(u)) & (last(u) ≤ s.n) || throw(BoundsError(s, u)))
    return CartesianPower(s.space, length(u))
end
Base.@propagate_inbounds function Base.getindex(s::CartesianPower, u::AbstractVector{Int})
    @boundscheck(all(i -> (1 ≤ i) & (i ≤ s.n), u) || throw(BoundsError(s, u)))
    return CartesianPower(s.space, length(u))
end
Base.@propagate_inbounds Base.getindex(s::CartesianPower, ::Colon) = s

#

Base.:(==)(s₁::CartesianPower, s₂::CartesianPower) =
    (s₁.n == s₂.n) & (s₁.space == s₂.space)
Base.issubset(s₁::CartesianPower, s₂::CartesianPower) =
    (s₁.n == s₂.n) & issubset(s₁.space, s₂.space)
function Base.intersect(s₁::CartesianPower, s₂::CartesianPower)
    s₁.n == s₂.n || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $(s₁.n) cartesian product(s), s₂ has $(s₂.n) cartesian product(s)"))
    return CartesianPower(intersect(s₁.space, s₂.space), s₁.n)
end
function Base.union(s₁::CartesianPower, s₂::CartesianPower)
    s₁.n == s₂.n || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $(s₁.n) cartesian product(s), s₂ has $(s₂.n) cartesian product(s)"))
    return CartesianPower(union(s₁.space, s₂.space), s₁.n)
end

dimension(s::CartesianPower) = dimension(s.space)*s.n
function dimension(s::CartesianPower, i::Int)
    (1 ≤ i) & (i ≤ s.n) || return throw(BoundsError(s, i))
    return dimension(s.space)
end
dimensions(s::CartesianPower) = fill(dimension(s.space), s.n)

# order, frequency

order(s::CartesianPower) = fill(order(s.space), s.n)
function order(s::CartesianPower, i::Int)
    (1 ≤ i) & (i ≤ s.n) || return throw(BoundsError(s, i))
    return order(s.space)
end

frequency(s::CartesianPower) = fill(frequency(s.space), s.n)
function frequency(s::CartesianPower, i::Int)
    (1 ≤ i) & (i ≤ s.n) || return throw(BoundsError(s, i))
    return frequency(s.space)
end

#

function _component_findposition(i::Int, s::CartesianPower)
    dim = dimension(s.space)
    x = (i-1)*dim
    return 1+x:dim+x
end
function _component_findposition(u::UnitRange{Int}, s::CartesianPower)
    dim = dimension(s.space)
    x = (first(u)-1)*dim
    return 1+x:dim*length(u)+x
end

# promotion

Base.convert(::Type{T}, s::T) where {T<:CartesianPower} = s
Base.convert(::Type{CartesianPower{T}}, s::CartesianPower) where {T} =
    CartesianPower{T}(convert(T, s.space), s.n)

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:CartesianPower} = T
Base.promote_rule(::Type{CartesianPower{T}}, ::Type{CartesianPower{S}}) where {T,S} =
    CartesianPower{promote_type(T, S)}

"""
    CartesianProduct{T<:Tuple{Vararg{VectorSpace}}} <: CartesianSpace

Cartesian space resulting from the cartesian product of some [`VectorSpace`](@ref).

Field:
- `spaces :: T`

Constructors:
- `CartesianProduct(::Tuple{Vararg{VectorSpace}})`
- `CartesianProduct(spaces::VectorSpace...)`: equivalent to `CartesianProduct(spaces)`
- `×(s₁::VectorSpace, s₂::VectorSpace)`: equivalent to `CartesianProduct((s₁, s₂))`
- `×(s₁::CartesianProduct, s₂::CartesianProduct)`: equivalent to `CartesianProduct((s₁.spaces..., s₂.spaces...))`
- `×(s₁::CartesianProduct, s₂::VectorSpace)`: equivalent to `CartesianProduct((s₁.spaces..., s₂))`
- `×(s₁::VectorSpace, s₂::CartesianProduct)`: equivalent to `CartesianProduct((s₁, s₂.spaces...))`

See also: [`×`](@ref), [`CartesianPower`](@ref), [`^(::VectorSpace, ::Int)`](@ref).

# Examples
```jldoctest
julia> s = CartesianProduct(Taylor(1), Fourier(2, 1.0), Chebyshev(3))
Taylor(1) × Fourier{Float64}(2, 1.0) × Chebyshev(3)

julia> spaces(s)
(Taylor(1), Fourier{Float64}(2, 1.0), Chebyshev(3))

julia> nspaces(s)
3
```
"""
struct CartesianProduct{T<:Tuple{Vararg{VectorSpace}}} <: CartesianSpace
    spaces :: T
    CartesianProduct{T}(spaces::T) where {N,T<:NTuple{N,VectorSpace}} = new{T}(spaces)
    CartesianProduct{Tuple{}}(::Tuple{}) = throw(ArgumentError("CartesianProduct is only defined for at least one VectorSpace"))
end

CartesianProduct(spaces::T) where {T<:Tuple{Vararg{VectorSpace}}} = CartesianProduct{T}(spaces)
CartesianProduct(spaces::VectorSpace...) = CartesianProduct(spaces)

spaces(s::CartesianProduct) = s.spaces

nspaces(::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N} = N

"""
    ×(::VectorSpace, ::VectorSpace)
    ×(::CartesianProduct, ::CartesianProduct)
    ×(::CartesianProduct, ::VectorSpace)
    ×(::VectorSpace, ::CartesianProduct)

Create a [`CartesianProduct`](@ref) from the cartesian product of some [`VectorSpace`](@ref).

See also: [`CartesianProduct`](@ref), [`CartesianPower`](@ref) and [`^(::VectorSpace, ::Int)`](@ref).

# Examples
```jldoctest
julia> Taylor(1) × Fourier(2, 1.0)
Taylor(1) × Fourier{Float64}(2, 1.0)

julia> Taylor(1) × Fourier(2, 1.0) × Chebyshev(3)
Taylor(1) × Fourier{Float64}(2, 1.0) × Chebyshev(3)

julia> (Taylor(1) × Fourier(2, 1.0)) × Chebyshev(3)
Taylor(1) × Fourier{Float64}(2, 1.0) × Chebyshev(3)

julia> Taylor(1) × (Fourier(2, 1.0) × Chebyshev(3))
Taylor(1) × Fourier{Float64}(2, 1.0) × Chebyshev(3)

julia> ParameterSpace()^2 × ((Taylor(1) ⊗ Fourier(2, 1.0)) × Chebyshev(3))^3
𝕂² × ((Taylor(1) ⊗ Fourier{Float64}(2, 1.0)) × Chebyshev(3))³
```
"""
×(s₁::VectorSpace, s₂::VectorSpace) = CartesianProduct((s₁, s₂))
×(s₁::CartesianProduct, s₂::CartesianProduct) = CartesianProduct((s₁.spaces..., s₂.spaces...))
×(s₁::CartesianProduct, s₂::VectorSpace) = CartesianProduct((s₁.spaces..., s₂))
×(s₁::VectorSpace, s₂::CartesianProduct) = CartesianProduct((s₁, s₂.spaces...))

Base.@propagate_inbounds Base.getindex(s::CartesianProduct, i::Int) = getindex(s.spaces, i)
Base.@propagate_inbounds Base.getindex(s::CartesianProduct, u::AbstractRange{Int}) = CartesianProduct(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::CartesianProduct, u::AbstractVector{Int}) = CartesianProduct(getindex(s.spaces, u))
Base.@propagate_inbounds Base.getindex(s::CartesianProduct, c::Colon) = CartesianProduct(getindex(s.spaces, c))

Base.front(s::CartesianProduct) = CartesianProduct(Base.front(s.spaces))
Base.tail(s::CartesianProduct) = CartesianProduct(Base.tail(s.spaces))

#

function Base.:(==)(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N}
    s₁[1] == s₂[1] && return Base.tail(s₁) == Base.tail(s₂)
    return false
end
Base.:(==)(s₁::CartesianProduct{<:Tuple{VectorSpace}}, s₂::CartesianProduct{<:Tuple{VectorSpace}}) =
    s₁[1] == s₂[1]
function Base.issubset(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N}
    issubset(s₁[1], s₂[1]) && return issubset(Base.tail(s₁), Base.tail(s₂))
    return false
end
Base.issubset(s₁::CartesianProduct{<:Tuple{VectorSpace}}, s₂::CartesianProduct{<:Tuple{VectorSpace}}) =
    issubset(s₁[1], s₂[1])
Base.intersect(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProduct(map(intersect, s₁.spaces, s₂.spaces))
Base.union(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProduct(map(union, s₁.spaces, s₂.spaces))

dimension(s::CartesianProduct{<:Tuple{VectorSpace,Vararg{VectorSpace}}}) = @inbounds dimension(s.spaces[1]) + dimension(Base.tail(s))
dimension(s::CartesianProduct{<:Tuple{VectorSpace}}) = @inbounds dimension(s.spaces[1])
dimension(s::CartesianProduct{<:Tuple{CartesianProduct}}) = @inbounds dimension(s.spaces[1])
dimension(s::CartesianProduct, i::Int) = dimension(s.spaces[i])
dimensions(s::CartesianProduct) = map(dimension, s.spaces)

# order, frequency

order(s::CartesianProduct) = map(order, s.spaces)
order(s::CartesianProduct, i::Int) = order(s.spaces[i])

frequency(s::CartesianProduct) = map(frequency, s.spaces)
frequency(s::CartesianProduct, i::Int) = frequency(s.spaces[i])

#

function _component_findposition(i::Int, s::CartesianProduct)
    dims = dimensions(s)
    dim = dims[i]
    x = mapreduce(j -> dims[j], +, 1:i-1; init=0)
    return 1+x:dim+x
end
function _component_findposition(u::UnitRange{Int}, s::CartesianProduct)
    dims = dimensions(s)
    dim = mapreduce(j -> dims[j], +, u)
    x = mapreduce(j -> dims[j], +, 1:first(u)-1; init=0)
    return 1+x:dim+x
end

# promotion

Base.convert(::Type{T}, s::T) where {T<:CartesianProduct} = s
Base.convert(::Type{CartesianProduct{T}}, s::CartesianProduct) where {T} =
    CartesianProduct{T}(convert(T, s.spaces))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:CartesianProduct} = T
Base.promote_rule(::Type{CartesianProduct{T}}, ::Type{CartesianProduct{S}}) where {T,S} =
    CartesianProduct{promote_type(T, S)}

#

_deep_nspaces(::VectorSpace) = 1
_deep_nspaces(s::CartesianPower) = s.n * _deep_nspaces(s.space)
_deep_nspaces(s::CartesianProduct) = sum(_deep_nspaces, s.spaces)

#

_iscompatible(::VectorSpace, ::VectorSpace) = false
_iscompatible(::ParameterSpace, ::ParameterSpace) = true
_iscompatible(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds _iscompatible(s₁[1], s₂[1]) & _iscompatible(Base.tail(s₁), Base.tail(s₂))
_iscompatible(s₁::TensorSpace{<:Tuple{BaseSpace}}, s₂::TensorSpace{<:Tuple{BaseSpace}}) =
    @inbounds _iscompatible(s₁[1], s₂[1])
_iscompatible(::Taylor, ::Taylor) = true
_iscompatible(s₁::Fourier, s₂::Fourier) = frequency(s₁) == frequency(s₂)
_iscompatible(::Chebyshev, ::Chebyshev) = true
_iscompatible(s₁::CartesianPower, s₂::CartesianPower) =
    (nspaces(s₁) == nspaces(s₂)) & _iscompatible(space(s₁), space(s₂))
_iscompatible(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N} =
    @inbounds _iscompatible(s₁[1], s₂[1]) & _iscompatible(Base.tail(s₁), Base.tail(s₂))
_iscompatible(s₁::CartesianProduct{<:Tuple{VectorSpace}}, s₂::CartesianProduct{<:Tuple{VectorSpace}}) =
    @inbounds _iscompatible(s₁[1], s₂[1])
_iscompatible(s₁::CartesianPower, s₂::CartesianProduct) =
    (nspaces(s₁) == nspaces(s₂)) & all(s₂ᵢ -> _iscompatible(space(s₁), s₂ᵢ), spaces(s₂))
_iscompatible(s₁::CartesianProduct, s₂::CartesianPower) =
    (nspaces(s₁) == nspaces(s₂)) & all(s₁ᵢ -> _iscompatible(s₁ᵢ, space(s₂)), spaces(s₁))

# show

Base.show(io::IO, ::MIME"text/plain", s::VectorSpace) = print(io, string_space(s))

string_space(s::VectorSpace) = string(s)

string_space(::ParameterSpace) = "𝕂"

string_space(s::TensorSpace) = string_space(s[1]) * " ⊗ " * string_space(Base.tail(s))
string_space(s::TensorSpace{<:NTuple{2,BaseSpace}}) = string_space(s[1]) * " ⊗ " * string_space(s[2])
string_space(s::TensorSpace{<:Tuple{BaseSpace}}) = "TensorSpace(" * string_space(s[1]) * ")"

string_space(s::Taylor) = "Taylor(" * string(order(s)) * ")"
string_space(s::Fourier) = string(typeof(s)) * "(" * string(order(s)) * ", " * string(frequency(s)) * ")"
string_space(s::Chebyshev) = "Chebyshev(" * string(order(s)) * ")"

string_space(s::CartesianPower) = string_space(space(s)) * _supscript(nspaces(s))
string_space(s::CartesianPower{<:TensorSpace}) = "(" * string_space(space(s)) * ")" * _supscript(nspaces(s))
string_space(s::CartesianPower{<:CartesianSpace}) = "(" * string_space(space(s)) * ")" * _supscript(nspaces(s))

string_space(s::CartesianProduct) = cartesian_string_space(s[1]) * " × " * string_space(Base.tail(s))
string_space(s::CartesianProduct{<:NTuple{2,VectorSpace}}) = cartesian_string_space(s[1]) * " × " * cartesian_string_space(s[2])
string_space(s::CartesianProduct{<:Tuple{VectorSpace}}) = "CartesianProduct(" * string_space(s[1]) * ")"
cartesian_string_space(s::VectorSpace) = string_space(s)
cartesian_string_space(s::TensorSpace) = "(" * string_space(s) * ")"
cartesian_string_space(s::CartesianProduct) = "(" * string_space(s) * ")"

function _supscript_digit(i::Int)
    if i == 0
        return "⁰"
    elseif i == 1
        return "¹"
    elseif i == 2
        return "²"
    elseif i == 3
        return "³"
    elseif i == 4
        return "⁴"
    elseif i == 5
        return "⁵"
    elseif i == 6
        return "⁶"
    elseif i == 7
        return "⁷"
    elseif i == 8
        return "⁸"
    else
        return "⁹"
    end
end
function _supscript(n::Int)
    if 0 ≤ n ≤ 9
        return _supscript_digit(n)
    else
        x = ""
        while n > 0
            n, d = divrem(n, 10)
            x = string(_supscript_digit(d), x)
        end
        return x
    end
end
