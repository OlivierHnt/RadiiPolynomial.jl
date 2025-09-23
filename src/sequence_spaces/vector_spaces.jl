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
_firstindex(s::VectorSpace) = first(indices(s))
_lastindex(s::VectorSpace) = last(indices(s))

_checkbounds_indices(α, s::VectorSpace) = α ∈ indices(s)
_checkbounds_indices(α::Union{AbstractRange,AbstractVector}, s::VectorSpace) = all(∈(indices(s)), α)
_checkbounds_indices(::Colon, ::VectorSpace) = true
_checkbounds_indices(α::VectorSpace, space::VectorSpace) = issubset(α, space)

_iscompatible(::VectorSpace, ::VectorSpace) = false

function _prettystring(s::VectorSpace, ::Bool)
    T = typeof(s)
    a, b... = fieldnames(T)
    str = mapreduce(f -> string(getproperty(s, f)), (x, y) -> x * ", " * y, b; init = string(getproperty(s, a)))
    return string(T) * "(" * str * ")"
end

#

"""
    EmptySpace <: VectorSpace

Empty vector space.

# Example

```jldoctest
julia> EmptySpace()
∅

julia> LinearOperator(EmptySpace(), EmptySpace(), [;;])
LinearOperator : ∅ → ∅ with coefficients Matrix{Any}:
```
"""
struct EmptySpace <: VectorSpace end

Base.:(==)(::EmptySpace, ::EmptySpace) = true
Base.issubset(::EmptySpace, ::EmptySpace) = true
Base.intersect(::EmptySpace, ::EmptySpace) = EmptySpace()
Base.union(::EmptySpace, ::EmptySpace) = EmptySpace()

indices(::EmptySpace) = Base.OneTo(0)

_findposition(i, ::EmptySpace) = i
_findposition(α::EmptySpace, s::EmptySpace) = _findposition(indices(α), s)

_prettystring(::EmptySpace, iscompact::Bool) = ifelse(iscompact, "∅", "EmptySpace()")

#

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

desymmetrize(s::ParameterSpace) = s

indices(::ParameterSpace) = Base.OneTo(1)

_findposition(i, ::ParameterSpace) = i
_findposition(α::ParameterSpace, s::ParameterSpace) = _findposition(indices(α), s)

_iscompatible(::ParameterSpace, ::ParameterSpace) = true

IntervalArithmetic.interval(::Type{T}, s::ParameterSpace) where {T} = s
IntervalArithmetic.interval(s::ParameterSpace) = s

_prettystring(::ParameterSpace, iscompact::Bool) = ifelse(iscompact, "𝕂", "ParameterSpace()")

#

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
    SymBaseSpace <: BaseSpace

Abstract type for all symmetric sequence spaces that are not a
[`TensorSpace`](@ref) but can be interlaced to form one.
"""
abstract type SymBaseSpace <: BaseSpace end

order(s::SymBaseSpace) = order(desymmetrize(s))
frequency(s::SymBaseSpace) = frequency(desymmetrize(s))

Base.issubset(s₁::SymBaseSpace, s₂::SymBaseSpace) = false
Base.issubset(s₁::SymBaseSpace, s₂::BaseSpace) = issubset(desymmetrize(s₁), s₂)
Base.union(s₁::SymBaseSpace, s₂::SymBaseSpace) = union(desymmetrize(s₁), desymmetrize(s₂))
Base.union(s₁::SymBaseSpace, s₂::BaseSpace) = union(desymmetrize(s₁), s₂)
Base.union(s₁::BaseSpace, s₂::SymBaseSpace) = union(s₁, desymmetrize(s₂))

"""
    TensorSpace{T<:Tuple{Vararg{BaseSpace}}} <: SequenceSpace

Sequence space resulting from the tensor product of some [`BaseSpace`](@ref).

Field:
- `spaces :: T`

Constructors:
- `TensorSpace(spaces::Tuple{Vararg{BaseSpace}})`
- `TensorSpace(spaces::BaseSpace...)`
- `⊗(s₁::BaseSpace, s₂::BaseSpace)`: equivalent to `TensorSpace((s₁, s₂))`
- `⊗(s₁::TensorSpace, s₂::TensorSpace)`: equivalent to `TensorSpace((s₁.spaces..., s₂.spaces...))`
- `⊗(s₁::TensorSpace, s₂::BaseSpace)`: equivalent to `TensorSpace((s₁.spaces..., s₂))`
- `⊗(s₁::BaseSpace, s₂::TensorSpace)`: equivalent to `TensorSpace((s₁, s₂.spaces...))`

See also: [`⊗`](@ref).

# Examples

```jldoctest
julia> s = TensorSpace(Taylor(1), Fourier(2, 1.0), Chebyshev(3))
Taylor(1) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)

julia> spaces(s)
(Taylor(1), Fourier(2, 1.0), Chebyshev(3))
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
Taylor(1) ⊗ Fourier(2, 1.0)

julia> Taylor(1) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)
Taylor(1) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)

julia> Taylor(1) ⊗ (Fourier(2, 1.0) ⊗ Chebyshev(3))
Taylor(1) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)

julia> (Taylor(1) ⊗ Fourier(2, 1.0)) ⊗ Chebyshev(3)
Taylor(1) ⊗ Fourier(2, 1.0) ⊗ Chebyshev(3)
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

dimension(s::TensorSpace, i::Int) = dimension(s.spaces[i])
dimensions(s::TensorSpace) = map(dimension, s.spaces)
_firstindex(s::TensorSpace) = map(_firstindex, s.spaces)
_lastindex(s::TensorSpace) = map(_lastindex, s.spaces)

order(s::TensorSpace) = map(order, s.spaces)
order(s::TensorSpace, i::Int) = order(s.spaces[i])

frequency(s::TensorSpace) = map(frequency, s.spaces)
frequency(s::TensorSpace, i::Int) = frequency(s.spaces[i])

desymmetrize(s::TensorSpace) = TensorSpace(map(desymmetrize, spaces(s)))

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

_compatible_space_with_constant_index(s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map(_compatible_space_with_constant_index, s.spaces))
_findindex_constant(s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = map(_findindex_constant, s.spaces)

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
_findposition(α::TensorSpace{<:NTuple{N,BaseSpace}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = _findposition(indices(α), s)

_iscompatible(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds _iscompatible(s₁[1], s₂[1]) & _iscompatible(Base.tail(s₁), Base.tail(s₂))
_iscompatible(s₁::TensorSpace{<:Tuple{BaseSpace}}, s₂::TensorSpace{<:Tuple{BaseSpace}}) =
    @inbounds _iscompatible(s₁[1], s₂[1])

IntervalArithmetic.interval(::Type{T}, s::TensorSpace) where {T} = TensorSpace(map(sᵢ -> interval(T, sᵢ), s.spaces))
IntervalArithmetic.interval(s::TensorSpace) = TensorSpace(map(interval, s.spaces))

_prettystring(s::TensorSpace, iscompact::Bool) = _prettystring(s[1], iscompact) * " ⊗ " * _prettystring(Base.tail(s), iscompact)
_prettystring(s::TensorSpace{<:NTuple{2,BaseSpace}}, iscompact::Bool) = _prettystring(s[1], iscompact) * " ⊗ " * _prettystring(s[2], iscompact)
_prettystring(s::TensorSpace{<:Tuple{BaseSpace}}, iscompact::Bool) = "TensorSpace(" * _prettystring(s[1], iscompact) * ")"

# promotion

Base.convert(::Type{TensorSpace{T}}, s::TensorSpace) where {T} =
    TensorSpace{T}(convert(T, s.spaces))

Base.promote_rule(::Type{TensorSpace{T}}, ::Type{TensorSpace{S}}) where {T,S} =
    TensorSpace{promote_type(T, S)}

#

"""
    Taylor <: BaseSpace

Sequence space whose elements are Taylor sequences of a prescribed order.

Field:
- `order :: Int`

Constructor:
- `Taylor(order::Int)`

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

desymmetrize(s::Taylor) = s

indices(s::Taylor) = 0:s.order

_compatible_space_with_constant_index(s::Taylor) = s
_findindex_constant(::Taylor) = 0

_findposition(i::Int, ::Taylor) = i + 1
_findposition(u::AbstractRange{Int}, ::Taylor) = u .+ 1
_findposition(u::AbstractVector{Int}, s::Taylor) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::Taylor) = c
_findposition(α::Taylor, s::Taylor) = _findposition(indices(α), s)

_iscompatible(::Taylor, ::Taylor) = true

IntervalArithmetic.interval(::Type{T}, s::Taylor) where {T} = s
IntervalArithmetic.interval(s::Taylor) = s

_prettystring(s::Taylor, ::Bool) = "Taylor(" * string(order(s)) * ")"

#

"""
    Fourier{T<:Real} <: BaseSpace

Sequence space whose elements are Fourier sequences of a prescribed order and frequency.

Fields:
- `order :: Int`
- `frequency :: T`

Constructor:
- `Fourier(order::Int, frequency::Real)`

See also: [`Taylor`](@ref) and [`Chebyshev`](@ref).

# Examples

```jldoctest
julia> s = Fourier(2, 1.0)
Fourier(2, 1.0)

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
        (order ≥ 0) & (inf(frequency) ≥ 0) || return throw(DomainError((order, frequency), "Fourier is only defined for positive orders and frequencies"))
        return new{T}(order, frequency)
    end
end

Fourier(order::Int, frequency::T) where {T<:Real} = Fourier{T}(order, frequency)

order(s::Fourier) = s.order

frequency(s::Fourier) = s.frequency

Base.:(==)(s₁::Fourier, s₂::Fourier) = _safe_isequal(s₁.frequency, s₂.frequency) & (s₁.order == s₂.order)
Base.issubset(s₁::Fourier, s₂::Fourier) = _safe_isequal(s₁.frequency, s₂.frequency) & (s₁.order ≤ s₂.order)
function Base.intersect(s₁::Fourier{T}, s₂::Fourier{S}) where {T<:Real,S<:Real}
    _safe_isequal(s₁.frequency, s₂.frequency) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $(s₁.frequency), s₂ has frequency $(s₂.frequency)"))
    R = promote_type(T, S)
    return Fourier(min(s₁.order, s₂.order), convert(R, s₁.frequency))
end
function Base.union(s₁::Fourier{T}, s₂::Fourier{S}) where {T<:Real,S<:Real}
    _safe_isequal(s₁.frequency, s₂.frequency) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $(s₁.frequency), s₂ has frequency $(s₂.frequency)"))
    R = promote_type(T, S)
    return Fourier(max(s₁.order, s₂.order), convert(R, s₁.frequency))
end

desymmetrize(s::Fourier) = s

indices(s::Fourier) = -s.order:s.order

_compatible_space_with_constant_index(s::Fourier) = s
_findindex_constant(::Fourier) = 0

_findposition(i::Int, s::Fourier) = i + s.order + 1
_findposition(u::AbstractRange{Int}, s::Fourier) = u .+ s.order .+ 1
_findposition(u::AbstractVector{Int}, s::Fourier) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::Fourier) = c
_findposition(α::Fourier, s::Fourier) = _findposition(indices(α), s)

_iscompatible(s₁::Fourier, s₂::Fourier) = _safe_isequal(frequency(s₁), frequency(s₂))

IntervalArithmetic.interval(::Type{T}, s::Fourier) where {T} = Fourier(order(s), interval(T, frequency(s)))
IntervalArithmetic.interval(s::Fourier) = Fourier(order(s), interval(frequency(s)))

_prettystring(s::Fourier, ::Bool) = "Fourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

# promotion

Base.convert(::Type{Fourier{T}}, s::Fourier) where {T<:Real} =
    Fourier{T}(s.order, convert(T, s.frequency))

Base.promote_rule(::Type{Fourier{T}}, ::Type{Fourier{S}}) where {T<:Real,S<:Real} =
    Fourier{promote_type(T, S)}

#

"""
    Chebyshev <: BaseSpace

Sequence space whose elements are Chebyshev sequences of a prescribed order.

Field:
- `order :: Int`

Constructor:
- `Chebyshev(order::Int)`

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

desymmetrize(s::Chebyshev) = s

indices(s::Chebyshev) = 0:s.order

_compatible_space_with_constant_index(s::Chebyshev) = s
_findindex_constant(::Chebyshev) = 0

_findposition(i::Int, ::Chebyshev) = i + 1
_findposition(u::AbstractRange{Int}, ::Chebyshev) = u .+ 1
_findposition(u::AbstractVector{Int}, s::Chebyshev) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::Chebyshev) = c
_findposition(α::Chebyshev, s::Chebyshev) = _findposition(indices(α), s)

_iscompatible(::Chebyshev, ::Chebyshev) = true

IntervalArithmetic.interval(::Type{T}, s::Chebyshev) where {T} = s
IntervalArithmetic.interval(s::Chebyshev) = s

_prettystring(s::Chebyshev, ::Bool) = "Chebyshev(" * string(order(s)) * ")"

#

"""
    CosFourier{T<:Real} <: SymBaseSpace

Sequence space whose elements are cosine sequences of a prescribed order and frequency.

Field:
- `space :: Fourier{T}`

Constructors:
- `CosFourier(space::Fourier)`
- `CosFourier(order::Int, frequency::Real)`

# Example

```jldoctest
julia> s = CosFourier(2, 1.0)
CosFourier(2, 1.0)

julia> order(s)
2

julia> frequency(s)
1.0
```
"""
struct CosFourier{T<:Real} <: SymBaseSpace
    space :: Fourier{T}
    CosFourier{T}(space::Fourier{T}) where {T<:Real} = new{T}(space)
end

CosFourier(space::Fourier{T}) where {T<:Real} = CosFourier{T}(space)
CosFourier{T}(order::Int, frequency::T) where {T<:Real} = CosFourier(Fourier{T}(order, frequency))
CosFourier(order::Int, frequency::Real) = CosFourier(Fourier(order, frequency))

desymmetrize(s::CosFourier) = s.space

Base.:(==)(s₁::CosFourier, s₂::CosFourier) = desymmetrize(s₁) == desymmetrize(s₂)
Base.issubset(s₁::CosFourier, s₂::CosFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
Base.intersect(s₁::CosFourier, s₂::CosFourier) = CosFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
Base.union(s₁::CosFourier, s₂::CosFourier) = CosFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

indices(s::CosFourier) = 0:order(s)

_compatible_space_with_constant_index(s::CosFourier) = s
_findindex_constant(::CosFourier) = 0

_findposition(i::Int, ::CosFourier) = i + 1
_findposition(u::AbstractRange{Int}, ::CosFourier) = u .+ 1
_findposition(u::AbstractVector{Int}, s::CosFourier) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::CosFourier) = c
_findposition(α::CosFourier, s::CosFourier) = _findposition(indices(α), s)

_iscompatible(s₁::CosFourier, s₂::CosFourier) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂))

IntervalArithmetic.interval(::Type{T}, s::CosFourier) where {T} = CosFourier(interval(T, desymmetrize(s)))
IntervalArithmetic.interval(s::CosFourier) = CosFourier(interval(desymmetrize(s)))

_prettystring(s::CosFourier, ::Bool) = "CosFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

# promotion

Base.convert(::Type{CosFourier{T}}, s::CosFourier) where {T<:Real} =
    CosFourier{T}(order(s), convert(T, frequency(s)))

Base.promote_rule(::Type{CosFourier{T}}, ::Type{CosFourier{S}}) where {T<:Real,S<:Real} =
    CosFourier{promote_type(T, S)}

#

"""
    SinFourier{T<:Real} <: SymBaseSpace

Sequence space whose elements are sine sequences of a prescribed order and frequency.

Field:
- `space :: Fourier{T}`

Constructors:
- `SinFourier(space::Fourier)`
- `SinFourier(order::Int, frequency::Real)`

# Example

```jldoctest
julia> s = SinFourier(2, 1.0)
SinFourier(2, 1.0)

julia> order(s)
2

julia> frequency(s)
1.0
```
"""
struct SinFourier{T<:Real} <: SymBaseSpace
    space :: Fourier{T}
    function SinFourier{T}(space::Fourier{T}) where {T<:Real}
        order(space) < 1 && return throw(DomainError(order, "SinFourier is only defined for orders greater or equal to 1"))
        return new{T}(space)
    end
end

SinFourier(space::Fourier{T}) where {T<:Real} = SinFourier{T}(space)
SinFourier{T}(order::Int, frequency::T) where {T<:Real} = SinFourier(Fourier{T}(order, frequency))
SinFourier(order::Int, frequency::Real) = SinFourier(Fourier(order, frequency)) # may fail since it can normalize to order 0

desymmetrize(s::SinFourier) = s.space

Base.:(==)(s₁::SinFourier, s₂::SinFourier) = desymmetrize(s₁) == desymmetrize(s₂)
Base.issubset(s₁::SinFourier, s₂::SinFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
Base.intersect(s₁::SinFourier, s₂::SinFourier) = SinFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
Base.union(s₁::SinFourier, s₂::SinFourier) = SinFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

indices(s::SinFourier) = 1:order(s)

_compatible_space_with_constant_index(s::SinFourier) = desymmetrize(s)

_findposition(i::Int, ::SinFourier) = i
_findposition(u::AbstractRange{Int}, ::SinFourier) = u
_findposition(u::AbstractVector{Int}, s::SinFourier) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::SinFourier) = c
_findposition(α::SinFourier, s::SinFourier) = _findposition(indices(α), s)

_iscompatible(s₁::SinFourier, s₂::SinFourier) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂))

IntervalArithmetic.interval(::Type{T}, s::SinFourier) where {T} = SinFourier(interval(T, desymmetrize(s)))
IntervalArithmetic.interval(s::SinFourier) = SinFourier(interval(desymmetrize(s)))

_prettystring(s::SinFourier, ::Bool) = "SinFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

# promotion

Base.convert(::Type{SinFourier{T}}, s::SinFourier) where {T<:Real} =
    SinFourier{T}(order(s), convert(T, frequency(s)))

Base.promote_rule(::Type{SinFourier{T}}, ::Type{SinFourier{S}}) where {T<:Real,S<:Real} =
    SinFourier{promote_type(T, S)}





# Cartesian spaces

"""
    CartesianSpace <: VectorSpace

Abstract type for all cartesian spaces.
"""
abstract type CartesianSpace <: VectorSpace end

_findposition(i::Union{Int,AbstractRange{Int},AbstractVector{Int},Colon}, ::CartesianSpace) = i

_component_findposition(u::AbstractRange{Int}, s::CartesianSpace) =
    mapreduce(i -> _component_findposition(i, s), union, u)
_component_findposition(u::AbstractVector{Int}, s::CartesianSpace) =
    mapreduce(i -> _component_findposition(i, s), union, u)
_component_findposition(c::Colon, ::CartesianSpace) = c

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

indices(s::CartesianPower) = Base.OneTo(dimension(s.space)*s.n)

function dimension(s::CartesianPower, i::Int)
    (1 ≤ i) & (i ≤ s.n) || return throw(BoundsError(s, i))
    return dimension(s.space)
end
dimensions(s::CartesianPower) = fill(dimension(s.space), s.n)

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

desymmetrize(s::CartesianPower) = CartesianPower(desymmetrize(space(s)), nspaces(s))

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

_iscompatible(s₁::CartesianPower, s₂::CartesianPower) =
    (nspaces(s₁) == nspaces(s₂)) & _iscompatible(space(s₁), space(s₂))

IntervalArithmetic.interval(::Type{T}, s::CartesianPower) where {T} = CartesianPower(interval(T, s.space), s.n)
IntervalArithmetic.interval(s::CartesianPower) = CartesianPower(interval(s.space), s.n)

_prettystring(s::CartesianPower, iscompact::Bool) = _prettystring(space(s), iscompact) * _supscript(nspaces(s))
_prettystring(s::CartesianPower{<:TensorSpace}, iscompact::Bool) = "(" * _prettystring(space(s), iscompact) * ")" * _supscript(nspaces(s))
_prettystring(s::CartesianPower{<:CartesianSpace}, iscompact::Bool) = "(" * _prettystring(space(s), iscompact) * ")" * _supscript(nspaces(s))

# promotion

Base.convert(::Type{CartesianPower{T}}, s::CartesianPower) where {T} =
    CartesianPower{T}(convert(T, s.space), s.n)

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
Taylor(1) × Fourier(2, 1.0) × Chebyshev(3)

julia> spaces(s)
(Taylor(1), Fourier(2, 1.0), Chebyshev(3))

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
Taylor(1) × Fourier(2, 1.0)

julia> Taylor(1) × Fourier(2, 1.0) × Chebyshev(3)
Taylor(1) × Fourier(2, 1.0) × Chebyshev(3)

julia> (Taylor(1) × Fourier(2, 1.0)) × Chebyshev(3)
Taylor(1) × Fourier(2, 1.0) × Chebyshev(3)

julia> Taylor(1) × (Fourier(2, 1.0) × Chebyshev(3))
Taylor(1) × Fourier(2, 1.0) × Chebyshev(3)

julia> ParameterSpace()^2 × ((Taylor(1) ⊗ Fourier(2, 1.0)) × Chebyshev(3))^3
𝕂² × ((Taylor(1) ⊗ Fourier(2, 1.0)) × Chebyshev(3))³
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

indices(s::CartesianProduct) = Base.OneTo(mapreduce(dimension, +, s.spaces))

dimension(s::CartesianProduct, i::Int) = dimension(s.spaces[i])
dimensions(s::CartesianProduct) = map(dimension, s.spaces)

order(s::CartesianProduct) = map(order, s.spaces)
order(s::CartesianProduct, i::Int) = order(s.spaces[i])

frequency(s::CartesianProduct) = map(frequency, s.spaces)
frequency(s::CartesianProduct, i::Int) = frequency(s.spaces[i])

desymmetrize(s::CartesianProduct) = CartesianProduct(map(desymmetrize, spaces(s)))

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

_iscompatible(s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N} =
    @inbounds _iscompatible(s₁[1], s₂[1]) & _iscompatible(Base.tail(s₁), Base.tail(s₂))
_iscompatible(s₁::CartesianProduct{<:Tuple{VectorSpace}}, s₂::CartesianProduct{<:Tuple{VectorSpace}}) =
    @inbounds _iscompatible(s₁[1], s₂[1])

IntervalArithmetic.interval(::Type{T}, s::CartesianProduct) where {T} = CartesianProduct(map(sᵢ -> interval(T, sᵢ), s.spaces))
IntervalArithmetic.interval(s::CartesianProduct) = CartesianProduct(map(interval, s.spaces))

_prettystring(s::CartesianProduct, iscompact::Bool) = _prettystring_cartesian(s[1], iscompact) * " × " * _prettystring(Base.tail(s), iscompact)
_prettystring(s::CartesianProduct{<:NTuple{2,VectorSpace}}, iscompact::Bool) = _prettystring_cartesian(s[1], iscompact) * " × " * _prettystring_cartesian(s[2], iscompact)
_prettystring(s::CartesianProduct{<:Tuple{VectorSpace}}, iscompact::Bool) = "CartesianProduct(" * _prettystring(s[1], iscompact) * ")"
_prettystring_cartesian(s::VectorSpace, iscompact::Bool) = _prettystring(s, iscompact)
_prettystring_cartesian(s::TensorSpace, iscompact::Bool) = "(" * _prettystring(s, iscompact) * ")"
_prettystring_cartesian(s::CartesianProduct, iscompact::Bool) = "(" * _prettystring(s, iscompact) * ")"

# promotion

Base.convert(::Type{CartesianProduct{T}}, s::CartesianProduct) where {T} =
    CartesianProduct{T}(convert(T, s.spaces))

Base.promote_rule(::Type{CartesianProduct{T}}, ::Type{CartesianProduct{S}}) where {T,S} =
    CartesianProduct{promote_type(T, S)}

# mix

_iscompatible(s₁::CartesianPower, s₂::CartesianProduct) =
    (nspaces(s₁) == nspaces(s₂)) & all(s₂ᵢ -> _iscompatible(space(s₁), s₂ᵢ), spaces(s₂))
_iscompatible(s₁::CartesianProduct, s₂::CartesianPower) =
    (nspaces(s₁) == nspaces(s₂)) & all(s₁ᵢ -> _iscompatible(s₁ᵢ, space(s₂)), spaces(s₁))

for f ∈ (:(==), :issubset)
    @eval begin
        function Base.$f(s₁::CartesianPower, s₂::CartesianProduct)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return false
            return all(s₂ᵢ -> $f(s₁.space, s₂ᵢ), s₂.spaces)
        end

        function Base.$f(s₁::CartesianProduct, s₂::CartesianPower)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return false
            return all(s₁ᵢ -> $f(s₁ᵢ, s₂.space), s₁.spaces)
        end
    end
end

for f ∈ (:intersect, :union)
    @eval begin
        function Base.$f(s₁::CartesianPower, s₂::CartesianProduct)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map(s₂ᵢ -> $f(s₁.space, s₂ᵢ), s₂.spaces))
        end

        function Base.$f(s₁::CartesianProduct, s₂::CartesianPower)
            n = nspaces(s₁)
            m = nspaces(s₂)
            n == m || return throw(ArgumentError("number of cartesian products must be equal: s₁ has $n cartesian product(s), s₂ has $m cartesian product(s)"))
            return CartesianProduct(map(s₁ᵢ -> $f(s₁ᵢ, s₂.space), s₁.spaces))
        end
    end
end

# additional methods

function _findposition(α::CartesianSpace, s::CartesianSpace)
    v = [_findposition(_iterate_space(α, 1), _iterate_space(s, 1));]
    for i ∈ 2:_deep_nspaces(α)
        offset = sum(dimension(_iterate_space(s, k)) for k in 1:i-1)
        append!(v, _findposition(_iterate_space(α, i), _iterate_space(s, i)) .+ offset)
    end
    return v
end

_iterate_space(s::CartesianPower, i) = _iterate_space(space(s), i)
function _iterate_space(s::CartesianProduct, i)
    _deep_nspaces(s[1]) ≥ i && return _iterate_space(s[1], i)
    for j ∈ 2:nspaces(s)-1
        _deep_nspaces(s[1:j]) ≥ i && return _iterate_space(s[j], i - _deep_nspaces(s[1:j-1]))
    end
    return _iterate_space(s[nspaces(s)], i - _deep_nspaces(s[1:nspaces(s)-1]))
end
_iterate_space(s, i) = s

_deep_nspaces(::VectorSpace) = 1
_deep_nspaces(s::CartesianPower) = s.n * _deep_nspaces(s.space)
_deep_nspaces(s::CartesianProduct) = sum(_deep_nspaces, s.spaces)





# show

Base.show(io::IO, ::MIME"text/plain", s::VectorSpace) = print(io, _prettystring(s, true))

Base.show(io::IO, s::VectorSpace) = print(io, _prettystring(s, get(io, :compact, false)))

function _supscript(n::Integer)
    if 0 ≤ n ≤ 9
        return _supscript_digit(n)
    else
        len = ndigits(n)
        x = Vector{Char}(undef, len)
        i = 0
        while n > 0
            n, d = divrem(n, 10)
            x[len-i] = _supscript_digit(d)
            i += 1
        end
        return join(x)
    end
end

function _supscript_digit(i::Integer)
    i == 0 && return '⁰'
    i == 1 && return '¹'
    i == 2 && return '²'
    i == 3 && return '³'
    i == 4 && return '⁴'
    i == 5 && return '⁵'
    i == 6 && return '⁶'
    i == 7 && return '⁷'
    i == 8 && return '⁸'
    return '⁹'
end
