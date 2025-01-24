"""
    AbstractSequence{T<:VectorSpace,S<:AbstractVector}

Abstract type for all sequences.
"""
abstract type AbstractSequence{T<:VectorSpace,S<:AbstractVector} end

"""
    Sequence{T<:VectorSpace,S<:AbstractVector}

Compactly supported sequence in the given space.

Fields:
- `space :: T`
- `coefficients :: S`

Constructors:
- `Sequence(::VectorSpace, ::AbstractVector)`
- `Sequence(coefficients::AbstractVector)`: equivalent to `Sequence(ParameterSpace()^length(coefficients), coefficients)`

# Examples

```jldoctest
julia> Sequence(Taylor(2), [1, 2, 1]) # 1 + 2x + x^2
Sequence in Taylor(2) with coefficients Vector{Int64}:
 1
 2
 1

julia> Sequence(Taylor(1) ⊗ Fourier(1, 1.0), [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]) # (1 + x) cos(y)
Sequence in Taylor(1) ⊗ Fourier(1, 1.0) with coefficients Vector{Float64}:
 0.5
 0.5
 0.0
 0.0
 0.5
 0.5

julia> Sequence([1, 2, 3])
Sequence in 𝕂³ with coefficients Vector{Int64}:
 1
 2
 3
```
"""
struct Sequence{T<:VectorSpace,S<:AbstractVector} <: AbstractSequence{T,S}
    space :: T
    coefficients :: S
    function Sequence{T,S}(space::T, coefficients::S) where {T<:VectorSpace,S<:AbstractVector}
        l = length(coefficients)
        Base.OneTo(l) == eachindex(coefficients) || return throw(ArgumentError("offset vectors are not supported"))
        d = dimension(space)
        d == l || return throw(DimensionMismatch("dimensions must match: space has dimension $d, coefficients has length $l"))
        return new{T,S}(space, coefficients)
    end
end

Sequence(space::T, coefficients::S) where {T<:VectorSpace,S<:AbstractVector} =
    Sequence{T,S}(space, coefficients)

Sequence(coefficients::AbstractVector) =
    Sequence(ParameterSpace()^length(coefficients), coefficients)

space(a::Sequence) = a.space

coefficients(a::Sequence) = a.coefficients

# utilities

Base.:(==)(a::Sequence, b::Sequence) =
    space(a) == space(b) && coefficients(a) == coefficients(b)

Base.iszero(a::Sequence) = iszero(coefficients(a))

Base.isapprox(a::Sequence, b::Sequence; kwargs...) =
    space(a) == space(b) && isapprox(coefficients(a), coefficients(b); kwargs...)

Base.copy(a::Sequence) = Sequence(space(a), copy(coefficients(a)))

Base.similar(a::Sequence) = Sequence(space(a), similar(coefficients(a)))
Base.similar(a::Sequence, ::Type{T}) where {T} = Sequence(space(a), similar(coefficients(a), T))

Base.zeros(s::VectorSpace) = Sequence(s, zeros(dimension(s)))
Base.zeros(::Type{T}, s::VectorSpace) where {T} = Sequence(s, zeros(T, dimension(s)))

Base.ones(s::VectorSpace) = Sequence(s, ones(dimension(s)))
Base.ones(::Type{T}, s::VectorSpace) where {T} = Sequence(s, ones(T, dimension(s)))

Base.fill(value, s::VectorSpace) = Sequence(s, fill(value, dimension(s)))

function Base.fill!(a::Sequence, value)
    fill!(coefficients(a), value)
    return a
end

IntervalArithmetic.interval(::Type{T}, a::Sequence, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) where {T} =
    Sequence(interval(T, space(a)), interval(T, coefficients(a), d; format = format))
IntervalArithmetic.interval(a::Sequence, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) =
    Sequence(interval(space(a)), interval(coefficients(a), d; format = format))
IntervalArithmetic.interval(::Type{T}, a::Sequence, d::AbstractVector{IntervalArithmetic.Decoration}; format::Symbol = :infsup) where {T} =
    Sequence(interval(T, space(a)), interval(T, coefficients(a), d; format = format))
IntervalArithmetic.interval(a::Sequence, d::AbstractVector{IntervalArithmetic.Decoration}; format::Symbol = :infsup) =
    Sequence(interval(space(a)), interval(coefficients(a), d; format = format))

Base.reverse(a::Sequence; dims = :) = Sequence(space(a), reverse(coefficients(a); dims = dims))

Base.reverse!(a::Sequence; dims = :) = Sequence(space(a), reverse!(coefficients(a); dims = dims))

Base.zero(a::Sequence) = zeros(eltype(a), space(a))
Base.zero(::Type{Sequence{T,S}}) where {T<:VectorSpace,S<:AbstractVector} = zeros(eltype(S), _zero_space(T))
_zero_space(::Type{TensorSpace{T}}) where {T<:Tuple} = TensorSpace(map(_zero_space, fieldtypes(T)))
_zero_space(::Type{Taylor}) = Taylor(0)
_zero_space(::Type{Fourier{T}}) where {T<:Real} = Fourier(0, one(T))
_zero_space(::Type{Chebyshev}) = Chebyshev(0)

Base.one(a::Sequence{ParameterSpace}) = Sequence(space(a), [one(eltype(a))])
function Base.one(a::Sequence{<:SequenceSpace})
    new_space = _compatible_space_with_constant_index(space(a))
    CoefType = eltype(a)
    c = zeros(CoefType, new_space)
    @inbounds c[_findindex_constant(new_space)] = one(CoefType)
    return c
end
Base.one(::Type{Sequence{T,S}}) where {T<:VectorSpace,S<:AbstractVector} = ones(eltype(S), _zero_space(T))

Base.float(a::Sequence) = Sequence(_float_space(space(a)), float.(coefficients(a)))

Base.big(a::Sequence) = Sequence(_big_space(space(a)), big.(coefficients(a)))

for (f, g) ∈ ((:float, :_float_space), (:big, :_big_space))
    @eval begin
        $g(s::VectorSpace) = s

        $g(s::TensorSpace) = TensorSpace(map($g, spaces(s)))

        $g(s::Fourier) = Fourier(order(s), $f(frequency(s)))

        $g(s::CartesianPower) = CartesianPower($g(space(s)), nspaces(s))

        $g(s::CartesianProduct) = CartesianPower(map($g, spaces(s)))
    end
end

for f ∈ (:complex, :real, :imag, :conj)
    @eval Base.$f(a::Sequence) = Sequence(space(a), $f.(coefficients(a)))
end

Base.conj!(a::Sequence) = Sequence(space(a), conj!(coefficients(a)))

Base.complex(a::Sequence, b::Sequence) = Sequence(image(+, space(a), space(b)), complex.(coefficients(a), coefficients(b)))

Base.complex(::Type{Sequence{T,S}}) where {T<:VectorSpace,S<:AbstractVector} = Sequence{T,Vector{complex(eltype(S))}}

Base.permutedims(a::Sequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    Sequence(space(a)[σ], vec(permutedims(_no_alloc_reshape(coefficients(a), dimensions(space(a))), σ)))

Base.@propagate_inbounds component(a::Sequence{<:CartesianSpace}, i) =
    Sequence(space(a)[i], view(coefficients(a), _component_findposition(i, space(a))))

eachcomponent(a::Sequence{<:CartesianSpace}) =
    (@inbounds(component(a, i)) for i ∈ Base.OneTo(nspaces(space(a))))

# setindex!

Base.@propagate_inbounds function Base.setindex!(a::Sequence, x, α)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(indices(space_a), α)))
    setindex!(coefficients(a), x, _findposition(α, space_a))
    return a
end
Base.@propagate_inbounds function Base.setindex!(a::Sequence, x, u::AbstractVector)
    for (i, uᵢ) ∈ enumerate(u)
        a[uᵢ] = x[i]
    end
    return a
end
Base.@propagate_inbounds function Base.setindex!(a::Sequence{TensorSpace{T}}, x, u::TensorIndices{<:NTuple{N,Any}}) where {N,T<:NTuple{N,BaseSpace}}
    for (i, uᵢ) ∈ enumerate(u)
        a[uᵢ] = x[i]
    end
    return a
end

# promotion

Base.convert(::Type{Sequence{T₁,S₁}}, a::Sequence{T₂,S₂}) where {T₁,S₁,T₂,S₂} =
    Sequence{T₁,S₁}(convert(T₁, space(a)), convert(S₁, coefficients(a)))

Base.promote_rule(::Type{Sequence{T₁,S₁}}, ::Type{Sequence{T₂,S₂}}) where {T₁,S₁,T₂,S₂} =
    Sequence{promote_type(T₁, T₂), promote_type(S₁, S₂)}

# show

function Base.show(io::IO, ::MIME"text/plain", a::Sequence)
    println(io, "Sequence in ", _prettystring(space(a)), " with coefficients ", typeof(coefficients(a)), ":")
    return Base.print_array(io, coefficients(a))
end

function Base.show(io::IO, a::Sequence)
    get(io, :compact, false) && return show(io, coefficients(a))
    return print(io, "Sequence(", space(a), ", ", coefficients(a), ")")
end





# General methods

# order, frequency

order(a::AbstractSequence) = order(space(a))
order(a::AbstractSequence, i::Int) = order(space(a), i)

frequency(a::AbstractSequence) = frequency(space(a))
frequency(a::AbstractSequence, i::Int) = frequency(space(a), i)

# utilities

Base.firstindex(a::AbstractSequence) = _firstindex(space(a))

Base.lastindex(a::AbstractSequence) = _lastindex(space(a))

Base.length(a::AbstractSequence) = length(coefficients(a))

Base.size(a::AbstractSequence) = size(coefficients(a)) # necessary for broadcasting

Base.iterate(a::AbstractSequence) = iterate(coefficients(a))
Base.iterate(a::AbstractSequence, i) = iterate(coefficients(a), i)

Base.eltype(a::AbstractSequence) = eltype(coefficients(a))
Base.eltype(::Type{<:AbstractSequence{<:VectorSpace,T}}) where {T<:AbstractVector} = eltype(T)

# getindex, view

Base.@propagate_inbounds function Base.getindex(a::AbstractSequence, α)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(indices(space_a), α)))
    return getindex(coefficients(a), _findposition(α, space_a))
end
Base.@propagate_inbounds function Base.getindex(a::AbstractSequence, u::AbstractVector)
    v = Vector{eltype(a)}(undef, length(u))
    for (i, uᵢ) ∈ enumerate(u)
        v[i] = a[uᵢ]
    end
    return v
end
Base.@propagate_inbounds function Base.getindex(a::AbstractSequence{TensorSpace{T}}, u::TensorIndices{<:NTuple{N,Any}}) where {N,T<:NTuple{N,BaseSpace}}
    v = Vector{eltype(a)}(undef, length(u))
    for (i, uᵢ) ∈ enumerate(u)
        v[i] = a[uᵢ]
    end
    return v
end

Base.@propagate_inbounds function Base.view(a::AbstractSequence, α)
    space_a = space(a)
    @boundscheck(_checkbounds_indices(α, space_a) || throw(BoundsError(indices(space_a), α)))
    return view(coefficients(a), _findposition(α, space_a))
end

# selectdim

Base.@propagate_inbounds function Base.selectdim(a::AbstractSequence{<:TensorSpace}, dim::Int, i)
    A = _no_alloc_reshape(coefficients(a), dimensions(space(a)))
    return selectdim(A, dim, _findposition(i, spaces(space(a))[dim]))
end
