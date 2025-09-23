"""
    InfiniteSequence{T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence

Infinite sequence in the given sequence space, with a norm and a truncation error.

Fields:
- `sequence :: Sequence{T,S}`
- `sequence_norm :: R`
- `sequence_error :: R`
- `banachspace :: U`

Constructors:
- `InfiniteSequence(sequence::Sequence{T,S}, sequence_error::R, banachspace::U)`
- `InfiniteSequence(sequence::Sequence{T,S}, banachspace::U)`
- `InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace)`
- `InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, sequence_error::Interval, banachspace::BanachSpace)`

# Example

```jldoctest
julia> InfiniteSequence(Sequence(Taylor(2), [1.0, 2.0, 1.0]), 0.1, Ell1())
Sequence in Taylor(2) with coefficients Vector{Float64}:
 1.0
 2.0
 1.0
Norm of the truncated sequence: 4.0
Sequence error: 0.1
Banach space: ℓ¹()
```
"""
struct InfiniteSequence{T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence
    sequence :: Sequence{T,S}
    sequence_norm :: R
    sequence_error :: R
    banachspace :: U
    global _unsafe_infinite_sequence(sequence::Sequence{T,S}, sequence_norm::R, sequence_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
        new{T,S,R,U}(sequence, sequence_norm, sequence_error, banachspace)
end

function InfiniteSequence{T,S,R,U}(sequence::Sequence{T,S}, sequence_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace}
    _iscompatbanachspace(space(sequence), banachspace) || return throw(ArgumentError("invalid norm for the sequence space"))
    inf(sequence_error) ≥ 0 || return throw(ArgumentError("sequence error must be positive"))
    return _unsafe_infinite_sequence(sequence, convert(R, norm(sequence, banachspace)), sequence_error, banachspace)
end

InfiniteSequence(sequence::Sequence{T,S}, sequence_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
    InfiniteSequence{T,S,R,U}(sequence, sequence_error, banachspace)

InfiniteSequence(sequence::Sequence, banachspace::BanachSpace) =
    InfiniteSequence(sequence, zero(real(eltype(sequence))), banachspace)

InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace) =
    InfiniteSequence(Sequence(space, coefficients), banachspace)

InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, sequence_error::Interval, banachspace::BanachSpace) =
    InfiniteSequence(Sequence(space, coefficients), sequence_error, banachspace)

_iscompatbanachspace(::SequenceSpace, ::BanachSpace) = false
_iscompatbanachspace(::SequenceSpace, ::Ell1{<:Weight}) = true
_iscompatbanachspace(::SequenceSpace, ::Ell2{<:Weight}) = true
_iscompatbanachspace(::SequenceSpace, ::EllInf{<:Weight}) = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::Ell1{<:NTuple{N,Weight}}) where {N} = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::Ell2{<:NTuple{N,Weight}}) where {N} = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::EllInf{<:NTuple{N,Weight}}) where {N} = true

sequence(a::InfiniteSequence) = a.sequence
sequence_norm(a::InfiniteSequence) = a.sequence_norm
sequence_error(a::InfiniteSequence) = a.sequence_error
banachspace(a::InfiniteSequence) = a.banachspace

space(a::InfiniteSequence) = space(sequence(a)) # needed for general methods

coefficients(a::InfiniteSequence) = coefficients(sequence(a)) # needed for general methods

# utilities

Base.eltype(a::InfiniteSequence) = eltype(coefficients(a))
Base.eltype(::Type{<:InfiniteSequence{<:SequenceSpace,T}}) where {T<:AbstractVector} = eltype(T)

Base.:(==)(a::InfiniteSequence, b::InfiniteSequence) = # by-pass default
    (sequence(a) == sequence(b)) & iszero(sequence_error(a)) & iszero(sequence_error(b))

Base.copy(a::InfiniteSequence) =
    _unsafe_infinite_sequence(copy(sequence(a)), sequence_norm(a), sequence_error(a), banachspace(a))

Base.zero(a::InfiniteSequence) = InfiniteSequence(zero(sequence(a)), banachspace(a))
Base.one(a::InfiniteSequence) = InfiniteSequence(one(sequence(a)), banachspace(a))

IntervalArithmetic.interval(::Type{T}, a::InfiniteSequence, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) where {T} =
    InfiniteSequence(interval(T, sequence(a), d; format = format), interval(T, sequence_error(a)), interval(T, banachspace(a)))
IntervalArithmetic.interval(a::InfiniteSequence, d::IntervalArithmetic.Decoration = com; format::Symbol = :infsup) =
    InfiniteSequence(interval(sequence(a), d; format = format), interval(sequence_error(a)), interval(banachspace(a)))
IntervalArithmetic.interval(::Type{T}, a::InfiniteSequence, d::AbstractVector{IntervalArithmetic.Decoration}; format::Symbol = :infsup) where {T} =
    InfiniteSequence(interval(T, sequence(a), d; format = format), interval(T, sequence_error(a)), interval(T, banachspace(a)))
IntervalArithmetic.interval(a::InfiniteSequence, d::AbstractVector{IntervalArithmetic.Decoration}; format::Symbol = :infsup) =
    InfiniteSequence(interval(sequence(a), d; format = format), interval(sequence_error(a)), interval(banachspace(a)))

Base.float(a::InfiniteSequence) = InfiniteSequence(float(sequence(a)), float(sequence_error(a)), banachspace(a))
for f ∈ (:complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::InfiniteSequence) = InfiniteSequence($f(sequence(a)), sequence_error(a), banachspace(a))
end

Base.permutedims(a::InfiniteSequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    _unsafe_infinite_sequence(permutedims(sequence(a), σ), sequence_norm(a), sequence_error(a), banachspace(a))

# show

function Base.show(io::IO, ::MIME"text/plain", a::InfiniteSequence)
    println(io, "Sequence in ", _prettystring(space(a), true), " with coefficients ", typeof(coefficients(a)), ":")
    Base.print_array(io, coefficients(a))
    println(io, "\nNorm of the truncated sequence: ", sequence_norm(a))
    println(io, "Sequence error: ", sequence_error(a))
    return print(io, "Banach space: ", _prettystring(banachspace(a)))
end

function Base.show(io::IO, a::InfiniteSequence)
    get(io, :compact, false) && return show(io, (coefficients(a), sequence_error(a), banachspace(a)))
    return print(io, "InfiniteSequence(", space(a), ", ", coefficients(a), ", ", sequence_norm(a), ", ", sequence_error(a), ", ", banachspace(a), ")")
end
