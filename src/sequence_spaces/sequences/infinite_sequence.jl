"""
    InfiniteSequence{T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence

Infinite sequence in the given sequence space, with error and norm bookkeeping.
The error is split into three independent non-negative upper bounds:
- `finite_error`: error on the finite part of the sequence
- `tail_error`: error on the tail part of the sequence
- `total_error`: error on the total sequence (no support hypothesis)

Fields:
- `sequence :: Sequence{T,S}`
- `sequence_norm :: R`
- `finite_error :: R`
- `tail_error :: R`
- `total_error :: R`
- `full_norm :: R`
- `banachspace :: U`

Constructors:
- `InfiniteSequence(sequence, finite_error, tail_error, total_error, banachspace)`
- `InfiniteSequence(sequence, banachspace; finite_error = 0, tail_error = 0, total_error = ...)`
- `InfiniteSequence(sequence, banachspace)`: all errors zero.
- `InfiniteSequence(space, coefficients, banachspace)`
- `InfiniteSequence(space, coefficients, finite_error, tail_error, total_error, banachspace)`

# Example

```jldoctest
julia> InfiniteSequence(Sequence(Taylor(2), [1.0, 2.0, 1.0]), 0.0, 0.1, 0.1, Ell1())
Sequence in Taylor(2) with coefficients Vector{Float64}:
 1.0
 2.0
 1.0
Norm of the truncated sequence: 4.0
Finite error: 0.0
Tail error: 0.1
Total error: 0.1
Banach space: ℓ¹()
```
"""
struct InfiniteSequence{T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence
    sequence :: Sequence{T,S}
    sequence_norm :: R
    finite_error :: R
    tail_error :: R
    total_error :: R
    full_norm :: R
    banachspace :: U
    global _unsafe_infinite_sequence(sequence::Sequence{T,S}, sequence_norm::R, finite_error::R, tail_error::R, total_error::R, full_norm::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
        new{T,S,R,U}(sequence, sequence_norm, finite_error, tail_error, total_error, full_norm, banachspace)
end

function InfiniteSequence{T,S,R,U}(sequence::Sequence{T,S}, finite_error::R, tail_error::R, total_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace}
    _iscompatbanachspace(space(sequence), banachspace) || return throw(ArgumentError("invalid norm for the sequence space"))
    (inf(finite_error) ≥ 0) & (inf(tail_error) ≥ 0) & (inf(total_error) ≥ 0) || return throw(ArgumentError("errors must be non-negative"))
    total_ = min(finite_error + tail_error, total_error)
    finite_, tail_ = ifelse(_safe_iszero(total_), (zero(R), zero(R)), (finite_error, tail_error))
    seq_norm = convert(R, norm(sequence, banachspace))
    return _unsafe_infinite_sequence(sequence, seq_norm, finite_, tail_, total_, seq_norm + total_, banachspace)
end

InfiniteSequence(sequence::Sequence{T,S}, finite_error::R, tail_error::R, total_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
    InfiniteSequence{T,S,R,U}(sequence, finite_error, tail_error, total_error, banachspace)

function InfiniteSequence(sequence::Sequence{<:SequenceSpace}, finite_error::Real, tail_error::Real, total_error::Real, banachspace::BanachSpace)
    fe, te, tote = promote(finite_error, tail_error, total_error)
    return InfiniteSequence(sequence, fe, te, tote, banachspace)
end

function InfiniteSequence(sequence::Sequence{<:SequenceSpace}, banachspace::BanachSpace; finite_error = nothing, tail_error = nothing, total_error = nothing)
    R_zero = zero(real(eltype(sequence)))
    if total_error !== nothing
        fe = something(finite_error, total_error)
        te = something(tail_error, total_error)
        return InfiniteSequence(sequence, fe, te, total_error, banachspace)
    else
        fe = something(finite_error, R_zero)
        te = something(tail_error, R_zero)
        return InfiniteSequence(sequence, fe, te, fe + te, banachspace)
    end
end

InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace) =
    InfiniteSequence(Sequence(space, coefficients), banachspace)

InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, finite_error::Real, tail_error::Real, total_error::Real, banachspace::BanachSpace) =
    InfiniteSequence(Sequence(space, coefficients), finite_error, tail_error, total_error, banachspace)

_iscompatbanachspace(::SequenceSpace, ::BanachSpace) = false
_iscompatbanachspace(::SequenceSpace, ::Ell1{<:Weight}) = true
_iscompatbanachspace(::SequenceSpace, ::Ell2{<:Weight}) = true
_iscompatbanachspace(::SequenceSpace, ::EllInf{<:Weight}) = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::Ell1{<:NTuple{N,Weight}}) where {N} = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::Ell2{<:NTuple{N,Weight}}) where {N} = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::EllInf{<:NTuple{N,Weight}}) where {N} = true

sequence(a::InfiniteSequence) = a.sequence
sequence_norm(a::InfiniteSequence) = a.sequence_norm
finite_error(a::InfiniteSequence) = a.finite_error
tail_error(a::InfiniteSequence) = a.tail_error
total_error(a::InfiniteSequence) = a.total_error
sequence_error(a::InfiniteSequence) = min(finite_error(a) + tail_error(a), total_error(a))
banachspace(a::InfiniteSequence) = a.banachspace

space(a::InfiniteSequence) = space(sequence(a)) # needed for general methods

coefficients(a::InfiniteSequence) = coefficients(sequence(a)) # needed for general methods

# utilities

Base.eltype(a::InfiniteSequence) = eltype(coefficients(a))
Base.eltype(::Type{<:InfiniteSequence{<:SequenceSpace,T}}) where {T<:AbstractVector} = eltype(T)

Base.:(==)(a::InfiniteSequence, b::InfiniteSequence) = # by-pass default
    (sequence(a) == sequence(b)) & iszero(sequence_error(a)) & iszero(sequence_error(b))

Base.zero(a::InfiniteSequence) = InfiniteSequence(zero(sequence(a)), banachspace(a))
Base.one(a::InfiniteSequence) = InfiniteSequence(one(sequence(a)), banachspace(a))

Base.float(a::InfiniteSequence) =
    _unsafe_infinite_sequence(float(sequence(a)), float(sequence_norm(a)), float(finite_error(a)), float(tail_error(a)), float(total_error(a)), float(a.full_norm), banachspace(a))
for f ∈ (:complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::InfiniteSequence) =
        _unsafe_infinite_sequence($f(sequence(a)), sequence_norm(a), finite_error(a), tail_error(a), total_error(a), a.full_norm, banachspace(a))
end

Base.permutedims(a::InfiniteSequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    _unsafe_infinite_sequence(permutedims(sequence(a), σ), sequence_norm(a), finite_error(a), tail_error(a), total_error(a), a.full_norm, banachspace(a))

# show

function Base.show(io::IO, ::MIME"text/plain", a::InfiniteSequence)
    println(io, "Sequence in ", _prettystring(space(a), true), " with coefficients ", typeof(coefficients(a)), ":")
    Base.print_array(io, coefficients(a))
    println(io, "\nNorm of the truncated sequence: ", sequence_norm(a))
    println(io, "Finite error: ", finite_error(a))
    println(io, "Tail error: ", tail_error(a))
    println(io, "Total error: ", total_error(a))
    return print(io, "Banach space: ", _prettystring(banachspace(a)))
end

function Base.show(io::IO, a::InfiniteSequence)
    get(io, :compact, false) && return show(io, (coefficients(a), finite_error(a), tail_error(a), total_error(a), banachspace(a)))
    return print(io, "InfiniteSequence(", space(a), ", ", coefficients(a), ", ", sequence_norm(a), ", ", finite_error(a), ", ", tail_error(a), ", ", total_error(a), ", ", banachspace(a), ")")
end
