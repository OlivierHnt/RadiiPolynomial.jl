"""
    InfiniteSequence{T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence

Infinite sequence in the given sequence space, with error and norm bookkeeping.

Fields:
- `sequence :: Sequence{T,S}`: finite truncation.
- `sequence_norm :: R`: norm of `sequence` in `banachspace`.
- `finite_error :: R`
- `tail_error :: R`
- `full_norm :: R`: tight upper bound on `‖a_∞‖_X`.
- `banachspace :: U`

Constructors:
- `InfiniteSequence(sequence, finite_error, tail_error, banachspace)`
- `InfiniteSequence(sequence, banachspace; finite_error = 0, tail_error = 0)`
- `InfiniteSequence(sequence, banachspace)`: both errors are zero.
- `InfiniteSequence(space, coefficients, banachspace)`
- `InfiniteSequence(space, coefficients, finite_error, tail_error, banachspace)`

# Example

```jldoctest
julia> InfiniteSequence(Sequence(Taylor(2), [1.0, 2.0, 1.0]), 0.0, 0.1, Ell1())
Sequence in Taylor(2) with coefficients Vector{Float64}:
 1.0
 2.0
 1.0
Norm of the truncated sequence: 4.0
Finite error: 0.0
Tail error: 0.1
Banach space: ℓ¹()
```
"""
struct InfiniteSequence{T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence
    sequence :: Sequence{T,S}
    sequence_norm :: R
    finite_error :: R
    tail_error :: R
    full_norm :: R
    banachspace :: U
    global _unsafe_infinite_sequence(sequence::Sequence{T,S}, sequence_norm::R, finite_error::R, tail_error::R, full_norm::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
        new{T,S,R,U}(sequence, sequence_norm, finite_error, tail_error, full_norm, banachspace)
end

function InfiniteSequence{T,S,R,U}(sequence::Sequence{T,S}, finite_error::R, tail_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace}
    _iscompatbanachspace(space(sequence), banachspace) || return throw(ArgumentError("invalid norm for the sequence space"))
    (inf(finite_error) ≥ 0) & (inf(tail_error) ≥ 0) || return throw(ArgumentError("errors must be non-negative"))
    seq_norm = convert(R, norm(sequence, banachspace))
    return _unsafe_infinite_sequence(sequence, seq_norm, finite_error, tail_error, seq_norm + finite_error + tail_error, banachspace)
end

InfiniteSequence(sequence::Sequence{T,S}, finite_error::R, tail_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
    InfiniteSequence{T,S,R,U}(sequence, finite_error, tail_error, banachspace)

function InfiniteSequence(sequence::Sequence{<:SequenceSpace}, finite_error::Real, tail_error::Real, banachspace::BanachSpace)
    fe, te = promote(finite_error, tail_error)
    return InfiniteSequence(sequence, fe, te, banachspace)
end

function InfiniteSequence(sequence::Sequence{<:SequenceSpace}, banachspace::BanachSpace; finite_error::Real = zero(real(eltype(sequence))), tail_error::Real = zero(real(eltype(sequence))))
    return InfiniteSequence(sequence, finite_error, tail_error, banachspace)
end

InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace) =
    InfiniteSequence(Sequence(space, coefficients), banachspace)

InfiniteSequence(space::SequenceSpace, coefficients::AbstractVector, finite_error::Real, tail_error::Real, banachspace::BanachSpace) =
    InfiniteSequence(Sequence(space, coefficients), finite_error, tail_error, banachspace)

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
sequence_error(a::InfiniteSequence) = finite_error(a) + tail_error(a)
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
    _unsafe_infinite_sequence(float(sequence(a)), float(sequence_norm(a)), float(finite_error(a)), float(tail_error(a)), float(a.full_norm), banachspace(a))
for f ∈ (:complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::InfiniteSequence) =
        _unsafe_infinite_sequence($f(sequence(a)), sequence_norm(a), finite_error(a), tail_error(a), a.full_norm, banachspace(a))
end

Base.permutedims(a::InfiniteSequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    _unsafe_infinite_sequence(permutedims(sequence(a), σ), sequence_norm(a), finite_error(a), tail_error(a), a.full_norm, banachspace(a))

# show

function Base.show(io::IO, ::MIME"text/plain", a::InfiniteSequence)
    println(io, "Sequence in ", _prettystring(space(a), true), " with coefficients ", typeof(coefficients(a)), ":")
    Base.print_array(io, coefficients(a))
    println(io, "\nNorm of the truncated sequence: ", sequence_norm(a))
    println(io, "Finite error: ", finite_error(a))
    println(io, "Tail error: ", tail_error(a))
    return print(io, "Banach space: ", _prettystring(banachspace(a)))
end

function Base.show(io::IO, a::InfiniteSequence)
    get(io, :compact, false) && return show(io, (coefficients(a), finite_error(a), tail_error(a), banachspace(a)))
    return print(io, "InfiniteSequence(", space(a), ", ", coefficients(a), ", ", sequence_norm(a), ", ", finite_error(a), ", ", tail_error(a), ", ", banachspace(a), ")")
end
