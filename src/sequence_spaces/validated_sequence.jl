struct ValidatedSequence{T<:SequenceSpace,R<:Interval,S<:AbstractVector{<:Union{R,Complex{R}}},U<:BanachSpace} <: AbstractSequence{T,S}
    sequence :: Sequence{T,S}
    sequence_norm :: R
    sequence_error :: R
    banachspace :: U
    global _unsafe_validated_sequence(sequence::Sequence{T,S}, sequence_norm::R, sequence_error::R, banachspace::U) where {T<:SequenceSpace,R<:Interval,S<:AbstractVector{<:Union{R,Complex{R}}},U<:BanachSpace} =
        new{T,R,S,U}(sequence, sequence_norm, sequence_error, banachspace)
end

function ValidatedSequence(sequence::Sequence{T,S}, sequence_error::R, banachspace::U) where {T<:SequenceSpace,R<:Interval,S<:AbstractVector{<:Union{R,Complex{R}}},U<:BanachSpace}
    if isempty_interval(sequence_error)
        seq = fill(emptyinterval(sequence_error), space(sequence))
        sequence_norm = emptyinterval(sequence_error)
        return _unsafe_validated_sequence(seq, sequence_norm, sequence_error, banachspace)
    elseif (inf(sequence_error) ≥ 0) & isbounded(sequence_error)
        sequence_norm = norm(sequence, banachspace)
        return _unsafe_validated_sequence(sequence, sequence_norm, sequence_error, banachspace)
    else
        return throw(ArgumentError("truncation error must be positive and finite"))
    end
end

ValidatedSequence(sequence::Sequence, banachspace::BanachSpace) =
    ValidatedSequence(sequence, interval(zero(real(eltype(sequence)))), banachspace)

ValidatedSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace) =
    ValidatedSequence(Sequence(space, coefficients), banachspace)

sequence(a::ValidatedSequence) = a.sequence
sequence_norm(a::ValidatedSequence) = a.sequence_norm
sequence_error(a::ValidatedSequence) = a.sequence_error
banachspace(a::ValidatedSequence) = a.banachspace

# needed for general methods

space(a::ValidatedSequence) = space(sequence(a))

coefficients(a::ValidatedSequence) = coefficients(sequence(a))

# utilities

Base.copy(a::ValidatedSequence) =
    _unsafe_validated_sequence(copy(sequence(a)), sequence_norm(a), sequence_error(a), banachspace(a))

Base.zero(a::ValidatedSequence) = ValidatedSequence(zero(sequence(a)), banachspace(a))
# Base.one(a::ValidatedSequence) = ValidatedSequence(one(sequence(a)), banachspace(a))

for f ∈ (:float, :complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::ValidatedSequence) = ValidatedSequence($f(sequence(a)), sequence_error(a), banachspace(a))
end

Base.permutedims(a::ValidatedSequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    _unsafe_validated_sequence(permutedims(sequence(a), σ), sequence_norm(a), sequence_error(a), banachspace(a))

# show

function Base.show(io::IO, ::MIME"text/plain", a::ValidatedSequence)
    println(io, "Sequence in ", _prettystring(space(a)), " with coefficients ", typeof(coefficients(a)), ":")
    Base.print_array(io, coefficients(a))
    println(io, "\nNorm of the truncated sequence: ", sequence_norm(a))
    println(io, "Truncation error: ", sequence_error(a))
    return print(io, "Banach space: ", _prettystring(banachspace(a)))
end

function Base.show(io::IO, a::ValidatedSequence)
    get(io, :compact, false) && return show(io, (coefficients(a), sequence_error(a), banachspace(a)))
    return print(io, "ValidatedSequence(", space(a), ", ", coefficients(a), ", ", sequence_norm(a), ", ", sequence_error(a), ", ", banachspace(a), ")")
end

#

_banachspace_identity(::Ell1) = Ell1()
_banachspace_identity(::Ell2) = Ell2()

_banachspace_intersect(X₁::Ell1, X₂::Ell1) = Ell1(_weight_min(weight(X₁), weight(X₂)))
_banachspace_intersect(X₁::Ell2, X₂::Ell2) = Ell2(_weight_min(weight(X₁), weight(X₂)))

_weight_min(::IdentityWeight, ::IdentityWeight) = IdentityWeight()
_weight_min(::IdentityWeight, ::Weight) = IdentityWeight()
_weight_min(::Weight, ::IdentityWeight) = IdentityWeight()
_weight_min(w₁::GeometricWeight, w₂::GeometricWeight) = GeometricWeight(min(rate(w₁), rate(w₂)))
_weight_min(w₁::AlgebraicWeight, w₂::AlgebraicWeight) = AlgebraicWeight(min(rate(w₁), rate(w₂)))
_weight_min(w₁::BesselWeight, w₂::BesselWeight) = BesselWeight(min(rate(w₁), rate(w₂)))
_weight_min(w₁::NTuple{N,Weight}, w₂::NTuple{N,Weight}) where {N} = map(_weight_min, w₁, w₂)

# arithmetic

Base.:+(a::ValidatedSequence) = ValidatedSequence(+(sequence(a)), sequence_error(a), banachspace(a))
Base.:+(a::ValidatedSequence, b::ValidatedSequence) =
    ValidatedSequence(sequence(a) + sequence(b), sequence_error(a) + sequence_error(b), _banachspace_intersect(banachspace(a), banachspace(b)))
Base.:+(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) + b, sequence_error(a), banachspace(a))
Base.:+(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a + sequence(b), sequence_error(b), banachspace(b))

Base.:-(a::ValidatedSequence) = ValidatedSequence(-(sequence(a)), sequence_error(a), banachspace(a))
Base.:-(a::ValidatedSequence, b::ValidatedSequence) =
    ValidatedSequence(sequence(a) - sequence(b), sequence_error(a) + sequence_error(b), _banachspace_intersect(banachspace(a), banachspace(b)))
Base.:-(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) - b, sequence_error(a), banachspace(a))
Base.:-(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a - sequence(b), sequence_error(b), banachspace(b))

function Base.:*(a::ValidatedSequence, b::ValidatedSequence)
    X = _banachspace_intersect(banachspace(a), banachspace(b))
    full_c = sequence(a) * sequence(b) # banach_rounding_mul(sequence(a), sequence(b), X)
    c = project(full_c, space(a) ∪ space(b))
    @inbounds view(full_c, indices(space(c))) .= 0
    return ValidatedSequence(c, norm(full_c, X) +
            sequence_norm(a) * sequence_error(b) +
            sequence_norm(b) * sequence_error(a) +
            sequence_error(a) * sequence_error(b),
        X)
end
Base.:*(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) * b, abs(sequence_error(a) * b), banachspace(a))
Base.:*(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a * sequence(b), abs(a * sequence_error(b)), banachspace(b))

function Base.:^(a::ValidatedSequence, n::Integer)
    n < 0 && return inv(a^(-n))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return a*a
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a *= a
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a *= a
        end
        c = c * a
    end
    return c
end

function Base.inv(a::ValidatedSequence)
    space_approx = _image_trunc(inv, space(a))
    A = fft(mid.(sequence(a)), fft_size(space_approx))
    A⁻¹ = inv.(A)
    seq_approx_a⁻¹ = _call_ifft!(A⁻¹, space_approx, eltype(a))

    X = banachspace(a)
    approx_a⁻¹ = ValidatedSequence(interval.(seq_approx_a⁻¹), X)

    f = approx_a⁻¹ * a - interval(real(eltype(seq_approx_a⁻¹)), 1)
    Y = norm(approx_a⁻¹ * f)
    Z₁ = norm(f)
    r = interval_of_existence(Y, Z₁, Inf; verbose = false)

    isempty_interval(r) || return ValidatedSequence(sequence(approx_a⁻¹), interval(real(eltype(seq_approx_a⁻¹)), inf(r)), X)
    fill!(sequence(approx_a⁻¹), emptyinterval(eltype(seq_approx_a⁻¹)))
    return ValidatedSequence(sequence(approx_a⁻¹), emptyinterval(real(eltype(seq_approx_a⁻¹))), X)
end

function Base.:/(a::ValidatedSequence, b::ValidatedSequence)
    space_approx = _image_trunc(/, space(a), space(b))
    A = fft(mid.(sequence(a)), fft_size(space_approx))
    B = fft(mid.(sequence(b)), fft_size(space_approx))
    AB⁻¹ = A ./ B
    B⁻¹ = inv.(B)
    seq_approx_ab⁻¹ = _call_ifft!(AB⁻¹, space_approx, promote_type(eltype(a), eltype(b)))
    seq_approx_b⁻¹ = _call_ifft!(B⁻¹, space_approx, eltype(b))

    X = _banachspace_intersect(banachspace(a), banachspace(b))
    approx_ab⁻¹ = ValidatedSequence(interval.(seq_approx_ab⁻¹), X)
    approx_b⁻¹ = ValidatedSequence(interval.(seq_approx_b⁻¹), X)

    Y = norm(approx_b⁻¹ * (approx_ab⁻¹ * b - a))
    Z₁ = norm(approx_b⁻¹ * b - interval(real(eltype(seq_approx_ab⁻¹)), 1))
    r = interval_of_existence(Y, Z₁, Inf; verbose = false)

    isempty_interval(r) || return ValidatedSequence(sequence(approx_ab⁻¹), interval(real(eltype(seq_approx_ab⁻¹)), inf(r)), X)
    fill!(sequence(approx_ab⁻¹), emptyinterval(eltype(seq_approx_ab⁻¹)))
    return ValidatedSequence(sequence(approx_ab⁻¹), emptyinterval(real(eltype(seq_approx_ab⁻¹))), X)
end
Base.:/(a::ValidatedSequence, b::Number) = ValidatedSequence(sequence(a) / b, abs(sequence_error(a) / b), banachspace(a))
Base.:/(a::Number, b::ValidatedSequence) = a * inv(b)

Base.:\(a::ValidatedSequence, b::ValidatedSequence) = b / a
Base.:\(a::ValidatedSequence, b::Number) = b / a
Base.:\(a::Number, b::ValidatedSequence) = b / a

function Base.sqrt(a::ValidatedSequence)
    space_approx = _image_trunc(sqrt, space(a))
    A = fft(mid.(sequence(a)), fft_size(space_approx))
    sqrtA = sqrt.(A)
    sqrtA⁻¹ = inv.(sqrtA)
    seq_approx_sqrta = _call_ifft!(sqrtA, space_approx, eltype(a))
    seq_approx_sqrta⁻¹ = _call_ifft!(sqrtA⁻¹, space_approx, eltype(a))

    X = banachspace(a)
    approx_sqrta = ValidatedSequence(interval.(seq_approx_sqrta), X)
    approx_sqrta⁻¹ = ValidatedSequence(interval.(seq_approx_sqrta⁻¹), X)

    Y = norm(approx_sqrta⁻¹ * (approx_sqrta ^ 2 - a))/interval(real(eltype(seq_approx_sqrta)), 2)
    Z₁ = norm(approx_sqrta⁻¹ * approx_sqrta - interval(real(eltype(seq_approx_sqrta)), 1))
    Z₂ = norm(approx_sqrta⁻¹)
    r = interval_of_existence(Y, Z₁, Z₂, Inf; verbose = false)

    isempty_interval(r) || return ValidatedSequence(sequence(approx_sqrta), interval(real(eltype(seq_approx_sqrta)), inf(r)), X)
    fill!(sequence(approx_sqrta), emptyinterval(eltype(seq_approx_sqrta)))
    return ValidatedSequence(sequence(approx_sqrta), emptyinterval(real(eltype(seq_approx_sqrta))), X)
end

Base.abs(a::ValidatedSequence) = sqrt(a^2)
Base.abs2(a::ValidatedSequence) = a^2

# special operators

# TODO: projection, integration, etc.

function differentiate(a::ValidatedSequence, α=1)
    sequence_error(a) == 0 || return throw(DomainError) # TODO: lift restriction
    c = differentiate(sequence(a), α)
    return ValidatedSequence(c, banachspace(a))
end

evaluate(a::ValidatedSequence, x) = _return_evaluate(evaluate(sequence(a), x), a)
_return_evaluate(c, a::ValidatedSequence) = interval(c, sequence_error(a); format = :midpoint)
function _return_evaluate(c::Sequence, a::ValidatedSequence)
    c .= interval.(c, sequence_error(a); format = :midpoint)
    return ValidatedSequence(c, sequence_error(a), banachspace(a))
end

# norm

LinearAlgebra.norm(a::ValidatedSequence) = sequence_norm(a) + sequence_error(a)
