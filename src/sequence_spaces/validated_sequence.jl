_check_interval_space(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    _check_interval_space(s₁[1], s₂[1]) & _check_interval_space(Base.tail(s₁), Base.tail(s₂))
_check_interval_space(s₁::TensorSpace{<:Tuple{BaseSpace}}, s₂::TensorSpace{<:Tuple{BaseSpace}}) =
    _check_interval_space(s₁[1], s₂[1])
_check_interval_space(::Taylor) = true
_check_interval_space(::Fourier{<:Interval}) = true
_check_interval_space(::Chebyshev) = true
_check_interval_space(::CosFourier{<:Interval}) = true
_check_interval_space(::SinFourier{<:Interval}) = true
_check_interval_space(::Fourier) = throw(ArgumentError("the frequency of the Fourier space must be an interval"))
_check_interval_space(::CosFourier) = throw(ArgumentError("the frequency of the CosFourier space must be an interval"))
_check_interval_space(::SinFourier) = throw(ArgumentError("the frequency of the SinFourier space must be an interval"))

_check_interval_banachspace(X::Union{Ell1,Ell2,EllInf}) = _check_interval_weight(weight(X))
_check_interval_weight(::IdentityWeight) = true
_check_interval_weight(::GeometricWeight{<:Interval}) = true
_check_interval_weight(::AlgebraicWeight{<:Interval}) = true
_check_interval_weight(::BesselWeight{<:Interval}) = true
_check_interval_weight(::GeometricWeight) = throw(ArgumentError("the geometric weight of the Banach space must be an interval"))
_check_interval_weight(::AlgebraicWeight) = throw(ArgumentError("the algebraic weight of the Banach space must be an interval"))
_check_interval_weight(::BesselWeight) = throw(ArgumentError("the Bessel weight of the Banach space must be an interval"))

struct ValidatedSequence{T<:SequenceSpace,S<:AbstractVector{<:Union{Interval,Complex{<:Interval}}},R<:Interval,U<:BanachSpace} <: AbstractSequence{T,S}
    sequence :: Sequence{T,S}
    sequence_norm :: R
    truncation_error :: R
    banachspace :: U
    global _unsafe_validated_sequence(sequence::Sequence{T,S}, sequence_norm::R, truncation_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector{<:Union{Interval,Complex{<:Interval}}},R<:Interval,U<:BanachSpace} =
        new{T,S,R,U}(sequence, sequence_norm, truncation_error, banachspace)
end

function ValidatedSequence{T,S,R,U}(sequence::Sequence{T,S}, truncation_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector{<:Union{Interval,Complex{<:Interval}}},R<:Interval,U<:BanachSpace}
    _check_interval_space(space(sequence))
    _check_interval_banachspace(banachspace)
    if isempty_interval(truncation_error)
        seq = fill(emptyinterval(truncation_error), space(sequence))
        sequence_norm = emptyinterval(truncation_error)
        return _unsafe_validated_sequence(seq, sequence_norm, truncation_error, banachspace)
    elseif (inf(truncation_error) ≥ 0) & isbounded(truncation_error)
        sequence_norm = norm(sequence, banachspace)
        return _unsafe_validated_sequence(sequence, sequence_norm, truncation_error, banachspace)
    else
        return throw(ArgumentError("truncation error must be positive and finite"))
    end
end

ValidatedSequence(sequence::Sequence{T,S}, truncation_error::R, banachspace::U) where {T<:SequenceSpace,S<:AbstractVector,R<:Interval,U<:BanachSpace} =
    ValidatedSequence{T,S,R,U}(sequence, truncation_error, banachspace)

ValidatedSequence(sequence::Sequence, banachspace::BanachSpace) =
    ValidatedSequence(sequence, abs(zero(eltype(sequence))), banachspace)

ValidatedSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace) =
    ValidatedSequence(Sequence(space, coefficients), banachspace)

sequence(a::ValidatedSequence) = a.sequence
sequence_norm(a::ValidatedSequence) = a.sequence_norm
truncation_error(a::ValidatedSequence) = a.truncation_error
banachspace(a::ValidatedSequence) = a.banachspace

# needed for general methods

space(a::ValidatedSequence) = space(sequence(a))

coefficients(a::ValidatedSequence) = coefficients(sequence(a))

# utilities

Base.copy(a::ValidatedSequence) =
    _unsafe_validated_sequence(copy(sequence(a)), sequence_norm(a), truncation_error(a), banachspace(a))

Base.zero(a::ValidatedSequence) = ValidatedSequence(zero(sequence(a)), banachspace(a))
Base.one(a::ValidatedSequence) = ValidatedSequence(one(sequence(a)), banachspace(a))

for f ∈ (:float, :complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::ValidatedSequence) = ValidatedSequence($f(sequence(a)), truncation_error(a), banachspace(a))
end

Base.permutedims(a::ValidatedSequence{<:TensorSpace}, σ::AbstractVector{<:Integer}) =
    _unsafe_validated_sequence(permutedims(sequence(a), σ), sequence_norm(a), truncation_error(a), banachspace(a))

# show

function Base.show(io::IO, ::MIME"text/plain", a::ValidatedSequence)
    println(io, "Sequence in ", _prettystring(space(a)), " with coefficients ", typeof(coefficients(a)), ":")
    Base.print_array(io, coefficients(a))
    println(io, "\nNorm of the truncated sequence: ", sequence_norm(a))
    println(io, "Truncation error: ", truncation_error(a))
    return print(io, "Banach space: ", _prettystring(banachspace(a)))
end

function Base.show(io::IO, a::ValidatedSequence)
    get(io, :compact, false) && return show(io, (coefficients(a), truncation_error(a), banachspace(a)))
    return print(io, "ValidatedSequence(", space(a), ", ", coefficients(a), ", ", sequence_norm(a), ", ", truncation_error(a), ", ", banachspace(a), ")")
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

# arithmetic

Base.:+(a::ValidatedSequence) = ValidatedSequence(+(sequence(a)), truncation_error(a), banachspace(a))
Base.:+(a::ValidatedSequence, b::ValidatedSequence) =
    ValidatedSequence(sequence(a) + sequence(b), truncation_error(a) + truncation_error(b), _banachspace_intersect(banachspace(a), banachspace(b)))
Base.:+(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) + b, truncation_error(a), banachspace(a))
Base.:+(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a + sequence(b), truncation_error(b), banachspace(b))

Base.:-(a::ValidatedSequence) = ValidatedSequence(-(sequence(a)), truncation_error(a), banachspace(a))
Base.:-(a::ValidatedSequence, b::ValidatedSequence) =
    ValidatedSequence(sequence(a) - sequence(b), truncation_error(a) + truncation_error(b), _banachspace_intersect(banachspace(a), banachspace(b)))
Base.:-(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) - b, truncation_error(a), banachspace(a))
Base.:-(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a - sequence(b), truncation_error(b), banachspace(b))

function Base.:*(a::ValidatedSequence, b::ValidatedSequence)
    X = _banachspace_intersect(banachspace(a), banachspace(b))
    c = sequence(a) * sequence(b) # banach_rounding_mul(sequence(a), sequence(b), X)
    return ValidatedSequence(c,
            sequence_norm(a) * truncation_error(b) +
            sequence_norm(b) * truncation_error(a) +
            truncation_error(a) * truncation_error(b),
        X)
end
Base.:*(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) * b, abs(truncation_error(a) * b), banachspace(a))
Base.:*(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a * sequence(b), abs(a * truncation_error(b)), banachspace(b))

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
    space_c = _image_trunc(inv, space(a))
    A = fft(mid.(sequence(a)), fft_size(space_c))
    C = inv.(A)
    c = _call_ifft!(C, space_c, eltype(a))
    X_ = banachspace(a)
    X = _banachspace_intersect(X_, _banachspace_identity(X_))
    _c_ = ValidatedSequence(interval.(c), X)

    val = _c_ * a - interval(real(eltype(c)), 1)
    Y = norm(_c_ * val)
    Z₁ = norm(val)
    r = interval_of_existence(Y, Z₁, Inf)

    isempty_interval(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), interval(inf(r)), X)
end

function Base.:/(a::ValidatedSequence, b::ValidatedSequence)
    space_c = _image_trunc(/, space(a), space(b))
    A = fft(mid.(sequence(a)), fft_size(space_c))
    B = fft(mid.(sequence(b)), fft_size(space_c))
    C = A ./ B
    c = _call_ifft!(C, space_c, promote_type(eltype(a), eltype(b)))
    B⁻¹ = inv.(B)
    approx_b⁻¹ = _call_ifft!(B⁻¹, space(b), eltype(b))
    X = _banachspace_intersect(banachspace(a), banachspace(b))
    _c_ = ValidatedSequence(interval.(c), X)
    _approx_b⁻¹_ = ValidatedSequence(interval.(approx_b⁻¹), X)

    Y = norm(_approx_b⁻¹_ * (_c_ * b - a))
    Z₁ = norm(_approx_b⁻¹_ * b - interval(real(eltype(c)), 1))
    r = interval_of_existence(Y, Z₁, Inf)

    isempty_interval(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), interval(inf(r)), X)
end
Base.:/(a::ValidatedSequence, b::Number) = ValidatedSequence(sequence(a) / b, abs(truncation_error(a) / b), banachspace(a))
function Base.:/(a::Number, b::ValidatedSequence)
    space_c = _image_trunc(inv, space(b))
    B = fft(mid.(sequence(b)), fft_size(space_c))
    C = mid(a) ./ B
    c = _call_ifft!(C, space_c, promote_type(typeof(a), eltype(b)))
    B⁻¹ = inv.(B)
    approx_b⁻¹ = _call_ifft!(B⁻¹, space_c, eltype(b))
    X = banachspace(b)
    _c_ = ValidatedSequence(interval.(c), X)
    _approx_b⁻¹_ = ValidatedSequence(interval.(approx_b⁻¹), X)

    Y = norm(_approx_b⁻¹_ * (_c_ * b - a))
    Z₁ = norm(_approx_b⁻¹_ * b - interval(real(eltype(c)), 1))
    r = interval_of_existence(Y, Z₁, Inf)

    isempty_interval(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), interval(inf(r)), X)
end

Base.:\(a::ValidatedSequence, b::ValidatedSequence) = b / a
Base.:\(a::ValidatedSequence, b::Number) = b / a
Base.:\(a::Number, b::ValidatedSequence) = b / a

function Base.sqrt(a::ValidatedSequence)
    space_c = _image_trunc(sqrt, space(a))
    A = fft(mid.(sequence(a)), fft_size(space_c))
    C = sqrt.(A)
    C⁻¹ = inv.(C)
    c = _call_ifft!(C, space_c, eltype(a))
    approx_c⁻¹ = _call_ifft!(C⁻¹, space_c, eltype(a))
    X = banachspace(a)
    _c_ = ValidatedSequence(interval.(c), X)
    _approx_c⁻¹_ = ValidatedSequence(interval.(approx_c⁻¹), X)

    Y = norm(_approx_c⁻¹_ * (_c_ ^ interval(real(eltype(c)), 2) - a))/interval(eltype(c), 2)
    Z₁ = norm(_approx_c⁻¹_ * _c_ - interval(real(eltype(c)), 1))
    Z₂ = norm(_approx_c⁻¹_)
    r = interval_of_existence(Y, Z₁, Z₂, Inf)

    isempty_interval(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), interval(inf(r)), X)
end

Base.abs(a::ValidatedSequence) = sqrt(a^2)
Base.abs2(a::ValidatedSequence) = a^2

# special operators

# TODO: projection, integration, etc.

function differentiate(a::ValidatedSequence, α=1)
    truncation_error(a) == 0 || return throw(DomainError) # TODO: lift restriction
    c = differentiate(sequence(a), α)
    return ValidatedSequence(c, banachspace(a))
end

evaluate(a::ValidatedSequence, x) = _return_evaluate(evaluate(sequence(a), x), a)
_return_evaluate(c, a::ValidatedSequence) = interval(c, truncation_error(a); format = :midpoint)
function _return_evaluate(c::Sequence, a::ValidatedSequence)
    c .= interval.(c, truncation_error(a); format = :midpoint)
    return ValidatedSequence(c, truncation_error(a), banachspace(a))
end

# norm

LinearAlgebra.norm(a::ValidatedSequence) = sequence_norm(a) + truncation_error(a)
