_to_interval(a::Sequence{<:VectorSpace,<:AbstractVector{<:Interval}}) = a
_to_interval(a::Sequence{<:VectorSpace,<:AbstractVector{<:Complex{<:Interval}}}) = a
_to_interval(a::Sequence) = Sequence(_to_interval(space(a)), Interval.(coefficients(a)))

_to_interval(s::VectorSpace) = s
_to_interval(s::TensorSpace) = TensorSpace(map(_to_interval, spaces(s)))
_to_interval(s::CartesianPower) = CartesianPower(_to_interval(space(s)), nspaces(s))
_to_interval(s::CartesianProduct) = CartesianProduct(map(_to_interval, spaces(s)))
_to_interval(s::Fourier) = Fourier(order(s), _to_interval(frequency(s)))

_to_interval(a::Interval) = a
_to_interval(a::Real) = Interval(a)

_to_interval(X::NormedCartesianSpace) = NormedCartesianSpace(map(_to_interval, X.inner), _to_interval(X.outer))
_to_interval(X::Ell1{IdentityWeight}) = X
_to_interval(X::Ell1{<:GeometricWeight{<:Interval}}) = X
_to_interval(X::Ell1{<:GeometricWeight}) = Ell1(GeometricWeight(Interval(X.weight.rate)))

struct ValidatedSequence{T<:VectorSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} <: AbstractSequence{T,S}
    sequence :: Sequence{T,S}
    sequence_norm :: R
    truncation_error :: R
    banachspace :: U
    global _unsafe_validated_sequence(sequence::Sequence{T,S}, sequence_norm::R, truncation_error::R, banachspace::U) where {T<:VectorSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
        new{T,S,R,U}(sequence, sequence_norm, truncation_error, banachspace)
end

function ValidatedSequence{T,S,R,U}(sequence::Sequence{T,S}, truncation_error::R, banachspace::U) where {T<:VectorSpace,S<:AbstractVector,R<:Real,U<:BanachSpace}
    ((truncation_error ≥ 0) & isfinite(truncation_error)) || return throw(ArgumentError("truncation error must be positive and finite"))
    seq = _to_interval(sequence)
    trunc = _to_interval(truncation_error)
    X = _to_interval(banachspace)
    sequence_norm = norm(seq, X)
    return _unsafe_validated_sequence(seq, sequence_norm, trunc, X)
end

function ValidatedSequence{T,S,R,U}(sequence::Sequence{T,S}, truncation_error::R, banachspace::U) where {T<:VectorSpace,S<:AbstractVector,R<:Interval,U<:BanachSpace}
    if isempty(truncation_error)
        seq = fill(emptyinterval(truncation_error), space(sequence))
        sequence_norm = emptyinterval(truncation_error)
        return _unsafe_validated_sequence(seq, sequence_norm, truncation_error, _to_interval(banachspace))
    elseif (truncation_error ≥ 0) & isfinite(truncation_error)
        seq = _to_interval(sequence)
        X = _to_interval(banachspace)
        sequence_norm = norm(seq, X)
        return _unsafe_validated_sequence(seq, sequence_norm, truncation_error, X)
    else
        return throw(ArgumentError("truncation error must be positive and finite"))
    end
end

ValidatedSequence(sequence::Sequence{T,S}, truncation_error::R, banachspace::U) where {T<:VectorSpace,S<:AbstractVector,R<:Real,U<:BanachSpace} =
    ValidatedSequence{T,S,R,U}(sequence, truncation_error, banachspace)

ValidatedSequence(sequence::Sequence, banachspace::BanachSpace) =
    ValidatedSequence(sequence, zero(eltype(sequence)), banachspace)

ValidatedSequence(space::VectorSpace, coefficients::AbstractVector, X::BanachSpace) =
    ValidatedSequence(Sequence(space, coefficients), X)

ValidatedSequence(coefficients::AbstractVector, X::BanachSpace) =
    ValidatedSequence(Sequence(coefficients), X)

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

Base.one(a::ValidatedSequence{ParameterSpace}) =
    _unsafe_validated_sequence(one(sequence(a)), one(eltype(a)), zero(eltype(a)), banachspace(a))

for f ∈ (:float, :complex, :real, :imag, :conj, :conj!)
    @eval Base.$f(a::ValidatedSequence) = ValidatedSequence($f(sequence(a)), truncation_error(a), banachspace(a))
end

Base.permutedims(a::ValidatedSequence{<:TensorSpace}, σ::AbstractVector{Int}) =
    _unsafe_validated_sequence(permutedims(sequence(a), σ), sequence_norm(a), truncation_error(a), banachspace(a))

Base.@propagate_inbounds component(a::ValidatedSequence{<:CartesianSpace}, i) =
    ValidatedSequence(component(sequence(a), i), truncation_error(a), banachspace(a))

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

# Arithmetic

_banachspace_intersect(X₁::Ell1, X₂::Ell1) = Ell1(_weight_min(X₁.weight, X₂.weight))
_banachspace_intersect(X₁::Ell2, X₂::Ell2) = Ell2(_weight_min(X₁.weight, X₂.weight))
_banachspace_intersect(X₁::EllInf, X₂::EllInf) = EllInf(_weight_min(X₁.weight, X₂.weight))

_weight_min(::IdentityWeight, ::IdentityWeight) = IdentityWeight()
_weight_min(::IdentityWeight, ::Weight) = IdentityWeight()
_weight_min(::Weight, ::IdentityWeight) = IdentityWeight()
_weight_min(w₁::GeometricWeight, w₂::GeometricWeight) = GeometricWeight(min(w₁.rate, w₂.rate))
_weight_min(w₁::AlgebraicWeight, w₂::AlgebraicWeight) = AlgebraicWeight(min(w₁.rate, w₂.rate))
_weight_min(w₁::BesselWeight, w₂::BesselWeight) = BesselWeight(min(w₁.rate, w₂.rate))

Base.:+(a::ValidatedSequence, b::ValidatedSequence) =
    ValidatedSequence(sequence(a) + sequence(b), truncation_error(a) + truncation_error(b), _banachspace_intersect(banachspace(a), banachspace(b)))
Base.:+(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) + b, truncation_error(a), banachspace(a))
Base.:+(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a + sequence(b), truncation_error(b), banachspace(b))

Base.:-(a::ValidatedSequence, b::ValidatedSequence) =
    ValidatedSequence(sequence(a) - sequence(b), truncation_error(a) + truncation_error(b), _banachspace_intersect(banachspace(a), banachspace(b)))
Base.:-(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) - b, truncation_error(a), banachspace(a))
Base.:-(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a - sequence(b), truncation_error(b), banachspace(b))

function Base.:*(a::ValidatedSequence, b::ValidatedSequence)
    c = sequence(a) * sequence(b)
    X = _banachspace_intersect(banachspace(a), banachspace(b))
    return ValidatedSequence(c,
            sequence_norm(a) * truncation_error(b) +
            sequence_norm(b) * truncation_error(a) +
            truncation_error(a) * truncation_error(b),
        X)
end
Base.:*(a::ValidatedSequence, b::Number) =
    ValidatedSequence(sequence(a) * b, truncation_error(a) * b, banachspace(a))
Base.:*(a::Number, b::ValidatedSequence) =
    ValidatedSequence(a * sequence(b), a * truncation_error(b), banachspace(b))

function Base.:^(a::ValidatedSequence, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
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

# Elementary operations

_call_ifft!(C, s, ::Type{<:Real}) = rifft!(C, s)
_call_ifft!(C, s, ::Type) = ifft!(C, s)

function Base.:/(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = space(a) ∪ space(b)
    A = fft(a, fft_size(space_c, 1))
    B = fft(b, fft_size(space_c, 1))
    C = A ./ B
    return _call_ifft!(C, space_c, promote_type(eltype(a), eltype(b)))
end

for f ∈ (:inv, :sqrt)
    @eval function Base.$f(a::Sequence{<:SequenceSpace})
        A = fft(a, fft_size(space(a), 1))
        C = $f.(A)
        return _call_ifft!(C, space(a), eltype(a))
    end
end

function Base.:/(a::ValidatedSequence{<:SequenceSpace}, b::ValidatedSequence{<:SequenceSpace})
    space_c = space(a) ∪ space(b)
    A = fft(mid.(sequence(a)), fft_size(space_c, 1))
    B = fft(mid.(sequence(b)), fft_size(space_c, 1))
    C = A ./ B
    c = _call_ifft!(C, space_c, promote_type(eltype(a), eltype(b)))
    B⁻¹ = inv.(B)
    approx_b⁻¹ = _call_ifft!(B⁻¹, space(b), eltype(b))
    X = _banachspace_intersect(banachspace(a), banachspace(b))
    _c_ = ValidatedSequence(c, X)
    _approx_b⁻¹_ = ValidatedSequence(approx_b⁻¹, X)

    Y = norm(_approx_b⁻¹_ * (_c_ * b - a))
    Z₁ = norm(_approx_b⁻¹_ * b - 1)
    r = interval_of_existence(Y, Z₁, Inf)

    isempty(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), Interval(inf(r)), X)
end
Base.:/(a::ValidatedSequence{<:SequenceSpace}, b::Number) = ValidatedSequence(sequence(a) / b, truncation_error(a) / b, banachspace(a))
Base.:/(a::Number, b::ValidatedSequence{<:SequenceSpace}) = a * inv(b)

Base.:\(a::ValidatedSequence{<:SequenceSpace}, b::ValidatedSequence{<:SequenceSpace}) = b / a
Base.:\(a::ValidatedSequence{<:SequenceSpace}, b::Number) = b / a
Base.:\(a::Number, b::ValidatedSequence{<:SequenceSpace}) = b / a

function Base.inv(a::ValidatedSequence{<:SequenceSpace})
    A = fft(mid.(sequence(a)), fft_size(space(a), 1))
    C = inv.(A)
    c = _call_ifft!(C, space(a), eltype(a))
    X = banachspace(a)
    _c_ = ValidatedSequence(c, X)

    val = _c_ * a - 1
    Y = norm(_c_ * val)
    Z₁ = norm(val)
    r = interval_of_existence(Y, Z₁, Inf)

    isempty(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), interval(inf(r)), X)
end

function Base.sqrt(a::ValidatedSequence{<:SequenceSpace})
    A = fft(mid.(sequence(a)), fft_size(space(a), 1))
    C = sqrt.(A)
    C⁻¹ = inv.(C)
    c = _call_ifft!(C, space(a), eltype(a))
    approx_c⁻¹ = _call_ifft!(C⁻¹, space(a), eltype(a))
    X = banachspace(a)
    _c_ = ValidatedSequence(c, X)
    _approx_c⁻¹_ = ValidatedSequence(approx_c⁻¹, X)

    Y = norm(_approx_c⁻¹_ * (_c_^2 - a))/2
    Z₁ = norm(_approx_c⁻¹_ * _c_ - 1)
    Z₂ = norm(_approx_c⁻¹_)
    r = interval_of_existence(Y, Z₁, Z₂, Inf)

    isempty(r) && return ValidatedSequence(emptyinterval.(sequence(_c_)), r, X)
    return ValidatedSequence(interval.(sequence(_c_), inf(r); format = :midpoint), interval(inf(r)), X)
end

# projection

function project(a::ValidatedSequence, space_dest::VectorSpace, ::Type{T}=eltype(a)) where {T}
    c = project(sequence(a), space_dest)
    X = banachspace(a)
    banach_rounding!(c, norm(a), X, order(a))
    return ValidatedSequence(c, truncation_error(a), X)
end

# norm

LinearAlgebra.norm(a::ValidatedSequence) = sequence_norm(a) + truncation_error(a)
