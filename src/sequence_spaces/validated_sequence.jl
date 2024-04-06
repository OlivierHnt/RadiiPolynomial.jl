struct ValidatedSequence{T<:SequenceSpace,R<:Interval,S<:AbstractVector{<:Union{R,Complex{R}}},U<:BanachSpace} <: AbstractSequence{T,S}
    sequence :: Sequence{T,S}
    sequence_norm :: R
    sequence_error :: R
    banachspace :: U
    global _unsafe_validated_sequence(sequence::Sequence{T,S}, sequence_norm::R, sequence_error::R, banachspace::U) where {T<:SequenceSpace,R<:Interval,S<:AbstractVector{<:Union{R,Complex{R}}},U<:BanachSpace} =
        new{T,R,S,U}(sequence, sequence_norm, sequence_error, banachspace)
end

function ValidatedSequence(sequence::Sequence{T,S}, sequence_error::R, banachspace::U) where {T<:SequenceSpace,R<:Interval,S<:AbstractVector{<:Union{R,Complex{R}}},U<:BanachSpace}
    _iscompatbanachspace(space(sequence), banachspace) || return throw(ArgumentError("invalid norm for the sequence space"))
    if isempty_interval(sequence_error)
        seq = fill(emptyinterval(eltype(sequence)), space(sequence))
        sequence_norm = emptyinterval(sequence_error)
        return _unsafe_validated_sequence(seq, sequence_norm, sequence_error, banachspace)
    elseif (inf(sequence_error) ≥ 0) & isbounded(sequence_error)
        sequence_norm = norm(sequence, banachspace)
        return _unsafe_validated_sequence(sequence, sequence_norm, sequence_error, banachspace)
    else
        return throw(ArgumentError("truncation error must be positive and finite"))
    end
end

_iscompatbanachspace(::SequenceSpace, ::Ell1{<:Weight}) = true
_iscompatbanachspace(::SequenceSpace, ::Ell2{<:Weight}) = true
_iscompatbanachspace(::SequenceSpace, ::EllInf{<:Weight}) = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::Ell1{<:NTuple{N,Weight}}) where {N} = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::Ell2{<:NTuple{N,Weight}}) where {N} = true
_iscompatbanachspace(::TensorSpace{<:NTuple{N,BaseSpace}}, ::EllInf{<:NTuple{N,Weight}}) where {N} = true

ValidatedSequence(sequence::Sequence, banachspace::BanachSpace) =
    ValidatedSequence(sequence, interval(zero(real(eltype(sequence)))), banachspace)

ValidatedSequence(space::SequenceSpace, coefficients::AbstractVector, banachspace::BanachSpace) =
    ValidatedSequence(Sequence(space, coefficients), banachspace)

ValidatedSequence(space::SequenceSpace, coefficients::AbstractVector, sequence_error::Interval, banachspace::BanachSpace) =
    ValidatedSequence(Sequence(space, coefficients), sequence_error, banachspace)

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
Base.one(a::ValidatedSequence) = ValidatedSequence(one(sequence(a)), banachspace(a))

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
    println(io, "Sequence error: ", sequence_error(a))
    return print(io, "Banach space: ", _prettystring(banachspace(a)))
end

function Base.show(io::IO, a::ValidatedSequence)
    get(io, :compact, false) && return show(io, (coefficients(a), sequence_error(a), banachspace(a)))
    return print(io, "ValidatedSequence(", space(a), ", ", coefficients(a), ", ", sequence_norm(a), ", ", sequence_error(a), ", ", banachspace(a), ")")
end

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



# rigorous evaluation of nonlinearities

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

#

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

#

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

#

function Base.cbrt(a::ValidatedSequence)
    space_approx = _image_trunc(cbrt, space(a))
    A = fft(mid.(sequence(a)), fft_size(space_approx))
    cbrtA = A .^ (1//3)
    cbrtA⁻² = inv.(cbrtA) .^ 2
    seq_approx_cbrta = _call_ifft!(cbrtA, space_approx, eltype(a))
    seq_approx_cbrta⁻² = _call_ifft!(cbrtA⁻², space_approx, eltype(a))

    X = banachspace(a)
    approx_cbrta = ValidatedSequence(interval.(seq_approx_cbrta), X)
    approx_cbrta⁻² = ValidatedSequence(interval.(seq_approx_cbrta⁻²), X)

    Y = norm(approx_cbrta⁻² * (approx_cbrta ^ 3 - a))/interval(real(eltype(seq_approx_cbrta)), 3)
    Z₁ = norm(approx_cbrta⁻² * approx_cbrta ^ 2 - interval(real(eltype(seq_approx_cbrta)), 1))
    v = 2mid(norm(approx_cbrta⁻²)) * mid(norm(approx_cbrta))
    w = 2mid(norm(approx_cbrta⁻²))
    R = max(2sup(Y), (-v + sqrt(v^2 - 2w*(mid(Z₁) - 1)))/ w) # could use: 0.1*sup( (1-Z₁)^2/(4Y * norm(approx_cbrta⁻²)) - norm(approx_cbrta) )
    Z₂ = interval(real(eltype(seq_approx_cbrta)), 2) * norm(approx_cbrta⁻²) * (norm(approx_cbrta) + interval(real(eltype(seq_approx_cbrta)), R))
    r = interval_of_existence(Y, Z₁, Z₂, R; verbose = false)

    isempty_interval(r) || return ValidatedSequence(sequence(approx_cbrta), interval(real(eltype(seq_approx_cbrta)), inf(r)), X)
    fill!(sequence(approx_cbrta), emptyinterval(eltype(seq_approx_cbrta)))
    return ValidatedSequence(sequence(approx_cbrta), emptyinterval(real(eltype(seq_approx_cbrta))), X)
end

#

Base.abs(a::ValidatedSequence) = sqrt(a^2)
Base.abs2(a::ValidatedSequence) = a^2

#

for f ∈ (:exp, :cos, :sin, :cosh, :sinh)
    @eval begin
        function Base.$f(a::ValidatedSequence{<:BaseSpace})
            banachspace(a) isa Ell1{<:GeometricWeight} || return throw(ArgumentError("only Ell1{<:GeometricWeight} is allowed"))

            space_approx = _image_trunc($f, space(a))
            mida = mid.(sequence(a))
            N_fft_ = fft_size(space_approx)
            A = fft(mida, N_fft_)
            fA = $f.(A)
            seq_approx_fa = _call_ifft!(fA, space_approx, eltype(a))

            seq_fa = interval.(seq_approx_fa)

            ν = rate(weight(banachspace(a)))

            ν_finite_part = interval(max(nextfloat(sup(rate(weight(banachspace(a))))), rate(geometricweight(seq_approx_fa))))
            ν_finite_part⁻¹ = inv(ν_finite_part)

            N_fft = 2 * N_fft_

            C = max(_contour($f, sequence(a), ν_finite_part, N_fft, eltype(seq_fa)),
                    _contour($f, sequence(a), ν_finite_part⁻¹, N_fft, eltype(seq_fa)))

            error = C * (
                mapreduce(k -> (_safe_pow(ν_finite_part⁻¹, k) + _safe_pow(ν_finite_part, k)) * _safe_pow(ν_finite_part, abs(k)), +, indices(space_approx)) / (_safe_pow(ν_finite_part, N_fft) - 1) +
                _safe_mul(2, ν_finite_part) / (ν_finite_part - ν) * _safe_pow(ν * ν_finite_part⁻¹, order(a) + 1))

            if !isthinzero(sequence_error(a))
                r_star = 1 + sequence_error(a)
                W = mapreduce(
                        θ -> max(_contour($f, sequence(a) + r_star * cispi(θ), ν_finite_part, N_fft, real(eltype(seq_fa))),
                                 _contour($f, sequence(a) + r_star * cispi(θ), ν_finite_part⁻¹, N_fft, real(eltype(seq_fa)))),
                        max,
                        mince(interval(IntervalArithmetic.numtype(ν), -1, 1), N_fft)
                    ) * (ν_finite_part + ν) / (ν_finite_part - ν)
                error += W * sequence_error(a)
            end

            return ValidatedSequence(seq_fa, error, banachspace(a))
        end
    end
end

function _contour(f, ū, ν_finite_part, N_fft, T)
    ū_contour = complex.(ū)
    val = sup(inv(interval(IntervalArithmetic.numtype(ν_finite_part), N_fft)))
    δ = interval(-val, val)
    for k ∈ indices(space(ū))
        ū_contour[k] *= _safe_pow(ν_finite_part, k) * cispi(_safe_mul(k, δ))
    end
    grid_ū_contour = fft(ū_contour, N_fft)
    contour_integral = zero(real(T))
    for v ∈ grid_ū_contour
        contour_integral += abs(f(v))
    end
    return interval(sup(_safe_div(contour_integral, N_fft)))
end
