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

# by-pass default
Base.:(==)(a::ValidatedSequence, b::ValidatedSequence) =
    (sequence(a) == sequence(b)) & iszero(sequence_error(a)) & iszero(sequence_error(b))

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

function differentiate(a::ValidatedSequence, α=1)
    sequence_error(a) == 0 || return throw(DomainError) # TODO: lift restriction
    c = differentiate(sequence(a), α)
    return ValidatedSequence(c, banachspace(a))
end



function integrate(a::ValidatedSequence, α=1)
    c = integrate(sequence(a), α)
    X = banachspace(a)
    return ValidatedSequence(c, _integral_error(X, space(a), space(c), α) * sequence_error(a), X)
end

_integral_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _integral_error(X, dom[1], codom[1], α[1]) * _integral_error(X, Base.tail(dom), Base.tail(codom), Base.tail(α))
_integral_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _integral_error(X, dom[1], codom[1], α[1])

_integral_error(X::Ell1{<:NTuple{N,Weight}}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _integral_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1]) * _integral_error(Ell1(Base.tail(weight(X))), Base.tail(dom), Base.tail(codom), Base.tail(α))
_integral_error(X::Ell1{<:Tuple{Weight}}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _integral_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1])

function _integral_error(X::Ell1, dom::Taylor, codom::Taylor, n::Int)
    v = __getindex(weight(X), codom, n)
    return v * _nzval(Integral(n), dom, codom, typeof(v), n, 0)
end
_integral_error(::Ell1, dom::Fourier{T}, codom::Fourier{S}, n::Int) where {T<:Real,S<:Real} =
    abs(_nzval(Integral(n), dom, codom, complex(promote_type(T, S)), 1, 1))
function _integral_error(X::Ell1, ::Chebyshev, codom::Chebyshev, n::Int)
    v = ExactReal(1) + __getindex(weight(X), codom, n)
    n == 0 && return one(v)
    n == 1 && return v
    return throw(DomainError) # TODO: lift restriction
end

__getindex(::IdentityWeight, ::BaseSpace, ::Int) = interval(1)
__getindex(w::Weight, s::BaseSpace, n::Int) = _getindex(w, s, n)



evaluate(a::ValidatedSequence, x) = _return_evaluate(evaluate(sequence(a), x), a)

_return_evaluate(c, a::ValidatedSequence) = interval(c, sequence_error(a); format = :midpoint)
_return_evaluate(c::Sequence, a::ValidatedSequence) = ValidatedSequence(c, sequence_error(a), banachspace(a))



# norm

LinearAlgebra.norm(a::ValidatedSequence) = sequence_norm(a) + sequence_error(a)



# rigorous enclosure of nonlinearities

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
    @inbounds view(full_c, indices(space(c))) .= zero(eltype(c))
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

    f = approx_a⁻¹ * a - ExactReal(1)
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
    Z₁ = norm(approx_b⁻¹ * b - ExactReal(1))
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

    Y = norm(approx_sqrta⁻¹ * (approx_sqrta ^ 2 - a)) / ExactReal(2)
    Z₁ = norm(approx_sqrta⁻¹ * approx_sqrta - ExactReal(1))
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

    approx_cbrta² = approx_cbrta ^ 2
    Y = norm(approx_cbrta⁻² * (approx_cbrta² * approx_cbrta - a)) / ExactReal(3)
    Z₁ = norm(approx_cbrta⁻² * approx_cbrta² - ExactReal(1))
    v = 2mid(norm(approx_cbrta⁻²)) * mid(norm(approx_cbrta))
    w = 2mid(norm(approx_cbrta⁻²))
    R = max(2sup(Y), (-v + sqrt(v^2 - 2w*(mid(Z₁) - 1))) / w) # could use: 0.1sup( (1-Z₁)^2/(4Y * norm(approx_cbrta⁻²)) - norm(approx_cbrta) )
    Z₂ = ExactReal(2) * norm(approx_cbrta⁻²) * (norm(approx_cbrta) + ExactReal(R))
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
            @assert !iszero(sequence(a)) # TODO: lift restriction
            banachspace(a) isa Ell1{<:GeometricWeight} || return throw(ArgumentError("only Ell1{<:GeometricWeight} is allowed"))

            space_approx = _image_trunc($f, space(a))
            N_fft = 2 * fft_size(space_approx)
            A = fft(sequence(a), N_fft)
            fA = $f.(A)
            seq_fa = _call_ifft!(fA, space_approx, eltype(a))

            ν = rate(weight(banachspace(a)))

            ν_finite_part = interval(max(nextfloat(sup(rate(weight(banachspace(a))))), rate(geometricweight(seq_fa))))
            ν_finite_part⁻¹ = inv(ν_finite_part)

            C = max(_contour($f, sequence(a), ν_finite_part, N_fft, eltype(seq_fa)),
                    _contour($f, sequence(a), ν_finite_part⁻¹, N_fft, eltype(seq_fa)))

            q = mapreduce(k -> (ν_finite_part⁻¹ ^ ExactReal(k) + ν_finite_part ^ ExactReal(k)) * ν_finite_part ^ ExactReal(abs(k)), +, indices(space_approx))
            error = C * (q / (ν_finite_part ^ ExactReal(N_fft) - ExactReal(1)) +
                         ExactReal(2) * ν_finite_part / (ν_finite_part - ν) * (ν * ν_finite_part⁻¹) ^ ExactReal(order(a) + 1))

            if !isthinzero(sequence_error(a))
                r_star = ExactReal(1) + sequence_error(a)
                # W = mapreduce(
                #         θ -> max(_contour($f, sequence(a) + r_star * cispi(θ), ν_finite_part, N_fft, eltype(seq_fa)),
                #                  _contour($f, sequence(a) + r_star * cispi(θ), ν_finite_part⁻¹, N_fft, eltype(seq_fa))),
                #         max,
                #         mince(interval(IntervalArithmetic.numtype(ν), -1, 1), N_fft)
                #     ) * (ν_finite_part + ν) / (ν_finite_part - ν)
                θ = interval(IntervalArithmetic.numtype(ν), -1, 1)
                W = max(_contour($f, sequence(a) + r_star * cispi(θ), ν_finite_part, N_fft, eltype(seq_fa)),
                        _contour($f, sequence(a) + r_star * cispi(θ), ν_finite_part⁻¹, N_fft, eltype(seq_fa))) * (ν_finite_part + ν) / (ν_finite_part - ν)
                error += W * sequence_error(a)
            end

            return ValidatedSequence(seq_fa, error, banachspace(a))
        end

        function Base.$f(a::ValidatedSequence{<:TensorSpace})
            @assert !iszero(sequence(a)) # TODO: lift restriction
            banachspace(a) isa Tuple{Vararg{Ell1{<:GeometricWeight}}} || return throw(ArgumentError("only Ell1{<:GeometricWeight} is allowed"))

            space_approx = _image_trunc($f, space(a))
            N_fft = 2 .* fft_size(space_approx)
            A = fft(sequence(a), N_fft)
            fA = $f.(A)
            seq_fa = _call_ifft!(fA, space_approx, eltype(a))

            ν = rate.(weight(banachspace(a)))

            ν_finite_part = interval.(max.(nextfloat.(sup.(rate.(weight(banachspace(a))))), rate.(geometricweight(seq_fa))))
            ν_finite_part⁻¹ = inv.(ν_finite_part)

            _t_ = tuple(ν_finite_part, ν_finite_part⁻¹)
            mix_ν = Iterators.product(getindex.(_t_, 1), getindex.(_t_, 2))

            C = mapreduce(μ -> _contour($f, sequence(a), μ, N_fft, eltype(seq_fa)), max, mix_ν)

            q = mapreduce(k -> mapreduce(μ -> prod(μ .^ ExactReal.(k)), +, mix_ν) * prod(ν_finite_part .^ ExactReal.(abs.(k))), +, indices(space_approx))
            error = C * (q / prod(ν_finite_part .^ ExactReal.(N_fft) .- ExactReal(1)) +
                         ExactReal(2^2) * prod(ν_finite_part) / prod(ν_finite_part .- ν) * prod((ν .* ν_finite_part⁻¹) .^ ExactReal.(order(a) .+ 1)))

            if !isthinzero(sequence_error(a))
                r_star = ExactReal(1) + sequence_error(a)
                # W = mapreduce(
                #         θ -> mapreduce(μ -> _contour($f, sequence(a) + r_star * cispi(θ), μ, N_fft, eltype(seq_fa)), max, mix_ν),
                #         max,
                #         mince(interval(promote_type(IntervalArithmetic.numtype.(ν)...), -1, 1), minimum(N_fft))
                #     ) * prod((ν_finite_part .+ ν) ./ (ν_finite_part .- ν))
                θ = interval(promote_type(IntervalArithmetic.numtype.(ν)...), -1, 1)
                W = mapreduce(μ -> _contour($f, sequence(a) + r_star * cispi(θ), μ, N_fft, eltype(seq_fa)),
                        max,
                        mix_ν) * prod((ν_finite_part .+ ν) ./ (ν_finite_part .- ν))
                error += W * sequence_error(a)
            end

            return ValidatedSequence(seq_fa, error, banachspace(a))
        end
    end
end

function _contour(f, ū, μ, N_fft, T)
    ū_δ = complex.(ū)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), N_fft)))
    δ = interval(-val, val)
    for k ∈ indices(space(ū))
        ū_δ[k] *= μ ^ ExactReal(k) * cispi(ExactReal(k) * δ)
    end
    grid_ū_δ = fft(ū_δ, N_fft)
    contour_integral = zero(real(T))
    for v ∈ grid_ū_δ
        contour_integral += abs(f(v))
    end
    return interval(sup(contour_integral / ExactReal(N_fft)))
end

function _contour(f, ū, μ, N_fft::Tuple, T)
    ū_δ = complex.(ū)
    val = sup.(inv.(interval.(IntervalArithmetic.numtype.(μ), N_fft)))
    δ = interval.(.- val, val)
    for k ∈ indices(space(ū))
        ū_δ[k] *= prod(μ .^ ExactReal.(k) .* cispi.(ExactReal.(k) .* δ))
    end
    grid_ū_δ = fft(ū_δ, N_fft)
    contour_integral = zero(real(T))
    for v ∈ grid_ū_δ
        contour_integral += abs(f(v))
    end
    return interval(sup(contour_integral / ExactReal(prod(N_fft))))
end
