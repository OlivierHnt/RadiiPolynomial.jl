struct ValidatedSequence{T<:SequenceSpace,R<:Interval,S<:Union{AbstractVector{R},AbstractVector{Complex{R}}},U<:BanachSpace} <: AbstractSequence{T,S}
    sequence :: Sequence{T,S}
    sequence_norm :: R
    sequence_error :: R
    banachspace :: U
    global _unsafe_validated_sequence(sequence::Sequence{T,S}, sequence_norm::R, sequence_error::R, banachspace::U) where {T<:SequenceSpace,R<:Interval,S<:Union{AbstractVector{R},AbstractVector{Complex{R}}},U<:BanachSpace} =
        new{T,R,S,U}(sequence, sequence_norm, sequence_error, banachspace)
end

function ValidatedSequence(sequence::Sequence{T,S}, sequence_error::R, banachspace::U) where {T<:SequenceSpace,R<:Interval,S<:Union{AbstractVector{R},AbstractVector{Complex{R}}},U<:BanachSpace}
    _iscompatbanachspace(space(sequence), banachspace) || return throw(ArgumentError("invalid norm for the sequence space"))
    if isempty_interval(sequence_error)
        seq = fill(emptyinterval(eltype(sequence)), space(sequence))
        sequence_norm = emptyinterval(sequence_error)
        return _unsafe_validated_sequence(seq, sequence_norm, sequence_error, banachspace)
    elseif inf(sequence_error) ≥ 0
        sequence_norm = norm(sequence, banachspace)
        return _unsafe_validated_sequence(sequence, sequence_norm, sequence_error, banachspace)
    else
        return throw(ArgumentError("truncation error must be positive"))
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
    c = differentiate(sequence(a), α)
    X = banachspace(a)
    seq_err = sequence_error(a)
    iszero(seq_err) && return ValidatedSequence(c, X)
    return ValidatedSequence(c, _derivative_error(X, space(a), space(c), α) * seq_err, X)
end

_derivative_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _derivative_error(X, dom[1], codom[1], α[1]) * _derivative_error(X, Base.tail(dom), Base.tail(codom), Base.tail(α))
_derivative_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _derivative_error(X, dom[1], codom[1], α[1])

_derivative_error(X::Ell1{<:NTuple{N,Weight}}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _derivative_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1]) * _derivative_error(Ell1(Base.tail(weight(X))), Base.tail(dom), Base.tail(codom), Base.tail(α))
_derivative_error(X::Ell1{<:Tuple{Weight}}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _derivative_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1])

function _derivative_error(X::Ell1{<:GeometricWeight}, ::Taylor, ::Taylor, n::Int)
    ν = rate(weight(X))
    n == 0 && return one(ν)
    n == 1 && return ν                                                               / (ν - ExactReal(1))^2
    n == 2 && return ν * (ν + ExactReal(1))                                          / (ν - ExactReal(1))^3
    n == 3 && return ν * (ν^2 + ExactReal(4)*ν + ExactReal(1))                       / (ν - ExactReal(1))^4
    n == 4 && return ν * (ν + ExactReal(1)) * (ν^2 + ExactReal(10)*ν + ExactReal(1)) / (ν - ExactReal(1))^5
    return throw(DomainError) # TODO: lift restriction
end
function _derivative_error(X::Ell1{<:GeometricWeight}, ::Fourier, codom::Fourier, n::Int)
    ν = rate(weight(X))
    n == 0 && return one(ν)
    n == 1 && return frequency(codom)   * ExactReal(2) * ν                                                               / (ν - ExactReal(1))^2
    n == 2 && return frequency(codom)^2 * ExactReal(2) * ν * (ν + ExactReal(1))                                          / (ν - ExactReal(1))^3
    n == 3 && return frequency(codom)^3 * ExactReal(2) * ν * (ν^2 + ExactReal(4)*ν + ExactReal(1))                       / (ν - ExactReal(1))^4
    n == 4 && return frequency(codom)^4 * ExactReal(2) * ν * (ν + ExactReal(1)) * (ν^2 + ExactReal(10)*ν + ExactReal(1)) / (ν - ExactReal(1))^5
    return throw(DomainError) # TODO: lift restriction
end
function _derivative_error(::Ell1{<:GeometricWeight}, ::Chebyshev, ::Chebyshev, n::Int)
    n == 0 && return interval(1)
    return throw(DomainError) # TODO: lift restriction
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
    # TODO: propagate "NG" flag

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
    # TODO: propagate "NG" flag

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
    # TODO: propagate "NG" flag

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
    # TODO: propagate "NG" flag

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

function Base.:^(a::ValidatedSequence, p::Real)
    seq_a = sequence(a)
    _isconstant(seq_a) && return ValidatedSequence(_at_value(x -> ^(x, p), seq_a), banachspace(a))
    ν_ = rate(banachspace(a))

    #

    space_c = _image_trunc(^, space(a))
    A = fft(seq_a, fft_size(space_c))
    C = A .^ p
    c = _call_ifft!(C, space_c, eltype(a))
    ν̄_ = interval.(_optimize_decay(x -> ^(x, mid(p)), mid.(c), mid.(seq_a), mid.(ν_)))
    _, N_v = _resolve_saturation!(x -> ^(x, p), c, a, ν̄_)

    #

    error = prod(_error(x -> ^(x, p), seq_a, c, ν_, ν̄_, N_v))

    #

    if !isthinzero(sequence_error(a))
        ν = tuple(ν_...)
        ν̄ = tuple(ν̄_...)
        ν̄⁻¹ = inv.(ν̄)

        _tuple_ = tuple(ν̄, ν̄⁻¹)
        _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(length(ν)))...)

        r_star = ExactReal(1) + sequence_error(a)
        circle = r_star * cispi(interval(IntervalArithmetic.numtype(r_star), -1, 1))
        W = maximum(μ -> _contour(x -> ^(x, p), seq_a + circle, ν), _mix_) * prod((ν̄ .+ ν) ./ (ν̄ .- ν))

        error += W * sequence_error(a)
    end

    #

    return ValidatedSequence(c, error, banachspace(a))
end

# function _cross_branch_cut_log(a, N_fft)
#     ν = rate(weight(banachspace(a)))
#     image_on_boundary = [a( (ν * cispi(θ) + cispi(-θ) / ν) / 2 ) for θ ∈ mince(interval(IntervalArithmetic.numtype(ν), -1, 1), N_fft)]
#     return any(piece -> !isdisjoint_interval(interval(-Inf, 0), piece), image_on_boundary)
# end

#

for f ∈ (:exp, :cos, :sin, :cosh, :sinh)
    @eval begin
        function Base.$f(a::ValidatedSequence)
            seq_a = sequence(a)
            _isconstant(seq_a) & isthinzero(sequence_error(a)) && return ValidatedSequence(_at_value($f, seq_a), banachspace(a))
            ν_ = rate(banachspace(a))

            #

            space_c = _image_trunc($f, space(a))
            A = fft(seq_a, fft_size(space_c))
            C = $f.(A)
            c = _call_ifft!(C, space_c, eltype(a))
            ν̄_ = interval.(_optimize_decay($f, mid.(c), mid.(seq_a), mid.(ν_)))
            _, N_v = _resolve_saturation!($f, c, a, ν̄_)

            #

            error = prod(_error($f, seq_a, c, ν_, ν̄_, N_v))

            #

            if !isthinzero(sequence_error(a))
                ν = tuple(ν_...)
                ν̄ = tuple(ν̄_...)
                ν̄⁻¹ = inv.(ν̄)

                _tuple_ = tuple(ν̄, ν̄⁻¹)
                _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(length(ν)))...)

                r_star = ExactReal(1) + sequence_error(a)
                circle = r_star * cispi(interval(IntervalArithmetic.numtype(r_star), -1, 1))
                W = maximum(μ -> _contour($f, seq_a + circle, ν), _mix_) * prod((ν̄ .+ ν) ./ (ν̄ .- ν))

                error += W * sequence_error(a)
            end

            #

            return ValidatedSequence(c, error, banachspace(a))
        end
    end
end

#

function _error(f, a, approx, ν, ν̄, N_v)
    ν̄⁻¹ = inv(ν̄)

    C = max(_contour(f, a, ν̄), _contour(f, a, ν̄⁻¹))

    q = sum(k -> (ν̄ ^ ExactReal(k) + ν̄⁻¹ ^ ExactReal(k)) * ν ^ ExactReal(abs(k)), -N_v:N_v)

    return C, q / prod(ν̄ ^ ExactReal( fft_size(space(approx)) ) - ExactReal(1)) + ExactReal(2) * ν̄ / (ν̄ - ν) * (ν * ν̄⁻¹) ^ ExactReal(N_v + 1)
end

function _error(f, a, approx, ν::NTuple{N}, ν̄, N_v) where {N}
    ν̄⁻¹ = inv.(ν̄)
    _tuple_ = tuple(ν̄, ν̄⁻¹)
    _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(N))...)

    C = maximum(μ -> _contour(f, a, μ), _mix_)

    q = sum(k -> sum(μ -> prod(μ .^ ExactReal.(k)), _mix_) * prod(ν .^ ExactReal.(abs.(k))), TensorIndices(ntuple(i -> -N_v[i]:N_v[i], Val(N))))

    return C, q / prod(ν̄ .^ ExactReal.( fft_size(space(approx)) ) .- ExactReal(1)) + ExactReal(2^N) * prod(ν̄ ./ (ν̄ .- ν) .* (ν .* ν̄⁻¹) .^ ExactReal.(N_v .+ 1))
end

# function _resolve_saturation!(f, c, a, ν::Interval)
#     ν⁻¹ = inv(ν)
#     C = max(_contour(f, a, ν), _contour(f, a, ν⁻¹))
#     min_ord = order(c)
#     if isfinite(mag(C))
#         CoefType = eltype(c)
#         for k ∈ indices(space(c))
#             if mag(c[k]) > mag(C / ν ^ abs(k))
#                 min_ord = min(min_ord, abs(k))
#                 c[k] = zero(CoefType)
#             end
#         end
#     end
#     return c, min_ord
# end

# function _resolve_saturation!(f, c, a, ν::NTuple{N,Interval}) where {N}
#     ν⁻¹ = inv.(ν)
#     _tuple_ = tuple(ν, ν⁻¹)
#     _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(N))...)
#     C = maximum(μ -> _contour(f, a, μ), _mix_)
#     min_ord = order(c)
#     if isfinite(mag(C))
#         CoefType = eltype(c)
#         for k ∈ indices(space(c))
#             if mag(c[k]) > mag(C / prod(ν .^ abs.(k)))
#                 min_ord = min.(min_ord, abs.(k))
#                 c[k] = zero(CoefType)
#             end
#         end
#     end
#     return c, min_ord
# end

#

_contour(f, a, ν::NTuple{1}) = _contour(f, a, ν[1])

# function _contour(f, a, ν::Interval)
#     # mid_ν = mid(ν)
#     # N_fft = min(fft_size(space(a)), prevpow(2, log( ifelse(mid_ν < 1, floatmin(mid_ν), floatmax(mid_ν)) ) / log(mid_ν))) # maybe there is a better N_fft value to consider
#     N_fft = fft_size(space(a))

#     CoefType = complex(eltype(a))
#     grid_a_δ = zeros(CoefType, N_fft)

#     A = coefficients(a)
#     view(grid_a_δ, eachindex(A)) .= A
#     _preprocess!(grid_a_δ, space(a))
#     _boxes!(grid_a_δ, ν)

#     _fft_pow2!(grid_a_δ)
#     contour_integral = sum(abs ∘ f, grid_a_δ)

#     return contour_integral / ExactReal(N_fft)
# end

# function _contour(f, a, ν::Tuple{Vararg{Interval}})
#     # mid_ν = mid.(ν)
#     # N_fft = min.(fft_size(space(a)), prevpow.(2, log.( ifelse.(mid_ν .< 1, floatmin.(mid_ν), floatmax.(mid_ν)) ) ./ log.(mid_ν)))
#     N_fft = fft_size(space(a))

#     CoefType = complex(eltype(a))
#     grid_a_δ = zeros(CoefType, N_fft)

#     A = _no_alloc_reshape(coefficients(a), dimensions(space(a)))
#     view(grid_a_δ, axes(A)...) .= A
#     _apply_preprocess!(grid_a_δ, space(a))
#     _apply_boxes!(grid_a_δ, ν)

#     _fft_pow2!(grid_a_δ)
#     contour_integral = sum(abs ∘ f, grid_a_δ)

#     return contour_integral / ExactReal(prod(N_fft))
# end

#

function _boxes!(C, μ::Interval)
    len = length(C)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len))) # 1/N_fft should be an exact operation
    δ = interval(-val, val)
    for k ∈ 1:len÷2-1
        C[k+1]     *= μ ^ ExactReal(-k) * cispi(ExactReal(-k) * δ)
        C[len+1-k] *= μ ^ ExactReal( k) * cispi(ExactReal( k) * δ)
    end
    return C
end

function _boxes!(C, μ::Interval, ::Val{D}) where {D}
    len = size(C, D)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len))) # 1/N_fft should be an exact operation
    δ = interval(-val, val)
    for k ∈ 1:len÷2-1
        selectdim(C, D, k+1)     .*= μ ^ ExactReal(-k) * cispi(ExactReal(-k) * δ)
        selectdim(C, D, len+1-k) .*= μ ^ ExactReal( k) * cispi(ExactReal( k) * δ)
    end
    return C
end
