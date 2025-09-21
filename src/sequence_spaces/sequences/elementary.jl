_maybe_interval(::Type, a) = a
_maybe_interval(::Type{<:RealOrComplexI}, a) = interval.(a)

_isguaranteed(a::InfiniteSequence) = all(isguaranteed, sequence(a)) & isguaranteed(sequence_norm(a)) & isguaranteed(sequence_error(a)) & a

# division

_codomain(::typeof(inv), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain(inv, sᵢ), spaces(s)))
_codomain(::typeof(inv), s::Taylor) = s
_codomain(::typeof(inv), s::Fourier) = s
_codomain(::typeof(inv), s::Chebyshev) = s
_codomain(::typeof(inv), s::CosFourier) = s
_codomain(::typeof(inv), s::SinFourier) = s

function Base.inv(a::Sequence)
    space_approx = _codomain(inv, space(a))
    _isconstant(a) && return _at_value(inv, a)
    A = fft(a, fft_size(space_approx))
    A .= inv.(A)
    return _call_ifft!(A, space_approx, eltype(a))
end

function Base.inv(a::InfiniteSequence)
    # TODO: propagate "NG" flag

    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(sequence_error(a)) && return InfiniteSequence(_at_value(inv, seq_a), banachspace(a))

    seq_approx_a⁻¹ = inv(mid.(seq_a))

    X = banachspace(a)
    approx_a⁻¹ = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_a⁻¹), X)

    f = approx_a⁻¹ * a - exact(1)
    Y = norm(approx_a⁻¹ * f)
    Z₁ = norm(f)
    r, _ = interval_of_existence(Y, Z₁, Inf; verbose = false)

    return InfiniteSequence(sequence(approx_a⁻¹), _maybe_interval(eltype(a), inf(r)), X)
end

_codomain(::typeof(/), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> _codomain(/, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
_codomain(::typeof(/), s₁::Taylor, s₂::Taylor) = union(s₁, s₂)
_codomain(::typeof(/), s₁::Fourier, s₂::Fourier) = union(s₁, s₂)
_codomain(::typeof(/), s₁::Chebyshev, s₂::Chebyshev) = union(s₁, s₂)
_codomain(::typeof(/), s₁::CosFourier, s₂::CosFourier) = union(s₁, s₂)
_codomain(::typeof(/), s₁::SinFourier, s₂::SinFourier) = CosFourier(union(desymmetrize(s₁), desymmetrize(s₂)))
_codomain(::typeof(/), s₁::CosFourier, s₂::SinFourier) = union(SinFourier(desymmetrize(s₁)), s₂)
_codomain(::typeof(/), s₁::SinFourier, s₂::CosFourier) = union(s₁, CosFourier(desymmetrize(s₂)))

function Base.:/(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    # TODO: propagate "NG" flag
    space_approx = _codomain(/, space(a), space(b))
    _isconstant(b) && return rdiv!(a, b[_findindex_constant(space(b))])
    A = fft(a, fft_size(space_approx))
    B = fft(b, fft_size(space_approx))
    A .= A ./ B
    return _call_ifft!(A, space_approx, promote_type(eltype(a), eltype(b)))
end
Base.:/(a::Number, b::Sequence{<:SequenceSpace}) = lmul!(a, inv(b))

function Base.:/(a::InfiniteSequence, b::InfiniteSequence)
    # TODO: propagate "NG" flag

    space_approx = _codomain(/, space(a), space(b))

    seq_a = sequence(a)
    seq_b = sequence(b)
    _isconstant(seq_b) & _safe_iszero(sequence_error(b)) && return InfiniteSequence(rdiv!(seq_a, seq_b[_findindex_constant(space(seq_b))]), banachspace(a))

    A = fft(mid.(seq_a), fft_size(space_approx))
    B = fft(mid.(seq_b), fft_size(space_approx))
    A .= A ./ B
    B .= inv.(B)
    CoefType = promote_type(eltype(a), eltype(b))
    seq_approx_ab⁻¹ = _call_ifft!(A, space_approx, CoefType)
    seq_approx_b⁻¹ = _call_ifft!(B, space_approx, eltype(b))

    X = banachspace(a) ∩ banachspace(b)
    approx_ab⁻¹ = InfiniteSequence(_maybe_interval(CoefType, seq_approx_ab⁻¹), X)
    approx_b⁻¹ = InfiniteSequence(_maybe_interval(CoefType, seq_approx_b⁻¹), X)

    Y = norm(approx_b⁻¹ * (approx_ab⁻¹ * b - a))
    Z₁ = norm(approx_b⁻¹ * b - exact(1))
    r, _ = interval_of_existence(Y, Z₁, Inf; verbose = false)

    return InfiniteSequence(sequence(approx_ab⁻¹), _maybe_interval(CoefType, inf(r)), X)
end
Base.:/(a::Number, b::InfiniteSequence) = lmul!(a, inv(b))


_codomain(::typeof(\), s₁::SequenceSpace, s₂::SequenceSpace) = codomain(/, s₂, s₁)

Base.:\(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}) = b / a
Base.:\(a::Sequence{<:SequenceSpace}, b::Number) = b / a

Base.:\(a::InfiniteSequence, b::InfiniteSequence) = b / a
Base.:\(a::InfiniteSequence, b::Number) = b / a



# absolute value

Base.abs(a::Sequence{<:SequenceSpace}) = abs(InfiniteSequence(a, Ell1()))
Base.abs(a::InfiniteSequence{<:SequenceSpace}) = sqrt(^(a, 2))

Base.abs2(a::Sequence{<:SequenceSpace}) = ^(a, 2)
Base.abs2(a::InfiniteSequence{<:SequenceSpace}) = ^(a, 2)



# square root

_codomain(::typeof(sqrt), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain(sqrt, sᵢ), spaces(s)))
_codomain(::typeof(sqrt), s::Taylor) = s
_codomain(::typeof(sqrt), s::Fourier) = s
_codomain(::typeof(sqrt), s::Chebyshev) = s
_codomain(::typeof(sqrt), s::CosFourier) = s

function Base.sqrt(a::Sequence{<:SequenceSpace})
    space_approx = _codomain(sqrt, space(a))
    _isconstant(a) && return _at_value(sqrt, a)
    A = fft(a, fft_size(space_approx))
    A .= sqrt.(A)
    return _call_ifft!(A, space_approx, eltype(a))
end

function Base.sqrt(a::InfiniteSequence)
    # TODO: propagate "NG" flag

    space_approx = _codomain(sqrt, space(a))

    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(sequence_error(a)) && return InfiniteSequence(_at_value(sqrt, seq_a), banachspace(a))

    A = fft(mid.(seq_a), fft_size(space_approx))
    sqrtA = sqrt.(A)
    A .= inv.(sqrtA)
    seq_approx_sqrta = _call_ifft!(sqrtA, space_approx, eltype(a))
    seq_approx_sqrta⁻¹ = _call_ifft!(A, space_approx, eltype(a))

    X = banachspace(a)
    approx_sqrta = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_sqrta), X)
    approx_sqrta⁻¹ = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_sqrta⁻¹), X)

    Y = norm(approx_sqrta⁻¹ * (approx_sqrta ^ 2 - a)) / exact(2)
    Z₁ = norm(approx_sqrta⁻¹ * approx_sqrta - exact(1))
    Z₂ = norm(approx_sqrta⁻¹)
    r, _ = interval_of_existence(Y, Z₁, Z₂, Inf; verbose = false)

    return InfiniteSequence(sequence(approx_sqrta), _maybe_interval(eltype(a), inf(r)), X)
end



# cubic root

_codomain(::typeof(cbrt), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain(cbrt, sᵢ), spaces(s)))
_codomain(::typeof(cbrt), s::Taylor) = s
_codomain(::typeof(cbrt), s::Fourier) = s
_codomain(::typeof(cbrt), s::Chebyshev) = s
_codomain(::typeof(cbrt), s::CosFourier) = s
_codomain(::typeof(cbrt), s::SinFourier) = s

function Base.cbrt(a::Sequence{<:SequenceSpace})
    space_approx = _codomain(cbrt, space(a))
    _isconstant(a) && return _at_value(cbrt, a)
    A = fft(a, fft_size(space_approx))
    A .= A .^ (1//3)
    return _call_ifft!(A, space_approx, eltype(a))
end

function Base.cbrt(a::InfiniteSequence)
    # TODO: propagate "NG" flag

    space_approx = _codomain(cbrt, space(a))

    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(sequence_error(a)) && return InfiniteSequence(_at_value(cbrt, seq_a), banachspace(a))

    A = fft(mid.(seq_a), fft_size(space_approx))
    cbrtA = A .^ (1//3)
    A .= inv.(cbrtA) .^ 2
    seq_approx_cbrta = _call_ifft!(cbrtA, space_approx, eltype(a))
    seq_approx_cbrta⁻² = _call_ifft!(A, space_approx, eltype(a))

    X = banachspace(a)
    approx_cbrta = InfiniteSequence(_maybe_interval.(eltype(a), seq_approx_cbrta) : seq_approx_cbrta, X)
    approx_cbrta⁻² = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_cbrta⁻²) : seq_approx_cbrta⁻², X)

    approx_cbrta² = approx_cbrta ^ 2
    Y = norm(approx_cbrta⁻² * (approx_cbrta² * approx_cbrta - a)) / exact(3)
    Z₁ = norm(approx_cbrta⁻² * approx_cbrta² - exact(1))
    v = 2mid(norm(approx_cbrta⁻²)) * mid(norm(approx_cbrta))
    w = 2mid(norm(approx_cbrta⁻²))
    R = max(2sup(Y), (-v + sqrt(v^2 - 2w*(mid(Z₁) - 1))) / w) # could use: 0.1sup( (1-Z₁)^2/(4Y * norm(approx_cbrta⁻²)) - norm(approx_cbrta) )
    Z₂ = exact(2) * norm(approx_cbrta⁻²) * (norm(approx_cbrta) + exact(R))
    r, _ = interval_of_existence(Y, Z₁, Z₂, R; verbose = false)

    return InfiniteSequence(sequence(approx_cbrta), _maybe_interval(eltype(a), inf(r)), X)
end



# general nonlinearites

struct Nonlinearity{U<:Function,T<:RealOrComplexI,S<:RealOrComplexI}
    f          :: U
    poles      :: Vector{T}
    branch_cut :: S
end

    export Nonlinearity



# general power

_codomain(::typeof(^), s::TensorSpace, p::Real) = TensorSpace(map(sᵢ -> _codomain(^, sᵢ, p), spaces(s)))
_codomain(::typeof(^), s::Taylor, ::Real) = s
_codomain(::typeof(^), s::Fourier, ::Real) = s
_codomain(::typeof(^), s::Chebyshev, ::Real) = s
_codomain(::typeof(^), s::CosFourier, ::Real) = s
# the returned space depends on the power for `SinFourier`

function Base.:^(a::Sequence{<:SequenceSpace}, p::Real)
    isinteger(p) && return a ^ Integer(p)
    p == 1//2 && return sqrt(a)
    p == 1//3 && return cbrt(a)
    return Nonlinearity(x -> x ^ p, Complex{Interval{Float64}}[], interval(-Inf, 0))(a; codomain = _codomain(^, space(a), p))
end

function Base.:^(a::InfiniteSequence, p::Real)
    isinteger(p) && return a ^ Integer(p)
    p == 1//2 && return sqrt(a)
    p == 1//3 && return cbrt(a)
    return Nonlinearity(x -> x ^ p, Complex{Interval{Float64}}[], interval(-Inf, 0))(a; codomain = _codomain(^, space(a), p))
end



# entire functions

_codomain(::typeof(exp), s::Taylor) = s
_codomain(::typeof(exp), s::Fourier) = s
_codomain(::typeof(exp), s::Chebyshev) = s
_codomain(::typeof(exp), s::CosFourier) = s
_codomain(::typeof(exp), s::SinFourier) = desymmetrize(s)

_codomain(::typeof(cos), s::Taylor) = s
_codomain(::typeof(cos), s::Fourier) = s
_codomain(::typeof(cos), s::Chebyshev) = s
_codomain(::typeof(cos), s::CosFourier) = s
_codomain(::typeof(cos), s::SinFourier) = CosFourier(desymmetrize(s))

_codomain(::typeof(sin), s::Taylor) = s
_codomain(::typeof(sin), s::Fourier) = s
_codomain(::typeof(sin), s::Chebyshev) = s
_codomain(::typeof(sin), s::CosFourier) = desymmetrize(s)
_codomain(::typeof(sin), s::SinFourier) = s

_codomain(::typeof(cosh), s::Taylor) = s
_codomain(::typeof(cosh), s::Fourier) = s
_codomain(::typeof(cosh), s::Chebyshev) = s
_codomain(::typeof(cosh), s::CosFourier) = s
_codomain(::typeof(cosh), s::SinFourier) = CosFourier(symmetrize(s))

_codomain(::typeof(sinh), s::Taylor) = s
_codomain(::typeof(sinh), s::Fourier) = s
_codomain(::typeof(sinh), s::Chebyshev) = s
_codomain(::typeof(sinh), s::CosFourier) = desymmetrize(s)
_codomain(::typeof(sinh), s::SinFourier) = s

for f ∈ (:exp, :cos, :sin, :cosh, :sinh)
    @eval begin
        _codomain(::typeof($f), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain($f, sᵢ), spaces(s)))

        Nonlinearity(::typeof($f)) = Nonlinearity($f, Complex{Interval{Float64}}[], emptyinterval()) # entire functions

        Base.$f(a::Sequence{<:SequenceSpace}) = Nonlinearity($f)(a)

        Base.$f(a::InfiniteSequence) = Nonlinearity($f)(a)
    end
end

function (nl::Nonlinearity)(a::Sequence{<:SequenceSpace}; codomain::SequenceSpace = _codomain(nl.f, space(a)))
    _isconstant(a) && return _at_value(nl.f, a)
    A = fft(a, fft_size(codomain))
    C = nl.f.(A)
    return _call_ifft!(C, codomain, eltype(a))
end

function (nl::Nonlinearity)(a::InfiniteSequence; codomain::SequenceSpace = _codomain(nl.f, space(a)))
    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(sequence_error(a)) && return InfiniteSequence(_at_value(nl.f, seq_a), banachspace(a))
    ν_ = rate(banachspace(a))

    if !isempty_interval(nl.branch_cut) || !isempty(nl.poles)
        _check_branch_cut_poles(a, ν_, nl.poles, nl.branch_cut) ||
            return throw(ArgumentError("image intersects a branch cut or contains at least one pole: analyticity violated"))
    end

    A = fft(seq_a, fft_size(codomain))
    C = nl.f.(A)
    c = _call_ifft!(C, codomain, eltype(a))

    ν̄_ = interval.(_optimize_decay(nl.f, mid.(c), mid.(seq_a), mid.(ν_), a, nl.poles, nl.branch_cut))

    _, N_v = _resolve_saturation!(nl.f, c, a, ν̄_)

    error = prod(_error(nl.f, seq_a, c, ν_, ν̄_, N_v))

    if !_safe_iszero(sequence_error(a))
        ν = tuple(ν_...)
        ν̄ = tuple(ν̄_...)
        ν̄⁻¹ = inv.(ν̄)
        _tuple_ = tuple(ν̄, ν̄⁻¹)
        _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(length(ν)))...)

        r_star = exact(1) + sequence_error(a)
        circle = r_star * cispi(interval(IntervalArithmetic.numtype(r_star), -1, 1))

        # contour estimate
        W = maximum(μ -> _contour(nl.f, seq_a + circle, ν), _mix_) * prod((ν̄ .+ ν) ./ (ν̄ .- ν))
        error += W * sequence_error(a)
    end

    return InfiniteSequence(c, error, banachspace(a))
end

function _check_branch_cut_poles(a, ν, poles, branch_cut)
    r = mince(interval(0,  ν), fft_size(space(a)))
    check = true
    for rⱼ ∈ r
        check &= __check_branch_cut_poles(a, rⱼ, poles, branch_cut)
        check || return false
    end
    return true
end

function __check_branch_cut_poles(a, r, poles, branch_cut)
    CoefType = complex(float(eltype(a)))
    C = zeros(CoefType, fft_size(space(a)))
    A = _no_alloc_reshape(sequence(a))
    @inbounds view(C, axes(A)...) .= A
    _apply!(_preprocess!, C, space(a))
    _apply_boxes!(C, r)
    _fft_pow2!(C)
    return all(C) do x
        y = interval(x, sequence_error(a); format = :midpoint)
        return isdisjoint_interval(y, branch_cut) & all(p -> isdisjoint_interval(y, p), poles)
    end
end





# to avoid numerical artefacts when the input sequence is constant

function _isconstant(a::Sequence)
    s = space(a)
    idx = _findindex_constant(s) # throws for `SinFourier`
    return all(k -> ifelse(k == idx, true, _safe_iszero(a[k])), indices(s))
end

function _at_value(f, a)
    c = one(a)
    idx = _findindex_constant(space(a))
    c[idx] = f(a[idx])
    return c
end



# to optmize ν̄

function _optimize_decay(f, c, a, ν, aa, poles, branch_cut)
    ν̄_max = max(_geometric_rate(space(a), coefficients(a))[1], _geometric_rate(space(c), coefficients(c))[1])
    return _golden_search(ν, ν̄_max) do μ
        if !isempty_interval(branch_cut) || !isempty(poles)
            _check_branch_cut_poles(aa, μ, poles, branch_cut) ||
                return Inf
        end

        c_copy = copy(c)
        _, N_v = _resolve_saturation!(f, c_copy, a, μ)
        return prod(_error(f, a, c_copy, ν, μ, N_v))
    end
end

function _golden_search(f, a, b)
    ϕ = (sqrt(5) - 1) / 2 # ≈ 0.618
    c = b - ϕ * (b - a)
    d = a + ϕ * (b - a)
    fc = f(c)
    fd = f(d)
    iter = 0

    while abs(b - a) > 1e-2 && iter < 20
        if !isfinite(fc) || !isfinite(fd) # infeasible at c or d → shrink right bound
            b = d
            d = c
            fd = fc
            c = b - ϕ * (b - a)
            fc = f(c)
        elseif fc < fd
            b = d
            d = c
            fd = fc
            c = b - ϕ * (b - a)
            fc = f(c)
        else
            a = c
            c = d
            fc = fd
            d = a + ϕ * (b - a)
            fd = f(d)
        end
        iter += 1
    end

    return (a + b) / 2
end

function _golden_search(f, lower::NTuple{N}, upper::NTuple{N}) where {N}
    x = ntuple(i -> (lower[i] + upper[i]) / 2, Val(N))
    n = length(x)
    for _ ∈ 1:20
        for i ∈ 1:n
            a, b = lower[i], upper[i]
            μᵢ = _golden_search(μ -> f(ntuple(j -> ifelse(j == i, μ, x[i]), Val(N))), a, b)
            x = ntuple(j -> ifelse(j == i, μᵢ, x[i]), Val(N))
        end
    end
    return x
end



# to prevent numerical plateau

function _resolve_saturation!(f, c, a, ν)
    ν⁻¹ = inv(ν)
    C = max(_contour(f, a, ν), _contour(f, a, ν⁻¹))
    min_ord = order(c)
    if isfinite(mag(C))
        CoefType = eltype(c)
        for k ∈ indices(space(c))
            if mag(c[k]) > mag(C / ν ^ abs(k))
                min_ord = min(min_ord, abs(k))
                c[k] = zero(CoefType)
            end
        end
    end
    return c, min_ord
end

function _resolve_saturation!(f, c, a, ν::NTuple{N}) where {N}
    ν⁻¹ = inv.(ν)
    _tuple_ = tuple(ν, ν⁻¹)
    _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(N))...)
    C = maximum(μ -> _contour(f, a, μ), _mix_)
    min_ord = order(c)
    if isfinite(mag(C))
        CoefType = eltype(c)
        for k ∈ indices(space(c))
            if mag(c[k]) > mag(C / prod(ν .^ abs.(k)))
                min_ord = min.(min_ord, abs.(k))
                c[k] = zero(CoefType)
            end
        end
    end
    return c, min_ord
end



#

function _contour(f, a, ν::Tuple)
    N_fft = fft_size(space(a))

    CoefType = complex(eltype(a))
    grid_a_δ = zeros(CoefType, N_fft)

    A = _no_alloc_reshape(a)
    @inbounds view(grid_a_δ, axes(A)...) .= A # exact.(mid.(A))
    _apply!(_preprocess!, grid_a_δ, space(a))
    _apply_boxes!(grid_a_δ, ν)

    _fft_pow2!(grid_a_δ)
    contour_integral = sum(abs ∘ f, grid_a_δ)

    return contour_integral / exact(prod(N_fft))
end

_apply_boxes!(C::AbstractArray{T,N₁}, ν::NTuple{N₂}) where {T,N₁,N₂} =
    @inbounds _boxes!(_apply_boxes!(C, Base.tail(ν)), ν[1], Val(N₁-N₂+1))
_apply_boxes!(C::AbstractArray{T,N}, ν::NTuple{1}) where {T,N} =
    @inbounds _boxes!(C, ν[1], Val(N))
_apply_boxes!(C::AbstractVector, ν) = _boxes!(C, ν)

function _boxes!(C, μ::Interval)
    len = length(C)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len))) # 1/N_fft should be an exact operation
    δ = interval(-val, val)
    @inbounds for k ∈ 1:len÷2-1
        C[k+1]     *= μ ^ exact(-k) * cispi(exact(-k) * δ)
        C[len+1-k] *= μ ^ exact( k) * cispi(exact( k) * δ)
    end
    return C
end
function _boxes!(C, ν)
    len = length(C)
    @inbounds for k ∈ 1:len÷2-1
        C[k+1]     *= ν ^ exact(-k)
        C[len+1-k] *= ν ^ exact( k)
    end
    return C
end

function _boxes!(C, μ::Interval, ::Val{D}) where {D}
    len = size(C, D)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len))) # 1/N_fft should be an exact operation
    δ = interval(-val, val)
    @inbounds for k ∈ 1:len÷2-1
        selectdim(C, D, k+1)     .*= μ ^ exact(-k) * cispi(exact(-k) * δ)
        selectdim(C, D, len+1-k) .*= μ ^ exact( k) * cispi(exact( k) * δ)
    end
    return C
end
function _boxes!(C, ν, ::Val{D}) where {D}
    len = size(C, D)
    @inbounds for k ∈ 1:len÷2-1
        selectdim(C, D, k+1)     .*= ν ^ exact(-k)
        selectdim(C, D, len+1-k) .*= ν ^ exact( k)
    end
    return C
end



# error on the FFT

function _error(f, a, approx, ν, ν̄, N_v)
    ν̄⁻¹ = inv(ν̄)

    C = max(_contour(f, a, ν̄), _contour(f, a, ν̄⁻¹))

    q = sum(k -> (ν̄ ^ exact(k) + ν̄⁻¹ ^ exact(k)) * ν ^ exact(abs(k)), -N_v:N_v)

    return C, q / prod(ν̄ ^ exact( fft_size(space(approx)) ) - exact(1)) + exact(2) * ν̄ / (ν̄ - ν) * (ν * ν̄⁻¹) ^ exact(N_v + 1)
end

function _error(f, a, approx, ν::NTuple{N}, ν̄, N_v) where {N}
    ν̄⁻¹ = inv.(ν̄)
    _tuple_ = tuple(ν̄, ν̄⁻¹)
    _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(N))...)

    C = maximum(μ -> _contour(f, a, μ), _mix_)

    q = sum(k -> sum(μ -> prod(μ .^ exact.(k)), _mix_) * prod(ν .^ exact.(abs.(k))), TensorIndices(ntuple(i -> -N_v[i]:N_v[i], Val(N))))

    return C, q / prod(ν̄ .^ exact.( fft_size(space(approx)) ) .- exact(1)) + exact(2^N) * prod(ν̄ ./ (ν̄ .- ν) .* (ν .* ν̄⁻¹) .^ exact.(N_v .+ 1))
end
