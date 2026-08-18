_maybe_interval(::Type, a) = a
_maybe_interval(::Type{<:RealOrComplexI}, a) = interval(a)

_isguaranteed(a::InfiniteSequence) = all(isguaranteed, sequence(a)) & isguaranteed(sequence_norm(a)) & isguaranteed(finite_error(a)) & isguaranteed(tail_error(a)) & isguaranteed(total_error(a))

# Every nonlinearity here is evaluated pointwise on a grid and transformed back, so the FFT
# returns the *aliased* coefficients Σ_j c_{k+jm} rather than c_k, m being the grid size. In
# `_error` the resulting aliasing term decays like ν̄^{-m} in the grid size, whereas the
# truncation term decays like (ν/ν̄)^N in the order N. Sampling at twice the grid the order
# alone would call for squares ν̄^{-m}, which is what keeps aliasing negligible next to
# truncation
_oversampled_grid_size(s::SequenceSpace) = fast_grid_size(2 .* grid_size(s), s)
_oversampled_fft_size(s::SequenceSpace) = _full_fft_size(_oversampled_grid_size(s), s)

# division

_codomain(::typeof(inv), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain(inv, sᵢ), spaces(s)))
_codomain(::typeof(inv), s::Taylor) = s
_codomain(::typeof(inv), s::Fourier) = s
_codomain(::typeof(inv), s::Chebyshev) = s

function Base.inv(a::Sequence)
    space_approx = _codomain(inv, space(a))
    _isconstant(a) && return _at_value(inv, a)
    A = to_grid(a, _oversampled_grid_size(space_approx))
    A .= inv.(A)
    return _call_to_coef!(A, space_approx, eltype(a))
end

function Base.inv(a::InfiniteSequence)
    # TODO: propagate "NG" flag

    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(total_error(a)) && return InfiniteSequence(_at_value(inv, seq_a), banachspace(a))

    seq_approx_a⁻¹ = inv(mid.(seq_a))

    X = banachspace(a)
    approx_a⁻¹ = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_a⁻¹), X)

    f = approx_a⁻¹ * a - exact(1)
    Y = norm(approx_a⁻¹ * f)
    Z₁ = norm(f)
    r, _ = interval_of_existence(Y, Z₁, Inf; verbose = false)

    err = _maybe_interval(eltype(a), inf(r))
    return InfiniteSequence(sequence(approx_a⁻¹), X; total_error = err)
end

_codomain(::typeof(/), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> _codomain(/, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))
_codomain(::typeof(/), s₁::Taylor, s₂::Taylor) = union(s₁, s₂)
_codomain(::typeof(/), s₁::Fourier, s₂::Fourier) = union(s₁, s₂)
_codomain(::typeof(/), s₁::Chebyshev, s₂::Chebyshev) = union(s₁, s₂)

function Base.:/(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    # TODO: propagate "NG" flag
    space_approx = _codomain(/, space(a), space(b))
    _isconstant(b) && return a / b[_findindex_constant(space(b))]
    A = to_grid(a, _oversampled_grid_size(space_approx))
    B = to_grid(b, _oversampled_grid_size(space_approx))
    A .= A ./ B
    return _call_to_coef!(A, space_approx, promote_type(eltype(a), eltype(b)))
end
Base.:/(a::Number, b::Sequence{<:SequenceSpace}) = lmul!(a, inv(b))

function Base.:/(a::InfiniteSequence, b::InfiniteSequence)
    # TODO: propagate "NG" flag

    space_approx = _codomain(/, space(a), space(b))

    seq_a = sequence(a)
    seq_b = sequence(b)
    _isconstant(seq_b) & _safe_iszero(total_error(b)) && return InfiniteSequence(seq_a / seq_b[_findindex_constant(space(seq_b))], banachspace(a))

    A = to_grid(mid.(seq_a), _oversampled_grid_size(space_approx))
    B = to_grid(mid.(seq_b), _oversampled_grid_size(space_approx))
    A .= A ./ B
    B .= inv.(B)
    CoefType = promote_type(eltype(a), eltype(b))
    seq_approx_ab⁻¹ = _call_to_coef!(A, space_approx, CoefType)
    seq_approx_b⁻¹ = _call_to_coef!(B, space_approx, eltype(b))

    X = banachspace(a) ∩ banachspace(b)
    approx_ab⁻¹ = InfiniteSequence(_maybe_interval(CoefType, seq_approx_ab⁻¹), X)
    approx_b⁻¹ = InfiniteSequence(_maybe_interval(CoefType, seq_approx_b⁻¹), X)

    Y = norm(approx_b⁻¹ * (approx_ab⁻¹ * b - a))
    Z₁ = norm(approx_b⁻¹ * b - exact(1))
    r, _ = interval_of_existence(Y, Z₁, Inf; verbose = false)

    err = _maybe_interval(CoefType, inf(r))
    return InfiniteSequence(sequence(approx_ab⁻¹), X; total_error = err)
end
Base.:/(a::Number, b::InfiniteSequence) = lmul!(a, inv(b))


_codomain(::typeof(\), s₁::SequenceSpace, s₂::SequenceSpace) = codomain(/, s₂, s₁)

Base.:\(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}) = b / a
Base.:\(a::Sequence{<:SequenceSpace}, b::Number) = b / a

Base.:\(a::InfiniteSequence, b::InfiniteSequence) = b / a
Base.:\(a::InfiniteSequence, b::Number) = b / a



# square root

_codomain(::typeof(sqrt), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain(sqrt, sᵢ), spaces(s)))
_codomain(::typeof(sqrt), s::Taylor) = s
_codomain(::typeof(sqrt), s::Fourier) = s
_codomain(::typeof(sqrt), s::Chebyshev) = s

function Base.sqrt(a::Sequence{<:SequenceSpace})
    space_approx = _codomain(sqrt, space(a))
    _isconstant(a) && return _at_value(sqrt, a)
    A = to_grid(a, _oversampled_grid_size(space_approx))
    A .= sqrt.(A)
    return _call_to_coef!(A, space_approx, eltype(a))
end

function Base.sqrt(a::InfiniteSequence)
    # TODO: propagate "NG" flag

    space_approx = _codomain(sqrt, space(a))

    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(total_error(a)) && return InfiniteSequence(_at_value(sqrt, seq_a), banachspace(a))

    A = to_grid(mid.(seq_a), _oversampled_grid_size(space_approx))
    sqrtA = sqrt.(A)
    A .= inv.(sqrtA)
    seq_approx_sqrta = _call_to_coef!(sqrtA, space_approx, eltype(a))
    seq_approx_sqrta⁻¹ = _call_to_coef!(A, space_approx, eltype(a))

    X = banachspace(a)
    approx_sqrta = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_sqrta), X)
    approx_sqrta⁻¹ = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_sqrta⁻¹), X)

    Y = norm(approx_sqrta⁻¹ * (approx_sqrta ^ 2 - a)) / exact(2)
    Z₁ = norm(approx_sqrta⁻¹ * approx_sqrta - exact(1))
    Z₂ = norm(approx_sqrta⁻¹)
    r, _ = interval_of_existence(Y, Z₁, Z₂, Inf; verbose = false)

    err = _maybe_interval(eltype(a), inf(r))
    return InfiniteSequence(sequence(approx_sqrta), X; total_error = err)
end



# cube root

_codomain(::typeof(cbrt), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain(cbrt, sᵢ), spaces(s)))
_codomain(::typeof(cbrt), s::Taylor) = s
_codomain(::typeof(cbrt), s::Fourier) = s
_codomain(::typeof(cbrt), s::Chebyshev) = s

function Base.cbrt(a::Sequence{<:SequenceSpace})
    space_approx = _codomain(cbrt, space(a))
    _isconstant(a) && return _at_value(cbrt, a)
    A = to_grid(a, _oversampled_grid_size(space_approx))
    A .= A .^ (1//3)
    return _call_to_coef!(A, space_approx, eltype(a))
end

function Base.cbrt(a::InfiniteSequence)
    # TODO: propagate "NG" flag

    space_approx = _codomain(cbrt, space(a))

    seq_a = sequence(a)
    _isconstant(seq_a) & _safe_iszero(total_error(a)) && return InfiniteSequence(_at_value(cbrt, seq_a), banachspace(a))

    A = to_grid(mid.(seq_a), _oversampled_grid_size(space_approx))
    cbrtA = A .^ (1//3)
    A .= inv.(cbrtA) .^ 2
    seq_approx_cbrta = _call_to_coef!(cbrtA, space_approx, eltype(a))
    seq_approx_cbrta⁻² = _call_to_coef!(A, space_approx, eltype(a))

    X = banachspace(a)
    approx_cbrta = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_cbrta), X)
    approx_cbrta⁻² = InfiniteSequence(_maybe_interval(eltype(a), seq_approx_cbrta⁻²), X)

    approx_cbrta² = approx_cbrta ^ 2
    Y = norm(approx_cbrta⁻² * (approx_cbrta² * approx_cbrta - a)) / exact(3)
    Z₁ = norm(approx_cbrta⁻² * approx_cbrta² - exact(1))
    R = 3/2 * sup(Y)/(1 - sup(Z₁))
    Z₂ = norm(approx_cbrta⁻²) * (exact(2) * norm(approx_cbrta) + exact(R))
    r, _ = interval_of_existence(Y, Z₁, Z₂, R; verbose = false)

    err = _maybe_interval(eltype(a), inf(r))
    return InfiniteSequence(sequence(approx_cbrta), X; total_error = err)
end



# general nonlinearites

"""
    Nonlinearity(f, poles, branch_cut)

A function `f`, together with the set where it fails to be analytic: the `poles`
and the `branch_cut`. Applied to a `Sequence` it evaluates `f` on a grid and
transforms back; applied to an `InfiniteSequence` it also bounds the error of
doing so, which is only possible while the contours it reads `f` off stay clear
of `poles` and of `branch_cut`.
"""
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

function Base.:^(a::Union{Sequence{<:SequenceSpace}, InfiniteSequence}, p::Real)
    isinteger(p) && return a ^ Integer(p)
    p == 1//2 && return sqrt(a)
    p == 1//3 && return cbrt(a)
    return Nonlinearity(x -> x ^ p, Complex{Interval{Float64}}[], interval(-Inf, 0))(a; codomain = _codomain(^, space(a), p))
end



# entire functions

_codomain(::typeof(exp), s::Taylor) = s
_codomain(::typeof(exp), s::Fourier) = s
_codomain(::typeof(exp), s::Chebyshev) = s

_codomain(::typeof(cos), s::Taylor) = s
_codomain(::typeof(cos), s::Fourier) = s
_codomain(::typeof(cos), s::Chebyshev) = s

_codomain(::typeof(sin), s::Taylor) = s
_codomain(::typeof(sin), s::Fourier) = s
_codomain(::typeof(sin), s::Chebyshev) = s

_codomain(::typeof(cosh), s::Taylor) = s
_codomain(::typeof(cosh), s::Fourier) = s
_codomain(::typeof(cosh), s::Chebyshev) = s

_codomain(::typeof(sinh), s::Taylor) = s
_codomain(::typeof(sinh), s::Fourier) = s
_codomain(::typeof(sinh), s::Chebyshev) = s

for f ∈ (:exp, :cos, :sin, :cosh, :sinh)
    @eval begin
        _codomain(::typeof($f), s::TensorSpace) = TensorSpace(map(sᵢ -> _codomain($f, sᵢ), spaces(s)))

        Nonlinearity(::typeof($f)) = Nonlinearity($f, Complex{Interval{Float64}}[], emptyinterval(Float64)) # entire functions

        Base.$f(a::Sequence{<:SequenceSpace}) = Nonlinearity($f)(a)

        Base.$f(a::InfiniteSequence) = Nonlinearity($f)(a)
    end
end

function (nl::Nonlinearity)(a::Sequence{<:SequenceSpace}; codomain::SequenceSpace = _codomain(nl.f, space(a)))
    _isconstant(a) && return _at_value(nl.f, a)
    return _image_coefficients(nl.f, a, codomain, eltype(a))
end

function (nl::Nonlinearity)(a::InfiniteSequence; codomain::SequenceSpace = _codomain(nl.f, space(a)))
    seq_a = sequence(a)
    ε = total_error(a)
    _isconstant(seq_a) & _safe_iszero(ε) && return InfiniteSequence(_at_value(nl.f, seq_a), banachspace(a))

    ν = _analyticity_rate(banachspace(a))
    _validate_rate(desymmetrize(space(a)), ν)

    ρ = _safe_iszero(ε) ? ε : exact(1) + ε

    _check_branch_cut_poles(nl, seq_a, ν, ρ) ||
        return throw(ArgumentError(_safe_iszero(ε) ?
            "the image of the ν = $(ν) contour intersects a branch cut or contains at least one pole: analyticity violated" :
            "the input error cannot be propagated: the Cauchy estimate needs f analytic on the disc of radius r⋆ = 1 + total_error(a) = $(ρ) around the image of the ν = $(ν) contour, and that disc meets a branch cut or a pole"))

    c = _image_coefficients(nl.f, seq_a, codomain, eltype(a))

    ν̄ = _maybe_interval.(eltype(a), _optimize_decay(nl, c, a, ρ))

    N_v = _saturation_order(nl.f, c, seq_a, ν̄)

    C, finite_alias, tail_alias = _error(nl.f, seq_a, c, ν, ν̄, N_v)
    finite_err = C * finite_alias
    tail_err   = C * tail_alias
    total_err  = finite_err + tail_err

    if N_v != order(c)
        c = project(c, _oforder(space(c), max.(N_v, 0)))
        if any(N_v .< 0)
            coefficients(c) .= zero(eltype(c))
            finite_err += tail_err
        end
    end

    if !_safe_iszero(ε)
        pert = _perturbation_bound(nl.f, seq_a, ρ, ν, ν̄) * ε
        finite_err += pert
        tail_err   += pert
        total_err  += pert
    end

    return InfiniteSequence(c, banachspace(a);
        finite_error = finite_err,
        tail_error   = tail_err,
        total_error  = total_err)
end

#-

_image_coefficients(f, a::Sequence, codomain::SequenceSpace, ::Type{T}) where {T} =
    _call_to_coef!(f.(to_grid(a, _oversampled_grid_size(codomain))), codomain, T)

#-

# one entry per direction, a 1-tuple for base spaces: ν keeps this shape throughout
_analyticity_rate(X::BanachSpace) = Tuple(_analyticity_rate(weight(X)))
_analyticity_rate(w::Tuple{Vararg{Weight}}) = map(_analyticity_rate, w)
_analyticity_rate(w::GeometricWeight) = rate(w)
_analyticity_rate(::IdentityWeight) = exact(1) # valid only for Taylor
_analyticity_rate(w::Weight) = throw(ArgumentError("analyticity check requires a geometric weight, i.e. ω_k = ν^{|k|}; got $(_prettystring(w))"))

#-

_mag(x::Union{Interval,Complex{<:Interval}}) = mag(x)
_mag(x) = abs(x)

#-

_polyannulus_corners(ν::NTuple{N}) where {N} =
    @inbounds Iterators.product(ntuple(i -> (ν[i], inv(ν[i])), Val(N))...)





# checking branch cuts and poles

# `image_radius` is how far around the image of `a` the exclusion sets must be
function _check_branch_cut_poles(nl::Nonlinearity, a::Sequence, ν::Tuple, image_radius)
    _isentire(nl) && return true
    s = space(a)
    _validate_rate(s, ν)
    n = _sweep_length(s, ν)
    sz = _sweep_fft_size(s, ν)
    for r ∈ Iterators.product(_radii_sweeps(s, ν, n)...)
        _sweep_segment(nl, a, r, image_radius, sz) || return false
    end
    return true
end

_sweep_segment(nl::Nonlinearity, a::Sequence, r::Tuple{Vararg{Interval}}, image_radius, sz) =
    __check_branch_cut_poles(nl, a, r, image_radius, sz)

function _sweep_segment(nl::Nonlinearity, a::Sequence, r::Tuple{Vararg{Tuple}}, image_radius, sz)
    # for float, given the multi-dim segment [r₁, r₂], the FFT only runs at r₂
    # to compensate the neighborhood of the image that must keep clear of the
    # exclusion sets is inflated by the largest drift the image can undergo
    # across the segment, so the r₂ test covers every radius left unevaluated
    r₁ = map(first, r)
    r₂ = map(last, r)
    return __check_branch_cut_poles(nl, a, r₂, image_radius + _segment_margin(a, r₁, r₂), sz)
end

function _segment_margin(a::Sequence, r₁::Tuple, r₂::Tuple)
    # a(ρ, θ) = Σₖ aₖ · (∏ᵢ ρᵢ^{kᵢ}) · e^{i k·θ}
    # |a(ρ, θ) − a(ρ′, θ)| ≤ Σₖ |aₖ| · (hiₖ − loₖ) where hiₖ and loₖ are the
    # largest and smallest values ∏ᵢ ρᵢ^{kᵢ} takes over the multi-box [r₁, r₂]
    radial = zero(real(float(eltype(a))))
    for k ∈ indices(space(a))
        c = abs(a[k])
        iszero(c) && continue
        p₁ = r₁ .^ k
        p₂ = r₂ .^ k
        radial += c * (prod(max.(p₁, p₂)) - prod(min.(p₁, p₂)))
    end
    return radial
end

function __check_branch_cut_poles(nl::Nonlinearity, a::Sequence, r::Tuple, image_radius, sz)
    CoefType = complex(float(promote_type(eltype(a), typeof.(r)...)))
    C = zeros(CoefType, sz)
    A = _no_alloc_reshape(a)
    @inbounds view(C, axes(A)...) .= A
    _apply!(_preprocess_to_grid!, C, space(a))
    _apply_boxes!(C, r)
    _fft!(C) # forward FFT = values at the conjugate nodes
    return all(C) do x
        _isfinite_grid_value(x) || return false
        y = interval(x, image_radius; format = :midpoint)
        return isdisjoint_interval(y, nl.branch_cut) & all(p -> isdisjoint_interval(y, p), nl.poles)
    end
end
_isfinite_grid_value(::Union{Interval,Complex{<:Interval}}) = true
_isfinite_grid_value(x) = isfinite(x)

#-

_isentire(nl::Nonlinearity) = isempty_interval(nl.branch_cut) & isempty(nl.poles)

_validate_rate(::Taylor,    ν::Real) = inf(ν) ≥ 1 || throw(ArgumentError("Taylor analyticity check requires ν ≥ 1; got ν = $ν"))
_validate_rate(::Fourier,   ν::Real) = inf(ν) > 1 || throw(ArgumentError("Fourier analyticity check requires ν > 1; got ν = $ν"))
_validate_rate(::Chebyshev, ν::Real) = inf(ν) > 1 || throw(ArgumentError("Chebyshev analyticity check requires ν > 1; got ν = $ν"))
_validate_rate(s::BaseSpace, ν::Tuple{Any}) = _validate_rate(s, ν[1])
_validate_rate(s::TensorSpace, ν::Tuple) = foreach(_validate_rate, spaces(s), ν)

#-

_sweep_length(s::SequenceSpace, ::Tuple{Vararg{Interval}}) =
    max(maximum(fft_size(s)), 64)
_sweep_length(::SequenceSpace, ::NTuple{N,Any}) where {N} =
    max(2, floor(Int, 64^(1/N)))

_sweep_fft_size(s::SequenceSpace, ::Tuple{Vararg{Interval}}) =
    _oversampled_fft_size(s)
_sweep_fft_size(s::SequenceSpace, ::NTuple{N,Any}) where {N} =
    2^cld(3, N) .* _oversampled_fft_size(s)


_radii_sweeps(s::BaseSpace, ν::Tuple{Any}, n) = (_direction_sweep(s, ν[1], n),)
_radii_sweeps(s::TensorSpace{<:NTuple{N,BaseSpace}}, ν::NTuple{N,Any}, n) where {N} =
    map((sᵢ, νᵢ) -> _direction_sweep(sᵢ, νᵢ, n), spaces(s), ν)

_direction_sweep(::Taylor,    ν::Interval, n) = mince(interval(0, sup(ν)), n)
_direction_sweep(::Fourier,   ν::Interval, n) = mince(hull(inv(ν), ν), n)
_direction_sweep(::Chebyshev, ν::Interval, n) = mince(interval(1, sup(ν)), n)

_direction_sweep(::Taylor,    ν, n) = _consecutive_pairs(LinRange(zero(ν), ν, n))
_direction_sweep(::Fourier,   ν, n) = _consecutive_pairs(LinRange(inv(ν), ν, n))
_direction_sweep(::Chebyshev, ν, n) = _consecutive_pairs(LinRange(one(ν), ν, n))
_consecutive_pairs(collection) = zip(collection, Iterators.drop(collection, 1))





# finding the auxiliary radius ν̄ by golden search

function _optimize_decay(nl::Nonlinearity, c, aa, ρ)
    a = sequence(aa)
    ν = _analyticity_rate(banachspace(aa))
    ν_sup = _mag.(ν)
    a_mid = mid.(a)
    c_mid = mid.(c)
    rate_a = Tuple(_geometric_rate(space(a), coefficients(a_mid))[1])
    rate_c = Tuple(_geometric_rate(space(c), coefficients(c_mid))[1])
    ν̄_max = max.(max.(rate_a, rate_c), 2 .* ν_sup)

    function _score(μ)
        N_v = _saturation_order(nl.f, c_mid, a_mid, μ)
        C, finite_alias, tail_alias = _error(nl.f, a_mid, c_mid, ν_sup, μ, N_v)
        return _mag(C * (finite_alias + tail_alias))
    end

    if _isentire(nl)
        ν̄ = _golden_search(_score, ν_sup, ν̄_max)
    else
        ρ_sup = _mag(ρ)
        ν̄ = _golden_search(ν_sup, ν̄_max) do μ
            _check_branch_cut_poles(nl, a_mid, μ, ρ_sup) ? _score(μ) : Inf
        end
        if !_check_branch_cut_poles(nl, a, interval.(ν̄), ρ)
            ν̄ = _golden_search(ν_sup, ν̄_max) do μ
                _check_branch_cut_poles(nl, a, interval.(μ), ρ) ? _score(μ) : Inf
            end
            _check_branch_cut_poles(nl, a, interval.(ν̄), ρ) ||
                return throw(ArgumentError("no auxiliary radius ν̄ between ν = $(ν_sup) and $(ν̄_max) keeps `f` analytic within $(ρ) of the contour: analyticity violated"))
        end
    end

    all(ν̄ .> ν_sup) || return throw(ArgumentError("the auxiliary radius ν̄ = $(ν̄) must exceed the analyticity rate ν = $(ν_sup) in every direction"))

    return ν̄
end

#

function _golden_search(f, a, b)
    ϕ = (sqrt(5) - 1) / 2 # ≈ 0.618
    # the objective is flat around its minimum, so a percent of the initial bracket is as far as
    # it is worth splitting hairs; an absolute tolerance would instead scale with ν̄_max
    tol = abs(b - a) / 100
    c = b - ϕ * (b - a)
    d = a + ϕ * (b - a)
    fc = f(c)
    fd = f(d)
    best_x, best_f = fd < fc ? (d, fd) : (c, fc)
    iter = 0

    while abs(b - a) > tol && iter < 12
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
        if fc < best_f
            best_f, best_x = fc, c
        end
        if fd < best_f
            best_f, best_x = fd, d
        end
        iter += 1
    end

    # check if midpoint of the final bracket is the best point
    m = (a + b) / 2
    fm = f(m)
    if fm < best_f
        best_f, best_x = fm, m
    end

    # only when every probe was infeasible is there nothing better to offer than the midpoint
    return isfinite(best_f) ? best_x : m
end

_golden_search(f, lower::NTuple{1}, upper::NTuple{1}) =
    (_golden_search(μ -> f((μ,)), lower[1], upper[1]),)

function _golden_search(f, lower::NTuple{N}, upper::NTuple{N}) where {N}
    x = ntuple(i -> (lower[i] + upper[i]) / 2, Val(N))
    for _ ∈ 1:8
        x_prev = x
        for i ∈ 1:N
            a, b = lower[i], upper[i]
            μᵢ = _golden_search(μ -> f(ntuple(j -> ifelse(j == i, μ, x[j]), Val(N))), a, b)
            x = ntuple(j -> ifelse(j == i, μᵢ, x[j]), Val(N))
        end
        all(ntuple(i -> abs(x[i] - x_prev[i]) ≤ abs(upper[i] - lower[i]) / 100, Val(N))) && break
    end
    return x
end





# polishing the coefficients

# to avoid numerical artefacts when the input sequence is constant
function _isconstant(a::Sequence)
    s = space(a)
    idx = _findindex_constant(s)
    return all(k -> ifelse(k == idx, true, _safe_iszero(a[k])), indices(s))
end
function _at_value(f, a)
    c = one(a)
    idx = _findindex_constant(space(a))
    c[idx] = f(a[idx])
    return c
end

# prevent numerical noise

function _saturation_order(f, c, a, ν::Tuple)
    C = maximum(μ -> _contour(f, a, μ), _polyannulus_corners(ν))
    N_v = order(c)
    isfinite(_mag(C)) || return N_v
    for k ∈ indices(space(c))
        if _mag(c[k]) > _mag(C / prod(ν .^ abs.(k)))
            N_v = min.(N_v, abs.(k) .- 1)
        end
    end
    return N_v
end

_oforder(::Taylor, n::Int) = Taylor(n)
_oforder(::Chebyshev, n::Int) = Chebyshev(n)
_oforder(s::Fourier, n::Int) = Fourier(n, frequency(s))
_oforder(s::TensorSpace, n::Tuple) = TensorSpace(map(_oforder, spaces(s), n))





# contour integrals

function _contour(f, a, ν::Tuple)
    N_fft = _oversampled_fft_size(space(a))

    CoefType = complex(eltype(a))
    grid_a_δ = zeros(CoefType, N_fft)

    A = _no_alloc_reshape(a)
    @inbounds view(grid_a_δ, axes(A)...) .= A
    _apply!(_preprocess_to_grid!, grid_a_δ, space(a))
    _apply_boxes!(grid_a_δ, ν)

    _fft!(grid_a_δ)
    contour_integral = sum(abs ∘ f, grid_a_δ)

    return contour_integral / exact(prod(N_fft))
end

_apply_boxes!(C::AbstractArray{T,N₁}, ν::NTuple{N₂}) where {T,N₁,N₂} =
    @inbounds _boxes!(_apply_boxes!(C, Base.tail(ν)), ν[1], Val(N₁-N₂+1))
_apply_boxes!(C::AbstractArray{T,N}, ν::NTuple{1}) where {T,N} =
    @inbounds _boxes!(C, ν[1], Val(N))
_apply_boxes!(C::AbstractVector, ν::NTuple{1}) = @inbounds _boxes!(C, ν[1])

function _boxes!(C, μ::Interval)
    len = length(C)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len))) # 1/N_fft should be an exact operation
    δ = interval(-val, val)
    @inbounds for k ∈ 1:len÷2-1
        C[k+1]     *= μ ^ exact( k) * cispi(exact( k) * δ)
        C[len+1-k] *= μ ^ exact(-k) * cispi(exact(-k) * δ)
    end
    return C
end

function _boxes!(C, μ::Interval, ::Val{D}) where {D}
    len = size(C, D)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len))) # 1/N_fft should be an exact operation
    δ = interval(-val, val)
    @inbounds for k ∈ 1:len÷2-1
        selectdim(C, D, k+1)     .*= μ ^ exact( k) * cispi(exact( k) * δ)
        selectdim(C, D, len+1-k) .*= μ ^ exact(-k) * cispi(exact(-k) * δ)
    end
    return C
end

function _boxes!(C, ν)
    len = length(C)
    @inbounds for k ∈ 1:len÷2-1
        C[k+1]     = _scaled(C[k+1],     ν ^ exact( k))
        C[len+1-k] = _scaled(C[len+1-k], ν ^ exact(-k))
    end
    return C
end
function _boxes!(C, ν, ::Val{D}) where {D}
    len = size(C, D)
    @inbounds for k ∈ 1:len÷2-1
        selectdim(C, D, k+1)     .= _scaled.(selectdim(C, D, k+1),     ν ^ exact( k))
        selectdim(C, D, len+1-k) .= _scaled.(selectdim(C, D, len+1-k), ν ^ exact(-k))
    end
    return C
end
_scaled(x, s) = iszero(x) ? x : x * s





# error of the grid evaluation

function _error(f, a, approx, ν::NTuple{N}, ν̄, N_v) where {N}
    ν̄⁻¹ = inv.(ν̄)
    corners = _polyannulus_corners(ν̄)

    C = maximum(μ -> _contour(f, a, μ), corners)

    # `init` because the box may be empty since a plateau reaching the constant
    # term keeps nothing and every index is charged to the tail
    q = sum(k -> sum(μ -> prod(μ .^ exact.(k)), corners) * prod(ν .^ exact.(abs.(k))),
            TensorIndices(ntuple(i -> -N_v[i]:N_v[i], Val(N))); init = zero(prod(ν̄) * prod(ν)))

    finite_alias = q / prod(ν̄ .^ exact.( _oversampled_fft_size(space(approx)) ) .- exact(1))
    tail_alias   = exact(2^N) * prod(ν̄ ./ (ν̄ .- ν) .* (ν .* ν̄⁻¹) .^ exact.(N_v .+ 1))

    return C, finite_alias, tail_alias
end

function _perturbation_bound(f, a::Sequence{<:SequenceSpace,<:AbstractVector{<:RealOrComplexI}}, ρ, ν, ν̄)
    circle = ρ * cispi(interval(IntervalArithmetic.numtype(ρ), -1, 1))
    return maximum(μ -> _contour(f, a + circle, μ), _polyannulus_corners(ν̄)) *
        prod((ν̄ .+ ν) ./ (ν̄ .- ν))
end

function _perturbation_bound(f, a::Sequence, ρ, ν, ν̄)
    n = 2^5
    W = maximum(j -> maximum(μ -> _contour(f, a + _mag(ρ) * cispi(2j / n), μ), _polyannulus_corners(ν̄)), 0:n-1)
    return W * prod((ν̄ .+ ν) ./ (ν̄ .- ν))
end
