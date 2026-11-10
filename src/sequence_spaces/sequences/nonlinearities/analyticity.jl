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

_sweep_segment(nl::Nonlinearity, a::Sequence, r::Tuple{Interval,Vararg{Interval}}, image_radius, sz) =
    __check_branch_cut_poles(nl, a, r, image_radius, sz)

function _sweep_segment(nl::Nonlinearity, a::Sequence, r::Tuple{Tuple,Vararg{Tuple}}, image_radius, sz)
    # in floating point only r₂ is evaluated: the image radius grows by the drift of the image over [r₁, r₂]
    r₁ = map(first, r)
    r₂ = map(last, r)
    return __check_branch_cut_poles(nl, a, r₂, image_radius + _segment_margin(a, r₁, r₂), sz)
end

function _segment_margin(a::Sequence, r₁::Tuple, r₂::Tuple)
    # |a(ρ, θ) − a(ρ′, θ)| ≤ Σₖ |aₖ| (max − min of ∏ᵢ ρᵢ^kᵢ over [r₁, r₂])
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
    return all(_radius_grid(a, r, sz)) do x
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
