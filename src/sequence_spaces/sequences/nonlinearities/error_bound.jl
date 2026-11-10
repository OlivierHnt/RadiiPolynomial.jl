# error of the grid evaluation of f ∘ a, by Cauchy's estimate on the polyannulus of radii μ

# finite error, tail error, input error, and the order of the coefficients kept
function _grid_error(f, a, c, ν, ε, ρ, μ, circle_points = 2^5)
    C = maximum(μᵢ -> _contour(f, a, μᵢ), _polyannulus_corners(μ))
    N_v = _noise_trim(c, CauchyCeiling(C, μ), ChargeTail(mid.(ν) ./ mid.(μ)))
    finite_alias, tail_alias = _alias_and_tail(_coef_fft_size(space(c), space(a)), ν, μ, N_v)
    pert = _safe_iszero(ε) ? zero(C * tail_alias) : _perturbation_bound(f, a, ρ, ν, μ, circle_points) * ε
    return C * finite_alias, C * tail_alias, pert, N_v
end

# ∏ νᵢ^|kᵢ| / ν̄ᵢ^|kᵢ| summed over the indices folded by the grid onto the box |kᵢ| ≤ Nᵢ, and over the
# complement of the box; per direction, the folds sum to Σ ((νᵢν̄ᵢ)^|k| + (νᵢ/ν̄ᵢ)^|k|) / (ν̄ᵢ^N_fftᵢ - 1)
function _alias_and_tail(N_fft::NTuple{N,Int}, ν::NTuple{N}, ν̄, N_v::NTuple{N,Int}) where {N}
    box = map(_geometric_box, ν ./ ν̄, N_v)
    folds = (map(_geometric_box, ν .* ν̄, N_v) .+ box) ./ (ν̄ .^ exact.(N_fft) .- exact(1))
    return _prod_with_one_of(box, folds), _prod_with_one_of(box, map(_geometric_tail, ν ./ ν̄, N_v))
end

# Σ_{|k| ≤ N} ρ^|k| and Σ_{|k| > N} ρ^|k|
_geometric_box(ρ, N) = N < 0 ? zero(ρ) : (exact(1) + ρ - exact(2) * ρ ^ exact(N + 1)) / (exact(1) - ρ)
_geometric_tail(ρ, N) = exact(2) * ρ ^ exact(N + 1) / (exact(1) - ρ)

# ∏ᵢ (xᵢ + yᵢ) - ∏ᵢ xᵢ without the cancellation
function _prod_with_one_of(x::NTuple{N}, y::NTuple{N}) where {N}
    p = zero(x[1] * y[1])
    p_x = one(x[1])
    @inbounds for i ∈ 1:N
        p = p * (x[i] + y[i]) + p_x * y[i]
        p_x = p_x * x[i]
    end
    return p
end

# contour constant of f within ρ of the image of a, carried from ν̄ to the norm at ν
_perturbation_bound(f, a::Sequence, ρ, ν, ν̄, circle_points = 2^5) =
    maximum(z -> maximum(μ -> _contour(f, a + z, μ), _polyannulus_corners(ν̄)), _circle(eltype(a), ρ, circle_points)) *
        prod((ν̄ .+ ν) ./ (ν̄ .- ν))

# circle of radius ρ: enclosed in intervals, sampled in floating point
_circle(::Type{<:RealOrComplexI}, ρ, _) = (ρ * cispi(interval(IntervalArithmetic.numtype(ρ), -1, 1)),)
_circle(::Type, ρ, n) = (_mag(ρ) * cispi(2j / n) for j ∈ 0:n-1)

_polyannulus_corners(ν::NTuple{N}) where {N} =
    @inbounds Iterators.product(ntuple(i -> (ν[i], inv(ν[i])), Val(N))...)

# mean of |f ∘ a| over the polycircle of radii μ
function _contour(f, a, μ::Tuple)
    A = _radius_grid(a, μ, _oversampled_fft_size(space(a)))
    return sum(abs ∘ f, A) / exact(length(A))
end

# values of a on the grid of the polycircle of radii μ; an interval radius encloses a on the arc around
# each node
_radius_grid(a::Sequence, μ::Tuple, N_fft) =
    _to_grid!(zeros(complex(float(promote_type(eltype(a), typeof.(μ)...))), N_fft), a, μ)

_apply_boxes!(C::AbstractArray{T,N₁}, ν::NTuple{N₂}) where {T,N₁,N₂} =
    @inbounds _boxes!(_apply_boxes!(C, Base.tail(ν)), ν[1], Val(N₁-N₂+1))
_apply_boxes!(C::AbstractArray{T,N}, ν::NTuple{1}) where {T,N} =
    @inbounds _boxes!(C, ν[1], Val(N))

# the mode k of direction D is scaled by μᵏ, and by the arc of angles k [-1/N_fft, 1/N_fft] for an interval μ
function _boxes!(C, μ::Interval, ::Val{D}) where {D}
    len = size(C, D)
    val = sup(inv(interval(IntervalArithmetic.numtype(μ), len)))
    δ = interval(-val, val)
    @inbounds for k ∈ 1:len÷2-1
        arc = cispi(exact(k) * δ)
        selectdim(C, D, k+1)     .*= μ ^ exact( k) * arc
        selectdim(C, D, len+1-k) .*= μ ^ exact(-k) * arc
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
_scaled(x, s) = iszero(x) ? x : x * s # 0 * Inf
