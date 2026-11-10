# a computed coefficient is noise below a floor, the rounding level, or above a ceiling, a bound on the
# true coefficient

abstract type NoiseBound end
abstract type NoiseCeiling <: NoiseBound end

# 64 ulps of the largest coefficient
struct NoiseFloor <: NoiseBound end

# Cauchy's estimate |cₖ| ≤ C / ∏ μᵢ^|kᵢ|
struct CauchyCeiling{T,M<:Tuple} <: NoiseCeiling
    C :: T
    μ :: M
end

# |cₖ| ≤ bound / wₖ, for a norm at most `bound` in X
struct BanachCeiling{T,S<:BanachSpace} <: NoiseCeiling
    bound :: T
    X     :: S
end

# ‖a b‖ ≤ ‖a‖ ‖b‖ in the space fitted to the factors
function BanachCeiling(a::Sequence, b::Sequence)
    X = Ell1(weight(a)) ∩ Ell1(weight(b))
    return BanachCeiling(norm(a, X) * norm(b, X), X)
end
function BanachCeiling(a::Sequence, n::Integer)
    X = Ell1(weight(a))
    return BanachCeiling(norm(a, X) ^ n, X)
end

_magnitude(c::Sequence) = maximum(x -> abs(mid(x)), coefficients(c))
_noise_floor(c::Sequence) = 64 * eps(_magnitude(c))

# ceiling of the constant coefficient
_scale(e::CauchyCeiling) = e.C
_scale(e::BanachCeiling) = e.bound

_ceiling(e::CauchyCeiling, ::SequenceSpace, k) = e.C / prod(e.μ .^ abs.(k))
_ceiling(e::BanachCeiling, s::SequenceSpace, k) = _coefficient_bound(e.X, e.bound, _getindex(weight(e.X), s, k))

# noise above a ceiling is clamped to it in a `Sequence`, and left to the tail, of decay ρ, in an
# `InfiniteSequence`
struct ClampCoefficients end
struct ChargeTail{R<:Tuple}
    ρ :: R
end

# the smallest order keeping every coefficient above the floor
function _noise_trim(c::Sequence, ::NoiseFloor)
    floor = _noise_floor(c)
    N = Tuple(order(space(c)))
    M = ntuple(i -> maximum(k -> ifelse(abs(mid(c[k])) > floor, abs(Tuple(k)[i]), 0), indices(space(c))), length(N))
    return M == N ? c : project(c, _oforder(space(c), M))
end

# Banach rounding: a coefficient whose ceiling is below one ulp of the scale becomes [-ceiling, ceiling]
function _noise_trim!(c::Sequence, e::NoiseCeiling, ::ClampCoefficients)
    s = space(c)
    scale = float(sup(_scale(e)))
    isfinite(scale) || return c
    level = eps(typeof(scale)) * scale
    @inbounds for k ∈ indices(s)
        μₖ = _ceiling(e, s, k)
        if sup(μₖ) ≤ level
            c[k] = _envelope_box(eltype(c), μₖ)
        end
    end
    return c
end

# the order of the box |kᵢ| ≤ Nᵢ leaving out the noise at the smallest tail
function _noise_trim(c::Sequence, e::NoiseCeiling, tail::ChargeTail)
    s = space(c)
    N = Tuple(order(s))
    noise = [Tuple(k) for k ∈ indices(s) if _mag(c[k]) > _mag(_ceiling(e, s, k))]
    return isempty(noise) ? N : _least_tail_box(noise, N, tail.ρ)
end

# every order of the first directions is tried; the last direction takes the largest order leaving out
# the noise they keep
function _least_tail_box(noise, N::NTuple{D,Int}, ρ::NTuple{D}) where {D}
    tail(box) = _prod_with_one_of(map(_geometric_box, ρ, box), map(_geometric_tail, ρ, box))
    best_box, best_tail = ntuple(_ -> -1, Val(D)), Inf
    function visit(box, kept)
        i = length(box) + 1
        if i == D
            b = (box..., minimum(k -> abs(k[D]) - 1, kept; init = N[D]))
            t = tail(b)
            if t < best_tail
                best_box, best_tail = b, t
            end
        else
            for Nᵢ ∈ -1:N[i]
                visit((box..., Nᵢ), filter(k -> abs(k[i]) ≤ Nᵢ, kept))
            end
        end
    end
    visit((), noise)
    return best_box
end
