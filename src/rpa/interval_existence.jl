"""
    interval_of_existence(Y::Interval, Z₁::Interval, R::Real; verbose::Bool=false)

Return an interval and a boolean value with the following meaning:
- `true`: the interval corresponds to ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r \\le 0`` for all ``r \\in I`` and ``Z_1 < 1``.
- `false`: otherwise, and the interval is empty.
"""
function interval_of_existence(Y::Interval, Z₁::Interval, R::Real; verbose::Bool=false)
    ie, success, msg = _interval_of_existence(Y, Z₁, zero(Z₁), R)
    verbose && @info (success ?
        "success: first-order radii polynomial Y + (Z₁ - 1) r\nY = $Y\nZ₁ = $Z₁ (bound on the ball of radius R)\nR = $R" :
        "failure: $msg")
    return ie, success
end

interval_of_existence(Y, Z₁, R; verbose::Bool=false) = interval_of_existence(interval(Y), interval(Z₁), R; verbose=verbose)

"""
    interval_of_existence(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real; verbose::Bool=false)

Return an interval and a boolean value with the following meaning:
- `true`: the interval corresponds to ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r + Z_2 r^2 / 2 \\le 0`` and ``Z_1 + Z_2 r < 1`` for all ``r \\in I``.
- `false`: otherwise, and the interval is empty.
"""
function interval_of_existence(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real; verbose::Bool=false)
    ie, success, msg = _interval_of_existence(Y, Z₁, Z₂, R)
    verbose && @info (success ?
        "success: second-order radii polynomial Y + (Z₁ - 1) r + Z₂ r² / 2\nY = $Y\nZ₁ = $Z₁ (bound at the approximation)\nZ₂ = $Z₂ (bound on the ball of radius R)\nR = $R" :
        "failure: $msg")
    return ie, success
end

interval_of_existence(Y, Z₁, Z₂, R; verbose::Bool=false) = interval_of_existence(interval(Y), interval(Z₁), interval(Z₂), R; verbose=verbose)

function _interval_of_existence(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real)
    T = float(promote_type(IntervalArithmetic.numtype(Y), IntervalArithmetic.numtype(Z₁), IntervalArithmetic.numtype(Z₂)))

    isvalid, msg = _check_inputs(Y, Z₁, Z₂, R)
    isvalid || return emptyinterval(T), false, msg

    # the radii polynomial p(r) = Y + (Z₁ - 1) r + Z₂ r² / 2 is increasing in Y, Z₁, Z₂ on [0, ∞)
    y, z₁, z₂ = interval(T, sup(Y)), interval(T, sup(Z₁)), interval(T, sup(Z₂))

    b = one(z₁) - z₁ # -p'(0) > 0
    Δ = b^2 - exact(2) * z₂ * y
    inf(Δ) < 0 && return emptyinterval(T), false, "radii polynomial has no real root"
    r = exact(2) * y / (b + sqrt(Δ)) # smallest root of p, reduces to Y / (1 - Z₁) when Z₂ = 0
    r_sup = sup(r)

    # largest radius x ≤ R at which p'(x) < 0, i.e. Z₂ x < 1 - Z₁, is certified
    x = T(R, RoundDown)
    if !isthinzero(z₂)
        x = min(x, inf(b / z₂))
        while !(sup(z₂ * x) < inf(b))
            x = prevfloat(x)
        end
    end

    isfinite(r_sup) & (r_sup ≤ x) || return emptyinterval(T), false, "root not in [0, R] or contraction fails"

    dec = min(decoration(Y), decoration(Z₁), decoration(Z₂))
    t = isguaranteed(Y) & isguaranteed(Z₁) & isguaranteed(Z₂)
    return _setguarantee(interval(T, r_sup, x, dec), t), true, ""
end

function _check_inputs(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real)
    (R < 0) | isnan(R) && return false, "invalid threshold R"
    isbounded(Y) & precedes(zero(Y), Y) || return false, "Y must be positive and bounded"
    precedes(zero(Z₁), Z₁) & strictprecedes(Z₁, one(Z₁)) || return false, "Z₁ must satisfy 0 ≤ Z₁ < 1"
    isbounded(Z₂) & precedes(zero(Z₂), Z₂) || return false, "Z₂ must be positive and bounded"
    return true, ""
end

"""
    set_of_radii(Y::AbstractVector{<:Interval}, Z::AbstractMatrix{<:Interval}, W::AbstractArray{<:Interval,3}, R::AbstractVector{<:Real}; verbose::Bool = false)

Return a set of radii, a test vector and a boolean value with the following meaning:
- `true`: the radii ``r = (r_1, \\dots, r_n) \\in [0, R_1] \\times \\ldots \\times [0, R_n]`` and the test vector ``\\eta \\in \\R^n_+`` satisfy ``Y_m + \\sum_{i=1}^n Z_{m,i} r_i + \\sum_{i,j=1}^n W_{m,i,j} r_i r_j / 2 - r_m \\le 0`` and ``\\sum_{i=1}^n Z_{m,i} \\eta_i + \\sum_{i,j=1}^n W_{m,i,j} \\eta_i r_j < \\eta_m`` for all ``m``.
- `false`: otherwise, and the radii and the test vector are empty.
"""
function set_of_radii(Y::AbstractVector{<:Interval}, Z::AbstractMatrix{<:Interval}, W::AbstractArray{<:Interval,3}, R::AbstractVector{<:Real}; verbose::Bool=false)
    T = float(promote_type(IntervalArithmetic.numtype(eltype(Y)), IntervalArithmetic.numtype(eltype(Z)), IntervalArithmetic.numtype(eltype(W))))

    isvalid, msg = _check_inputs(Y, Z, W, R)
    isvalid || return _rpa_failure(msg, T, verbose)

    M = length(Y)

    # radii polynomials P(r)ₘ = Yₘ + Σᵢ Zₘᵢ rᵢ + Σᵢⱼ Wₘᵢⱼ rᵢ rⱼ / 2 - rₘ and contraction matrix A(r)ₘᵢ = Zₘᵢ + Σⱼ Wₘᵢⱼ rⱼ
    B = r -> reshape(reshape(W, M * M, M) * r, M, M)
    P = r -> Y .+ Z * r .+ B(r) * r ./ exact(2) .- r
    A = r -> Z .+ B(r)

    # floating-point copies for the heuristic search only; the certification below uses P and A on the intervals
    Yf, Zf, Wf = Float64.(sup.(Y)), Float64.(sup.(Z)), Float64.(sup.(W))
    Bf = r -> reshape(reshape(Wf, M * M, M) * r, M, M)
    Bfᵀ = r -> reshape(reshape(permutedims(Wf, (1, 3, 2)), M * M, M) * r, M, M)
    Pf = r -> Yf .+ Zf * r .+ Bf(r) * r ./ 2 .- r
    DPf = r -> Zf .+ (Bf(r) .+ Bfᵀ(r)) ./ 2 .- LinearAlgebra.I(M)

    r = _search_radii(Pf, DPf, max.(Yf, floatmin(Float64)), r -> all(p -> !isempty_interval(p) & precedes(p, zero(p)), P(r)))
    isnothing(r) && return _rpa_failure("radii polynomials not simultaneously nonpositive", T, verbose)

    η = _perron_vector(A(r))
    ρ = sup(maximum(A(r) * η ./ η)) # Collatz-Wielandt upper bound on the spectral radius of A(r)
    ρ < 1 || return _rpa_failure("inclusion found, but no contraction: spectral radius bound $ρ", T, verbose)
    all(r .≤ R) || return _rpa_failure("radii exceed the thresholds R", T, verbose)

    verbose && @info "success\nr = $r\nη = $η\nspectral radius bound = $ρ"

    dec = min(minimum(decoration, Y), minimum(decoration, Z), minimum(decoration, W))
    t = all(isguaranteed, Y) & all(isguaranteed, Z) & all(isguaranteed, W)
    return [_setguarantee(interval(T, rᵢ, rᵢ, dec), t) for rᵢ ∈ r], η, true
end

set_of_radii(Y, Z, W, R; verbose::Bool=false) = set_of_radii(interval.(Y), interval.(Z), interval.(W), R; verbose=verbose)

# from the smallest root of the radii polynomials, step outward along a direction in which they all decrease until `nonpositive` holds
function _search_radii(Pf, DPf, r₀, nonpositive)
    r₀ = try
        first(newton(r -> (Pf(r), DPf(r)), r₀))
    catch e
        e isa LinearAlgebra.SingularException || rethrow()
        r₀
    end
    r₀ = max.(r₀, floatmin(Float64))

    F = LinearAlgebra.lu(DPf(r₀); check = false) # DPf(r₀) d = -r₀ gives the relative direction d, else a uniform scaling
    d = LinearAlgebra.issuccess(F) ? -(F \ r₀) ./ r₀ : ones(length(r₀))
    all(isfinite, d) & (maximum(d) > 0) || (d = ones(length(r₀)))
    d ./= maximum(d)

    for n ∈ -52:150
        r = r₀ .* (1 .+ d .* 2.0^n)
        all(rᵢ -> isfinite(rᵢ) & (rᵢ > 0), r) || return nothing # an infinite radius would make the interval evaluations empty, and the checks vacuous
        nonpositive(r) && return r
    end
    return nothing
end

# Perron eigenvector of the midpoint matrix, kept positive, as test vector for the contraction
function _perron_vector(A)
    F = LinearAlgebra.eigen(max.(Float64.(mid.(A)), 10eps(Float64)))
    v = abs.(F.vectors[:, argmax(abs.(F.values))])
    return max.(v, eps(Float64))
end

function _rpa_failure(msg::AbstractString, ::Type{T}, verbose::Bool) where {T}
    verbose && @info "failure: $msg"
    return Interval{T}[], Float64[], false
end

function _check_inputs(Y::AbstractVector{<:Interval}, Z::AbstractMatrix{<:Interval}, W::AbstractArray{<:Interval,3}, R::AbstractVector{<:Real})
    M = length(Y)
    (size(Z) == (M, M)) & (size(W) == (M, M, M)) & (length(R) == M) || return false, "inconsistent sizes"
    any(<(0), R) | any(isnan, R) && return false, "invalid thresholds R"
    all(Yₘ -> isbounded(Yₘ) & precedes(zero(Yₘ), Yₘ), Y) & all(Zₘᵢ -> isbounded(Zₘᵢ) & precedes(zero(Zₘᵢ), Zₘᵢ), Z) & all(Wₘᵢⱼ -> isbounded(Wₘᵢⱼ) & precedes(zero(Wₘᵢⱼ), Wₘᵢⱼ), W) ||
        return false, "each entry of Y, Z, W must be positive and bounded"
    return true, ""
end
