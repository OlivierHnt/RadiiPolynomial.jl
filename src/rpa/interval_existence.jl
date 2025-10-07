"""
    interval_of_existence(Y::Interval, Z₁::Interval, R::Real; verbose::Bool = false)

Return an interval and a boolean value with the following meaning:
    - `true`: the interval corresponds to ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r \\le 0`` for all ``r \\in I`` and ``Z_1 < 1``.
    - `false`: otherwise, and the interval is empty.
"""
function interval_of_existence(Y::Interval, Z₁::Interval, R::Real; verbose::Bool = false)
    r = Y/(one(Z₁) - Z₁)

    isvalid, msg = _check_inputs(Y, Z₁, R)
    isvalid || return _rpa_failure(msg, r, verbose)

    r_sup = sup(r)
    0 ≤ r_sup ≤ R || return _rpa_failure("root not in [0, R]", r, verbose)

    verbose && @info "success: interval found\nY = $Y\nZ₁ = $Z₁\nR = $R"

    t = isguaranteed(Y) & isguaranteed(Z₁)
    return _setguarantee(interval(IntervalArithmetic.numtype(r), r_sup, R, min(decoration(Y), decoration(Z₁))), t), true
end

interval_of_existence(Y, Z₁, R; verbose::Bool = false) = interval_of_existence(interval(Y), interval(Z₁), R; verbose = verbose)

function _check_inputs(Y::Interval, Z₁::Interval, R::Real)
    R < 0 || isnan(R) && return false, "invalid threshold R"
    isbounded(Y) & precedes(zero(Y), Y) & precedes(zero(Z₁), Z₁) & strictprecedes(Z₁, one(Z₁)) ||
        return false, "Y must be positive and 0 ≤ Z₁ < 1"
    return true, ""
end

function _rpa_failure(msg::AbstractString, r::Interval, verbose::Bool)
    verbose && @info "failure: $msg"
    return emptyinterval(r), false
end

"""
    interval_of_existence(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real; verbose::Bool = false)

Return an interval and a boolean value with the following meaning:
    - `true`: the interval corresponds to ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r + Z_2 r^2 / 2 \\le 0`` and ``Z_1 + Z_2 r < 1`` for all ``r \\in I``.
    - `false`: otherwise, and the interval is empty.
"""
function interval_of_existence(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real; verbose::Bool = false)
    isthinzero(Z₂) && return interval_of_existence(Y, Z₁, R; verbose = verbose)

    b = Z₁ - one(Z₁)
    Δ = b^2 - exact(2)*Z₂*Y
    sqrtΔ = sqrt(Δ)
    r₁ = (-b - sqrtΔ)/Z₂
    r₂ = (-b + sqrtΔ)/Z₂

    isvalid, msg = _check_inputs(Y, Z₁, Z₂, R)
    isvalid || return _rpa_failure(msg, r₁, verbose)

    r₁_sup = sup(r₁)

    inf(Δ) < 0                                && return _rpa_failure("discriminant negative → complex roots", r₁, verbose)
    0 ≤ r₁_sup ≤ R && sup(Z₁ + Z₂*r₁_sup) < 1 || return _rpa_failure("roots not in [0, R] or contraction fails", r₁, verbose)

    verbose && @info "success: interval found\nY = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR = $R\nΔ = $Δ\nroots = ($r₁, $r₂)"
    z = inf(-b/Z₂)
    x = float(z)
    while x > z
        x = prevfloat(x)
    end

    t = isguaranteed(Y) & isguaranteed(Z₁) & isguaranteed(Z₂)
    return _setguarantee(interval(IntervalArithmetic.numtype(r₁), r₁_sup, min(R, x), min(decoration(Y), decoration(Z₁), decoration(Z₂))), t), true
end

interval_of_existence(Y, Z₁, Z₂, R; verbose::Bool = false) = interval_of_existence(interval(Y), interval(Z₁), interval(Z₂), R; verbose = verbose)

function _check_inputs(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real)
    R < 0 || isnan(R) && return false, "invalid threshold R"
    isbounded(Y) & precedes(zero(Y), Y) & precedes(zero(Z₁), Z₁) & strictprecedes(Z₁, one(Z₁)) & isbounded(Z₂) & precedes(zero(Z₂), Z₂) ||
        return false, "Y, Z₂ must be positive and 0 ≤ Z₁ < 1"
    return true, ""
end
