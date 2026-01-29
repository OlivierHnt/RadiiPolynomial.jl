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
    (R < 0) | isnan(R) && return false, "invalid threshold R"
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
    (R < 0) | isnan(R) && return false, "invalid threshold R"
    isbounded(Y) & precedes(zero(Y), Y) & precedes(zero(Z₁), Z₁) & strictprecedes(Z₁, one(Z₁)) & isbounded(Z₂) & precedes(zero(Z₂), Z₂) ||
        return false, "Y, Z₂ must be positive and 0 ≤ Z₁ < 1"
    return true, ""
end

"""
    set_of_radii(Y::AbstractVector{<:Interval}, Z::AbstractMatrix{<:Interval}, W::AbstractArray{<:Interval,3}, R::AbstractVector{<:Real}; verbose::Bool = false)

Return a set of radii, a vector and a boolean value with the following meaning:
    - `true`: the set of radii corresponds to ``r = (r_1, \\dots, r_n) \\in [0, R_1] \\times \\ldots \\times [0, R_n]`` and the test vector ``\\eta \in \\R^n_+`` such that ``Y_m + \\sum_{i=1}^n Z_{m,i} r_i + \\sum_{i,j=1}^n W_{m,i,j} r_i r_j / 2 - r_m \\le 0`` and ``\\sum_{i=1}^n Z_{m,i} \\eta_i + \\sum_{i,j=1}^n W_{m,i,j} \\eta_i r_j < \\eta_m``.
    - `false`: otherwise, and the set of radii and the vector are empty.
"""
function set_of_radii(Y::AbstractVector{<:Interval}, Z::AbstractMatrix{<:Interval}, W::AbstractArray{<:Interval,3}, R::AbstractVector{<:Real}; verbose::Bool = false)
    isvalid, msg = _check_inputs(Y, Z, W, R)
    isvalid || return _rpa_failure(msg, verbose)

    M = length(Y)

    I_mat = I(M)

    Yf = sup.(Y)
    Yf = max.(Yf, floatmin(Float64)) # avoid zeros

    Z₁f = sup.(Z₁)
    Z₂f = sup.(Z₂)

    WWf = r -> reshape(reshape(Z₂f, :, M) * r, M, M)
    WW2f = r -> reshape(reshape(permutedims(Z₂f, (1, 3, 2)), :, M) * r, M, M)

    Pf = r -> Yf .+ Z₁f * r .+ (WWf(r) * r) ./ 2 .- r
    DPf = r -> Z₁f .+ (WWf(r) .+ WW2f(r)) ./ 2 .- I_mat

    WW = r -> reshape(reshape(Z₂, :, M) * r, M, M)
    Pintval = r -> Y .+ Z₁ * r .+ (WW(r) * r) ./ 2 .- r

    Mat = r -> Z₁ .+ WW(r)

    # Newton iteration to find approximate zero
    r0 = Yf
    r0, newton_success = newton(r -> (Pf(r), DPf(r)), r0)
    if !(all(>(0), r0)) || !newton_success
        println("no good set of radii found for inclusion")
    end

    r0 = max.(r0, floatmin(Float64))

    direction = DPf(r0) \ r0
    direction ./= .-r0 # normalize the direction
    direction ./= maximum(direction)

    # search for negative point of radii polynomials
    success = false
    partialsuccess = false
    rmin = NaN
    eta = NaN

    if all(>=(0), r0)
        r2 = copy(r0)
        n = -52
        while n <= 150 && !partialsuccess && all(>(0), r2)
            if all(isstrictless.(Pintval(r2), interval(0)))
                rmin = r2
                partialsuccess = true
            else
                r2 = r0 .* (ones(M) .+ direction * 2.0^n)
                n += 1
            end
        end
    end

    if partialsuccess
        if M == 1
            if Mat(rmin) - 1 < 0
                success = true
                eta = 1
            end
        else
            dominant, eta = _collatz_wielandt(Mat(rmin))
            if dominant - 1 < 0
                success = true
            else
                println("inclusion found, but no contraction.")
            end
        end
    end

    if !success
        if M == 1
            println("radii polynomial not negative")
        else
            println("radii polynomials not negative simultaneously")
        end
        rmin = NaN
        eta = NaN
    end

    verbose && @info "success: $msg\nr = $(rmin)\nη = $(eta)"
    return rmin, eta, success
end

function _collatz_wielandt(A)
    # Compute upper bound on spectral radius (Perron-Frobenius) of nonnegative matrix A

    Aeps = max.(mid.(A), 10 * eps(Float64))

    F = eigen(Aeps)
    eigenvectors = F.vectors
    eigenvalues = F.values

    m = argmax(abs.(eigenvalues))
    y = abs.(eigenvectors[:, m])

    # testvector for Collatz-Wielandt bound
    testvector = max.(y, eps(Float64)) # avoid division by 0

    PFupperbound = maximum((A * testvector) ./ testvector)

    if eltype(A) <: Interval
        PFupperbound = sup(PFupperbound)
    end

    return PFupperbound, testvector
end

set_of_radii(Y, Z, W, R; verbose::Bool = false) = set_of_radii(interval.(Y), interval.(Z₁), interval.(Z₂), R; verbose = verbose)

function _rpa_failure(msg::AbstractString, verbose::Bool)
    verbose && @info "failure: $msg"
    return Interval[], Interval[], false
end

function _check_inputs(Y::AbstractVector{<:Interval}, Z::AbstractMatrix{<:Interval}, W::AbstractArray{<:Interval,3}, R::AbstractVector{<:Real})
    any(<(0), R) | any(isnan, R) && return false, "invalid threshold R"
    all(isbounded, Y) & all(Y_m -> precedes(zero(Y_m), Y_m), Y) & all(Z_mi -> precedes(zero(Z_mi), Z_mi), Z) & all(Z_mi -> strictprecedes(Z_mi, one(Z_mi)), Z) & all(isbounded, W) & all(W_mij -> precedes(zero(W_mij), W_mij), W) ||
        return false, "each entry of Y, W must be positive and each entry of Z must be in [0, 1)"
    return true, ""
end
