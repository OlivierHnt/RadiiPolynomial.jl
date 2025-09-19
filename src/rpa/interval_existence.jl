"""
    interval_of_existence(Y::Interval, Z₁::Interval, R::Real; verbose::Bool = false)

Return an interval and a boolean value with the following meaning:
    - `true`: the interval corresponds to ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r \\le 0`` for all ``r \\in I`` and ``Z_1 < 1``.
    - `false`: otherwise, and the interval is empty.
"""
function interval_of_existence(Y::Interval, Z₁::Interval, R::Real; verbose::Bool=false)
    r = Y / (one(Z₁) - Z₁)

    isvalid, msg = _check_inputs(Y, Z₁, R)
    isvalid || return _rpa_failure(msg, r, verbose)

    r_sup = sup(r)
    0 ≤ r_sup ≤ R || return _rpa_failure("root not in [0, R]", r, verbose)

    verbose && @info "success: interval found\nY = $Y\nZ₁ = $Z₁\nR = $R"

    t = isguaranteed(Y) & isguaranteed(Z₁)
    return _setguarantee(interval(IntervalArithmetic.numtype(r), r_sup, R, min(decoration(Y), decoration(Z₁))), t), true
end

interval_of_existence(Y, Z₁, R; verbose::Bool=false) = interval_of_existence(interval(Y), interval(Z₁), R; verbose=verbose)

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
function interval_of_existence(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real; verbose::Bool=false)
    isthinzero(Z₂) && return interval_of_existence(Y, Z₁, R; verbose=verbose)

    b = Z₁ - one(Z₁)
    Δ = b^2 - exact(2) * Z₂ * Y
    sqrtΔ = sqrt(Δ)
    r₁ = (-b - sqrtΔ) / Z₂
    r₂ = (-b + sqrtΔ) / Z₂

    isvalid, msg = _check_inputs(Y, Z₁, Z₂, R)
    isvalid || return _rpa_failure(msg, r₁, verbose)

    r₁_sup = sup(r₁)

    inf(Δ) < 0 && return _rpa_failure("discriminant negative → complex roots", r₁, verbose)
    0 ≤ r₁_sup ≤ R && sup(Z₁ + Z₂ * r₁_sup) < 1 || return _rpa_failure("roots not in [0, R] or contraction fails", r₁, verbose)

    verbose && @info "success: interval found\nY = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR = $R\nΔ = $Δ\nroots = ($r₁, $r₂)"
    z = inf(-b / Z₂)
    x = float(z)
    while x > z
        x = prevfloat(x)
    end

    t = isguaranteed(Y) & isguaranteed(Z₁) & isguaranteed(Z₂)
    return _setguarantee(interval(IntervalArithmetic.numtype(r₁), r₁_sup, min(R, x), min(decoration(Y), decoration(Z₁), decoration(Z₂))), t), true
end

interval_of_existence(Y, Z₁, Z₂, R; verbose::Bool=false) = interval_of_existence(interval(Y), interval(Z₁), interval(Z₂), R; verbose=verbose)

function _check_inputs(Y::Interval, Z₁::Interval, Z₂::Interval, R::Real)
    R < 0 || isnan(R) && return false, "invalid threshold R"
    isbounded(Y) & precedes(zero(Y), Y) & precedes(zero(Z₁), Z₁) & strictprecedes(Z₁, one(Z₁)) & isbounded(Z₂) & precedes(zero(Z₂), Z₂) ||
        return false, "Y, Z₂ must be positive and 0 ≤ Z₁ < 1"
    return true, ""
end

"""
    interval_of_existence(Y::AbstractArray{<:Interval}, Z₁::AbstractArray{<:Interval}, Z₂::AbstractArray{<:Interval}, R::AbstractArray{<:Real})

Return an array of intervals of existence ``I \\subset [0, R]ᵐ`` such that ``Y + ∑ Z_1 r + ∑∑ Z_2 r^2 / 2 - rᵐ \\le 0`` and ``∑ Z_1 η + ∑ ∑ Z_2 η r < 1`` for all ``r \\in I``.
"""
function _rpa_failure(msg::AbstractString, verbose::Bool)
    return verbose && @info "failure: $msg"
end

function interval_of_existence(Y::AbstractArray{<:Interval}, Z₁::AbstractArray{<:Interval}, Z₂::AbstractArray{<:Interval}, R::AbstractArray{<:Real}; verbose::Bool=true)
    isvalid, msg = _check_inputs(Y, Z₁, Z₂, R)
    isvalid || return _rpa_failure(msg, verbose)

    M = length(Y)

    # dense identity matrix for linear solves
    I_mat = Matrix(I, M, M)

    # floating point versions (take upper bounds of intervals)
    Yf = sup.(Y)
    Yf = max.(Yf, floatmin(Float64))      # avoid zeros

    Z₁f = sup.(Z₁)
    Z₂f = sup.(Z₂)

    # Functions for contractions with Z₂ (floating)
    WWf = r -> reshape(reshape(Z₂f, :, M) * r, M, M)
    WW2f = r -> reshape(reshape(permutedims(Z₂f, (1, 3, 2)), :, M) * r, M, M)

    # Radii polynomial and derivative (floating point)
    Pf = r -> Yf .+ Z₁f * r .+ (WWf(r) * r) ./ 2 .- r
    DPf = r -> Z₁f .+ (WWf(r) .+ WW2f(r)) ./ 2 .- I_mat

    # Interval arithmetic versions (keep intervals)
    WW = r -> reshape(reshape(Z₂, :, M) * r, M, M)
    Pintval = r -> Y .+ Z₁ * r .+ (WW(r) * r) ./ 2 .- r

    # Matrix function (interval/dense)
    Mat = r -> Z₁ .+ WW(r)

    # --- Newton iteration to find approximate zero ---
    r0 = zeros(M)
    newtoncounter = 0
    maxnewtoniterates = 15
    newtontolerance = 1e-15
    convergencetolerance = 1e-13
    dr0 = ones(M)       # start with nonzero step
    nanflag = false

    while newtoncounter <= maxnewtoniterates && maximum(abs.(dr0)) > newtontolerance
        dr0 = -(DPf(r0) \ Pf(r0))
        if any(isnan, dr0)
            dr0 = map(x -> isnan(x) ? 0.0 : x, dr0)
            nanflag = true
        else
            nanflag = false
        end
        r0 .= r0 .+ dr0
        newtoncounter += 1
    end

    if !(all(>(0), r0)) || maximum(abs.(dr0)) > convergencetolerance || nanflag
        println("No good set of radii found for inclusion")
    end

    r0 = max.(r0, floatmin(Float64))

    # --- Determine search direction ---
    direction = -(DPf(r0) \ r0)
    direction = direction ./ r0     # Normalize the direction
    direction = direction ./ maximum(direction)

    # --- Search for negative point of radii polynomials ---
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
            dominant, eta = CollatzWielandt(Mat(rmin))
            if dominant - 1 < 0
                success = true
            else
                println("Inclusion found, but no contraction.")
            end
        end
    end

    if !success
        if M == 1
            println("Radii polynomial not negative")
        else
            println("Radii polynomials not negative simultaneously")
        end
        rmin = NaN
        eta = NaN
    end

    return rmin, eta
end

interval_of_existence(Y::AbstractArray{<:Number}, Z₁::AbstractArray{<:Number}, Z₂::AbstractArray{<:Number}, R::AbstractArray{<:Real}; verbose::Bool=true) = interval_of_existence(interval.(Y), interval.(Z₁), interval.(Z₂), R; verbose=verbose)

function _check_inputs(Y::AbstractArray{<:Interval}, Z₁::AbstractArray{<:Interval}, Z₂::AbstractArray{<:Interval}, R::AbstractArray{<:Real})
    # Check R for negatives or NaNs
    if any(R .< 0) || any(isnan.(R))
        return false, "invalid threshold R"
    end

    # Check that all intervals are bounded and positive where required
    if !all(isbounded.(Y)) || !all(precedes.(zero.(Y), Y)) ||
       !all(precedes.(zero.(Z₁), Z₁)) || !all(strictprecedes.(Z₁, one.(Z₁))) ||
       !all(isbounded.(Z₂)) || !all(precedes.(zero.(Z₂), Z₂))
        return false, "Y, Z₂ must be positive and 0 ≤ Z₁ < 1"
    end

    return true, ""
end
