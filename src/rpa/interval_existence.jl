"""
    interval_of_existence(Y::Real, Z₁::Real, R::Real)

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r \\le 0`` and ``Z_1 < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y::Real, Z₁::Real, R::Real; verbose::Bool=false)
    r = interval(Y)/(interval(one(Z₁)) - interval(Z₁))
    r_sup = sup(r)

    if R < 0 || isnan(R)
        verbose && @info "failure: the error threshold R is invalid\nY  = $Y\nZ₁ = $Z₁\nR  = $R\nroot = $r"
        return emptyinterval(r)
    else
        if 0 ≤ r_sup ≤ R
            verbose && @info "success: the root and R delimit an interval of existence\nY  = $Y\nZ₁ = $Z₁\nR  = $R\nroot = $r"
            ie = IntervalArithmetic._unsafe_bareinterval(IntervalArithmetic.numtype(r), r_sup, R)
            return IntervalArithmetic._unsafe_interval(ie, min(decoration(r), decoration(ie)), isguaranteed(r))
        else
            verbose && @info "failure: the root is not in [0, R]\nY  = $Y\nZ₁ = $Z₁\nR  = $R\nroot = $r"
            return emptyinterval(r)
        end
    end
end

interval_of_existence(::Real, ::Real, R::Interval; verbose::Bool=false) = throw(ArgumentError("R cannot be an interval, got $R"))

"""
    interval_of_existence(Y::Real, Z₁::Real, Z₂::Real, R::Real)

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r + Z_2 r^2 / 2 \\le 0`` and ``Z_1 + Z_2 r < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y::Real, Z₁::Real, Z₂::Real, R::Real; verbose::Bool=false)
    iszero(Z₂) && return interval_of_existence(Y, Z₁, R; verbose = verbose)

    b = interval(Z₁) - interval(one(Z₁))
    Δ = b*b - _safe_mul(2, interval(Z₂))*interval(Y)

    sqrtΔ = sqrt(Δ)

    r₁ = (-b - sqrtΔ)/Z₂
    r₁_sup = sup(r₁)

    r₂ = (-b + sqrtΔ)/Z₂
    r₂_inf = inf(r₂)

    if inf(Δ) < 0
        verbose && @info "failure: both roots are complex\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ"
        return emptyinterval(r₁)
    elseif r₁_sup > r₂_inf
        verbose && @info "failure: could not compute the roots sufficently accuratly\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ"
        return emptyinterval(r₁)
    else
        if R < 0 | isnan(R)
            verbose && @info "failure: the threshold R is invalid\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ\nroots = ($r₁, $r₂)"
            return emptyinterval(r₁)
        else
            if 0 ≤ r₁_sup ≤ R && sup(interval(Z₁) + interval(Z₂) * r₁_sup) < 1
                if 0 ≤ r₂_inf ≤ R && sup(interval(Z₁) + interval(Z₂) * r₂_inf) < 1
                    verbose && @info "success: both roots delimit an interval of existence\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ\nroots = ($r₁, $r₂)"
                    ie = IntervalArithmetic._unsafe_bareinterval(IntervalArithmetic.numtype(r₁), r₁_sup, r₂_inf)
                elseif sup(interval(Z₁) + interval(Z₂) * R) < 1
                    verbose && @info "success: the smallest root and R delimit an interval of existence\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ\nroots = ($r₁, $r₂)"
                    ie = IntervalArithmetic._unsafe_bareinterval(IntervalArithmetic.numtype(r₁), r₁_sup, R)
                else
                    z = inf(-b/interval(Z₂))
                    x = float(z)
                    while x > z
                        x = prevfloat(x)
                    end
                    if r₁_sup ≤ x ≤ min(r₂_inf, R)
                        verbose && @info "success: the smallest root and a slightly smaller float than (1-Z₁)/Z₂ delimit an interval of existence\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ\nroots = ($r₁, $r₂)"
                        ie = IntervalArithmetic._unsafe_bareinterval(IntervalArithmetic.numtype(r₁), r₁_sup, x)
                    else
                        verbose && @info "success: the smallest root provides a (singleton) interval of existence\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\ndiscrimant: (Z₁ - 1)² - 2Z₂Y = $Δ\nroots = ($r₁, $r₂)"
                        ie = IntervalArithmetic._unsafe_bareinterval(IntervalArithmetic.numtype(r₁), r₁_sup, r₁_sup)
                    end
                end
                return IntervalArithmetic._unsafe_interval(ie, min(decoration(r₁), decoration(ie)), isguaranteed(r₁))
            else
                verbose && @info "failure: both roots are either not contained in [0, R], and/or too large\nY  = $Y\nZ₁ = $Z₁\nZ₂ = $Z₂\nR  = $R\n(Z₁ - 1)² - 2Z₂Y = $Δ\nroots = ($r₁, $r₂)"
                return emptyinterval(r₁)
            end
        end
    end
end

interval_of_existence(::Real, ::Real, ::Real, R::Interval; verbose::Bool=false) = throw(ArgumentError("R cannot be an interval, got $R"))
