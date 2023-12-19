"""
    interval_of_existence(Y::Real, Z₁::Real, R::Real)

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r \\le 0`` and ``Z_1 < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y_::Real, Z₁_::Real, R::Real)
    Y, Z₁ = sup(Y_), sup(Z₁_)
    NewType = float(promote_type(typeof(Y), typeof(Z₁), typeof(R)))
    if !(Y ≥ 0 && isfinite(Y) && Z₁ ≥ 0 && isfinite(Z₁) && R ≥ 0)
        return emptyinterval(NewType) # throw(DomainError((Y, Z₁, R), "Y and Z₁ must be positive and finite, R must be positive"))
    elseif Z₁ ≥ 1
        return emptyinterval(NewType)
    else
        r = Y/(one(Interval{NewType}) - Z₁)
        r_sup = NewType(sup(r), RoundUp)
        if 0 ≤ r_sup ≤ R
            return interval(NewType, r_sup, R)
        else
            return emptyinterval(NewType)
        end
    end
end

interval_of_existence(Y::Real, Z₁::Real, R::Interval) = interval_of_existence(Y, Z₁, sup(R))

"""
    interval_of_existence(Y::Real, Z₁::Real, Z₂::Real, R::Real)

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r + Z_2 r^2 / 2 \\le 0`` and ``Z_1 + Z_2 r < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y_::Real, Z₁_::Real, Z₂_::Real, R::Real)
    Y, Z₁, Z₂ = sup(Y_), sup(Z₁_), sup(Z₂_)
    NewType = float(promote_type(typeof(Y), typeof(Z₁), typeof(Z₂), typeof(R)))
    if Z₂ == 0
        return interval_of_existence(Y, Z₁, R)
    elseif !(Y ≥ 0 && isfinite(Y) && Z₁ ≥ 0 && isfinite(Z₁) && Z₂ ≥ 0 && isfinite(Z₂) && R ≥ 0)
        return emptyinterval(NewType) # throw(DomainError((Y, Z₁, Z₂, R), "Y, Z₁ and Z₂ must be positive and finite, R must be positive"))
    else
        b = Z₁ - one(Interval{NewType})
        Δ = b*b - 2*interval(Z₂)*Y
        if inf(Δ) < 0 # complex roots
            return emptyinterval(NewType)
        else # real roots
            sqrtΔ = sqrt(Δ)
            r₁ = -(sqrtΔ + b)/Z₂
            r₁_sup = NewType(sup(r₁), RoundUp)
            if 0 ≤ r₁_sup ≤ R && sup(Z₁ + interval(Z₂) * r₁_sup) < 1
                r₂ = (sqrtΔ - b)/Z₂
                r₂_inf = NewType(inf(r₂), RoundDown)
                if r₁_sup > r₂_inf
                    return emptyinterval(NewType)
                elseif 0 ≤ r₂_inf ≤ R && sup(Z₁ + interval(Z₂) * r₂_inf) < 1
                    return interval(NewType, r₁_sup, r₂_inf)
                elseif sup(Z₁ + interval(Z₂) * R) < 1
                    return interval(NewType, r₁_sup, R)
                else
                    x = NewType(prevfloat(inf(-b/Z₂)), RoundDown)
                    if r₁_sup ≤ x ≤ r₂_inf && x ≤ R
                        return interval(NewType, r₁_sup, x)
                    else
                        return interval(NewType, r₁_sup)
                    end
                end
            else
                return emptyinterval(NewType)
            end
        end
    end
end

interval_of_existence(Y::Real, Z₁::Real, Z₂::Real, R::Interval) = interval_of_existence(Y, Z₁, Z₂, sup(R))
