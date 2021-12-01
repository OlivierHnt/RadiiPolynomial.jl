"""
    interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, R::T) where {T<:Real}

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r \\leq 0`` and ``Z_1 < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, R::T) where {T<:Real}
    if inf(Y) < 0 || inf(Z₁) < 0 || R < 0
        return throw(DomainError)
    else
        r = Y/(one(Interval{T}) - Z₁)
        if 0 ≤ sup(r) ≤ R && sup(Z₁) < 1
            return Interval(sup(r), R)
        else
            return emptyinterval(T)
        end
    end
end

struct C¹Condition end

"""
    interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, R::T, ::C¹Condition) where {T<:Real}

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r + Z_2 r^2 / 2 \\leq 0`` and ``Z_1 + Z_2 r < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, R::T, ::C¹Condition) where {T<:Real}
    if inf(Y) < 0 || inf(Z₁) < 0 || inf(Z₂) < 0 || R < 0
        return throw(DomainError)
    elseif Z₂ == 0
        return interval_of_existence(Y, Z₁, R)
    else
        b = Z₁ - one(Interval{T})
        Δ = b*b - 2Z₂*Y
        if inf(Δ) < 0 # complex roots: d_ = - b - sqrt(complex(Δ)), r₁_ = d_/Z₂, r₂_ = 2Y/d_
            return emptyinterval(T)
        else # real roots
            d = sqrt(Δ) - b
            r₁ = 2Y/d
            if 0 ≤ sup(r₁) ≤ R && sup(Z₁ + Z₂ * r₁) < 1
                r₂ = d/Z₂
                if 0 ≤ sup(r₂) ≤ R && sup(Z₁ + Z₂ * r₂) < 1
                    return Interval(sup(r₁), inf(r₂))
                elseif sup(Z₁ + Z₂ * R) < 1
                    return Interval(sup(r₁), R)
                else
                    x = prevfloat(inf(-b/Z₂))
                    if sup(r₁) ≤ x ≤ inf(r₂) && x ≤ R
                        return Interval(sup(r₁), x)
                    else
                        return Interval(sup(r₁))
                    end
                end
            else
                return emptyinterval(T)
            end
        end
    end
end

struct C²Condition end

"""
    interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, R::T, ::C²Condition) where {T<:Real}

Return an interval of existence ``I \\subset [0, R]`` such that ``Y + (Z_1 - 1) r + Z_2 r^2 / 2 \\leq 0`` and ``Z_2 r < 1`` for all ``r \\in I``.
"""
function interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, R::T, ::C²Condition) where {T<:Real}
    if inf(Y) < 0 || inf(Z₁) < 0 || inf(Z₂) < 0 || R < 0
        return throw(DomainError)
    elseif Z₂ == 0
        return interval_of_existence(Y, Z₁, R)
    else
        b = Z₁ - one(Interval{T})
        Δ = b*b - 2Z₂*Y
        if inf(Δ) < 0 # complex roots: d_ = - b - sqrt(complex(Δ)), r₁_ = d_/Z₂, r₂_ = 2Y/d_
            return emptyinterval(T)
        else # real roots
            d = sqrt(Δ) - b
            r₁ = 2Y/d
            if 0 ≤ sup(r₁) ≤ R && sup(Z₂ * r₁) < 1
                r₂ = d/Z₂
                if 0 ≤ sup(r₂) ≤ R && sup(Z₂ * r₂) < 1
                    return Interval(sup(r₁), inf(r₂))
                elseif sup(Z₂ * R) < 1
                    return Interval(sup(r₁), R)
                else
                    x = prevfloat(inf(inv(Z₂)))
                    if sup(r₁) ≤ x ≤ inf(r₂) && x ≤ R
                        return Interval(sup(r₁), x)
                    else
                        return Interval(sup(r₁))
                    end
                end
            else
                return emptyinterval(T)
            end
        end
    end
end
