"""
    interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, R::T) where {T<:Real}

Return the interval of existence associated with the polynomial ``Y + (Z_1 - 1)r``.
"""
function interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, R::T) where {T<:Real}
    if inf(Y) < 0 || inf(Z₁) < 0 || R < 0
        return throw(DomainError)
    elseif Y == 0 # exact solution
        if Z₁ ≥ 1
            return Interval(zero(T))
        else
            return Interval(zero(T), R)
        end
    elseif Z₁ ≥ 1
        return emptyinterval(T)
    else
        r = Y/(one(Interval{T}) - Z₁)
        if r ≤ R
            return Interval(sup(r), R)
        else
            return emptyinterval(T)
        end
    end
end

"""
    interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, R::T) where {T<:Real}

Return the interval of existence associated with the polynomial ``Y + (Z_1 - 1)r + Z_2 r^2 / 2``.
"""
function interval_of_existence(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, R::T) where {T<:Real}
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
            r₂ = d/Z₂
            if Y == 0 # exact solution
                if Z₁ ≥ 1
                    return Interval(zero(T))
                elseif r₂ ≤ R && r₂ * Z₂ < 1
                    return Interval(zero(T), inf(r₂))
                elseif R * Z₂ < 1
                    return Interval(zero(T), R)
                else
                    Z₂⁻¹ = prevfloat(inf(inv(Z₂)))
                    if Z₂⁻¹ ≤ r₂ && Z₂⁻¹ ≤ R
                        return Interval(zero(T), Z₂⁻¹)
                    else
                        return Interval(zero(T))
                    end
                end
            elseif Z₁ ≥ 1 # Z₁ ≠ 1 (since Y > 0 and Z₁ = 1 implies complex roots)
                return emptyinterval(T)
            else
                r₁ = 2Y/d
                if r₁ ≤ R
                    if r₂ ≤ R && r₂ * Z₂ < 1
                        return Interval(sup(r₁), inf(r₂))
                    elseif R * Z₂ < 1
                        return Interval(sup(r₁), R)
                    else
                        Z₂⁻¹ = prevfloat(inf(inv(Z₂)))
                        if Z₂⁻¹ ≤ r₂ && Z₂⁻¹ ≤ R
                            return Interval(sup(r₁), Z₂⁻¹)
                        elseif r₁ * Z₂ < 1
                            return Interval(sup(r₁))
                        else
                            return emptyinterval(T)
                        end
                    end
                else
                    return emptyinterval(T)
                end
            end
        end
    end
end
