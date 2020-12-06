function roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, r₀::T) where {T<:Real}
    # p(r) = Y + (Z₁-1) r
    if iszero(Y)
        Z₁ ≥ 1 && return _success_rpa([zero(T)])
        return _success_rpa([zero(T), Inf])
    else
        if isone(Z₁)
            printstyled(string("The radii polynomial theorem is not satisfied: the radii polynomial has constant value ", Y, ".\n"); color = :red)
            return T[]
        else
            r₁ = Y/(one(Z₁) - Z₁)
            if r₁ < 0
                printstyled(string("The radii polynomial theorem is not satisfied: single negative root r₁ = ", r₁, ".\n"); color = :red)
                return T[]
            else
                r₁ ≤ r₀ && return _success_rpa([sup(r₁), r₀])
                printstyled(string("The radii polynomial theorem is not satisfied: r₁ > r₀ where r₁ = ", r₁, ".\n"); color = :red)
                return T[]
            end
        end
    end
end

function roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, r₀::T) where {T<:Real}
    # p(r) = Y + (Z₁-1) r + (Z₂/2) r^2
    iszero(Z₂) && return roots_radii_polynomial(Y, Z₁, r₀)
    if iszero(Y)
        Z₁ ≥ 1 && return _success_rpa([zero(T)])
        r₂ = -2b/Z₂
        r₂ ≤ r₀ && r₂ * Z₂ < 1 && return _success_rpa([zero(T), sup(r₂)])
        r₀ * Z₂ < 1 && return _success_rpa([zero(T), r₀])
        return _success_rpa([zero(T), sup(inv(Z₂))])
    else
        b = Z₁ - one(Z₁)
        Δ = b^2 - 2Z₂*Y
        if Δ < 0
            d_ = - b - sqrt(complex(Δ))
            r₁_ = d_/Z₂
            r₂_ = 2Y/d_
            printstyled(string("The radii polynomial theorem is not satisfied: two complex conjugate roots r₁ = ", r₁_, " and r₂ = ", r₂_, ".\n"); color = :red)
            return T[]
        else
            d = - b - sqrt(Δ)
            r₁ = d/Z₂
            if r₁ ≤ r₀ && r₁ * Z₂ < 1
                if Δ == 0
                    Z₁ ≤ 1 && return _success_rpa([sup(r₁)])
                    printstyled(string("The radii polynomial theorem is not satisfied: single negative root r₁ = ", r₁, ".\n"); color = :red)
                    return T[]
                else
                    r₂ = 2Y/d
                    if Z₁ > 1
                        printstyled(string("The radii polynomial theorem is not satisfied: two negative roots r₁ = ", r₁, " and r₂ = ", r₂, ".\n"); color = :red)
                        return T[]
                    else
                        r₂ ≤ r₀ && r₂ * Z₂ < 1 && return _success_rpa([sup(r₁), sup(r₂)])
                        r₀ * Z₂ < 1 && return _success_rpa([sup(r₁), r₀])
                        return _success_rpa([sup(r₁), sup(inv(Z₂))])
                    end
                end
            else
                if r₁ > r₀
                    printstyled(string("The radii polynomial theorem is not satisfied: r₁ > r₀ where r₁ = ", r₁, ".\n"); color = :red)
                else
                    printstyled(string("The radii polynomial theorem is not satisfied: r₁ * Z₂ ≥ 1 where r₁ = ", r₁, ".\n"); color = :red)
                end
                return T[]
            end
        end
    end
end

function _success_rpa(a)
    printstyled(string("The radii polynomial theorem is satisfied with the interval of existence ", a, ".\n"); color = :green)
    return a
end
