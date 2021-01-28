"""
    roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, r₀::T) where {T<:Real}

Return the interval of existence garanted by the Radii Polynomial Theorem from the root of the polynomial Y + (Z₁-1)r.
"""
function roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, r₀::T) where {T<:Real}
    @info "Applying the Radii Polynomial Theoreom of order 1: Y + (Z₁-1)r" r₀ Y Z₁
    if iszero(Y) # exact solution
        if Z₁ ≥ 1
            @info """VALID ROOT FOUND:
                → Y = 0 and Z₁ ≥ 1
                ✓ the radii polynomial is negative in (-∞,0]
                ⇒ the interval of existence is {0}"""
            return [zero(T)]
        else
            @info """VALID ROOT FOUND:
                → Y = 0 and 0 ≤ Z₁ < 1
                ✓ the radii polynomial is negative in [0,+∞)
                ⇒ the interval of existence is [0,r₀]"""
            return [zero(T), r₀]
        end
    else # not exact solution
        if isone(Z₁)
            @info """NO VALID ROOT FOUND:
                → Y ≥ 0 and Z₁ = 1
                ✗ the radii polynomial is positive with constant value Y
                ⇒ the interval of existence is empty""" Y
            return T[]
        else
            r₁ = Y/(one(Z₁) - Z₁)
            if Z₁ > 1
                @info """NO VALID ROOT FOUND:
                    → Y ≥ 0 and Z₁ > 1
                    ✓ the radii polynomial is negative in (-∞,r₁]
                    ✗ r₁ < 0
                    ⇒ the interval of existence is empty""" r₁
                return T[]
            else
                if r₁ ≤ r₀
                    @info """VALID ROOT FOUND:
                        → Y ≥ 0 and 0 ≤ Z₁ < 1
                        ✓ the radii polynomial is negative in [r₁,+∞)
                        ✓ r₁ ≤ r₀
                        ⇒ the interval of existence is [r₁,r₀]""" r₁
                    return [sup(r₁), r₀]
                else
                    @info """NO VALID ROOT FOUND:
                        → Y ≥ 0 and 0 ≤ Z₁ < 1
                        ✓ the radii polynomial is negative in [r₁,+∞)
                        ✗ r₁ > r₀
                        ⇒ the interval of existence is empty""" r₁
                    return T[]
                end
            end
        end
    end
end

"""
    roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, r₀::T) where {T<:Real}

Return the interval of existence garanted by the Radii Polynomial Theorem from the root(s) of the polynomial Y + (Z₁-1)r + Z₂r²/2.
"""
function roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, r₀::T) where {T<:Real}
    @info "Applying the Radii Polynomial Theoreom of order 2: Y + (Z₁-1)r + Z₂r²/2" r₀ Y Z₁ Z₂
    iszero(Z₂) && return roots_radii_polynomial(Y, Z₁, r₀)
    b = Z₁ - one(Z₁)
    Δ = b*b - 2Z₂*Y
    if inf(Δ) < 0 # complex roots
        d_ = - b - sqrt(complex(Δ))
        r₁_ = d_/Z₂
        r₂_ = 2Y/d_
        @info """NO VALID ROOTS FOUND:
            → Y ≥ 0, Z₁ ≥ 0, Z₂ ≥ 0
            ✗ the radii polynomial is positive in (-∞,+∞)
            ⇒ the interval of existence is empty""" r₁_ r₂_
        return T[]
    else # real roots
        d = - b - sqrt(Δ)
        r₁ = d/Z₂
        if iszero(Y) # exact solution
            if Z₁ ≥ 1
                @info """VALID ROOTS FOUND:
                    → Y = 0, Z₁ ≥ 1 and Z₂ ≥ 0
                    ✓ the radii polynomial is negative in [r₁,0]
                    ✓ 0 ≤ r₀ and 0 × Z₂ < 1
                    ✗ r₁ < 0
                    ⇒ the interval of existence is {0}""" r₁
                return [zero(T)]
            else
                if r₁ ≤ r₀ && r₁ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y = 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [0,r₁]
                        ✓ r₁ ≤ r₀ and r₁ × Z₂ < 1
                        ✓ 0 ≤ r₁
                        ⇒ the interval of existence is [0,r₁]""" r₁
                    return [zero(T),inf(r₁)]
                elseif r₀ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y = 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [0,r₁]
                        ✗ r₁ > r₀ or r₁ × Z₂ ≥ 1
                        ✓ r₀ × Z₂ < 1
                        ✓ 0 ≤ r₀
                        ⇒ the interval of existence is [0,r₀]""" r₁
                    return [zero(T),r₀]
                else
                    @info """VALID ROOTS FOUND:
                        → Y = 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [0,r₁]
                        ✗ r₁ > r₀ or r₁ × Z₂ ≥ 1
                        ✗ r₀ × Z₂ ≥ 1
                        ⇒ the interval of existence is [0,Z₂⁻¹)""" r₁
                    return [zero(T),inf(inv(Z₂))]
                end
            end
        else # not exact solution and Z₁ ≠ 1 (since for Y ≥ 0 it implies complex roots)
            r₂ = 2Y/d
            if Z₁ > 1
                @info """NO VALID ROOTS FOUND:
                    → Y ≥ 0, Z₁ > 1 and Z₂ ≥ 0
                    ✓ the radii polynomial is negative in [r₁,r₂]
                    ✗ r₁ ≤ r₂ < 0
                    ⇒ the interval of existence is empty""" r₁ r₂
                return T[]
            else
                if r₂ ≤ r₀ && r₂ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [r₁,r₂]
                        ✓ r₂ ≤ r₀ and r₂ × Z₂ < 1
                        ✓ 0 ≤ r₁ ≤ r₂
                        ⇒ the interval of existence is [r₁,r₂]""" r₁ r₂
                    return [sup(r₁),inf(r₂)]
                elseif r₁ ≤ r₀ && r₀ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [r₁,r₂]
                        ✗ r₂ > r₀ or r₂ × Z₂ ≥ 1
                        ✓ r₀ × Z₂ < 1
                        ✓ r₁ ≤ r₀
                        ⇒ the interval of existence is [r₁,r₀]""" r₁ r₂
                    return [sup(r₁),r₀]
                elseif r₁ ≤ r₀ && r₁ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [r₁,r₂]
                        ✗ r₂ > r₀ or r₂ × Z₂ ≥ 1
                        ✗ r₀ × Z₂ ≥ 1
                        ✓ r₁ ≤ r₀ and r₁ × Z₂ < 1
                        ⇒ the interval of existence is {r₁}""" r₁ r₂
                    return [sup(r₁)]
                else
                    @info """NO VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✓ the radii polynomial is negative in [r₁,r₂]
                        ✗ r₂ > r₀ or r₂ × Z₂ ≥ 1
                        ✗ r₁ > r₀ or r₀ × Z₂ ≥ 1
                        ✗ r₁ > r₀ or r₁ × Z₂ ≥ 1
                        ⇒ the interval of existence is empty""" r₁ r₂
                    return T[]
                end
            end
        end
    end
end
