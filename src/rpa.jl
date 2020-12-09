function roots_radii_polynomial(Y::Interval{T}, Z::Interval{T}, r₀::T) where {T<:Real}
    @info "Applying the Radii Polynomial Theoreom of order 1: p(r) := Y + (Z-1)r" r₀ Y Z
    if iszero(Y) # exact solution
        if Z ≥ 1
            @info """VALID ROOT FOUND:
                → Y = 0 and Z ≥ 1
                ✔ the radii polynomial is negative in (-∞,0]
                ⇒ the interval of existence is {0}"""
            return [zero(T)]
        else
            @info """VALID ROOT FOUND:
                → Y = 0 and 0 ≤ Z < 1
                ✔ the radii polynomial is negative in [0,+∞)
                ⇒ the interval of existence is [0,r₀]"""
            return [zero(T), r₀]
        end
    else # not exact solution
        if isone(Z)
            @info """NO VALID ROOT FOUND:
                → Y ≥ 0 and Z = 1
                ✗ the radii polynomial is positive with constant value Y
                ⇒ the interval of existence is ∅""" Y
            return T[]
        else
            r₁ = Y/(one(Z) - Z)
            if Z > 1
                @info """NO VALID ROOT FOUND:
                    → Y ≥ 0 and Z > 1
                    ✔ the radii polynomial is negative in (-∞,r₁]
                    ✗ r₁ < 0
                    ⇒ the interval of existence is ∅""" r₁
                return T[]
            else
                if r₁ ≤ r₀
                    @info """VALID ROOT FOUND:
                        → Y ≥ 0 and 0 ≤ Z < 1
                        ✔ the radii polynomial is negative in [r₁,+∞)
                        ✔ r₁ ≤ r₀
                        ⇒ the interval of existence is [r₁,r₀]""" r₁
                    return [sup(r₁), r₀]
                else
                    @info """NO VALID ROOT FOUND:
                        → Y ≥ 0 and 0 ≤ Z < 1
                        ✔ the radii polynomial is negative in [r₁,+∞)
                        ✗ r₁ > r₀
                        ⇒ the interval of existence is ∅""" r₁
                    return T[]
                end
            end
        end
    end
end

function roots_radii_polynomial(Y::Interval{T}, Z₁::Interval{T}, Z₂::Interval{T}, r₀::T) where {T<:Real}
    @info "Applying the Radii Polynomial Theoreom of order 2: p(r) := Y + (Z₁-1)r + Z₂r²/2" r₀ Y Z₁ Z₂
    iszero(Z₂) && return roots_radii_polynomial(Y, Z₁, r₀)
    b = Z₁ - one(Z₁)
    Δ = b^2 - 2Z₂*Y
    if Δ < 0 # complex roots
        d_ = - b - sqrt(complex(Δ))
        r₁_ = d_/Z₂
        r₂_ = 2Y/d_
        @info """NO VALID ROOTS FOUND:
            → Y ≥ 0, Z₁ ≥ 0, Z₂ ≥ 0
            ✗ the radii polynomial is positive in (-∞,+∞)
            ⇒ the interval of existence is ∅""" r₁_ r₂_
        return T[]
    else # real roots
        d = - b - sqrt(Δ)
        r₁ = d/Z₂
        if iszero(Y) # exact solution
            if Z₁ ≥ 1
                @info """VALID ROOTS FOUND:
                    → Y = 0, Z₁ ≥ 1 and Z₂ ≥ 0
                    ✔ the radii polynomial is negative in [r₁,0]
                    ✔ 0 ≤ r₀ and 0 × Z₂ < 1
                    ✗ r₁ < 0
                    ⇒ the interval of existence is {0}""" r₁
                return [zero(T)]
            else
                if r₁ ≤ r₀ && r₁ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y = 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [0,r₁]
                        ✔ r₁ ≤ r₀ and r₁ × Z₂ < 1
                        ✔ 0 ≤ r₁
                        ⇒ the interval of existence is [0,r₁]""" r₁
                    return [zero(T),inf(r₁)]
                elseif r₀ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y = 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [0,r₁]
                        ✗ r₁ > r₀ or r₁ × Z₂ ≥ 1
                        ✔ r₀ × Z₂ < 1
                        ✔ 0 ≤ r₀
                        ⇒ the interval of existence is [0,r₀]""" r₁
                    return [zero(T),r₀]
                else
                    @info """VALID ROOTS FOUND:
                        → Y = 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [0,r₁]
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
                    ✔ the radii polynomial is negative in [r₁,r₂]
                    ✗ r₁ ≤ r₂ < 0
                    ⇒ the interval of existence is ∅""" r₁ r₂
                return T[]
            else
                if r₂ ≤ r₀ && r₂ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [r₁,r₂]
                        ✔ r₂ ≤ r₀ and r₂ × Z₂ < 1
                        ✔ 0 ≤ r₁ ≤ r₂
                        ⇒ the interval of existence is [r₁,r₂]""" r₁ r₂
                    return [sup(r₁),inf(r₂)]
                elseif r₁ ≤ r₀ && r₀ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [r₁,r₂]
                        ✗ r₂ > r₀ or r₂ × Z₂ ≥ 1
                        ✔ r₀ × Z₂ < 1
                        ✔ r₁ ≤ r₀
                        ⇒ the interval of existence is [r₁,r₀]""" r₁ r₂
                    return [sup(r₁),r₀]
                elseif r₁ ≤ r₀ && r₁ * Z₂ < 1
                    @info """VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [r₁,r₂]
                        ✗ r₂ > r₀ or r₂ × Z₂ ≥ 1
                        ✗ r₀ × Z₂ ≥ 1
                        ✔ r₁ ≤ r₀ and r₁ × Z₂ < 1
                        ⇒ the interval of existence is {r₁}""" r₁ r₂
                    return [sup(r₁)]
                else
                    @info """NO VALID ROOTS FOUND:
                        → Y ≥ 0, 0 ≤ Z₁ < 1 and Z₂ ≥ 0
                        ✔ the radii polynomial is negative in [r₁,r₂]
                        ✗ r₂ > r₀ or r₂ × Z₂ ≥ 1
                        ✗ r₁ > r₀ or r₀ × Z₂ ≥ 1
                        ✗ r₁ > r₀ or r₁ × Z₂ ≥ 1
                        ⇒ the interval of existence is ∅""" r₁ r₂
                    return T[]
                end
            end
        end
    end
end

## RPA finite dimensional case

_make_interval_x_pm_r₀(x::Real, r₀::Real) = x ± r₀
_make_interval_x_pm_r₀(x::Complex, r₀::Real) = (real(x) ± r₀) + im*(imag(x) ± r₀)

function rpa_finite_dimension(x::T, F::Function, DF::Function, r₀::Real=Inf) where {T}
    x_interval = @interval(x)
    x_pm_r₀ = _make_interval_x_pm_r₀(x, r₀)

    F_interval = F(x_interval)
    DF_interval = DF(x_pm_r₀)
    DF_ = DF(x)

    Y = abs(DF_ \ F_interval)
    Z = abs(DF_ \ DF_interval - one(T))

    return roots_radii_polynomial(Y, Z, r₀)
end

function rpa_finite_dimension(x::Vector{T}, F::Function, DF::Function, r₀::Real=Inf) where {T}
    x_interval = [@interval(xᵢ) for xᵢ ∈ x]
    x_pm_r₀ = [_make_interval_x_pm_r₀(xᵢ, r₀) for xᵢ ∈ x]

    F_interval = F(x_interval)
    DF_interval = DF(x_pm_r₀)
    DF_ = DF(x)

    Y = norm(DF_ \ F_interval, Inf)
    Z = opnorm(DF_ \ DF_interval - I*one(T), Inf)

    return roots_radii_polynomial(Y, Z, r₀)
end

function rpa_finite_dimension(x::T, F::Function, DF::Function, D²F::Function, r₀::Real=Inf) where {T}
    x_interval = @interval(x)
    x_pm_r₀ = _make_interval_x_pm_r₀(x, r₀)

    F_interval = F(x_interval)
    DF_interval = DF(x_interval)
    DF_ = DF(x)
    D²F_ = D²F(x_pm_r₀)

    Y = abs(DF_ \ F_interval)
    Z₁ = abs(DF_ \ DF_interval - one(T))
    Z₂ = abs(DF_ \ D²F_)

    return roots_radii_polynomial(Y, Z₁, Z₂, r₀)
end

function rpa_finite_dimension(x::Vector{T}, F::Function, DF::Function, D²F::Function, r₀::Real=Inf) where {T}
    x_interval = [@interval(xᵢ) for xᵢ ∈ x]
    x_pm_r₀ = [_make_interval_x_pm_r₀(xᵢ, r₀) for xᵢ ∈ x]

    F_interval = F(x_interval)
    DF_interval = DF(x_interval)
    DF_ = DF(x)
    D²F_ = D²F(x_pm_r₀)

    Y = norm(DF_ \ F_interval, Inf)
    Z₁ = opnorm(DF_ \ DF_interval - I*one(T), Inf)
    Z₂ = opnorm(DF_ \ D²F_, Inf)

    return roots_radii_polynomial(Y, Z₁, Z₂, r₀)
end

## Generic newton method

function newton(x₀, F_DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=false)
    verbose && return _newton_verbose(x₀, F_DF, tol, maxiter)
    return _newton_silent(x₀, F_DF, tol, maxiter)
end

function _newton_verbose(x₀::Number, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm_F = norm(F, Inf)
    printstyled("    → iteration ", 0, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
    norm_F ≤ tol && return x₀
    x = x₀
    for i ∈ 1:maxiter
        x -= DF \ F
        F, DF = F_DF(x)
        norm_F = norm(F, Inf)
        printstyled("    → iteration ", i, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
        norm_F ≤ tol && return x
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end

function _newton_silent(x₀::Number, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm(F, Inf) ≤ tol && return x₀
    x = x₀
    for i ∈ 1:maxiter
        x -= DF \ F
        F, DF = F_DF(x)
        norm(F, Inf) ≤ tol && return x
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end

function _newton_verbose(x₀::Vector, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm_F = norm(F, Inf)
    printstyled("    → iteration ", 0, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
    norm_F ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm_F = norm(F, Inf)
        printstyled("    → iteration ", i, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
        norm_F ≤ tol && return x
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end

function _newton_silent(x₀::Vector, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm(F, Inf) ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm(F, Inf) ≤ tol && return x
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end
