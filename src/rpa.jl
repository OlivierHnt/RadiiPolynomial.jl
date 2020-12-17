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
                ✔ the radii polynomial is negative in (-∞,0]
                ⇒ the interval of existence is {0}"""
            return [zero(T)]
        else
            @info """VALID ROOT FOUND:
                → Y = 0 and 0 ≤ Z₁ < 1
                ✔ the radii polynomial is negative in [0,+∞)
                ⇒ the interval of existence is [0,r₀]"""
            return [zero(T), r₀]
        end
    else # not exact solution
        if isone(Z₁)
            @info """NO VALID ROOT FOUND:
                → Y ≥ 0 and Z₁ = 1
                ✗ the radii polynomial is positive with constant value Y
                ⇒ the interval of existence is ∅""" Y
            return T[]
        else
            r₁ = Y/(one(Z₁) - Z₁)
            if Z₁ > 1
                @info """NO VALID ROOT FOUND:
                    → Y ≥ 0 and Z₁ > 1
                    ✔ the radii polynomial is negative in (-∞,r₁]
                    ✗ r₁ < 0
                    ⇒ the interval of existence is ∅""" r₁
                return T[]
            else
                if r₁ ≤ r₀
                    @info """VALID ROOT FOUND:
                        → Y ≥ 0 and 0 ≤ Z₁ < 1
                        ✔ the radii polynomial is negative in [r₁,+∞)
                        ✔ r₁ ≤ r₀
                        ⇒ the interval of existence is [r₁,r₀]""" r₁
                    return [sup(r₁), r₀]
                else
                    @info """NO VALID ROOT FOUND:
                        → Y ≥ 0 and 0 ≤ Z₁ < 1
                        ✔ the radii polynomial is negative in [r₁,+∞)
                        ✗ r₁ > r₀
                        ⇒ the interval of existence is ∅""" r₁
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

##

"""
"""
function Y end

"""
"""
function Z₁ end

"""
"""
function Z₂ end

## RPA finite dimensional case

"""
    FixedPointProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅}

Fields:
- `x₀ :: T₁`
- `G :: T₂`
- `DG :: T₃`
- `D²G :: T₄`
- `r₀ :: T₅`
"""
mutable struct FixedPointProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅}
    x₀ :: T₁
    G :: T₂
    DG :: T₃
    D²G :: T₄
    r₀ :: T₅
end

Y(pb::FixedPointProblemFiniteDimension) = norm(pb.G - pb.x₀, Inf)

Z₁(pb::FixedPointProblemFiniteDimension) = opnorm(pb.DG, Inf)

Z₂(pb::FixedPointProblemFiniteDimension) = opnorm(pb.D²G, Inf)

#

"""
    ZeroFindingProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅}

Fields:
- `F :: T₁`
- `DF :: T₂`
- `A :: T₃`
- `D²F :: T₄`
- `r₀ :: T₅`
"""
mutable struct ZeroFindingProblemFiniteDimension{T₁,T₂,T₃,T₄,T₅}
    F :: T₁
    DF :: T₂
    A :: T₃
    D²F :: T₄
    r₀ :: T₅
end

Y(pb::ZeroFindingProblemFiniteDimension) = norm(pb.A * pb.F, Inf)

Z₁(pb::ZeroFindingProblemFiniteDimension) = opnorm(pb.A * pb.DF - I, Inf)

Z₂(pb::ZeroFindingProblemFiniteDimension) = opnorm(pb.A * pb.D²F, Inf)

## RPA infinite dimensional case

"""
    TailProblem{T₁,T₂,T₃,T₄,T₅,T₆}

Fields:
- `∞f :: T₁`
- `Df :: T₂`
- `D²_abs_f :: T₃`
- `∞L∞_bound :: T₄`
- `ν :: T₅`
- `r₀ :: T₆`
"""
mutable struct TailProblem{T₁,T₂,T₃,T₄,T₅,T₆}
    ∞f :: T₁
    Df :: T₂
    D²_abs_f :: T₃
    ∞L∞_bound :: T₄
    ν :: T₅
    r₀ :: T₆
end

prove(pb::TailProblem) =
    roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)

Y(pb::TailProblem{T}) where {T<:Sequence} =
    pb.∞L∞_bound * norm(pb.∞f, pb.ν)

Y(pb::TailProblem{T}) where {T<:AbstractVector{<:Sequence}} =
    pb.∞L∞_bound * norm(norm.(pb.∞f, pb.ν), Inf)

Z₁(pb::TailProblem{T}) where {T<:Sequence} =
    pb.∞L∞_bound * norm(pb.Df, pb.ν)

Z₁(pb::TailProblem{T}) where {T<:AbstractVector{<:Sequence}} =
    pb.∞L∞_bound * opnorm(norm.(pb.Df, pb.ν), Inf)

Z₂(pb::TailProblem{T}) where {T<:Sequence} =
    pb.∞L∞_bound * pb.D²_abs_f

Z₂(pb::TailProblem{T}) where {T<:AbstractVector{<:Sequence}} =
    pb.∞L∞_bound * opnorm(pb.D²_abs_f, Inf)

#

"""
    ZeroFindingProblemCategory1{T₁,T₂,T₃,T₄,T₅,T₆,T₇,T₈,T₉,T₁₀}

Fields:
- `∞f :: T₁`
- `Df :: T₂`
- `D²_abs_f :: T₃`
- `ᴺF :: T₄`
- `ᴺDF :: T₅`
- `ᴺAᴺ :: T₆`
- `ᴺL∞_bound :: T₇`
- `∞L⁻¹∞_bound :: T₈`
- `ν :: T₉`
- `r₀ :: T₁₀`
"""
mutable struct ZeroFindingProblemCategory1{T₁,T₂,T₃,T₄,T₅,T₆,T₇,T₈,T₉,T₁₀}
    ∞f :: T₁
    Df :: T₂
    D²_abs_f :: T₃
    ᴺF :: T₄
    ᴺDF :: T₅
    ᴺAᴺ :: T₆
    ᴺL∞_bound :: T₇
    ∞L⁻¹∞_bound :: T₈
    ν :: T₉
    r₀ :: T₁₀
end

prove(pb::ZeroFindingProblemCategory1) =
    roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), pb.r₀)

Y(pb::ZeroFindingProblemCategory1{T}) where {T<:Sequence} =
    norm(pb.ᴺAᴺ * pb.ᴺF, pb.ν) + pb.∞L⁻¹∞_bound * norm(pb.∞f, pb.ν)

Y(pb::ZeroFindingProblemCategory1{T}) where {T<:AbstractVector{<:Sequence}} =
    norm(norm.(pb.ᴺAᴺ * pb.ᴺF, pb.ν), Inf) + pb.∞L⁻¹∞_bound * norm(norm.(pb.∞f, pb.ν), Inf)

Z₁(pb::ZeroFindingProblemCategory1{T}) where {T<:Sequence} =
    opnorm(pb.ᴺAᴺ * pb.ᴺDF -̄ I, pb.ν, pb.ν) + pb.ᴺL∞_bound * opnorm(pb.ᴺAᴺ, pb.ν, pb.ν) + pb.∞L⁻¹∞_bound * norm(pb.Df, pb.ν)

Z₁(pb::ZeroFindingProblemCategory1{T}) where {T<:AbstractVector{<:Sequence}} =
    opnorm(opnorm.(pb.ᴺAᴺ * pb.ᴺDF -̄ I, pb.ν, pb.ν), Inf) + pb.ᴺL∞_bound * opnorm(opnorm.(pb.ᴺAᴺ, pb.ν, pb.ν), Inf) + pb.∞L⁻¹∞_bound * opnorm(norm.(pb.Df, pb.ν), Inf)

Z₂(pb::ZeroFindingProblemCategory1{T}) where {T<:Sequence} =
    (opnorm(pb.ᴺAᴺ, pb.ν, pb.ν) + pb.∞L⁻¹∞_bound) * pb.D²_abs_f

Z₂(pb::ZeroFindingProblemCategory1{T}) where {T<:AbstractVector{<:Sequence}} =
    (opnorm(opnorm.(pb.ᴺAᴺ, pb.ν, pb.ν), Inf) + pb.∞L⁻¹∞_bound) * opnorm(pb.D²_abs_f, Inf)

## Generic newton method

function newton(x₀, F_DF; tol::Real=1e-12, maxiter::Int=15, verbose::Bool=false)
    verbose && return _newton_verbose(x₀, F_DF, tol, maxiter)
    return _newton_silent(x₀, F_DF, tol, maxiter)
end

function _newton_verbose(x₀::Number, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm_F = abs(F)
    printstyled("    → iteration ", 0, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
    if norm_F ≤ tol
        printstyled("    ... succes!\n"; color = :green)
        return x₀
    end
    x = x₀
    for i ∈ 1:maxiter
        x -= DF \ F
        F, DF = F_DF(x)
        norm_F = abs(F)
        printstyled("    → iteration ", i, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
        if norm_F ≤ tol
            printstyled("    ... succes!\n"; color = :green)
            return x
        end
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end

function _newton_verbose(x₀::Vector, F_DF, tol, maxiter)
    printstyled("Starting Newton's method ...\n"; color = :blue)
    F, DF = F_DF(x₀)
    norm_F = norm(F, Inf)
    printstyled("    → iteration ", 0, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
    if norm_F ≤ tol
        printstyled("    ... succes!\n"; color = :green)
        return x₀
    end
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm_F = norm(F, Inf)
        printstyled("    → iteration ", i, ": ||F(x)|| = ", norm_F, "\n"; color = :yellow)
        if norm_F ≤ tol
            printstyled("    ... succes!\n"; color = :green)
            return x
        end
    end
    printstyled("    ... failure!\n"; color = :red)
    return x
end

function _newton_silent(x₀::Number, F_DF, tol, maxiter)
    F, DF = F_DF(x₀)
    abs(F) ≤ tol && return x₀
    x = x₀
    for i ∈ 1:maxiter
        x -= DF \ F
        F, DF = F_DF(x)
        abs(F) ≤ tol && return x
    end
    return x
end

function _newton_silent(x₀::Vector, F_DF, tol, maxiter)
    F, DF = F_DF(x₀)
    norm(F, Inf) ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        F, DF = F_DF(x)
        norm(F, Inf) ≤ tol && return x
    end
    return x
end
