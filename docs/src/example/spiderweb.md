# Spiderweb central configurations

In this example we will prove the existence and local uniqueness of a class of relative equilibria called *central configurations* in the ``n``-body problem.

```@example
using LinearAlgebra, IntervalArithmetic, RadiiPolynomial

function newton(x₀::Vector, F_DF::Function, tol::Real, maxiter::Int)
    F, DF = F_DF(x₀)
    norm(F, Inf) ≤ tol && return x₀
    x = copy(x₀)
    for i ∈ 1:maxiter
        x .-= ldiv!(lu!(DF), F)
        sort!(x)
        F, DF = F_DF(x)
        norm(F, Inf) ≤ tol && return x
    end
    return x
end

function F_DF(x::Vector{T}, m₀::T, m::Vector{T}, λ::T, n::Int, l::Int) where {T}
    # compute the map F whose zero gives a spiderweb central configuration
    # and its jacobian DF
    sqrt2 = sqrt(2)
    F = λ*x
    DF = Matrix{T}(I*λ, n, n)
    indices = eachindex(x)
    l_range = 0:l-1
    @inbounds for j ∈ indices
        xⱼ² = x[j]^2
        @inbounds for i ∈ indices
            xᵢ², xᵢxⱼ = x[i]^2, x[i]*x[j]
            xᵢ³ = xᵢ²*x[i]
            if j == 1
                F[i] += m₀/xᵢ²
            end
            if i == j
                DF[i,j] += -2m₀/xᵢ³
            end
            @inbounds for k ∈ l_range
                θ = 360k/l
                cosdθ = cosd(θ)
                cosd2θ = cosd(2θ)
                if i ≠ j
                    d = sqrt(xᵢ² + xⱼ² - 2xᵢxⱼ*cosdθ)
                    d⁵2 = 2d^5
                    F[i] += m[j]*(x[i] - x[j]*cosdθ)/d^3
                    DF[i,i] += -m[j]*(4xᵢ² + xⱼ² - 8xᵢxⱼ*cosdθ + 3xⱼ²*cosd2θ)/d⁵2
                    DF[i,j] += -m[j]*(-4(xᵢ² + xⱼ²)*cosdθ + xᵢxⱼ*(7 + cosd2θ))/d⁵2
                elseif i == j && k > 0
                    ζ = sqrt2*sqrt(1-cosdθ)
                    F[i] += m[i]/(2xᵢ²*ζ)
                    DF[i,j] += -m[i]/(xᵢ³*ζ)
                end
            end
        end
    end
    return F, DF
end

function F_DF_D²F(x::Vector{T}, xᵣ::Vector{T}, m₀::T, m::Vector{T}, λ::T, n::Int, l::Int) where {T<:Interval}
    # compute with interval arithmetic the map F, its jacobian DF and the hessian D²F
    π2_l = 2convert(T, π)/l
    sqrt2 = sqrt(convert(T, 2))
    F = λ*x
    DF = Matrix{T}(I*λ, n, n)
    D²F = zeros(T, n, n*n)
    Dⱼ²Fᵢ = zeros(T, n, n)
    indices = eachindex(x)
    l_range = 0:l-1
    @inbounds for j ∈ indices
        xⱼ² = pow(x[j], 2)
        xᵣⱼ² = pow(xᵣ[j], 2)
        @inbounds for i ∈ indices
            xᵢ², xᵢxⱼ = pow(x[i], 2), x[i]*x[j]
            xᵢ³ = xᵢ²*x[i]
            xᵣᵢ², xᵣᵢxᵣⱼ = pow(xᵣ[i], 2), xᵣ[i]*xᵣ[j]
            xᵣᵢ⁴ = xᵣᵢ²*xᵣᵢ²
            if j == 1
                F[i] += m₀/xᵢ²
            end
            if i == j
                DF[i,j] += -2m₀/xᵢ³
                Dⱼ²Fᵢ[i,j] = 6m₀/xᵣᵢ⁴
            end
            @inbounds for k ∈ l_range
                θ = k*π2_l
                cosθ = cos(θ)
                cos2θ = cos(2θ)
                if i ≠ j
                    d = sqrt(xᵢ² + xⱼ² - 2xᵢxⱼ*cosθ)
                    d⁵2 = 2pow(d, 5)
                    F[i] += m[j]*(x[i] - x[j]*cosθ)/pow(d, 3)
                    DF[i,i] += -m[j]*(4xᵢ² + xⱼ² - 8xᵢxⱼ*cosθ + 3xⱼ²*cos2θ)/d⁵2
                    DF[i,j] += -m[j]*(-4(xᵢ² + xⱼ²)*cosθ + xᵢxⱼ*(7 + cos2θ))/d⁵2
                    dᵣ = sqrt(xᵣᵢ² + xᵣⱼ² - 2xᵣᵢxᵣⱼ*cosθ)
                    dᵣ⁷2 = 2pow(dᵣ, 7)
                    Dⱼ²Fᵢ[i,i] += 3m[j]*(xᵣ[i] - xᵣ[j]*cosθ)*(4xᵣᵢ² - xᵣⱼ² - 8xᵣᵢxᵣⱼ*cosθ + 5xᵣⱼ²*cos2θ)/dᵣ⁷2
                    Dⱼ²Fᵢ[i,j] += -3m[j]*(xᵣ[j]*(8xᵣⱼ² + 23xᵣᵢ²)*cosθ - xᵣ[i]*(20xᵣⱼ² + 2xᵣᵢ² + (4xᵣⱼ² + 6xᵣᵢ²)*cos2θ - xᵣᵢxᵣⱼ*cos(3θ)))/(2dᵣ⁷2)
                elseif i == j && k > 0
                    ζ = sqrt2*sqrt(1-cosθ)
                    F[i] += m[i]/(2xᵢ²*ζ)
                    DF[i,j] += -m[i]/(xᵢ³*ζ)
                    Dⱼ²Fᵢ[i,j] += 3m[i]/(xᵣᵢ⁴*ζ)
                end
            end
        end
    end
    @inbounds for k ∈ indices
        s = (k-1)*n
        view(D²F, n*s+1:n+1:n*s+n*n) .= view(Dⱼ²Fᵢ, k, :)
        Dⱼ²Fᵢ_view = view(Dⱼ²Fᵢ, :, k)
        view(D²F, k, 1+s:n+s) .= Dⱼ²Fᵢ_view
        view(D²F, :, k+s) .= Dⱼ²Fᵢ_view
    end
    return F, DF, D²F
end

n = 18 # number of circles
l = 100 # number of masses per circle

m₀ = 0.0 # central mass
m = ones(n)/l # vector of masses
λ = -1.0

# numerical solution

x₀ = newton(float.(1:n), x -> F_DF(x, m₀, m, λ, n, l), 1e-12, 50)

# proof

m₀_interval = @interval(0.0)
m_interval = [@interval(1/l) for _ ∈ 1:n]
λ_interval = @interval(-1.0)

x₀_interval = [@interval(xᵢ) for xᵢ ∈ x₀]

r₀ = 1e-5
xᵣ_interval = [xᵢ ± r₀ for xᵢ ∈ x₀]

F, DF, D²F = F_DF_D²F(x₀_interval, xᵣ_interval, m₀_interval, m_interval, λ_interval, n, l)
A = inv(mid.(DF))

pb = ZeroFindingProblemFiniteDimension(x₀_interval, r₀, F, DF, A, D²F)

error = roots_radii_polynomial(Y(pb), Z₁(pb), Z₂(pb), r₀)
```
