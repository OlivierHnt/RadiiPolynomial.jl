_linear_regression_log_abs(x) = log(abs(x))
_linear_regression_log_abs(x::Interval) = log(mag(x))
_linear_regression_log_abs(x::Complex{<:Interval}) = log(mag(x))

"""
    Weight

Abstract type for all weights.
"""
abstract type Weight end

_getindex(weight::NTuple{N,Weight}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _getindex(weight[1], s[1], α[1]) * _getindex(Base.tail(weight), Base.tail(s), Base.tail(α))
_getindex(weight::Tuple{Weight}, s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _getindex(weight[1], s[1], α[1])

"""
    IdentityWeight <: Weight

Identity weight.
"""
struct IdentityWeight <: Weight end

"""
    GeometricWeight{T<:Real} <: Weight

Geometric weight associated with a given `rate` satisfying `isfinite(rate) & (rate ≥ 1)`.

Field:
- `rate :: T`

See also: [`geometricweight`](@ref), [`IdentityWeight`](@ref),
[`AlgebraicWeight`](@ref), [`algebraicweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> w = GeometricWeight(1.0)
GeometricWeight(1.0)

julia> rate(w)
1.0
```
"""
struct GeometricWeight{T<:Real} <: Weight
    rate :: T
    function GeometricWeight{T}(rate::T) where {T<:Real}
        isfinite(inf(rate)) & (inf(rate) ≥ 1) || return throw(DomainError(rate, "rate must be finite and greater or equal to one"))
        return new{T}(rate)
    end
end

GeometricWeight(rate::T) where {T<:Real} = GeometricWeight{T}(rate)

rate(weight::GeometricWeight) = weight.rate

_getindex(weight::GeometricWeight, ::Taylor, i::Int) = rate(weight) ^ ExactReal(i)

_getindex(weight::GeometricWeight, ::Fourier, i::Int) = rate(weight) ^ ExactReal(abs(i))

_getindex(weight::GeometricWeight, ::Chebyshev, i::Int) = rate(weight) ^ ExactReal(i)

"""
    geometricweight(a::Sequence{<:SequenceSpace})

Compute an approximation of the geometric decay rate of `a` by performing the
ordinary least squares method on the logarithm of the absolute value of the
coefficients of `a`.

See also: [`GeometricWeight`](@ref), [`IdentityWeight`](@ref),
[`AlgebraicWeight`](@ref), [`algebraicweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> rate(geometricweight(Sequence(Taylor(10), [inv(2.0^i) for i in 0:10]))) ≈ 2
true

julia> rate.(geometricweight(Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv(2.0^i * 3.0^abs(j)) for i in 0:10, j in -3:3])))) .≈ (2, 3)
(true, true)
```
"""
geometricweight(::Sequence{<:SequenceSpace})

function geometricweight(a::Sequence{<:BaseSpace})
    s = space(a)
    A = coefficients(a)
    rate, _ = _geometric_max_rate(s, A)
    return GeometricWeight(rate)
end

function geometricweight(a::Sequence{<:TensorSpace})
    s = space(a)
    A = coefficients(a)
    rate, _ = _geometric_rate(s, A)
    return GeometricWeight.(rate)
end

function _geometric_rate(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    log_abs_A = _linear_regression_log_abs.(filter(!iszero, A))
    len = length(log_abs_A)
    len == 0 && return ntuple(_ -> one(eltype(log_abs_A)), Val(N)), zero(eltype(log_abs_A))
    x = ones(Float64, len, N+1)
    n = 0
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if !iszero(A[i])
            view(x, i-n, 2:N+1) .= abs.(α) .+ 1
        else
            n += 1
        end
    end
    r = LinearAlgebra.svd(x) \ log_abs_A
    rate = ntuple(Val(N)) do i
        @inbounds rᵢ₊₁ = r[i+1]
        v = ifelse(isfinite(rᵢ₊₁) & (rᵢ₊₁ < 0), rᵢ₊₁, zero(rᵢ₊₁))
        return exp(-v)
    end
    err = norm(mul!(log_abs_A, x, r, 1, -1), 2)
    return rate, err
end

# Taylor

function _geometric_max_rate(s::Taylor, A)
    i = length(A)
    @inbounds abs_Aᵢ = abs(A[i])
    n = 1
    mean = abs_Aᵢ / n
    δ = abs(abs_Aᵢ - mean)
    while i ≥ 2 && δ ≤ 1e-12
        i -= 1
        @inbounds abs_Aᵢ = abs(A[i])
        check_nz = !iszero(abs_Aᵢ)
        n_prev = n
        n += check_nz
        mean = (n_prev * mean + abs_Aᵢ) / n
        δ = ifelse(check_nz, abs(abs_Aᵢ - mean), δ)
    end
    @inbounds rate, err = _geometric_rate(s, view(A, 1:i), mean)
    return rate, err
end

function _geometric_rate(::Taylor, A, threshold)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ) && abs(Aᵢ) ≥ threshold
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += i
            u += i*log_abs_Aᵢ
            t += i*i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    β = ifelse(isfinite(r) & (r < 0), r, zero(r))
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            err += (_linear_regression_log_abs(Aᵢ) - α - β * i)^2
        end
    end
    return exp(-β), err
end

# Fourier

function _geometric_max_rate(s::Fourier, A)
    ord = order(s)
    len = i = length(A)
    @inbounds abs_A₋ᵢ = abs(A[len-i+1])
    @inbounds abs_Aᵢ = abs(A[i])
    n_neg = 1
    n_pos = 1
    mean_neg = abs_A₋ᵢ / n_neg
    mean_pos = abs_Aᵢ / n_pos
    δ_neg = abs(abs_A₋ᵢ - mean_neg)
    δ_pos = abs(abs_Aᵢ - mean_pos)
    while i ≥ ord+2 && min(δ_neg, δ_pos) ≤ 1e-12
        i -= 1
        @inbounds abs_A₋ᵢ = abs(A[len-i+1])
        @inbounds abs_Aᵢ = abs(A[i])
        check_nz_neg = !iszero(abs_A₋ᵢ)
        check_nz_pos = !iszero(abs_Aᵢ)
        n_prev_neg = n_neg
        n_prev_pos = n_pos
        n_neg += check_nz_neg
        n_pos += check_nz_pos
        mean_neg = (n_prev_neg * mean_neg + abs_A₋ᵢ) / n_neg
        mean_pos = (n_prev_pos * mean_pos + abs_Aᵢ) / n_pos
        δ_neg = ifelse(check_nz_neg, abs(abs_A₋ᵢ - mean_neg), δ_neg)
        δ_pos = ifelse(check_nz_pos, abs(abs_Aᵢ - mean_pos), δ_pos)
    end
    @inbounds rate, err = _geometric_rate(Fourier(i - ord - 1, frequency(s)), view(A, (len-i+1):i), min(mean_pos, mean_neg))
    return rate, err
end

function _geometric_rate(s::Fourier, A, threshold)
    ord = order(s)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ) && abs(Aᵢ) ≥ threshold
            abs_i = abs(i-ord-1)+1
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += abs_i
            u += abs_i*log_abs_Aᵢ
            t += abs_i*abs_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    β = ifelse(isfinite(r) & (r < 0), r, zero(r))
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            err += (_linear_regression_log_abs(Aᵢ) - α - β * (abs(i-ord-1)+1))^2
        end
    end
    return exp(-β), err
end

# Chebyshev

function _geometric_max_rate(s::Chebyshev, A)
    i = length(A)
    @inbounds abs_Aᵢ = abs(A[i])
    n = 1
    mean = abs_Aᵢ / n
    δ = abs(abs_Aᵢ - mean)
    while i ≥ 2 && δ ≤ 1e-12
        i -= 1
        @inbounds abs_Aᵢ = abs(A[i])
        check_nz = !iszero(abs_Aᵢ)
        n_prev = n
        n += check_nz
        mean = (n_prev * mean + abs_Aᵢ) / n
        δ = ifelse(check_nz, abs(abs_Aᵢ - mean), δ)
    end
    @inbounds rate, err = _geometric_rate(s, view(A, 1:i), mean)
    return rate, err
end

function _geometric_rate(::Chebyshev, A, threshold)
    sum_x = t = n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ) && abs(Aᵢ) ≥ threshold
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += i
            u += i*log_abs_Aᵢ
            t += i*i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    β = ifelse(isfinite(r) & (r < 0), r, zero(r))
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            err += (_linear_regression_log_abs(Aᵢ) - α - β * i)^2
        end
    end
    return exp(-β), err
end

"""
    AlgebraicWeight{T<:Real} <: Weight

Algebraic weight associated with a given `rate` satisfying `isfinite(rate) & (rate ≥ 0)`.

Field:
- `rate :: T`

See also: [`algebraicweight`](@ref), [`IdentityWeight`](@ref),
[`GeometricWeight`](@ref), [`geometricweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> w = AlgebraicWeight(1.0)
AlgebraicWeight(1.0)

julia> rate(w)
1.0
```
"""
struct AlgebraicWeight{T<:Real} <: Weight
    rate :: T
    function AlgebraicWeight{T}(rate::T) where {T<:Real}
        isfinite(inf(rate)) & (inf(rate) ≥ 0) || return throw(DomainError(rate, "rate must be finite and positive"))
        return new{T}(rate)
    end
end

AlgebraicWeight(rate::T) where {T<:Real} = AlgebraicWeight{T}(rate)

rate(weight::AlgebraicWeight) = weight.rate

_getindex(weight::AlgebraicWeight, ::Taylor, i::Int) = ExactReal(1 + i) ^ rate(weight)

_getindex(weight::AlgebraicWeight, ::Fourier, i::Int) = ExactReal(1 + abs(i)) ^ rate(weight)

_getindex(weight::AlgebraicWeight, ::Chebyshev, i::Int) = ExactReal(1 + i) ^ rate(weight)

"""
    algebraicweight(a::Sequence{<:SequenceSpace})

Compute an approximation of the algebraic decay rate of `a` by performing the
ordinary least squares method on the logarithm of the absolute value of the
coefficients of `a`.

See also: [`AlgebraicWeight`](@ref), [`IdentityWeight`](@ref),
[`GeometricWeight`](@ref), [`geometricweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> rate(algebraicweight(Sequence(Taylor(10), [inv((1.0 + i)^2) for i in 0:10]))) ≈ 2
true

julia> rate.(algebraicweight(Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv((1.0 + i)^2 * (1.0 + abs(j))^3) for i in 0:10, j in -3:3])))) .≈ (2, 3)
(true, true)
```
"""
algebraicweight(::Sequence{<:SequenceSpace})

function algebraicweight(a::Sequence{<:BaseSpace})
    s = space(a)
    A = coefficients(a)
    rate, _ = _algebraic_max_rate(s, A)
    return AlgebraicWeight(rate)
end

function algebraicweight(a::Sequence{<:TensorSpace})
    s = space(a)
    A = coefficients(a)
    rate, _ = _algebraic_rate(s, A)
    return AlgebraicWeight.(rate)
end

function _algebraic_rate(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    log_abs_A = _linear_regression_log_abs.(filter(!iszero, A))
    len = length(log_abs_A)
    len == 0 && return ntuple(_ -> zero(eltype(log_abs_A)), Val(N)), zero(eltype(log_abs_A))
    x = ones(Float64, len, N+1)
    n = 0
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if !iszero(A[i])
            view(x, i-n, 2:N+1) .= log.(abs.(α) .+ 1)
        else
            n += 1
        end
    end
    r = LinearAlgebra.svd(x) \ log_abs_A
    rate = ntuple(Val(N)) do i
        @inbounds rᵢ₊₁ = r[i+1]
        return ifelse(isfinite(rᵢ₊₁) & (rᵢ₊₁ < 0), -rᵢ₊₁, zero(rᵢ₊₁))
    end
    err = norm(mul!(log_abs_A, x, r, 1, -1), 2)
    return rate, err
end

# Taylor

function _algebraic_max_rate(s::Taylor, A)
    i = length(A)
    @inbounds abs_Aᵢ = abs(A[i])
    mean = abs_Aᵢ/1
    n = 0 + !iszero(abs_Aᵢ)
    δ = abs(abs_Aᵢ - mean)
    while i ≥ 2 && δ ≤ 1e-12
        i -= 1
        @inbounds abs_Aᵢ = abs(A[i])
        check_nz = !iszero(abs_Aᵢ)
        n_prev = n
        n += check_nz
        mean = (n_prev * mean + abs_Aᵢ) / n
        δ = ifelse(check_nz, abs(abs_Aᵢ - mean), δ)
    end
    @inbounds rate, err = _algebraic_rate(s, view(A, 1:i), mean)
    return rate, err
end

function _algebraic_rate(::Taylor, A, threshold)
    sum_x = t = 0.0
    n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0.0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ) && abs(Aᵢ) ≥ threshold
            log_i = log(i)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += log_i
            u += log_i*log_abs_Aᵢ
            t += log_i*log_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    β = ifelse(isfinite(r) & (r < 0), -r, zero(r))
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            err += (_linear_regression_log_abs(Aᵢ) - α - β * log(i))^2
        end
    end
    return β, err
end

# Fourier

function _algebraic_max_rate(s::Fourier, A)
    ord = order(s)
    len = i = length(A)
    @inbounds abs_A₋ᵢ = abs(A[len-i+1])
    @inbounds abs_Aᵢ = abs(A[i])
    mean_neg = abs_A₋ᵢ/1
    mean_pos = abs_Aᵢ/1
    n_neg = 0 + !iszero(abs_A₋ᵢ)
    n_pos = 0 + !iszero(abs_Aᵢ)
    δ_neg = abs(abs_A₋ᵢ - mean_neg)
    δ_pos = abs(abs_Aᵢ - mean_pos)
    while i ≥ ord+2 && min(δ_neg, δ_pos) ≤ 1e-12
        i -= 1
        @inbounds abs_A₋ᵢ = abs(A[len-i+1])
        @inbounds abs_Aᵢ = abs(A[i])
        check_nz_neg = !iszero(abs_A₋ᵢ)
        check_nz_pos = !iszero(abs_Aᵢ)
        n_prev_neg = n_neg
        n_prev_pos = n_pos
        n_neg += check_nz_neg
        n_pos += check_nz_pos
        mean_neg = (n_prev_neg * mean_neg + abs_A₋ᵢ) / n_neg
        mean_pos = (n_prev_pos * mean_pos + abs_Aᵢ) / n_pos
        δ_neg = ifelse(check_nz_neg, abs(abs_A₋ᵢ - mean_neg), δ_neg)
        δ_pos = ifelse(check_nz_pos, abs(abs_Aᵢ - mean_pos), δ_pos)
    end
    @inbounds rate, err = _algebraic_rate(Fourier(i - ord - 1, frequency(s)), view(A, (len-i+1):i), min(mean_pos, mean_neg))
    return rate, err
end

function _algebraic_rate(s::Fourier, A, threshold)
    ord = order(s)
    sum_x = t = 0.0
    n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0.0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ) && abs(Aᵢ) ≥ threshold
            log_abs_i = log(abs(i-ord-1)+1)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += log_abs_i
            u += log_abs_i*log_abs_Aᵢ
            t += log_abs_i*log_abs_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    β = ifelse(isfinite(r) & (r < 0), -r, zero(r))
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            err += (_linear_regression_log_abs(Aᵢ) - α - β * log(abs(i-ord-1)+1))^2
        end
    end
    return β, err
end

# Chebyshev

function _algebraic_max_rate(s::Chebyshev, A)
    i = length(A)
    @inbounds abs_Aᵢ = abs(A[i])
    mean = abs_Aᵢ/1
    n = 0 + !iszero(abs_Aᵢ)
    δ = abs(abs_Aᵢ - mean)
    while i ≥ 2 && δ ≤ 1e-12
        i -= 1
        @inbounds abs_Aᵢ = abs(A[i])
        check_nz = !iszero(abs_Aᵢ)
        n_prev = n
        n += check_nz
        mean = (n_prev * mean + abs_Aᵢ) / n
        δ = ifelse(check_nz, abs(abs_Aᵢ - mean), δ)
    end
    @inbounds rate, err = _algebraic_rate(s, view(A, 1:i), mean)
    return rate, err
end

function _algebraic_rate(::Chebyshev, A, threshold)
    sum_x = t = 0.0
    n = 0
    sum_log_abs_A = zero(_linear_regression_log_abs(one(eltype(A))))
    u = 0.0*sum_log_abs_A
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ) && abs(Aᵢ) ≥ threshold
            log_i = log(i)
            log_abs_Aᵢ = _linear_regression_log_abs(Aᵢ)
            sum_x += log_i
            u += log_i*log_abs_Aᵢ
            t += log_i*log_i
            sum_log_abs_A += log_abs_Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_log_abs_A)/(t - sum_x*x̄)
    β = ifelse(isfinite(r) & (r < 0), -r, zero(r))
    α = sum_log_abs_A/n - β * x̄
    err = zero(sum_log_abs_A)
    for (i, Aᵢ) ∈ enumerate(A)
        if !iszero(Aᵢ)
            err += (_linear_regression_log_abs(Aᵢ) - α - β * log(i))^2
        end
    end
    return β, err
end

# retrieve the optimal weight

function weight(a::Sequence{<:BaseSpace})
    s = space(a)
    A = coefficients(a)
    geo_rate, geo_err = _geometric_max_rate(s, A)
    alg_rate, alg_err = _algebraic_max_rate(s, A)
    geo_err ≤ alg_err && return GeometricWeight(geo_rate)
    return AlgebraicWeight(alg_rate)
end

function weight(a::Sequence{<:TensorSpace})
    s = space(a)
    A = coefficients(a)
    geo_rate, geo_err = _geometric_rate(s, A)
    alg_rate, alg_err = _algebraic_rate(s, A)
    geo_err ≤ alg_err && return GeometricWeight.(geo_rate)
    return AlgebraicWeight.(alg_rate)
end

"""
    BesselWeight{T<:Real} <: Weight

Bessel weight associated with a given `rate` satisfying `isfinite(rate) & (rate ≥ 0)`.

Field:
- `rate :: T`

See also: [`IdentityWeight`](@ref), [`GeometricWeight`](@ref), [`geometricweight`](@ref),
[`AlgebraicWeight`](@ref) and [`algebraicweight`](@ref).

# Examples

```jldoctest
julia> w = BesselWeight(1.0)
BesselWeight(1.0)

julia> rate(w)
1.0
```
"""
struct BesselWeight{T<:Real} <: Weight
    rate :: T
    function BesselWeight{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate ≥ 0) || return throw(DomainError(rate, "rate must be finite and positive"))
        return new{T}(rate)
    end
end

BesselWeight(rate::T) where {T<:Real} = BesselWeight{T}(rate)

rate(weight::BesselWeight) = weight.rate

_getindex(weight::BesselWeight, ::TensorSpace{<:NTuple{N,Fourier}}, α::NTuple{N,Int}) where {N} =
    (one(rate(weight)) + mapreduce(abs2, +, α)) ^ rate(weight)
_getindex(weight::BesselWeight{<:Interval}, ::TensorSpace{<:NTuple{N,Fourier}}, α::NTuple{N,Int}) where {N} =
    (one(rate(weight)) + interval(mapreduce(abs2, +, α))) ^ rate(weight)

_getindex(weight::BesselWeight, ::Fourier, i::Int) = (one(rate(weight)) + i*i) ^ rate(weight)
_getindex(weight::BesselWeight{<:Interval}, ::Fourier, i::Int) = (one(rate(weight)) + interval(i*i)) ^ rate(weight)

# show

Base.show(io::IO, ::MIME"text/plain", weight::Weight) = print(io, _prettystring(weight))

_prettystring(weight::Weight) = string(weight)
_prettystring(weight::GeometricWeight) = "GeometricWeight(" * string(rate(weight)) * ")"
_prettystring(weight::AlgebraicWeight) = "AlgebraicWeight(" * string(rate(weight)) * ")"
_prettystring(weight::BesselWeight) = "BesselWeight(" * string(rate(weight)) * ")"

#

"""
    BanachSpace

Abstract type for all Banach spaces.
"""
abstract type BanachSpace end

"""
    Ell1{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^1`` space.

Field:
- `weight :: T`

Constructors:
- `Ell1(::Weight)`
- `Ell1(::Tuple{Vararg{Weight}})`
- `Ell1()`: equivalent to `Ell1(IdentityWeight())`
- `Ell1(weight::Weight...)`: equivalent to `Ell1(weight)`

Unicode alias [`ℓ¹`](@ref) can be typed by `\\ell<tab>\\^1<tab>` in the Julia REPL
and in many editors.

See also: [`Ell2`](@ref) and [`EllInf`](@ref).

# Examples

```jldoctest
julia> Ell1()
ℓ¹()

julia> Ell1(GeometricWeight(1.0))
ℓ¹(GeometricWeight(1.0))

julia> Ell1(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
struct Ell1{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    Ell1{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    Ell1{Tuple{}}(::Tuple{}) = throw(ArgumentError("Ell1 is only defined for at least one Weight"))
end

Ell1(weight::T) where {T<:Weight} = Ell1{T}(weight)
Ell1(weight::T) where {T<:Tuple{Vararg{Weight}}} = Ell1{T}(weight)
Ell1() = Ell1{IdentityWeight}(IdentityWeight())
Ell1(weight::Weight...) = Ell1(weight)

weight(X::Ell1) = X.weight

"""
    ℓ¹(::Weight)
    ℓ¹(::Tuple{Vararg{Weight}})
    ℓ¹()
    ℓ¹(::Weight...)

Unicode alias of [`Ell1`](@ref) representing the weighted ``\\ell^1`` space.

# Examples

```jldoctest
julia> ℓ¹()
ℓ¹()

julia> ℓ¹(GeometricWeight(1.0))
ℓ¹(GeometricWeight(1.0))

julia> ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
const ℓ¹ = Ell1

"""
    Ell2{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^2`` space.

Field:
- `weight :: T`

Constructors:
- `Ell2(::Weight)`
- `Ell2(::Tuple{Vararg{Weight}})`
- `Ell2()`: equivalent to `Ell2(IdentityWeight())`
- `Ell2(weight::Weight...)`: equivalent to `Ell2(weight)`

Unicode alias [`ℓ²`](@ref) can be typed by `\\ell<tab>\\^2<tab>` in the Julia REPL
and in many editors.

See also: [`Ell1`](@ref) and [`EllInf`](@ref).

# Examples

```jldoctest
julia> Ell2()
ℓ²()

julia> Ell2(BesselWeight(1.0))
ℓ²(BesselWeight(1.0))

julia> Ell2(BesselWeight(1.0), GeometricWeight(2.0))
ℓ²(BesselWeight(1.0), GeometricWeight(2.0))
```
"""
struct Ell2{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    Ell2{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    Ell2{Tuple{}}(::Tuple{}) = throw(ArgumentError("Ell2 is only defined for at least one Weight"))
end

Ell2(weight::T) where {T<:Weight} = Ell2{T}(weight)
Ell2(weight::T) where {T<:Tuple{Vararg{Weight}}} = Ell2{T}(weight)
Ell2() = Ell2{IdentityWeight}(IdentityWeight())
Ell2(weight::Weight...) = Ell2(weight)

weight(X::Ell2) = X.weight

"""
    ℓ²(::Weight)
    ℓ²(::Tuple{Vararg{Weight}})
    ℓ²()
    ℓ²(::Weight...)

Unicode alias of [`Ell2`](@ref) representing the weighted ``\\ell^2`` space.

# Examples

```jldoctest
julia> ℓ²()
ℓ²()

julia> ℓ²(BesselWeight(1.0))
ℓ²(BesselWeight(1.0))

julia> ℓ²(BesselWeight(1.0), GeometricWeight(2.0))
ℓ²(BesselWeight(1.0), GeometricWeight(2.0))
```
"""
const ℓ² = Ell2

"""
    EllInf{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^\\infty`` space.

Field:
- `weight :: T`

Constructors:
- `EllInf(::Weight)`
- `EllInf(::Tuple{Vararg{Weight}})`
- `EllInf()`: equivalent to `EllInf(IdentityWeight())`
- `EllInf(weight::Weight...)`: equivalent to `EllInf(weight)`

Unicode alias [`ℓ∞`](@ref) can be typed by `\\ell<tab>\\infty<tab>` in the Julia REPL
and in many editors.

See also: [`Ell1`](@ref) and [`Ell2`](@ref).

# Examples

```jldoctest
julia> EllInf()
ℓ∞()

julia> EllInf(GeometricWeight(1.0))
ℓ∞(GeometricWeight(1.0))

julia> EllInf(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ∞(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
struct EllInf{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    EllInf{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    EllInf{Tuple{}}(::Tuple{}) = throw(ArgumentError("EllInf is only defined for at least one Weight"))
end

EllInf(weight::T) where {T<:Weight} = EllInf{T}(weight)
EllInf(weight::T) where {T<:Tuple{Vararg{Weight}}} = EllInf{T}(weight)
EllInf() = EllInf{IdentityWeight}(IdentityWeight())
EllInf(weight::Weight...) = EllInf(weight)

weight(X::EllInf) = X.weight

"""
    ℓ∞(::Weight)
    ℓ∞(::Tuple{Vararg{Weight}})
    ℓ∞()
    ℓ∞(::Weight...)

Unicode alias of [`EllInf`](@ref) representing the weighted ``\\ell^\\infty`` space.

# Examples

```jldoctest
julia> ℓ∞()
ℓ∞()

julia> ℓ∞(GeometricWeight(1.0))
ℓ∞(GeometricWeight(1.0))

julia> ℓ∞(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ∞(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
const ℓ∞ = EllInf

# normed cartesian space

"""
    NormedCartesianSpace{T<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},S<:BanachSpace} <: BanachSpace

Cartesian Banach space.

Fields:
- `inner :: T`
- `outer :: S`

See also: [`Ell1`](@ref), [`Ell2`](@ref) and [`EllInf`](@ref).

# Examples

```jldoctest
julia> NormedCartesianSpace(Ell1(), EllInf())
NormedCartesianSpace(ℓ¹(), ℓ∞())

julia> NormedCartesianSpace((Ell1(), Ell2()), EllInf())
NormedCartesianSpace((ℓ¹(), ℓ²()), ℓ∞())
```
"""
struct NormedCartesianSpace{T<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},S<:BanachSpace} <: BanachSpace
    inner :: T
    outer :: S
end

# show

Base.show(io::IO, ::MIME"text/plain", X::BanachSpace) = print(io, _prettystring(X))

_prettystring(X::BanachSpace) = string(X)

_prettystring(::Ell1{IdentityWeight}) = "ℓ¹()"
_prettystring(X::Ell1{<:Weight}) = "ℓ¹(" * _prettystring(weight(X)) * ")"
_prettystring(X::Ell1{<:Tuple{Vararg{Weight}}}) = "ℓ¹(" * mapreduce(_prettystring, (x, y) -> x * ", " * y, weight(X)) * ")"

_prettystring(::Ell2{IdentityWeight}) = "ℓ²()"
_prettystring(X::Ell2{<:Weight}) = "ℓ²(" * _prettystring(weight(X)) * ")"
_prettystring(X::Ell2{<:Tuple{Vararg{Weight}}}) = "ℓ²(" * mapreduce(_prettystring, (x, y) -> x * ", " * y, weight(X)) * ")"

_prettystring(::EllInf{IdentityWeight}) = "ℓ∞()"
_prettystring(X::EllInf{<:Weight}) = "ℓ∞(" * _prettystring(weight(X)) * ")"
_prettystring(X::EllInf{<:Tuple{Vararg{Weight}}}) = "ℓ∞(" * mapreduce(_prettystring, (x, y) -> x * ", " * y, weight(X)) * ")"

_prettystring(X::NormedCartesianSpace{<:BanachSpace}) =
    "NormedCartesianSpace(" * _prettystring(X.inner) * ", " *  _prettystring(X.outer) * ")"

_prettystring(X::NormedCartesianSpace{<:Tuple{Vararg{BanachSpace}}}) =
    "NormedCartesianSpace((" * mapreduce(_prettystring, (x, y) -> x * ", " * y, X.inner) * "), " *  _prettystring(X.outer) * ")"

#

"""
    norm(a::AbstractSequence, p::Real=Inf)

Compute the `p`-norm of `a`. Only `p` equal to `1`, `2` or `Inf` is supported.

This is equivalent to:
- `norm(a, Ell1(IdentityWeight()))` if `p == 1`
- `norm(a, Ell2(IdentityWeight()))` if `p == 2`
- `norm(a, EllInf(IdentityWeight()))` if `p == Inf`

See also: [`norm(::Sequence, ::BanachSpace)`](@ref).
"""
function norm(a::AbstractSequence, p::Real=Inf)
    if p == 1
        return norm(a, Ell1(IdentityWeight()))
    elseif p == 2
        return norm(a, Ell2(IdentityWeight()))
    elseif p == Inf
        return norm(a, EllInf(IdentityWeight()))
    else
        return throw(ArgumentError("p-norm is only supported for p equal to 1, 2 or Inf"))
    end
end

"""
    opnorm(A::LinearOperator, p::Real=Inf)

Compute the operator norm of `A` induced by the `p`-norm. Only `p` equal to `1`,
`2` or `Inf` is supported.

This is equivalent to:
- `opnorm(A, Ell1(IdentityWeight()))` if `p == 1`
- `opnorm(A, Ell2(IdentityWeight()))` if `p == 2`
- `opnorm(A, EllInf(IdentityWeight()))` if `p == Inf`

See also: [`opnorm(::LinearOperator, ::BanachSpace)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)`](@ref).
"""
function opnorm(A::LinearOperator, p::Real=Inf)
    if p == 1
        return opnorm(A, Ell1(IdentityWeight()))
    elseif p == 2
        return opnorm(A, Ell2(IdentityWeight()))
    elseif p == Inf
        return opnorm(A, EllInf(IdentityWeight()))
    else
        return throw(ArgumentError("p-norm is only supported for p equal to 1, 2 or Inf"))
    end
end

"""
    opnorm(A::LinearOperator, X::BanachSpace, Y::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
`domain(A)` and `Y` the Banach space corresponding to `codomain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)`](@ref).
"""
function opnorm(A::LinearOperator, X::BanachSpace, Y::BanachSpace)
    codomain_A = codomain(A)
    A_ = coefficients(A)
    @inbounds v₁ = norm(Sequence(codomain_A, view(A_, :, 1)), Y)
    T = typeof(v₁)
    sz = size(A_, 2)
    v = Vector{T}(undef, sz)
    @inbounds v[1] = v₁
    @inbounds for i ∈ 2:sz
        A_view = view(A_, :, i)
        if all(iszero, A_view)
            v[i] = zero(T)
        else
            v[i] = norm(Sequence(codomain_A, A_view), Y)
        end
    end
    return opnorm(LinearOperator(domain(A), ParameterSpace(), transpose(v)), X)
end

"""
    opnorm(A::LinearOperator, X::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
both `domain(A)` and `codomain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)`](@ref).
"""
opnorm(A::LinearOperator, X::BanachSpace) = opnorm(A, X, X)

#

"""
    norm(a::Sequence, X::BanachSpace)

Compute the norm of `a` by interpreting `space(a)` as `X`.

See also: [`norm(::Sequence, ::Real=Inf)`](@ref).
"""
norm(::Sequence, ::BanachSpace)

"""
    opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::BanachSpace)

Compute the operator norm of `A` where `X` is the Banach space corresponding to
`domain(A)`.

See also: [`opnorm(::LinearOperator, ::Real=Inf)`](@ref),
[`opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)`](@ref) and
[`opnorm(::LinearOperator, ::BanachSpace)`](@ref).
"""
opnorm(::LinearOperator{<:VectorSpace,ParameterSpace}, ::BanachSpace)

for T ∈ (:Ell1, :Ell2, :EllInf)
    @eval begin
        norm(a::Sequence, X::$T{<:Weight}) = _apply(X, space(a), coefficients(a))
        norm(a::Sequence{TensorSpace{S}}, X::$T{<:Weight}) where {N,S<:NTuple{N,BaseSpace}} =
            norm(a, $T(ntuple(_ -> weight(X), Val(N))))
        function norm(a::Sequence{TensorSpace{S}}, X::$T{<:NTuple{N,Weight}}) where {N,S<:NTuple{N,BaseSpace}}
            space_a = space(a)
            A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
            return _apply(X, space_a, A)
        end
        _apply(X::$T, space::TensorSpace, A) =
            @inbounds _apply($T(weight(X)[1]), space[1], _apply($T(Base.tail(weight(X))), Base.tail(space), A))
        _apply(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply($T(weight(X)[1]), space[1], A)

        opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::$T{<:Weight}) =
            _apply_dual(X, domain(A), vec(coefficients(A)))
        opnorm(A::LinearOperator{TensorSpace{S},ParameterSpace}, X::$T{<:Weight}) where {N,S<:NTuple{N,BaseSpace}} =
            opnorm(A, $T(ntuple(_ -> weight(X), Val(N))))
        function opnorm(A::LinearOperator{TensorSpace{S},ParameterSpace}, X::$T{<:NTuple{N,Weight}}) where {N,S<:NTuple{N,BaseSpace}}
            domain_A = domain(A)
            A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
            return _apply_dual(X, domain_A, A_)
        end
        _apply_dual(X::$T, space::TensorSpace, A) =
            @inbounds _apply_dual($T(weight(X)[1]), space[1], _apply_dual($T(Base.tail(weight(X))), Base.tail(space), A))
        _apply_dual(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply_dual($T(weight(X)[1]), space[1], A)
    end
end

#

function opnorm(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ::Ell1{IdentityWeight}, ::Ell1{IdentityWeight})
    r = z = abs(zero(eltype(A)))
    A_ = coefficients(A)
    for j ∈ axes(A_, 2)
        s = z
        @inbounds for i ∈ axes(A_, 1)
            s += abs(A_[i,j])
        end
        r = max(r, s)
    end
    return r
end

opnorm(A::LinearOperator, ::Ell2{IdentityWeight}, ::Ell2{IdentityWeight}) =
    sqrt(opnorm(A, Ell1(IdentityWeight())) * opnorm(A, EllInf(IdentityWeight())))

function opnorm(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ::EllInf{IdentityWeight}, ::EllInf{IdentityWeight})
    r = z = abs(zero(eltype(A)))
    A_ = coefficients(A)
    for i ∈ axes(A_, 1)
        s = z
        @inbounds for j ∈ axes(A_, 2)
            s += abs(A_[i,j])
        end
        r = max(r, s)
    end
    return r
end

# ParameterSpace

_apply(::Ell1{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::Ell1{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])

_apply(::Ell2{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::Ell2{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])

_apply(::EllInf{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::EllInf{IdentityWeight}, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])

# SequenceSpace

# Taylor

_apply(::Ell1{IdentityWeight}, ::Taylor, A::AbstractVector) = sum(abs, A)
function _apply(::Ell1{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs.(selectdim(A, N, i))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(EllInf(IdentityWeight()), space, A)

function _apply(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ ord-1:-1:0
        s .= s .* ν .+ abs.(selectdim(A, N, i+1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i+1]) * νⁱ)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) * _getindex(weight(X), space, i)
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) / _getindex(weight(X), space, i))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Taylor, A::AbstractVector) = sqrt(sum(abs2, A))
function _apply(::Ell2{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs2.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs2.(selectdim(A, N, i))
    end
    s .= sqrt.(s)
    return s
end
_apply_dual(X::Ell2{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(X, space, A)

_apply(::EllInf{IdentityWeight}, ::Taylor, A::AbstractVector) = maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, ::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Taylor, A::AbstractArray) = _apply(Ell1(IdentityWeight()), space, A)

function _apply(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ = one(ν)
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i+1]) * νⁱ)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ ord-1:-1:0
        s .= s .* ν .+ abs.(selectdim(A, N, i+1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) * _getindex(weight(X), space, i))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) / _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i)
    end
    return s
end

# Fourier

_apply(::Ell1{IdentityWeight}, ::Fourier, A::AbstractVector) = sum(abs, A)
function _apply(::Ell1{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs.(selectdim(A, N, i))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(EllInf(IdentityWeight()), space, A)

function _apply(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    if ord == 0
        @inbounds s = abs(A[1]) * one(ν)
    else
        @inbounds s = (abs(A[1]) + abs(A[2ord+1])) * one(ν)
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[ord+1-i]) + abs(A[ord+1+i])
        end
        @inbounds s = s * ν + abs(A[ord+1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₋ₙ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₋ₙ))
    if ord == 0
        s .= abs.(A₋ₙ)
    else
        @inbounds s .= abs.(selectdim(A, N, 2ord+1)) .+ abs.(A₋ₙ)
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))
        end
        @inbounds s .= s .* ν .+ abs.(selectdim(A, N, ord+1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * νⁱ
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s = max(s, abs(A[ord+1+i]) * νⁱ, abs(A[ord+1-i]) * νⁱ)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* νⁱ, abs.(selectdim(A, N, ord+1+i)) .* νⁱ)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) * _getindex(weight(X), space, i)
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) .* _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _getindex(weight(X), space, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _getindex(weight(X), space, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Fourier, A::AbstractVector) = sqrt(sum(abs2, A))
function _apply(::Ell2{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs2.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .+= abs2.(selectdim(A, N, i))
    end
    s .= sqrt.(s)
    return s
end
_apply_dual(X::Ell2{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(X, space, A)

function _apply(X::Ell2{<:BesselWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) * _getindex(weight(X), space, i)
    end
    return sqrt(x)
end
function norm(a::Sequence{TensorSpace{T}}, X::Ell2{<:BesselWeight}) where {N,T<:NTuple{N,Fourier}}
    space_a = space(a)
    x = zero(abs2(zero(eltype(a)))*_getindex(weight(X), space_a, ntuple(_ -> 0, Val(N))))
    @inbounds for α ∈ indices(space_a)
        x += abs2(a[α]) * _getindex(weight(X), space_a, α)
    end
    return sqrt(x)
end
function _apply_dual(X::Ell2{<:BesselWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) / _getindex(weight(X), space, i)
    end
    return sqrt(x)
end
function opnorm(A::LinearOperator{TensorSpace{T},ParameterSpace}, X::Ell2{<:BesselWeight}) where {N,T<:NTuple{N,Fourier}}
    domain_A = domain(A)
    x = zero(abs2(zero(eltype(A)))/_getindex(weight(X), domain_A, ntuple(_ -> 0, Val(N))))
    @inbounds for α ∈ indices(domain_A)
        x += abs2(A[1,α]) / _getindex(weight(X), domain_A, α)
    end
    return sqrt(x)
end

_apply(::EllInf{IdentityWeight}, ::Fourier, A::AbstractVector) = maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, ::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁)
    @inbounds for i ∈ 2:size(A, N)
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Fourier, A::AbstractArray) = _apply(Ell1(IdentityWeight()), space, A)

function _apply(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ = one(ν)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * νⁱ
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s = max(s, abs(A[ord+1+i]) * νⁱ, abs(A[ord+1-i]) * νⁱ)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ = one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* νⁱ, abs.(selectdim(A, N, ord+1+i)) .* νⁱ)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    if ord == 0
        @inbounds s = abs(A[1]) * one(ν)
    else
        @inbounds s = (abs(A[1]) + abs(A[2ord+1])) * one(ν)
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[ord+1-i]) + abs(A[ord+1+i])
        end
        @inbounds s = s * ν + abs(A[ord+1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds A₋ₙ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₋ₙ))
    if ord == 0
        s .= abs.(A₋ₙ)
    else
        @inbounds s .= abs.(selectdim(A, N, 2ord+1)) .+ abs.(A₋ₙ)
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))
        end
        @inbounds s .= s .* ν .+ abs.(selectdim(A, N, ord+1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        x = _getindex(weight(X), space, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = _getindex(weight(X), space, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) / _getindex(weight(X), space, i)
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) ./ _getindex(weight(X), space, i)
    end
    return s
end

# Chebyshev

_apply(::Ell1{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds abs(A[1]) + ExactReal(2) * sum(abs, view(A, 2:length(A)))
function _apply(::Ell1{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= ExactReal(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds max(abs(A[1]), maximum(abs, view(A, 2:length(A))) / ExactReal(2))
function _apply_dual(::Ell1{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(s ./ ExactReal(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = (ExactReal(2) * ν) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= (ExactReal(2) * ν) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / ExactReal(2)
    @inbounds s = abs(A[1]) * one(νⁱ½)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s = max(s, abs(A[i+1]) * νⁱ½)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / ExactReal(2)
    CoefType = typeof(abs(zero(T))*νⁱ½)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ½)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) * _getindex(weight(X), space, i)
        end
        @inbounds s = ExactReal(2) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i)
        end
        @inbounds s .= ExactReal(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) / _getindex(weight(X), space, i))
        end
        @inbounds s = max(s / ExactReal(2), abs(A[1]))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(s ./ ExactReal(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + ExactReal(2) * sum(abs2, view(A, 2:length(A))))
function _apply(::Ell2{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(ExactReal(2) .* s .+ abs2.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::Ell2{IdentityWeight}, ::Chebyshev, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + sum(abs2, view(A, 2:length(A))) / ExactReal(2))
function _apply_dual(::Ell2{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(s ./ ExactReal(2) .+ abs2.(selectdim(A, N, 1)))
    return s
end

_apply(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractVector) =
    @inbounds max(abs(A[1]), ExactReal(2) * maximum(abs, view(A, 2:length(A))))
function _apply(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(ExactReal(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractVector) =
    @inbounds abs(A[1]) + sum(abs, view(A, 2:length(A))) / ExactReal(2)
function _apply_dual(::EllInf{IdentityWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s ./ ExactReal(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ2 = ExactReal(2) * one(ν)
    @inbounds s = abs(A[1]) * one(νⁱ)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s = max(s, abs(A[i+1]) * νⁱ2)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ2 = ExactReal(2) * one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ2)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ2)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractVector)
    ν = inv(rate(weight(X)))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = s * (ν / ExactReal(2)) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s .* (ν / ExactReal(2)) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) * _getindex(weight(X), space, i))
        end
        @inbounds s = max(ExactReal(2) * s, abs(A[1]))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(ExactReal(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) / _getindex(weight(X), space, i)
        end
        @inbounds s = s / ExactReal(2) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i)
        end
        @inbounds s .= s ./ ExactReal(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

# CartesianSpace

_apply(::Ell1{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = sum(abs, A)
_apply_dual(::Ell1{IdentityWeight}, space::CartesianSpace, A::AbstractVector) = _apply(EllInf(IdentityWeight()), space, A)

_apply(::Ell2{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = sqrt(sum(abs2, A))
_apply_dual(::Ell2{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = sqrt(sum(abs2, A))

_apply(::EllInf{IdentityWeight}, ::CartesianSpace, A::AbstractVector) = maximum(abs, A)
_apply_dual(::EllInf{IdentityWeight}, space::CartesianSpace, A::AbstractVector) = _apply(Ell1(IdentityWeight()), space, A)

for X ∈ (:Ell1, :EllInf)
    @eval begin
        function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _norm(a, X)
        end
        function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _norm(a, X)
        end

        function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,$X{IdentityWeight}})
            0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
            return _opnorm(A, X)
        end
        function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$X{IdentityWeight}}) where {N}
            0 < nspaces(domain(A)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
            return _opnorm(A, X)
        end
    end
end

function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(space(a)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_norm2(a, X))
end
function norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    0 < nspaces(space(a)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
    return sqrt(_norm2(a, X))
end

function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    0 < nspaces(domain(A)) || return throw(ArgumentError("number of cartesian products must be strictly positive"))
    return sqrt(_opnorm2(A, X))
end
function opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    0 < nspaces(domain(A)) == N || return throw(ArgumentError("number of cartesian products must be strictly positive and equal to the number of inner Banach spaces"))
    return sqrt(_opnorm2(A, X))
end

# Ell1

function _norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}})
    @inbounds r = norm(component(a, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(space(a))
        r += norm(component(a, i), X.inner)
    end
    return r
end
_norm(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds norm(component(a, 1), X.inner) + _norm(component(a, 2:N), X)
_norm(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner)
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell1{IdentityWeight}}) where {N} =
    @inbounds norm(component(a, 1), X.inner[1]) + _norm(component(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell1{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner[1])

function _opnorm(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}})
    @inbounds r = opnorm(component(A, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(domain(A))
        r = max(r, opnorm(component(A, i), X.inner))
    end
    return r
end
_opnorm(A::LinearOperator{CartesianProduct{T},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds max(opnorm(component(A, 1), X.inner), _opnorm(component(A, 2:N), X))
_opnorm(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell1{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner)
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell1{IdentityWeight}}) where {N} =
    @inbounds max(opnorm(component(A, 1), X.inner[1]), _opnorm(component(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer)))
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell1{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner[1])

# Ell2

function _norm2(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = norm(component(a, 1), X.inner)
    r = v*v
    @inbounds for i ∈ 2:nspaces(space(a))
        v = norm(component(a, i), X.inner)
        r += v*v
    end
    return r
end
function _norm2(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds v = norm(component(a, 1), X.inner)
    return @inbounds v*v + _norm2(component(a, 2:N), X)
end
function _norm2(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = norm(component(a, 1), X.inner)
    return v*v
end
function _norm2(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    @inbounds v = norm(component(a, 1), X.inner[1])
    return @inbounds v*v + _norm2(component(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
end
function _norm2(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell2{IdentityWeight}})
    @inbounds v = norm(component(a, 1), X.inner[1])
    return v*v
end

function _opnorm2(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = opnorm(component(A, 1), X.inner)
    r = v*v
    @inbounds for i ∈ 2:nspaces(domain(A))
        v = opnorm(component(A, i), X.inner)
        r += v*v
    end
    return r
end
function _opnorm2(A::LinearOperator{CartesianProduct{T},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds v = opnorm(component(A, 1), X.inner)
    return @inbounds v*v + _opnorm2(component(A, 2:N), X)
end
function _opnorm2(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,Ell2{IdentityWeight}})
    @inbounds v = opnorm(component(A, 1), X.inner)
    return v*v
end
function _opnorm2(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},Ell2{IdentityWeight}}) where {N}
    @inbounds v = opnorm(component(A, 1), X.inner[1])
    return @inbounds v*v + _opnorm2(component(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
end
function _opnorm2(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},Ell2{IdentityWeight}})
    @inbounds v = opnorm(component(A, 1), X.inner[1])
    return v*v
end

# EllInf

function _norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}})
    @inbounds r = norm(component(a, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(space(a))
        r = max(r, norm(component(a, i), X.inner))
    end
    return r
end
_norm(a::Sequence{CartesianProduct{T}}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds max(norm(component(a, 1), X.inner), _norm(component(a, 2:N), X))
_norm(a::Sequence{<:CartesianProduct{<:Tuple{VectorSpace}}}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner)
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},EllInf{IdentityWeight}}) where {N} =
    @inbounds max(norm(component(a, 1), X.inner[1]), _norm(component(a, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer)))
_norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},EllInf{IdentityWeight}}) =
    @inbounds norm(component(a, 1), X.inner[1])

function _opnorm(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}})
    @inbounds r = opnorm(component(A, 1), X.inner)
    @inbounds for i ∈ 2:nspaces(domain(A))
        r += opnorm(component(A, i), X.inner)
    end
    return r
end
_opnorm(A::LinearOperator{CartesianProduct{T},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) where {N,T<:NTuple{N,VectorSpace}} =
    @inbounds opnorm(component(A, 1), X.inner) + _opnorm(component(A, 2:N), X)
_opnorm(A::LinearOperator{<:CartesianProduct{<:Tuple{VectorSpace}},ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,EllInf{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner)
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},EllInf{IdentityWeight}}) where {N} =
    @inbounds opnorm(component(A, 1), X.inner[1]) + _opnorm(component(A, 2:N), NormedCartesianSpace(Base.tail(X.inner), X.outer))
_opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:Tuple{BanachSpace},EllInf{IdentityWeight}}) =
    @inbounds opnorm(component(A, 1), X.inner[1])
