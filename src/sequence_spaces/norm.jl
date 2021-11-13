"""
    Norm

Abstract type for all norms.
"""
abstract type Norm end

# fallback method for operators

LinearAlgebra.opnorm(A::LinearOperator, d::Norm) = opnorm(A, d, d)

function LinearAlgebra.opnorm(A::LinearOperator, d_domain::Norm, d_codomain::Norm)
    codomain_A = codomain(A)
    A_ = coefficients(A)
    @inbounds v₁ = norm(Sequence(codomain_A, view(A_, :, 1)), d_codomain)
    len = size(A, 2)
    v = Matrix{typeof(v₁)}(undef, 1, len)
    @inbounds v[1] = v₁
    @inbounds for i ∈ 2:len
        v[i] = norm(Sequence(codomain_A, view(A_, :, i)), d_codomain)
    end
    return opnorm(LinearOperator(domain(A), ParameterSpace(), v), d_domain)
end

# Sequence spaces

# weights

abstract type Weights end

struct GeometricWeights{T<:Real} <: Weights
    rate :: T
    function GeometricWeights{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate > 0) || return throw(DomainError)
        return new{T}(rate)
    end
end
GeometricWeights(rate::T) where {T<:Real} = GeometricWeights{T}(rate)
rate(weights::GeometricWeights) = weights.rate

struct AlgebraicWeights{T<:Real} <: Weights
    rate :: T
    function AlgebraicWeights{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate ≥ 0) || return throw(DomainError)
        return new{T}(rate)
    end
end
AlgebraicWeights(rate::T) where {T<:Real} = AlgebraicWeights{T}(rate)
rate(weights::AlgebraicWeights) = weights.rate

function weight(s::TensorSpace{<:NTuple{N,BaseSpace}}, weights::NTuple{N,Weights}, α::NTuple{N,Int}) where {N}
    _is_space_index(s, α) || return throw(BoundsError)
    return _weight(s, weights, α)
end
function weight(s::BaseSpace, weights::Weights, i::Int)
    _is_space_index(s, i) || return throw(BoundsError)
    return _weight(s, weights, i)
end

_is_space_index(s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    _is_space_index(s[1], α[1]) & _is_space_index(Base.tail(s), Base.tail(α))
_is_space_index(s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    _is_space_index(s[1], α[1])
@generated function _weight(s::TensorSpace{<:NTuple{N,BaseSpace}}, weights::NTuple{N,Weights}, α::NTuple{N,Int}) where {N}
    p = :(_weight(s[1], weights[1], α[1]))
    for i ∈ 2:N
        p = :(_weight(s[$i], weights[$i], α[$i]) * $p)
    end
    return p
end

_is_space_index(::Taylor, i::Int) = i ≥ 0
_weight(::Taylor, weights::GeometricWeights, i::Int) = weights.rate ^ i
_weight(::Taylor, weights::GeometricWeights{<:Interval}, i::Int) = pow(weights.rate, i)
_weight(::Taylor, weights::AlgebraicWeights, i::Int) = (one(weights.rate) + i) ^ weights.rate
_weight(::Taylor, weights::AlgebraicWeights{<:Interval}, i::Int) = pow(one(weights.rate) + i, weights.rate)

_is_space_index(::Fourier, ::Int) = true
_weight(::Fourier, weights::GeometricWeights, i::Int) = weights.rate ^ abs(i)
_weight(::Fourier, weights::GeometricWeights{<:Interval}, i::Int) = pow(weights.rate, abs(i))
_weight(::Fourier, weights::AlgebraicWeights, i::Int) = (one(weights.rate) + abs(i)) ^ weights.rate
_weight(::Fourier, weights::AlgebraicWeights{<:Interval}, i::Int) = pow(one(weights.rate) + abs(i), weights.rate)

_is_space_index(::Chebyshev, i::Int) = i ≥ 0
function _weight(::Chebyshev, weights::GeometricWeights, i::Int)
    x = weights.rate ^ i
    return ifelse(i == 0, x, 2x)
end
function _weight(::Chebyshev, weights::GeometricWeights{<:Interval}, i::Int)
    x = pow(weights.rate, i)
    return ifelse(i == 0, x, 2x)
end
function _weight(::Chebyshev, weights::AlgebraicWeights, i::Int)
    x = (one(weights.rate) + i) ^ weights.rate
    return ifelse(i == 0, x, 2x)
end
function _weight(::Chebyshev, weights::AlgebraicWeights{<:Interval}, i::Int)
    x = pow(one(weights.rate) + i, weights.rate)
    return ifelse(i == 0, x, 2x)
end

function geometricweights(a::Sequence{<:BaseSpace})
    rate = _geometric_rate(space(a), coefficients(a))
    return GeometricWeights(rate)
end
function geometricweights(a::Sequence{<:TensorSpace})
    rate = _geometric_rate(space(a), coefficients(a))
    return GeometricWeights.(rate)
end

function algebraicweights(a::Sequence{<:BaseSpace})
    rate = _algebraic_rate(space(a), coefficients(a))
    return AlgebraicWeights(max(zero(rate), rate))
end
function algebraicweights(a::Sequence{<:TensorSpace})
    rate = _algebraic_rate(space(a), coefficients(a))
    return AlgebraicWeights.(max.(zero.(rate), rate))
end

_geometric_rate(s::BaseSpace, A::AbstractVector{<:Interval}) = Interval(exp(-_linear_regression(s, log.(mag.(A)))))
_geometric_rate(s::BaseSpace, A::AbstractVector{<:Complex{<:Interval}}) = Interval(exp(-_linear_regression(s, log.(mag.(A)))))
_geometric_rate(s::BaseSpace, A::AbstractVector) = exp(-_linear_regression(s, log.(abs.(A))))
_geometric_rate(s::TensorSpace, A::AbstractVector{<:Interval}) = Interval.(exp.((-).(_linear_regression(s, log.(mag.(A))))))
_geometric_rate(s::TensorSpace, A::AbstractVector{<:Complex{<:Interval}}) = Interval.(exp.((-).(_linear_regression(s, log.(mag.(A))))))
_geometric_rate(s::TensorSpace, A::AbstractVector) = exp.((-).(_linear_regression(s, log.(abs.(A)))))

_algebraic_rate(s::BaseSpace, A::AbstractVector{<:Interval}) = Interval(-_log_linear_regression(s, log.(mag.(A))))
_algebraic_rate(s::BaseSpace, A::AbstractVector{<:Complex{<:Interval}}) = Interval(-_log_linear_regression(s, log.(mag.(A))))
_algebraic_rate(s::BaseSpace, A::AbstractVector) = -_log_linear_regression(s, log.(abs.(A)))
_algebraic_rate(s::TensorSpace, A::AbstractVector{<:Interval}) = Interval.((-).(_log_linear_regression(s, log.(mag.(A)))))
_algebraic_rate(s::TensorSpace, A::AbstractVector{<:Complex{<:Interval}}) = Interval.((-).(_log_linear_regression( s, log.(mag.(A)))))
_algebraic_rate(s::TensorSpace, A::AbstractVector) = (-).(_log_linear_regression(s, log.(abs.(A))))

function _linear_regression(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    A_ = filter(isfinite, A)
    x = ones(Int, length(A_), N+1)
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if isfinite(A[i])
            view(x, i, 2:N+1) .= abs.(α) .+ 1
        end
    end
    x_T = transpose(x)
    r = (x_T * x) \ x_T * A_
    return @inbounds ntuple(i -> r[i+1], Val(N))
end

function _log_linear_regression(s::TensorSpace{<:NTuple{N,BaseSpace}}, A) where {N}
    A_ = filter(isfinite, A)
    x = ones(Float64, length(A_), N+1)
    @inbounds for (i, α) ∈ enumerate(indices(s))
        if isfinite(A[i])
            view(x, i, 2:N+1) .= log.(abs.(α) .+ 1)
        end
    end
    x_T = transpose(x)
    r = (x_T * x) \ x_T * A_
    return @inbounds ntuple(i -> r[i+1], Val(N))
end

# Taylor

function _linear_regression(::Taylor, A)
    n = sum_x = 0
    u = t = sum_A = zero(eltype(A))
    for (i, Aᵢ) ∈ enumerate(A)
        if isfinite(Aᵢ)
            sum_x += i
            u += i*Aᵢ
            t += i*i
            sum_A += Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r), r, zero(r))
end

function _log_linear_regression(::Taylor, A)
    n = 0
    sum_x = u = t = sum_A = zero(promote_type(Float64, eltype(A)))
    for (i, Aᵢ) ∈ enumerate(A)
        if isfinite(Aᵢ)
            log_i = log(i)
            sum_x += log_i
            u += log_i*Aᵢ
            t += log_i*log_i
            sum_A += Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r), r, zero(r))
end

# Fourier

function _linear_regression(s::Fourier, A)
    ord = order(s)
    n = sum_x = 0
    u = t = sum_A = zero(eltype(A))
    for (i, Aᵢ) ∈ enumerate(A)
        if isfinite(Aᵢ)
            abs_i = abs(i-ord-1)+1
            sum_x += abs_i
            u += abs_i*Aᵢ
            t += abs_i*abs_i
            sum_A += Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r), r, zero(r))
end

function _log_linear_regression(s::Fourier, A)
    ord = order(s)
    n = 0
    sum_x = u = t = sum_A = zero(promote_type(Float64, eltype(A)))
    for (i, Aᵢ) ∈ enumerate(A)
        if isfinite(Aᵢ)
            log_abs_i = log(abs(i-ord-1)+1)
            sum_x += log_abs_i
            u += log_abs_i*Aᵢ
            t += log_abs_i*log_abs_i
            sum_A += Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r), r, zero(r))
end

# Chebyshev

function _linear_regression(::Chebyshev, A)
    n = sum_x = 0
    u = t = sum_A = zero(eltype(A))
    for (i, Aᵢ) ∈ enumerate(A)
        if isfinite(Aᵢ)
            sum_x += i
            u += i*Aᵢ
            t += i*i
            sum_A += Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r), r, zero(r))
end

function _log_linear_regression(::Chebyshev, A)
    n = 0
    sum_x = u = t = sum_A = zero(promote_type(Float64, eltype(A)))
    for (i, Aᵢ) ∈ enumerate(A)
        if isfinite(Aᵢ)
            log_i = log(i)
            sum_x += log_i
            u += log_i*Aᵢ
            t += log_i*log_i
            sum_A += Aᵢ
            n += 1
        end
    end
    x̄ = sum_x/n
    r = (u - x̄*sum_A)/(t - sum_x*x̄)
    return ifelse(isfinite(r), r, zero(r))
end

# ℓᵖ norm

"""
    ℓᵖNorm{T<:Real} <: Norm

Norm of the ``\\ell^p`` space.

Fields:
- `p :: T`
"""
struct ℓᵖNorm{T<:Real} <: Norm
    p :: T
    function ℓᵖNorm{T}(p::T) where {T<:Real}
        p ≥ 1 || return throw(DomainError(p, "ℓᵖNorm is only defined for real numbers greater than 1 and possibly infinite"))
        return new{T}(p)
    end
end

ℓᵖNorm(p::T) where {T<:Real} = ℓᵖNorm{T}(p)

LinearAlgebra.norm(a::Sequence, p::Real=2) = norm(a, ℓᵖNorm(p))
LinearAlgebra.opnorm(A::LinearOperator, p::Real=2) = opnorm(A, ℓᵖNorm(p))

LinearAlgebra.norm(a::Sequence, d::ℓᵖNorm) = norm(coefficients(a), d.p)

LinearAlgebra.opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, d::ℓᵖNorm) =
    opnorm(coefficients(A), d.p)

function LinearAlgebra.opnorm(A::LinearOperator, d_domain::ℓᵖNorm, d_codomain::ℓᵖNorm)
    d_domain.p == d_codomain.p && return opnorm(coefficients(A), d_domain.p)
    v = map(Aᵢ -> norm(Aᵢ, d_codomain.p), eachcol(coefficients(A)))
    return opnorm(transpose(v), d_domain.p)
end

for T ∈ (:Interval, :(Complex{<:Interval}))
    @eval begin
        LinearAlgebra.norm(a::Sequence{<:VectorSpace,<:AbstractVector{<:$T}}, d::ℓᵖNorm) = norm(Interval.(mag.(coefficients(a))), d.p)

        LinearAlgebra.opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace,<:AbstractMatrix{<:$T}}, d::ℓᵖNorm) =
            opnorm(Interval.(mag.(coefficients(A))), d.p)

        function LinearAlgebra.opnorm(A::LinearOperator{<:VectorSpace,<:VectorSpace,<:AbstractMatrix{<:$T}}, d_domain::ℓᵖNorm, d_codomain::ℓᵖNorm)
            mA = Interval.(mag.(coefficients(A)))
            d_domain.p == d_codomain.p && return opnorm(mA, d.p)
            v = map(mAᵢ -> Interval(mag(norm(mAᵢ, d_codomain.p))), eachcol(mA))
            return opnorm(transpose(v), d_domain.p)
        end
    end
end

# weighted ℓ¹ norm

"""
    Weightedℓ¹Norm{T<:Union{Weights,Tuple{Vararg{Weights}}}} <: Norm

Norm of the weighted ``\\ell^1`` space.

Fields:
- `weights :: T`
"""
struct Weightedℓ¹Norm{T<:Union{Weights,Tuple{Vararg{Weights}}}} <: Norm
    weights :: T
end

LinearAlgebra.norm(a::Sequence{<:BaseSpace}, d::Weightedℓ¹Norm{<:Weights}) =
    _apply(d, space(a), coefficients(a))

function LinearAlgebra.norm(a::Sequence{TensorSpace{T}}, d::Weightedℓ¹Norm{<:NTuple{N,Weights}}) where {N,T<:NTuple{N,BaseSpace}}
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    return _apply(d, space_a, A)
end

_apply(d::Weightedℓ¹Norm, space::TensorSpace, A) =
    @inbounds _apply(Weightedℓ¹Norm(d.weights[1]), space[1], _apply(Weightedℓ¹Norm(Base.tail(d.weights)), Base.tail(space), A))

_apply(d::Weightedℓ¹Norm, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply(Weightedℓ¹Norm(d.weights[1]), space[1], A)

LinearAlgebra.opnorm(A::LinearOperator{<:BaseSpace,ParameterSpace}, d::Weightedℓ¹Norm{<:Weights}) =
    _apply_dual(d, domain(A), vec(coefficients(A)))

function LinearAlgebra.opnorm(A::LinearOperator{TensorSpace{T},ParameterSpace}, d::Weightedℓ¹Norm{<:NTuple{N,Weights}}) where {N,T<:NTuple{N,BaseSpace}}
    domain_A = domain(A)
    A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
    return _apply_dual(d, domain_A, A_)
end

_apply_dual(d::Weightedℓ¹Norm, space::TensorSpace, A) =
    @inbounds _apply_dual(Weightedℓ¹Norm(d.weights[1]), space[1], _apply_dual(Weightedℓ¹Norm(Base.tail(d.weights)), Base.tail(space), A))

_apply_dual(d::Weightedℓ¹Norm, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply_dual(Weightedℓ¹Norm(d.weights[1]), space[1], A)

# Taylor

function _apply(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Taylor, A::AbstractVector)
    ν = rate(d.weights)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _weight(space, d.weights, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) * _weight(space, d.weights, i)
    end
    return s
end

function _apply(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(d.weights)
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
function _apply(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_weight(space, d.weights, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) .* _weight(space, d.weights, i)
    end
    return s
end

function _apply_dual(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Taylor, A::AbstractVector{T}) where {T}
    ν = rate(d.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs(A[1]) * ν⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i+1]) * ν⁻ⁱ)
    end
    return s
end
function _apply_dual(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Taylor, A::AbstractVector{T}) where {T}
    @inbounds s = abs(A[1]) / _weight(space, d.weights, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) / _weight(space, d.weights, i))
    end
    return s
end

function _apply_dual(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(d.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)
    CoefType = typeof(ν⁻¹)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* ν⁻ⁱ)
    end
    return s
end
function _apply_dual(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_weight(space, d.weights, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _weight(space, d.weights, i))
    end
    return s
end

# Fourier

function _apply(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Fourier, A::AbstractVector)
    ν = rate(d.weights)
    ord = order(space)
    if ord == 0
        return abs(A[1]) * one(ν)
    else
        @inbounds s = (abs(A[1]) + abs(A[2ord+1])) * one(ν)
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[ord+1-i]) + abs(A[ord+1+i])
        end
        @inbounds s = s * ν + abs(A[ord+1])
        return s
    end
end
function _apply(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _weight(space, d.weights, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) * _weight(space, d.weights, i)
    end
    return s
end

function _apply(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(d.weights)
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
function _apply(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_weight(space, d.weights, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) .* _weight(space, d.weights, i)
    end
    return s
end

function _apply_dual(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Fourier, A::AbstractVector{T}) where {T}
    ν = rate(d.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * ν⁻ⁱ
    @inbounds for i ∈ 1:ord
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[ord+1+i]) * ν⁻ⁱ, abs(A[ord+1-i]) * ν⁻ⁱ)
    end
    return s
end
function _apply_dual(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Fourier, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _weight(space, d.weights, 0)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _weight(space, d.weights, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end

function _apply_dual(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(d.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)
    CoefType = typeof(ν⁻¹)
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        ν⁻ⁱ *= ν⁻¹
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* ν⁻ⁱ, abs.(selectdim(A, N, ord+1+i)) .* ν⁻ⁱ)
    end
    return s
end
function _apply_dual(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_weight(space, d.weights, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _weight(space, d.weights, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end

# Chebyshev

function _apply(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Chebyshev, A::AbstractVector)
    ν = rate(d.weights)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:1
        s = s * ν + abs(A[i+1])
    end
    @inbounds s = 2s * ν + abs(A[1])
    return s
end
function _apply(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Chebyshev, A::AbstractVector)
    @inbounds s = abs(A[1]) * _weight(space, d.weights, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) * _weight(space, d.weights, i)
    end
    return s
end

function _apply(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(d.weights)
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= s .* ν .+ abs.(selectdim(A, N, i+1))
    end
    @inbounds s .= 2 .* s .* ν .+ abs.(selectdim(A, N, 1))
    return s
end
function _apply(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_weight(space, d.weights, 0))
    @inbounds Aᵢ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) .* _weight(space, d.weights, i)
    end
    return s
end

function _apply_dual(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ν = rate(d.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)/2
    @inbounds s = abs(A[1]) * one(ν⁻ⁱ)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i+1]) * ν⁻ⁱ)
    end
    return s
end
function _apply_dual(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Chebyshev, A::AbstractVector{T}) where {T}
    @inbounds s = abs(A[1]) / _weight(space, d.weights, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) / _weight(space, d.weights, i))
    end
    return s
end

function _apply_dual(d::Weightedℓ¹Norm{<:GeometricWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(d.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)/2
    CoefType = typeof(ν⁻ⁱ)
    @inbounds Aᵢ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* ν⁻ⁱ)
    end
    return s
end
function _apply_dual(d::Weightedℓ¹Norm{<:AlgebraicWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_weight(space, d.weights, 0))
    @inbounds Aᵢ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _weight(space, d.weights, i))
    end
    return s
end

# Hˢ Sobolev norm

"""
    𝐻ˢNorm{T<:Real} <: Norm

Norm of the ``H^s`` Sobolev space.

Fields:
- `exponent :: T`
"""
struct 𝐻ˢNorm{T<:Real} <: Norm
    exponent :: T
    function 𝐻ˢNorm{T}(exponent::T) where {T<:Real}
        isfinite(exponent) & (exponent > 0) || return throw(DomainError(exponent, "𝐻ˢNorm is only defined for real numbers greater than 1"))
        return new{T}(exponent)
    end
end

𝐻ˢNorm(exponent::T) where {T<:Real} = 𝐻ˢNorm{T}(exponent)

LinearAlgebra.norm(a::Sequence{<:BaseSpace}, d::𝐻ˢNorm) =
    _apply(d, space(a), coefficients(a))

function LinearAlgebra.norm(a::Sequence{<:TensorSpace}, d::𝐻ˢNorm)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    return _apply(d, space_a, A)
end

_apply(d::𝐻ˢNorm, space::TensorSpace, A) =
    @inbounds _apply(d, space[1], _apply(d, Base.tail(space), A))

_apply(d::𝐻ˢNorm, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply(d, space[1], A)

LinearAlgebra.opnorm(A::LinearOperator{<:BaseSpace,ParameterSpace}, d::𝐻ˢNorm) =
    _apply_dual(d, domain(A), vec(coefficients(A)))

function LinearAlgebra.opnorm(A::LinearOperator{<:TensorSpace,ParameterSpace}, d::𝐻ˢNorm)
    domain_A = domain(A)
    A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
    return _apply_dual(d, domain_A, A_)
end

_apply_dual(d::𝐻ˢNorm, space::TensorSpace, A) =
    @inbounds _apply_dual(d, space[1], _apply_dual(d, Base.tail(space), A))

_apply_dual(d::𝐻ˢNorm, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply_dual(d, space[1], A)

# Fourier

function _apply(d::𝐻ˢNorm, space::Fourier, A::AbstractVector)
    s = d.exponent
    un = one(eltype(A))
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) * (un + 0)^s
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) * (un + i*i)^s
    end
    return sqrt(x)
end

function _apply(d::𝐻ˢNorm, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    s = d.exponent
    un = one(T)
    CoefType = typeof(sqrt(abs2(zero(T))*(un+0)^s))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    x = Array{CoefType,N-1}(undef, size(Aᵢ))
    x .= abs.(Aᵢ)
    @inbounds for i ∈ 1:ord
        x .+= (abs2.(selectdim(A, N, ord+1+i)) .+ abs2.(selectdim(A, N, ord+1-i))) .* (un + i*i)^s
    end
    x .= sqrt.(x)
    return x
end

function _apply_dual(d::𝐻ˢNorm, space::Fourier, A::AbstractVector)
    s = d.exponent
    un = one(eltype(A))
    ord = order(space)
    @inbounds x = abs2(A[ord+1]) / (un + 0)^s
    @inbounds for i ∈ 1:ord
        x += (abs2(A[ord+1-i]) + abs2(A[ord+1+i])) / (un + i*i)^s
    end
    return sqrt(x)
end

function _apply_dual(d::𝐻ˢNorm, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    s = d.exponent
    un = one(T)
    CoefType = typeof(sqrt(abs2(zero(T))/(un+0)^s))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    x = Array{CoefType,N-1}(undef, size(Aᵢ))
    x .= abs2.(Aᵢ)
    @inbounds for i ∈ 1:ord
        x .+= (abs2.(selectdim(A, N, ord+1+i)) .+ abs2.(selectdim(A, N, ord+1-i))) ./ (un + i*i)^s
    end
    x .= sqrt.(x)
    return x
end

# Cartesian spaces

"""
    CartesianPowerNorm{T<:Norm,S<:Norm} <: Norm

Norm of a cartesian space comprised of spaces with the same norm.

Fields:
- `inner :: T`
- `outer :: S`
"""
struct CartesianPowerNorm{T<:Norm,S<:Norm} <: Norm
    inner :: T
    outer :: S
end

function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, d::CartesianPowerNorm)
    s = CartesianPower(ParameterSpace(), nb_cartesian_product(space(a)))
    v = map(aᵢ -> norm(aᵢ, d.inner), eachcomponent(a))
    return norm(Sequence(s, v), d.outer)
end

function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, d::CartesianPowerNorm)
    s = CartesianPower(ParameterSpace(), nb_cartesian_product(domain(A)))
    v = map(Aᵢ -> opnorm(Aᵢ, d.inner), eachcomponent(A))
    return opnorm(LinearOperator(s, ParameterSpace(), v), d.outer)
end

"""
    CartesianProductNorm{T<:Tuple{Vararg{Norm}}, S<:Norm} <: Norm

Norm of a cartesian space comprised of spaces with different norms.

Fields:
- `inner :: T`
- `outer :: S`
"""
struct CartesianProductNorm{T<:Tuple{Vararg{Norm}}, S<:Norm} <: Norm
    inner :: T
    outer :: S
end

function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, d::CartesianProductNorm{<:NTuple{N,Norm}}) where {N}
    n = nb_cartesian_product(space(a))
    n == N || return throw(DimensionMismatch)
    s = CartesianPower(ParameterSpace(), n)
    v = map((aᵢ, dᵢ) -> norm(aᵢ, dᵢ), eachcomponent(a), d.inner)
    return norm(Sequence(s, v), d.outer)
end

function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, d::CartesianProductNorm{<:NTuple{N,Norm}}) where {N}
    n = nb_cartesian_product(domain(A))
    n == N || return throw(DimensionMismatch)
    s = CartesianPower(ParameterSpace(), n)
    v = map((Aᵢ, dᵢ) -> opnorm(Aᵢ, dᵢ), eachcomponent(A), d.inner)
    return opnorm(LinearOperator(s, ParameterSpace(), transpose(v)), d.outer)
end
