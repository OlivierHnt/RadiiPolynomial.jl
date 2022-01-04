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
_weight(::Chebyshev, weights::GeometricWeights, i::Int) = weights.rate ^ i
_weight(::Chebyshev, weights::GeometricWeights{<:Interval}, i::Int) = pow(weights.rate, i)
_weight(::Chebyshev, weights::AlgebraicWeights, i::Int) = (one(weights.rate) + i) ^ weights.rate
_weight(::Chebyshev, weights::AlgebraicWeights{<:Interval}, i::Int) = pow(one(weights.rate) + i, weights.rate)

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

#

"""
    BanachSpace

Abstract type for all Banach spaces.
"""
abstract type BanachSpace end

LinearAlgebra.opnorm(A::LinearOperator, X::BanachSpace) = opnorm(A, X, X)

function LinearAlgebra.opnorm(A::LinearOperator, X_domain::BanachSpace, X_codomain::BanachSpace)
    codomain_A = codomain(A)
    A_ = coefficients(A)
    @inbounds v₁ = norm(Sequence(codomain_A, view(A_, :, 1)), X_codomain)
    sz = size(A_, 2)
    v = Vector{typeof(v₁)}(undef, sz)
    @inbounds v[1] = v₁
    @inbounds for i ∈ 2:sz
        v[i] = norm(Sequence(codomain_A, view(A_, :, i)), X_codomain)
    end
    return opnorm(LinearOperator(domain(A), ParameterSpace(), transpose(v)), X_domain)
end

#

"""
    ℓ¹ <: BanachSpace

``\\ell^1`` Banach space.
"""
struct ℓ¹ <: BanachSpace end

"""
    ℓ∞ <: BanachSpace

``\\ell^\\infty`` Banach space.
"""
struct ℓ∞ <: BanachSpace end

function LinearAlgebra.norm(a::Sequence, p::Real=Inf)
    if p == 1
        return norm(a, ℓ¹())
    elseif p == Inf
        return norm(a, ℓ∞())
    else
        return throw(ArgumentError)
    end
end

function LinearAlgebra.opnorm(A::LinearOperator, p::Real=Inf)
    if p == 1
        return opnorm(A, ℓ¹())
    elseif p == Inf
        return opnorm(A, ℓ∞())
    else
        return throw(ArgumentError)
    end
end

for T ∈ (:ℓ¹, :ℓ∞)
    @eval begin
        LinearAlgebra.norm(a::Sequence, X::$T) = _apply(X, space(a), coefficients(a))

        function LinearAlgebra.norm(a::Sequence{<:TensorSpace}, X::$T)
            space_a = space(a)
            A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
            return _apply(X, space_a, A)
        end

        _apply(X::$T, space::TensorSpace, A::AbstractArray) =
            @inbounds _apply(X, space[1], _apply(X, Base.tail(space), A))

        _apply(X::$T, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
            @inbounds _apply(X, space[1], A)

        LinearAlgebra.opnorm(A::LinearOperator{<:VectorSpace,ParameterSpace}, X::$T) =
            _apply_dual(X, domain(A), vec(coefficients(A)))

        function LinearAlgebra.opnorm(A::LinearOperator{<:TensorSpace,ParameterSpace}, X::$T)
            domain_A = domain(A)
            A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
            return _apply_dual(X, domain_A, A_)
        end
    end
end

# ParameterSpace

_apply(::ℓ¹, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::ℓ¹, ::ParameterSpace, A::AbstractArray) = @inbounds abs(A[1])

_apply(::ℓ∞, ::ParameterSpace, A::AbstractVector) = @inbounds abs(A[1])
_apply_dual(::ℓ∞, ::ParameterSpace, A::AbstractArray) = @inbounds abs(A[1])

# Taylor, Fourier

for S ∈ (:Taylor, :Fourier)
    @eval begin
        _apply(::ℓ¹, ::$S, A::AbstractVector) = sum(abs, A)
        function _apply(::ℓ¹, ::$S, A::AbstractArray{T,N}) where {T,N}
            CoefType = typeof(abs(zero(T)))
            @inbounds A₁ = selectdim(A, N, 1)
            s = Array{CoefType,N-1}(undef, size(A₁))
            s .= abs.(A₁)
            @inbounds for i ∈ 2:size(A, N)
                s .+= abs.(selectdim(A, N, i))
            end
            return s
        end
        _apply_dual(::ℓ¹, space::$S, A::AbstractArray) = _apply(ℓ∞(), space, A)

        _apply(::ℓ∞, ::$S, A::AbstractVector) = maximum(abs, A)
        function _apply(::ℓ∞, ::$S, A::AbstractArray{T,N}) where {T,N}
            CoefType = typeof(abs(zero(T)))
            @inbounds A₁ = selectdim(A, N, 1)
            s = Array{CoefType,N-1}(undef, size(A₁))
            s .= abs.(A₁)
            for i ∈ 2:size(A, N)
                s .= max.(s, abs.(selectdim(A, N, i)))
            end
            return s
        end
        _apply_dual(::ℓ∞, space::$S, A::AbstractArray) = _apply(ℓ¹(), space, A)
    end
end

# Chebyshev

function _apply(::ℓ¹, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1])
    @inbounds for i ∈ ord-1:-1:1
        s += abs(A[i+1])
    end
    return @inbounds 2s + abs(A[1])
end
function _apply(::ℓ¹, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1))
    end
    @inbounds s .= 2 .* s .+ abs.(selectdim(A, N, 1))
    return s
end
function _apply_dual(::ℓ¹, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1])
    @inbounds for i ∈ ord-1:-1:1
        s = max(s, abs(A[i+1]))
    end
    return @inbounds max(s/2, abs(A[1]))
end
function _apply_dual(::ℓ¹, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)))
    end
    @inbounds s .= max.(s ./ 2, abs.(selectdim(A, N, 1)))
    return s
end

function _apply(::ℓ∞, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1])
    @inbounds for i ∈ ord-1:-1:1
        s = max(s, abs(A[i+1]))
    end
    return @inbounds max(2s, abs(A[1]))
end
function _apply(::ℓ∞, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)))
    end
    @inbounds s .= max.(2 .* s, abs.(selectdim(A, N, 1)))
    return s
end
function _apply_dual(::ℓ∞, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1])
    @inbounds for i ∈ ord-1:-1:1
        s += abs(A[i+1])
    end
    return @inbounds s/2 + abs(A[1])
end
function _apply_dual(::ℓ∞, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1))
    end
    @inbounds s .= s ./ 2 .+ abs.(selectdim(A, N, 1))
    return s
end

# Sequence spaces

# weighted ℓ¹

"""
    Weightedℓ¹{T<:Union{Weights,Tuple{Vararg{Weights}}}} <: BanachSpace

Weighted ``\\ell^1`` Banach space.

Fields:
- `weights :: T`
"""
struct Weightedℓ¹{T<:Union{Weights,Tuple{Vararg{Weights}}}} <: BanachSpace
    weights :: T
end

LinearAlgebra.norm(a::Sequence{<:BaseSpace}, X::Weightedℓ¹{<:Weights}) =
    _apply(X, space(a), coefficients(a))

function LinearAlgebra.norm(a::Sequence{TensorSpace{T}}, X::Weightedℓ¹{<:NTuple{N,Weights}}) where {N,T<:NTuple{N,BaseSpace}}
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    return _apply(X, space_a, A)
end

_apply(X::Weightedℓ¹, space::TensorSpace, A) =
    @inbounds _apply(Weightedℓ¹(X.weights[1]), space[1], _apply(Weightedℓ¹(Base.tail(X.weights)), Base.tail(space), A))

_apply(X::Weightedℓ¹, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply(Weightedℓ¹(X.weights[1]), space[1], A)

LinearAlgebra.opnorm(A::LinearOperator{<:BaseSpace,ParameterSpace}, X::Weightedℓ¹{<:Weights}) =
    _apply_dual(Weightedℓ¹(X.weights), domain(A), vec(coefficients(A)))

function LinearAlgebra.opnorm(A::LinearOperator{TensorSpace{T},ParameterSpace}, X::Weightedℓ¹{<:NTuple{N,Weights}}) where {N,T<:NTuple{N,BaseSpace}}
    domain_A = domain(A)
    A_ = _no_alloc_reshape(coefficients(A), dimensions(domain_A))
    return _apply_dual(Weightedℓ¹(X.weights), domain_A, A_)
end

_apply_dual(X::Weightedℓ¹, space::TensorSpace, A) =
    @inbounds _apply_dual(Weightedℓ¹(X.weights[1]), space[1], _apply_dual(Weightedℓ¹(Base.tail(X.weights)), Base.tail(space), A))

_apply_dual(X::Weightedℓ¹, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply_dual(Weightedℓ¹(X.weights[1]), space[1], A)

# Taylor

function _apply(X::Weightedℓ¹{<:GeometricWeights}, space::Taylor, A::AbstractVector)
    ν = rate(X.weights)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end
function _apply(X::Weightedℓ¹{<:AlgebraicWeights}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) * _weight(space, X.weights, 0)
    @inbounds for i ∈ 1:order(space)
        s += abs(A[i+1]) * _weight(space, X.weights, i)
    end
    return s
end

function _apply(X::Weightedℓ¹{<:GeometricWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weights)
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
function _apply(X::Weightedℓ¹{<:AlgebraicWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_weight(space, X.weights, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .+= abs.(selectdim(A, N, i+1)) .* _weight(space, X.weights, i)
    end
    return s
end

function _apply_dual(X::Weightedℓ¹{<:GeometricWeights}, space::Taylor, A::AbstractVector{T}) where {T}
    ν = rate(X.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)
    @inbounds s = abs(A[1]) * ν⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i+1]) * ν⁻ⁱ)
    end
    return s
end
function _apply_dual(X::Weightedℓ¹{<:AlgebraicWeights}, space::Taylor, A::AbstractVector)
    @inbounds s = abs(A[1]) / _weight(space, X.weights, 0)
    @inbounds for i ∈ 1:order(space)
        s = max(s, abs(A[i+1]) / _weight(space, X.weights, i))
    end
    return s
end

function _apply_dual(X::Weightedℓ¹{<:GeometricWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weights)
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
function _apply_dual(X::Weightedℓ¹{<:AlgebraicWeights}, space::Taylor, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_weight(space, X.weights, 0))
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _weight(space, X.weights, i))
    end
    return s
end

# Fourier

function _apply(X::Weightedℓ¹{<:GeometricWeights}, space::Fourier, A::AbstractVector)
    ν = rate(X.weights)
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
function _apply(X::Weightedℓ¹{<:AlgebraicWeights}, space::Fourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _weight(space, X.weights, 0)
    @inbounds for i ∈ 1:ord
        s += (abs(A[ord+1-i]) + abs(A[ord+1+i])) * _weight(space, X.weights, i)
    end
    return s
end

function _apply(X::Weightedℓ¹{<:GeometricWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weights)
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
function _apply(X::Weightedℓ¹{<:AlgebraicWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_weight(space, X.weights, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    @inbounds s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        s .+= (abs.(selectdim(A, N, ord+1-i)) .+ abs.(selectdim(A, N, ord+1+i))) .* _weight(space, X.weights, i)
    end
    return s
end

function _apply_dual(X::Weightedℓ¹{<:GeometricWeights}, space::Fourier, A::AbstractVector{T}) where {T}
    ν = rate(X.weights)
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
function _apply_dual(X::Weightedℓ¹{<:AlgebraicWeights}, space::Fourier, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _weight(space, X.weights, 0)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _weight(space, X.weights, i)
        s = max(s, abs(A[ord+1+i]) * x, abs(A[ord+1-i]) * x)
    end
    return s
end

function _apply_dual(X::Weightedℓ¹{<:GeometricWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weights)
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
function _apply_dual(X::Weightedℓ¹{<:AlgebraicWeights}, space::Fourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_weight(space, X.weights, 0))
    ord = order(space)
    @inbounds A₀ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:ord
        x = abs(one(T)) / _weight(space, X.weights, i)
        s .= max.(s, abs.(selectdim(A, N, ord+1-i)) .* x, abs.(selectdim(A, N, ord+1+i)) .* x)
    end
    return s
end

# Chebyshev

function _apply(X::Weightedℓ¹{<:GeometricWeights}, space::Chebyshev, A::AbstractVector)
    ν = rate(X.weights)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:1
        s = s * ν + abs(A[i+1])
    end
    return @inbounds 2s * ν + abs(A[1])
end
function _apply(X::Weightedℓ¹{<:AlgebraicWeights}, space::Chebyshev, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _weight(space, X.weights, ord)
    @inbounds for i ∈ ord-1:-1:1
        s += abs(A[i+1]) * _weight(space, X.weights, i)
    end
    return @inbounds 2s + abs(A[1])
end

function _apply(X::Weightedℓ¹{<:GeometricWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weights)
    CoefType = typeof(2abs(zero(T))*ν)
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
function _apply(X::Weightedℓ¹{<:AlgebraicWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T))*_weight(space, X.weights, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _weight(space, X.weights, ord)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i+1)) .* _weight(space, X.weights, i)
    end
    @inbounds s .= 2 .* s .+ abs.(selectdim(A, N, 1))
    return s
end

function _apply_dual(X::Weightedℓ¹{<:GeometricWeights}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ν = rate(X.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)/2
    @inbounds s = abs(A[1]) * one(ν⁻ⁱ)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s = max(s, abs(A[i+1]) * ν⁻ⁱ)
    end
    return s
end
function _apply_dual(X::Weightedℓ¹{<:AlgebraicWeights}, space::Chebyshev, A::AbstractVector{T}) where {T}
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _weight(space, X.weights, ord)
    @inbounds for i ∈ ord-1:-1:1
        s = max(s, abs(A[i+1]) / _weight(space, X.weights, i))
    end
    return @inbounds max(s/2, abs(A[1]))
end

function _apply_dual(X::Weightedℓ¹{<:GeometricWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weights)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ = one(ν⁻¹)/2
    CoefType = typeof(ν⁻ⁱ)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ *= ν⁻¹
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* ν⁻ⁱ)
    end
    return s
end
function _apply_dual(X::Weightedℓ¹{<:AlgebraicWeights}, space::Chebyshev, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof((abs(zero(T))/_weight(space, X.weights, 0))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _weight(space, X.weights, ord)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _weight(space, X.weights, i))
    end
    @inbounds s .= max.(s ./ 2, abs.(selectdim(A, N, 1)))
    return s
end

# Hˢ Sobolev norm

"""
    Hˢ{T<:Real} <: BanachSpace

``H^s`` Sobolev space.

Fields:
- `exponent :: T`
"""
struct Hˢ{T<:Real} <: BanachSpace
    exponent :: T
    function Hˢ{T}(exponent::T) where {T<:Real}
        isfinite(exponent) & (exponent > 0) || return throw(DomainError(exponent, "Hˢ is only defined for real numbers greater than 1"))
        return new{T}(exponent)
    end
end

Hˢ(exponent::T) where {T<:Real} = Hˢ{T}(exponent)

function LinearAlgebra.norm(a::Sequence{<:Fourier}, X::Hˢ)
    s = X.exponent
    un = one(s)
    ord = order(space(a))
    @inbounds x = abs2(a[0]) * (un + 0)^s
    @inbounds for i ∈ 1:ord
        x += (abs2(a[-i]) + abs2(a[i])) * (un + i*i)^s
    end
    return sqrt(x)
end

function LinearAlgebra.norm(a::Sequence{<:TensorSpace{<:Tuple{Vararg{Fourier}}}}, X::Hˢ) where {N}
    s = X.exponent
    un = one(s)
    space_a = space(a)
    ord = order(space_a)
    @inbounds x = zero(abs2(zero(eltype(a)))*(un+0)^s)
    @inbounds for α ∈ indices(space_a)
        x += abs2(a[α]) * (un + mapreduce(abs2, +, α))^s
    end
    return sqrt(x)
end

function LinearAlgebra.opnorm(A::LinearOperator{<:Fourier,ParameterSpace}, X::Hˢ)
    s = X.exponent
    un = one(s)
    ord = order(domain(A))
    @inbounds x = abs2(A[1,0]) / (un + 0)^s
    @inbounds for i ∈ 1:ord
        x += (abs2(A[1,-i]) + abs2(A[1,i])) / (un + i*i)^s
    end
    return sqrt(x)
end

function LinearAlgebra.opnorm(A::LinearOperator{<:TensorSpace{<:Tuple{Vararg{Fourier}}},ParameterSpace}, X::Hˢ)
    s = X.exponent
    un = one(s)
    domain_A = domain(A)
    ord = order(domain_A)
    @inbounds x = zero(abs2(zero(eltype(A)))/(un+0)^s)
    @inbounds for α ∈ indices(domain_A)
        x += abs2(A[1,α]) / (un + mapreduce(abs2, +, α))^s
    end
    return sqrt(x)
end

# Cartesian spaces

"""
    NormedCartesianSpace{T<:BanachSpace,S<:BanachSpace} <: BanachSpace

Cartesian Banach space.

Fields:
- `inner :: T`
- `outer :: S`
"""
struct NormedCartesianSpace{T<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},S<:BanachSpace} <: BanachSpace
    inner :: T
    outer :: S
end

function LinearAlgebra.norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,ℓ¹})
    @inbounds r = norm(component(a, 1), X.inner)
    @inbounds for i ∈ 2:nb_cartesian_product(space(a))
        r += norm(component(a, i), X.inner)
    end
    return r
end

function LinearAlgebra.norm(a::Sequence{<:CartesianPower}, X::NormedCartesianSpace{<:BanachSpace,ℓ∞})
    @inbounds r = norm(component(a, 1), X.inner)
    @inbounds for i ∈ 2:nb_cartesian_product(space(a))
        r = max(r, norm(component(a, i), X.inner))
    end
    return r
end

function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,ℓ¹})
    @inbounds r = opnorm(component(A, 1), X.inner)
    @inbounds for i ∈ 2:nb_cartesian_product(domain(A))
        r = max(r, opnorm(component(A, i), X.inner))
    end
    return r
end

function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianPower,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,ℓ∞})
    @inbounds r = opnorm(component(A, 1), X.inner)
    @inbounds for i ∈ 2:nb_cartesian_product(domain(A))
        r += opnorm(component(A, i), X.inner)
    end
    return r
end

for T ∈ (:ℓ¹, :ℓ∞)
    @eval begin
        function LinearAlgebra.norm(a::Sequence{<:CartesianProduct}, X::NormedCartesianSpace{<:BanachSpace,$T})
            @inbounds r = norm(component(a, 1), X.inner)
            return _apply(X, r, Val(2), a)
        end

        function LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$T}) where {N}
            nb_cartesian_product(space(a)) == N || return throw(DimensionMismatch)
            @inbounds r = norm(component(a, 1), X.inner[1])
            return _apply(X, r, Val(2), a)
        end

        function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianProduct,ParameterSpace}, X::NormedCartesianSpace{<:BanachSpace,$T})
            @inbounds r = opnorm(component(A, 1), X.inner)
            return _apply_dual(X, r, Val(2), A)
        end

        function LinearAlgebra.opnorm(A::LinearOperator{<:CartesianSpace,ParameterSpace}, X::NormedCartesianSpace{<:NTuple{N,BanachSpace},$T}) where {N}
            nb_cartesian_product(domain(A)) == N || return throw(DimensionMismatch)
            @inbounds r = opnorm(component(A, 1), X.inner[1])
            return _apply_dual(X, r, Val(2), A)
        end
    end
end

function _apply(X::NormedCartesianSpace{<:BanachSpace,ℓ¹}, r, ::Val{D}, a) where {D}
    if D ≤ nb_cartesian_product(space(a))
        @inbounds r += norm(component(a, D), X.inner)
        return _apply(X, r, Val(D+1), a)
    else
        return r
    end
end

function _apply(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ¹}, r, ::Val{D}, a) where {N,D}
    @inbounds r += norm(component(a, D), X.inner[D])
    return _apply(X, r, Val(D+1), a)
end
function _apply(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ¹}, r, ::Val{N}, a) where {N}
    @inbounds r += norm(component(a, N), X.inner[N])
    return r
end

function _apply(X::NormedCartesianSpace{<:BanachSpace,ℓ∞}, r, ::Val{D}, a) where {D}
    if D ≤ nb_cartesian_product(space(a))
        @inbounds r = max(r, norm(component(a, D), X.inner))
        return _apply(X, r, Val(D+1), a)
    else
        return r
    end
end

function _apply(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ∞}, r, ::Val{D}, a) where {N,D}
    @inbounds r = max(r, norm(component(a, D), X.inner[D]))
    return _apply(X, r, Val(D+1), a)
end
function _apply(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ∞}, r, ::Val{N}, a) where {N}
    @inbounds r = max(r, norm(component(a, N), X.inner[N]))
    return r
end

function _apply_dual(X::NormedCartesianSpace{<:BanachSpace,ℓ¹}, r, ::Val{D}, A) where {D}
    if D ≤ nb_cartesian_product(domain(A))
        @inbounds r = max(r, opnorm(component(A, D), X.inner))
        return _apply_dual(X, r, Val(D+1), A)
    else
        return r
    end
end

function _apply_dual(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ¹}, r, ::Val{D}, A) where {N,D}
    @inbounds r = max(r, opnorm(component(A, D), X.inner[D]))
    return _apply_dual(X, r, Val(D+1), A)
end
function _apply_dual(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ¹}, r, ::Val{N}, A) where {N}
    @inbounds r = max(r, opnorm(component(A, N), X.inner[N]))
    return r
end

function _apply_dual(X::NormedCartesianSpace{<:BanachSpace,ℓ∞}, r, ::Val{D}, A) where {D}
    if D ≤ nb_cartesian_product(domain(A))
        @inbounds r += opnorm(component(A, D), X.inner)
        return _apply_dual(X, r, Val(D+1), A)
    else
        return r
    end
end

function _apply_dual(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ∞}, r, ::Val{D}, A) where {N,D}
    @inbounds r += opnorm(component(A, D), X.inner[D])
    return _apply_dual(X, r, Val(D+1), A)
end
function _apply_dual(X::NormedCartesianSpace{<:NTuple{N,BanachSpace},ℓ∞}, r, ::Val{N}, A) where {N}
    @inbounds r += opnorm(component(A, N), X.inner[N])
    return r
end
