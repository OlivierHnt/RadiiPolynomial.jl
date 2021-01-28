struct Evaluation{T}
    value :: T
end

(ℰ::Evaluation)(a) = *(ℰ, a)

# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(ℰ::Evaluation, a::Sequence) = evaluate(a, ℰ.value)
# signature needed to resolve ambiguity due to *(b, A::Operator)
Base.:*(ℰ::Evaluation, A::Operator) = evaluate(A, ℰ.value)

+̄(ℰ::Evaluation, A::Operator) = +(Operator(domain(A), codomain(A), ℰ), A)
+̄(A::Operator, ℰ::Evaluation) = +(A, Operator(domain(A), codomain(A), ℰ))
-̄(ℰ::Evaluation, A::Operator) = -(Operator(domain(A), codomain(A), ℰ), A)
-̄(A::Operator, ℰ::Evaluation) = -(A, Operator(domain(A), codomain(A), ℰ))
*̄(ℰ::Evaluation, A::Operator) = *(Operator(codomain(A), ParameterSpace(1), ℰ), A)

# sequence space

function Operator(domain::Taylor, ℰ::Evaluation{T}) where {T}
    A = Operator(domain, ParameterSpace(1), Matrix{T}(undef, 1, dimension(domain)))
    @inbounds A[1,0] = one(T)
    @inbounds for i ∈ 1:order(domain)
        A[1,i] = ℰ.value ^ i
    end
    return A
end

function Operator(domain::Fourier, ℰ::Evaluation)
    iωx = im*domain.frequency*ℰ.value
    CoefType = float(typeof(iωx))
    A = Operator(domain, ParameterSpace(1), Matrix{CoefType}(undef, 1, dimension(domain)))
    @inbounds A[1,0] = one(CoefType)
    @inbounds for j ∈ 1:order(domain)
        eiωxj = exp(iωx*j)
        A[1,j] = eiωxj
        A[1,-j] = conj(eiωxj)
    end
    return A
end

function Operator(domain::Chebyshev, ℰ::Evaluation{T}) where {T}
    A = Operator(domain, ParameterSpace(1), Matrix{T}(undef, 1, dimension(domain)))
    ord = order(domain)
    @inbounds A[1,0] = one(T)
    ord == 0 && return A
    x2 = 2ℰ.value
    @inbounds A[1,1] = x2
    ord == 1 && return A
    @inbounds A[1,2] = x2*A[1,1] - 2A[1,0]
    @inbounds for j ∈ 3:ord
        A[1,j] = x2*A[1,j-1] - A[1,j-2]
    end
    return A
end


##

# sequence space

(a::Sequence{<:UnivariateSpace})(x) = evaluate(a, x)
(a::Sequence{<:TensorSpace})(x; dims=:) = evaluate(a, x, dims)
(a::Sequence{<:TensorSpace})(x...; dims=:) = evaluate(a, x, dims)

function evaluate(a::Sequence{Taylor}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    iszero(x) && return @inbounds convert(CoefType, a[0])
    ord = order(a)
    @inbounds s = convert(CoefType, a[ord])
    @inbounds for i ∈ ord-1:-1:0
        s = s * x + a[i]
    end
    return s
end

function evaluate(a::Sequence{<:Fourier}, x)
    ord = order(a)
    eiωx = cis(frequency(a)*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(eltype(a), typeof(eiωx))
    @inbounds s₊ = convert(CoefType, a[ord])
    @inbounds s₋ = convert(CoefType, a[-ord])
    @inbounds for j ∈ ord-1:-1:1
        s₊ = s₊ * eiωx + a[j]
        s₋ = s₋ * eiωx_conj + a[-j]
    end
    return @inbounds s₋ * eiωx_conj + a[0] + s₊ * eiωx
end

function evaluate(a::Sequence{Chebyshev}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    ord = order(a)
    x2 = 2x
    result = s = zero(CoefType)
    @inbounds t = convert(CoefType, 2a[ord])
    @inbounds for i ∈ ord-1:-1:1
        result = t
        t = x2 * t - s + 2a[i]
        s = result
    end
    @inbounds result = x * t - s + a[0]
    return result
end

function evaluate(a::Sequence{<:TensorSpace}, x, dims=:)
    A = reshape(a.coefficients, dimensions(a.space))
    return _evaluate(a.space, dims, A, x)
end

_evaluate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, A::AbstractArray{T,N}, x) where {N,T} =
    Sequence(TensorSpace(tuple(space.spaces[1:dims-1]..., space.spaces[dims+1:N]...)), vec(_evaluate(space[dims], Val(dims), A, x)))
function _evaluate(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, dims::Int, A::AbstractArray{T,2}, x) where {T}
    space_ = dims == 1 ? space[2] : space[dims-1]
    return Sequence(space_, vec(_evaluate(space[dims], Val(dims), A, x)))
end
_evaluate(space, ::Colon, A, x) = _apply_evaluate(space, A, x)[1]

_apply_evaluate(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, x::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _evaluate(space[1], Val(N₂-N₁+1), _apply_evaluate(Base.tail(space), A, Base.tail(x)), x[1])

_apply_evaluate(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, x::NTuple{2,Any}) where {T,N} =
    @inbounds _evaluate(space[1], Val(N-1), _evaluate(space[2], Val(N), A, x[2]), x[1])

function _evaluate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    iszero(x) && return @inbounds convert(Array{CoefType,N}, selectdim(A, D, 1:1))
    ord = order(space)
    @inbounds s = convert(Array{CoefType,N}, selectdim(A, D, ord+1:ord+1))
    @inbounds for i ∈ ord-1:-1:0
        Aᵢ = selectdim(A, D, i+1:i+1)
        @. s = s * x + Aᵢ
    end
    return s
end

function _evaluate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, x) where {D,T,N}
    ord = order(space)
    eiωx = cis(frequency(space)*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(T, typeof(eiωx))
    @inbounds s₊ = convert(Array{CoefType,N}, selectdim(A, D, 2ord+1:2ord+1))
    @inbounds s₋ = convert(Array{CoefType,N}, selectdim(A, D, 1:1))
    @inbounds for j ∈ ord-1:-1:1
        Aⱼ = selectdim(A, D, ord+1+j:ord+1+j)
        A₋ⱼ = selectdim(A, D, ord+1-j:ord+1-j)
        @. s₊ = s₊ * eiωx + Aⱼ
        @. s₋ = s₋ * eiωx_conj + A₋ⱼ
    end
    @inbounds A₀ = selectdim(A, D, ord+1:ord+1)
    @. s₊ = s₋ * eiωx_conj + A₀ + s₊ * eiωx
    return s₊
end

function _evaluate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    ord = order(space)
    x2 = 2x
    result = Array{CoefType,N}(undef, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
    s = zeros(CoefType, axes(result))
    t = similar(result)
    @inbounds t .= 2 .* selectdim(A, D, ord+1:ord+1)
    @inbounds for i ∈ ord-1:-1:1
        Aᵢ = selectdim(A, D, i+1:i+1)
        result .= t
        @. t = x2 * t - s + 2Aᵢ
        s .= result
    end
    @inbounds A₀ = selectdim(A, D, 1:1)
    @. result = x * t - s + A₀
    return result
end

# cartesian space

(a::Sequence{<:CartesianSpace})(x) = evaluate(a, x)
(a::Sequence{<:CartesianSpace})(x...) = evaluate(a, x)

# TODO: fix type instability
evaluate(a::Sequence{<:CartesianSpace}, x) = map(aᵢ -> evaluate(aᵢ, x), eachcomponent(a))
