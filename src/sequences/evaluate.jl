(a::Sequence{<:UnivariateSpace})(x) = evaluate(a, x)
(a::Sequence{<:TensorSpace})(x; dims=:) = evaluate(a, x, dims)
(a::Sequence{<:TensorSpace})(x...; dims=:) = evaluate(a, x, dims)

function evaluate(a::Sequence{Taylor}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    iszero(x) && return @inbounds convert(CoefType, a[0])
    # Horner's rule
    ord = order(a)
    @inbounds s = convert(CoefType, a[ord])
    @inbounds for i ∈ ord-1:-1:0
        s = muladd(s, x, a[i])
    end
    return s
end

function evaluate(a::Sequence{<:Fourier}, x)
    # Horner's rule
    ord = order(a)
    eiωx = exp(im*frequency(a)*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(eltype(a), typeof(eiωx))
    @inbounds s₊ = convert(CoefType, a[ord])
    @inbounds s₋ = convert(CoefType, a[-ord])
    @inbounds for j ∈ ord-1:-1:1
        s₊ = muladd(s₊, eiωx, a[j])
        s₋ = muladd(s₋, eiωx_conj, a[-j])
    end
    return @inbounds s₋ * eiωx_conj + a[0] + s₊ * eiωx
end

function evaluate(a::Sequence{Chebyshev}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    # Clenshaw's rule
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
    A = reshape(a.coefficients, size(a.space))
    return _evaluate(a.space, dims, A, x)
end

_evaluate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, A::AbstractArray{T,N}, x) where {N,T} =
    Sequence(space[1:dims-1] ⊗ space[dims+1:N], vec(_evaluate(space[dims], Val(dims), A, x)))
_evaluate(space, ::Colon, A, x) = _apply_evaluate(space, A, x)[1]

_apply_evaluate(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, x::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _evaluate(space[1], Val(N₂-N₁+1), _apply_evaluate(Base.tail(space), A, Base.tail(x)), x[1])

_apply_evaluate(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, x::NTuple{2,Any}) where {T,N} =
    @inbounds _evaluate(space[1], Val(N-1), _evaluate(space[2], Val(N), A, x[2]), x[1])

function _evaluate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    iszero(x) && return @inbounds convert(Array{CoefType,N}, _selectdim(space, A, D, 0:0))
    # Horner's rule
    ord = order(space)
    @inbounds s = convert(Array{CoefType,N}, _selectdim(space, A, D, ord:ord))
    @inbounds for i ∈ ord-1:-1:0
        Aᵢ = _selectdim(space, A, D, i:i)
        @. s = muladd(s, x, Aᵢ)
    end
    return s
end

function _evaluate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, x) where {D,T,N}
    # Horner's rule
    ord = order(space)
    eiωx = exp(im*frequency(space)*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(T, typeof(eiωx))
    @inbounds s₊ = convert(Array{CoefType,N}, _selectdim(space, A, D, ord:ord))
    @inbounds s₋ = convert(Array{CoefType,N}, _selectdim(space, A, D, -ord:-ord))
    @inbounds for j ∈ ord-1:-1:1
        Aⱼ = _selectdim(space, A, D, j:j)
        A₋ⱼ = _selectdim(space, A, D, -j:-j)
        @. s₊ = muladd(s₊, eiωx, Aⱼ)
        @. s₋ = muladd(s₋, eiωx_conj, A₋ⱼ)
    end
    @inbounds A₀ = _selectdim(space, A, D, 0:0)
    @. s₊ = s₋ * eiωx_conj + A₀ + s₊ * eiωx
    return s₊
end

function _evaluate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    # Clenshaw's rule
    ord = order(space)
    x2 = 2x
    result = Array{CoefType,N}(undef, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
    s = zeros(CoefType, axes(result))
    t = similar(result)
    @inbounds t .= 2 .* _selectdim(space, A, D, ord:ord)
    @inbounds for i ∈ ord-1:-1:1
        Aᵢ = _selectdim(space, A, D, i:i)
        result .= t
        @. t = x2 * t - s + 2Aᵢ
        s .= result
    end
    @inbounds A₀ = _selectdim(space, A, D, 0:0)
    @. result = x * t - s + A₀
    return result
end
