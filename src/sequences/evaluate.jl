(a::Sequence)(x) = evaluate(a, x)

(a::Sequence{<:TensorSpace})(x...) = evaluate(a, x)

function evaluate(a::Sequence{Taylor}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    iszero(x) && return @inbounds convert(CoefType, a[0])
    # Horner's rule
    ord = order(a.space)
    @inbounds s = convert(CoefType, a[ord])
    @inbounds for i ∈ ord-1:-1:0
        s = muladd(s, x, a[i])
    end
    return s
end

function evaluate(a::Sequence{<:Fourier}, x)
    # Horner's rule
    ord = order(a.space)
    eiωx = exp(im*space.frequency*x)
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
    ord = order(a.space)
    x2 = 2x
    s = zero(CoefType)
    @inbounds t = convert(CoefType, a[ord])
    @inbounds for i ∈ ord-1:-1:1
        t, s = x2 * t - s + a[i], t
    end
    return @inbounds x * t - s + a[0]
end

function evaluate(a::Sequence{<:TensorSpace}, x; dims=:)
    A = reshape(a.coefficients, size(a.space))
    return _evaluate(a.space, dims, A, x)
end

_evaluate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, A::Array{T,N}, x) where {N,T} =
    Sequence(space[1:dims-1] ⊗ space[dims+1:N], vec(_evaluate(space[dims], Val(dims), A, x)))
_evaluate(space, ::Colon, A, x) = _apply_evaluate(space, A, x)[1]

_apply_evaluate(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::Array{T,N₂}, x::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _evaluate(space[1], Val(N₂-N₁+1), _apply_evaluate(space[2:N₁], A, x[2:N₁]), x[1])

_apply_evaluate(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::Array{T,N}, x::NTuple{2,Any}) where {T,N} =
    @inbounds _evaluate(space[1], Val(N-1), _evaluate(space[2], Val(N), A, x[2]), x[1])

function _evaluate(space::Taylor, ::Val{D}, A::Array{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    iszero(x) && return @inbounds convert(Array{CoefType,N}, selectdim(A, D, 1:1))
    # Horner's rule
    ord = order(space)
    @inbounds s = convert(Array{CoefType,N}, selectdim(A, D, ord+1:ord+1))
    @inbounds for i ∈ ord:-1:1
        Aᵢ = selectdim(A, D, i:i)
        @. s = muladd(s, x, Aᵢ)
    end
    return s
end

function _evaluate(space::Fourier, ::Val{D}, A::Array{T,N}, x) where {D,T,N}
    # Horner's rule
    ord = order(space)
    eiωx = exp(im*space.frequency*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(T, typeof(eiωx))
    idx₀ = ord+1
    @inbounds s₊ = convert(Array{CoefType,N}, selectdim(A, D, idx₀+ord:idx₀+ord))
    @inbounds s₋ = convert(Array{CoefType,N}, selectdim(A, D, 1:1))
    @inbounds for j ∈ ord-1:-1:1
        idxⱼ = idx₀+j
        idx₋ⱼ = idx₀-j
        Aⱼ = selectdim(A, D, idxⱼ:idxⱼ)
        A₋ⱼ = selectdim(A, D, idx₋ⱼ:idx₋ⱼ)
        @. s₊ = muladd(s₊, eiωx, Aⱼ)
        @. s₋ = muladd(s₋, eiωx_conj, A₋ⱼ)
    end
    @inbounds A₀ = selectdim(A, D, idx₀:idx₀)
    @. s₊ = s₋ * eiωx_conj + A₀ + s₊ * eiωx
    return s₊
end

function _evaluate(space::Chebyshev, ::Val{D}, A::Array{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    # Clenshaw's rule
    ord = order(space)
    x2 = 2x
    s = zeros(CoefType, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
    @inbounds t = convert(Array{CoefType,N}, selectdim(A, D, ord+1:ord+1))
    @inbounds for i ∈ ord:-1:2
        Aᵢ = selectdim(A, D, i:i)
        t, s = x2 .* t .- s .+ Aᵢ, t
    end
    return @inbounds x .* t .- s .+ selectdim(A, D, 1:1)
end
