struct Rescale{T}
    γ :: T
end

(ℛ::Rescale)(a) = *(ℛ, a)

# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(ℛ::Rescale, a::Sequence) = rescale(a, ℛ.γ)
# signature needed to resolve ambiguity due to *(b, A::Operator)
Base.:*(ℛ::Rescale, A::Operator) = rescale(A, ℛ.γ)

# sequence space

function Operator(domain::Taylor, codomain::Taylor, ℛ::Rescale{T}) where {T}
    A = Operator(domain, codomain, Matrix{T}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ 0:min(domain.order, codomain.order)
        A[i,i] = ℛ.γ ^ i
    end
    return A
end

function Operator(domain::TensorSpace{NTuple{N,Taylor}}, codomain::TensorSpace{NTuple{N,Taylor}}, ℛ::Rescale{NTuple{N,T}}) where {N,T}
    A = Operator(domain, codomain, Matrix{T}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(T)
    @inbounds for α ∈ allindices(domain ∩ codomain)
        A[α,α] = mapreduce(^, *, ℛ.γ, α)
    end
    return A
end

##

# sequence space

function rescale(a::Sequence{Taylor}, γ)
    CoefType = promote_type(eltype(a), typeof(γ))
    c = Sequence(a.space, Vector{CoefType}(undef, dimension(a.space)))
    @. c.coefficients = a.coefficients
    return rescale!(c, γ)
end

function rescale(a::Sequence{TensorSpace{T}}, γ, dims=:) where {N,T<:NTuple{N,Taylor}}
    CoefType = promote_type(eltype(a), eltype(γ))
    c = Sequence(a.space, Vector{CoefType}(undef, dimension(a.space)))
    @. c.coefficients = a.coefficients
    return rescale!(c, γ, dims)
end

function rescale!(a::Sequence{Taylor}, γ)
    isone(γ) && return a
    @inbounds for i ∈ 1:order(a)
        a[i] *= γ^i
    end
    return a
end

function rescale!(a::Sequence{TensorSpace{T}}, γ, dims=:) where {N,T<:NTuple{N,Taylor}}
    A = reshape(a.coefficients, dimensions(a.space))
    _rescale!(a.space, dims, A, γ)
    return a
end

_rescale!(space, dims::Int, A, γ) = @inbounds _rescale!(space[dims], Val(dims), A, γ)
_rescale!(space, dims::Tuple{Int}, A, γ) = @inbounds _rescale!(space[dims[1]], Val(dims[1]), A, γ)
_rescale!(space, ::Colon, A, γ) = _apply_rescale!(space, A, γ)
_rescale!(space, dims::NTuple{N,Int}, A, γ) where {N} = @inbounds _apply_rescale!(space[[dims...]], dims, A, γ)
_rescale!(space, dims::Vector{Int}, A, γ) = @inbounds _apply_rescale!(space[dims], (dims...,), A, γ)

_apply_rescale!(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, γ::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _rescale!(space[1], Val(N₂-N₁+1), _apply_rescale!(Base.tail(space), A, Base.tail(γ)), γ[1])

_apply_rescale!(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, γ::NTuple{2,Any}) where {T,N} =
    @inbounds _rescale!(space[1], Val(N-1), _rescale!(space[2], Val(N), A, γ[2]), γ[1])

_apply_rescale!(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, dims::NTuple{N₁,Int}, A::AbstractArray{T,N₂}, γ::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _rescale!(space[1], Val(dims[1]), _apply_rescale!(Base.tail(space), Base.tail(dims), A, Base.tail(γ)), γ[1])

_apply_rescale!(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, dims::NTuple{2,Int}, A::AbstractArray{T,N}, γ::NTuple{2,Any}) where {T,N} =
    @inbounds _rescale!(space[1], Val(dims[1]), _rescale!(space[2], Val(dims[2]), A, γ[2]), γ[1])

function _rescale!(space::Taylor, ::Val{D}, A, γ) where {D}
    isone(γ) && return A
    @inbounds for i ∈ 1:order(space)
        selectdim(A, D, i+1) .*= γ^i
    end
    return A
end
