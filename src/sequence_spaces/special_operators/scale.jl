struct Scale{T}
    value :: T
end

## arithmetic

function +̄(𝒮::Scale{T}, A::Operator{Taylor,Taylor}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    γⁱ = one(CoefType)
    @inbounds C[0,0] += γⁱ
    @inbounds for i ∈ 1:min(order(domain), order(codomain))
        γⁱ *= 𝒮.value
        C[i,i] += γⁱ
    end
    return C
end

+̄(A::Operator{Taylor,Taylor}, 𝒮::Scale{T}) where {T} = +̄(𝒮, A)

function -̄(𝒮::Scale{T}, A::Operator{Taylor,Taylor}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = -A.coefficients
    γⁱ = one(CoefType)
    @inbounds C[0,0] += γⁱ
    @inbounds for i ∈ 1:min(order(domain), order(codomain))
        γⁱ *= 𝒮.value
        C[i,i] += γⁱ
    end
    return C
end

function -̄(A::Operator{Taylor,Taylor}, 𝒮::Scale{T}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    γⁱ = one(CoefType)
    @inbounds C[0,0] -= γⁱ
    @inbounds for i ∈ 1:min(order(domain), order(codomain))
        γⁱ *= 𝒮.value
        C[i,i] -= γⁱ
    end
    return C
end

function +̄(𝒮::Scale{T}, A::Operator{TensorSpace{NTuple{N,Taylor}},TensorSpace{NTuple{N,Taylor}}}) where {T,N}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ allindices(domain(A) ∩ codomain(A))
        C[α,α] += mapreduce(^, *, 𝒮.value, α)
    end
    return C
end

+̄(A::Operator{TensorSpace{NTuple{N,Taylor}},TensorSpace{NTuple{N,Taylor}}}, 𝒮::Scale{T}) where {T,N} = +̄(𝒮, A)

function -̄(𝒮::Scale{T}, A::Operator{TensorSpace{NTuple{N,Taylor}},TensorSpace{NTuple{N,Taylor}}}) where {T,N}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = -A.coefficients
    @inbounds for α ∈ allindices(domain(A) ∩ codomain(A))
        C[α,α] += mapreduce(^, *, 𝒮.value, α)
    end
    return C
end

function -̄(A::Operator{TensorSpace{NTuple{N,Taylor}},TensorSpace{NTuple{N,Taylor}}}, 𝒮::Scale{T}) where {T,N}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ allindices(domain(A) ∩ codomain(A))
        C[α,α] -= mapreduce(^, *, 𝒮.value, α)
    end
    return C
end

#

function project(𝒮::Scale{T}, domain::Taylor, codomain::Taylor) where {T}
    A = Operator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    γⁱ = one(T)
    @inbounds A[0,0] = γⁱ
    @inbounds for i ∈ 1:min(order(domain), order(codomain))
        γⁱ *= 𝒮.value
        A[i,i] = γⁱ
    end
    return A
end

function project(𝒮::Scale{NTuple{N,T}}, domain::TensorSpace{NTuple{N,Taylor}}, codomain::TensorSpace{NTuple{N,Taylor}}) where {N,T}
    A = Operator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    @inbounds for α ∈ allindices(domain ∩ codomain)
        A[α,α] = mapreduce(^, *, 𝒮.value, α)
    end
    return A
end

## action

(𝒮::Scale)(a::Sequence) = *(𝒮, a)
# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(𝒮::Scale, a::Sequence) = scale(a, 𝒮.value)

function scale(a::Sequence{Taylor}, γ)
    CoefType = promote_type(eltype(a), typeof(γ))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    @. c.coefficients = a.coefficients
    return scale!(c, γ)
end

function scale(a::Sequence{TensorSpace{NTuple{N,Taylor}}}, γ::NTuple{N,Any}) where {N}
    CoefType = promote_type(eltype(a), eltype(γ))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    @. c.coefficients = a.coefficients
    return scale!(c, γ)
end

function scale!(a::Sequence{Taylor}, γ)
    isone(γ) && return a
    γⁱ = one(γ)
    @inbounds for i ∈ 1:order(a)
        γⁱ *= γ
        a[i] *= γⁱ
    end
    return a
end

function scale!(a::Sequence{TensorSpace{NTuple{N,Taylor}}}, γ::NTuple{N,Any}) where {N}
    all(isone, γ) && return a
    @inbounds for α ∈ allindices(space(a))
        a[α] *= mapreduce(^, *, γ, α)
    end
    return a
end
