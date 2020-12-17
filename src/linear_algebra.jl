+̄(a::AbstractArray{<:Sequence}, b::AbstractArray{<:Sequence}) = a .+̄ b
-̄(a::AbstractArray{<:Sequence}, b::AbstractArray{<:Sequence}) = a .-̄ b

#

+̄(A::AbstractArray{<:Functional}, B::AbstractArray{<:Functional}) = A .+̄ B
-̄(A::AbstractArray{<:Functional}, B::AbstractArray{<:Functional}) = A .-̄ B

#

+̄(A::AbstractArray{<:Operator}, B::AbstractArray{<:Operator}) = A .+̄ B
-̄(A::AbstractArray{<:Operator}, B::AbstractArray{<:Operator}) = A .-̄ B

#

function +̄(A::AbstractMatrix{T}, J::UniformScaling) where {T<:Operator}
    aux = A[1,1] +̄ J.λ
    C = Matrix{promote_type(typeof(aux), T)}(undef, size(A))
    @inbounds for j ∈ axes(A, 2), i ∈ axes(A, 1)
        if i == j
            C[i,j] = A[i,j] +̄ J.λ
        else
            C[i,j] = copy(A[i,j])
        end
    end
    return C
end

function +̄(J::UniformScaling, A::AbstractMatrix{T}) where {T<:Operator}
    aux = A[1,1] +̄ J.λ
    C = Matrix{promote_type(typeof(aux), T)}(undef, size(A))
    @inbounds for j ∈ axes(A, 2), i ∈ axes(A, 1)
        if i == j
            C[i,j] = J.λ +̄ A[i,j]
        else
            C[i,j] = copy(A[i,j])
        end
    end
    return C
end

function -̄(A::AbstractMatrix{T}, J::UniformScaling) where {T<:Operator}
    aux = A[1,1] +̄ J.λ
    C = Matrix{promote_type(typeof(aux), T)}(undef, size(A))
    @inbounds for j ∈ axes(A, 2), i ∈ axes(A, 1)
        if i == j
            C[i,j] = A[i,j] -̄ J.λ
        else
            C[i,j] = copy(A[i,j])
        end
    end
    return C
end

function -̄(J::UniformScaling, A::AbstractMatrix{T}) where {T<:Operator}
    aux = A[1,1] +̄ J.λ
    C = Matrix{promote_type(typeof(aux), T)}(undef, size(A))
    @inbounds for j ∈ axes(A, 2), i ∈ axes(A, 1)
        if i == j
            C[i,j] = J.λ -̄ A[i,j]
        else
            C[i,j] = -A[i,j]
        end
    end
    return C
end

## inverse

function Base.inv(A::Matrix{T}) where {T<:Operator}
    @assert size(A, 1) == size(A, 2)
    domain_ = map(Aᵢ -> mapreduce(domain, ∪, Aᵢ), eachcol(A))
    codomain_ = map(Aᵢ -> mapreduce(codomain, ∪, Aᵢ), eachrow(A))
    A_ = mapreduce(j -> mapreduce(i -> coefficients(project(A[i,j], domain_[j], codomain_[i])), vcat, axes(A, 1)), hcat, axes(A, 2))
    A_⁻¹ = inv(A_)
    return [Operator(codomain_[j], domain_[i], A_⁻¹[1+(i-1)*length(domain_[i]):i*length(domain_[i]), 1+(j-1)*length(codomain_[j]):j*length(codomain_[j])]) for i ∈ axes(A, 1), j ∈ axes(A, 2)]
end
