# condition number

LinearAlgebra.cond(A::LinearOperator, p::Real=2) = cond(coefficients(A), p)

# transpose, adjoint

LinearAlgebra.transpose(A::LinearOperator) = transpose(coefficients(A))

LinearAlgebra.adjoint(A::LinearOperator) = adjoint(coefficients(A))

# factorize

LinearAlgebra.factorize(A::LinearOperator) = factorize(coefficients(A))

# lu

LinearAlgebra.lu(A::LinearOperator; kwargs...) =
    LinearOperator(domain(A), codomain(A), lu(coefficients(A); kwargs...))

LinearAlgebra.lu!(A::LinearOperator; kwargs...) =
    LinearOperator(domain(A), codomain(A), lu!(coefficients(A); kwargs...))

# eigen

LinearAlgebra.eigvals(A::LinearOperator; kwargs...) = eigvals(coefficients(A); kwargs...)

function LinearAlgebra.eigvecs(A::LinearOperator; kwargs...)
    Ξ = eigvecs(coefficients(A); kwargs...)
    Ξ_ = Vector{Sequence{typeof(domain(A)),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(domain(A), Ξ[:,i])
    end
    return Ξ_
end

function LinearAlgebra.eigen(A::LinearOperator; kwargs...)
    Λ, Ξ = eigen(coefficients(A); kwargs...)
    Ξ_ = Vector{Sequence{typeof(domain(A)),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(domain(A), Ξ[:,i])
    end
    return Λ, Ξ_
end

# kernel

function LinearAlgebra.nullspace(A::LinearOperator; kwargs...)
    Ξ = nullspace(coefficients(A); kwargs...)
    Ξ_ = Vector{Sequence{typeof(domain(A)),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(domain(A), Ξ[:,i])
    end
    return Ξ_
end
