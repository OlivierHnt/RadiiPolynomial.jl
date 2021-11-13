# condition number

LinearAlgebra.cond(A::LinearOperator, p::Real=2) = cond(coefficients(A), p)

# transpose, adjoint

LinearAlgebra.transpose(A::LinearOperator) = transpose(coefficients(A))

LinearAlgebra.adjoint(A::LinearOperator) = adjoint(coefficients(A))

# eigen

LinearAlgebra.eigvals(A::LinearOperator; kwargs...) = eigvals(coefficients(A); kwargs...)

function LinearAlgebra.eigvecs(A::LinearOperator; kwargs...)
    domain(A) == codomain(A) || throw(DimensionMismatch)
    Ξ = eigvecs(coefficients(A); kwargs...)
    return LinearOperator(ParameterSpace()^size(Ξ, 2), domain(A), Ξ)
end

function LinearAlgebra.eigen(A::LinearOperator; kwargs...)
    Λ, Ξ = eigen(coefficients(A); kwargs...)
    return Λ, LinearOperator(ParameterSpace()^size(Ξ, 2), domain(A), Ξ)
end

# kernel

function LinearAlgebra.nullspace(A::LinearOperator; kwargs...)
    Ξ = nullspace(coefficients(A); kwargs...)
    return LinearOperator(ParameterSpace()^size(Ξ, 2), domain(A), Ξ)
end
