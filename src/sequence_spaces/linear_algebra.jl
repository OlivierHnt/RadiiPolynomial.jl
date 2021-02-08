## factorization

LinearAlgebra.lu(A::Operator) = Operator(A.domain, A.codomain, lu(A.coefficients))
LinearAlgebra.lu!(A::Operator) = Operator(A.domain, A.codomain, lu!(A.coefficients))
function LinearAlgebra.ldiv!(A::Operator, b::Sequence)
    b.space == A.codomain || return throw(DimensionMismatch)
    return Sequence(A.domain, ldiv!(A.coefficients, b.coefficients))
end

## eigen

LinearAlgebra.eigvals(A::Operator) =
    eigvals(A.coefficients)

function LinearAlgebra.eigvecs(A::Operator)
    Ξ = eigvecs(A.coefficients)
    Ξ_ = Vector{Sequence{typeof(A.domain),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(A.domain, Ξ[:,i])
    end
    return Ξ_
end

function LinearAlgebra.eigen(A::Operator)
    Λ, Ξ = eigen(A.coefficients)
    Ξ_ = Vector{Sequence{typeof(A.domain),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(A.domain, Ξ[:,i])
    end
    return Λ, Ξ_
end
