## lu

LinearAlgebra.lu(A::Operator; check::Bool=true) = Operator(domain(A), codomain(A), lu(coefficients(A); check=check))
LinearAlgebra.lu!(A::Operator; check::Bool=true) = Operator(domain(A), codomain(A), lu!(coefficients(A); check=check))

## in-place div

function LinearAlgebra.ldiv!(A::Operator, b::Sequence)
    codomain(A) == space(b) || return throw(DimensionMismatch)
    return Sequence(domain(A), ldiv!(coefficients(A), coefficients(b)))
end

function LinearAlgebra.ldiv!(A::Operator, B::Operator)
    codomain(A) == codomain(B) || return throw(DimensionMismatch)
    return Operator(domain(B), domain(A), ldiv!(coefficients(A), coefficients(B)))
end

function LinearAlgebra.rdiv!(A::Operator, B::Operator)
    codomain(A) == codomain(B) || return throw(DimensionMismatch)
    return Operator(codomain(B), codomain(A), rdiv!(coefficients(A), coefficients(B)))
end

## eigen

LinearAlgebra.eigvals(A::Operator; kwargs...) = eigvals(coefficients(A); kwargs...)

function LinearAlgebra.eigvecs(A::Operator; kwargs...)
    Ξ = eigvecs(coefficients(A))
    Ξ_ = Vector{Sequence{typeof(domain(A)),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(domain(A), Ξ[:,i])
    end
    return Ξ_
end

function LinearAlgebra.eigen(A::Operator; kwargs...)
    Λ, Ξ = eigen(coefficients(A))
    Ξ_ = Vector{Sequence{typeof(domain(A)),Vector{eltype(Ξ)}}}(undef, size(Ξ, 2))
    @inbounds for i ∈ axes(Ξ, 2)
        Ξ_[i] = Sequence(domain(A), Ξ[:,i])
    end
    return Λ, Ξ_
end
