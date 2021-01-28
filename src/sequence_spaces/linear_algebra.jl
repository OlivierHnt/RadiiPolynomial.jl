## eigen

function LinearAlgebra.eigen(A::Operator{<:CartesianSpace,<:CartesianSpace})
    Λ, Ξ = eigen(A.coefficients)
    @inbounds Ξ_ = map(i -> Sequence(A.domain, Ξ[:,i]), axes(Ξ, 2))
    return Λ, Ξ_
end
