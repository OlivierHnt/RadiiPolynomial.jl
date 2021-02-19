## fallback methods

(A::Operator)(b::Sequence) = *(A, b)

function Base.:*(A::Operator, b::Sequence)
    domain_A = domain(A)
    codomain_A = codomain(A)
    space_b = space(b)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(codomain_A, Vector{CoefType}(undef, dimension(codomain_A)))
    if domain_A == space_b
        mul!(coefficients(c), coefficients(A), coefficients(b))
    else
        coefficients(c) .= zero(CoefType)
        @inbounds for β ∈ allindices(domain_A ∩ space_b), α ∈ allindices(codomain_A)
            c[α] += A[α,β]*b[β]
        end
    end
    return c
end

function Base.:\(A::Operator, b::Sequence)
    codomain_A = codomain(A)
    space_b = space(b)
    space_b == codomain_A && return Sequence(domain(A), \(coefficients(A), coefficients(b)))
    space_b ⊆ codomain_A && return \(A, project(b, codomain_A))
    codomain_A ⊆ space_b && return \(project(A, domain(A), space_b), b)
    union_space = ∪(space_b, codomain_A)
    return \(project(A, domain(A), union_space), project(b, union_space))
end

## cartesian space

function Base.:*(A::Operator{<:CartesianSpace,<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    space_b = space(b)
    nb_cartesian_product(domain_A) == nb_cartesian_product(space_b) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(codomain_A, zeros(CoefType, dimension(codomain_A)))
    @inbounds for j ∈ 1:nb_cartesian_product(domain_A)
        bⱼ = component(b, j)
        @inbounds for i ∈ 1:nb_cartesian_product(codomain_A)
            coefficients(component(c, i)) .+= coefficients(*(component(A, i, j), bⱼ))
        end
    end
    return c
end

function Base.:*(A::Operator{<:CartesianSpace,<:VectorSpace}, b::Sequence{<:CartesianSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    space_b = space(b)
    nb_cartesian_product(domain_A) == nb_cartesian_product(space_b) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(codomain_A, Vector{CoefType}(undef, dimension(codomain_A)))
    coefficients(c) .= coefficients(*(component(A, 1), component(b, 1)))
    @inbounds for j ∈ 2:nb_cartesian_product(domain_A)
        coefficients(c) .+= coefficients(*(component(A, j), component(b, j)))
    end
    return c
end

function Base.:*(A::Operator{<:VectorSpace,<:CartesianSpace}, b::Sequence{<:VectorSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    space_b = space(b)
    nb_cartesian_product(codomain_A) == nb_cartesian_product(space_b) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(codomain_A, Vector{CoefType}(undef, dimension(codomain_A)))
    @inbounds for i ∈ 1:nb_cartesian_product(codomain_A)
        coefficients(component(c, i)) .= coefficients(*(component(A, i), b))
    end
    return c
end
