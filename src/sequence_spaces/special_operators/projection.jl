## fallback methods

function project(a::Sequence, space_dest::VectorSpace)
    space_a = space(a)
    CoefType = eltype(a)
    c = Sequence(space_dest, Vector{CoefType}(undef, dimension(space_dest)))
    if space_a == space_dest
        coefficients(c) .= coefficients(a)
    elseif space_dest ⊆ space_a
        indices = allindices(space_dest)
        @inbounds view(c, indices) .= view(a, indices)
    else
        coefficients(c) .= zero(CoefType)
        @inbounds for α ∈ allindices(space_a ∩ space_dest)
            c[α] = a[α]
        end
    end
    return c
end

function project(A::Operator, domain_dest::VectorSpace, codomain_dest::VectorSpace)
    domain_A = domain(A)
    codomain_A = codomain(A)
    CoefType = eltype(A)
    C = Operator(domain_dest, codomain_dest, Matrix{CoefType}(undef, dimension(codomain_dest), dimension(domain_dest)))
    if domain_A == domain_dest && codomain_A == codomain_dest
        coefficients(C) .= coefficients(A)
    elseif domain_dest ⊆ domain_A && codomain_dest ⊆ codomain_A
        indices_dest_domain = allindices(domain_dest)
        indices_dest_codomain = allindices(codomain_dest)
        @inbounds view(C, indices_dest_codomain, indices_dest_domain) .= view(A, indices_dest_codomain, indices_dest_domain)
    else
        coefficients(C) .= zero(CoefType)
        @inbounds for β ∈ allindices(domain_A ∩ domain_dest), α ∈ allindices(codomain_A ∩ codomain_dest)
            C[α,β] = A[α,β]
        end
    end
    return C
end

## cartesian space

function project(a::Sequence{<:CartesianSpace}, space_dest::CartesianSpace)
    space_a = space(a)
    nb_cartesian_product(space_a) == nb_cartesian_product(space_dest) || return throw(DimensionMismatch)
    CoefType = eltype(a)
    c = Sequence(space_dest, Vector{CoefType}(undef, dimension(space_dest)))
    if space_a == space_dest
        coefficients(c) .= coefficients(a)
    else
        @inbounds for i ∈ 1:nb_cartesian_product(space_dest)
            coefficients(component(c, i)) .= coefficients(project(component(a, i), space_dest[i]))
        end
    end
    return c
end

function project(A::Operator{<:CartesianSpace,<:CartesianSpace}, domain_dest::CartesianSpace, codomain_dest::CartesianSpace)
    domain_A = domain(A)
    codomain_A = codomain(A)
    nb_cartesian_product(domain_A) == nb_cartesian_product(domain_dest) && nb_cartesian_product(codomain_A) == nb_cartesian_product(codomain_dest) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain_dest, codomain_dest, Matrix{CoefType}(undef, dimension(codomain_dest), dimension(domain_dest)))
    if domain_A == domain_dest && codomain_A == codomain_dest
        coefficients(C) .= coefficients(A)
    else
        @inbounds for j ∈ 1:nb_cartesian_product(domain_dest), i ∈ 1:nb_cartesian_product(codomain_dest)
            coefficients(component(C, i, j)) .= coefficients(project(component(A, i, j), domain_dest[j], codomain_dest[i]))
        end
    end
    return C
end

function project(A::Operator{<:CartesianSpace,<:VectorSpace}, domain_dest::CartesianSpace, codomain_dest::VectorSpace)
    domain_A = domain(A)
    codomain_A = codomain(A)
    nb_cartesian_product(domain_A) == nb_cartesian_product(domain_dest) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain_dest, codomain_dest, Matrix{CoefType}(undef, dimension(codomain_dest), dimension(domain_dest)))
    if domain_A == domain_dest && codomain_A == codomain_dest
        coefficients(C) .= coefficients(A)
    else
        @inbounds for j ∈ 1:nb_cartesian_product(domain_dest)
            coefficients(component(C, j)) .= coefficients(project(component(A, j), domain_dest[j], codomain_dest))
        end
    end
    return C
end

function project(A::Operator{<:VectorSpace,<:CartesianSpace}, domain_dest::VectorSpace, codomain_dest::CartesianSpace)
    domain_A = domain(A)
    codomain_A = codomain(A)
    nb_cartesian_product(codomain_A) == nb_cartesian_product(codomain_dest) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain_dest, codomain_dest, Matrix{CoefType}(undef, dimension(codomain_dest), dimension(domain_dest)))
    if domain_A == domain_dest && codomain_A == codomain_dest
        coefficients(C) .= coefficients(A)
    else
        @inbounds for i ∈ 1:nb_cartesian_product(codomain_dest)
            coefficients(component(C, i)) .= coefficients(project(component(A, i), domain_dest, codomain_dest[i]))
        end
    end
    return C
end
