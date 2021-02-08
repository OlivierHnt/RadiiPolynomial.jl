## fallback methods

function project(a::Sequence, space::VectorSpace)
    CoefType = eltype(a)
    c = Sequence(space, Vector{CoefType}(undef, dimension(space)))
    if a.space == space
        @. c.coefficients = a.coefficients
        return c
    elseif space ⊆ a.space
        @inbounds for α ∈ allindices(space)
            c[α] = a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(a.space ∩ space)
            c[α] = a[α]
        end
        return c
    end
end

function project(A::Operator, domain::VectorSpace, codomain::VectorSpace)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    elseif A.domain ⊆ domain && A.codomain == codomain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(A.domain)
            view(C, :, α) .= view(A, :, α)
        end
        return c
    elseif domain ⊆ A.domain && A.codomain == codomain
        @inbounds for α ∈ allindices(domain)
            view(C, :, α) .= view(A, :, α)
        end
        return C
    elseif A.domain == domain && A.codomain ⊆ codomain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(A.codomain)
            view(C, α, :) .= view(A, α, :)
        end
        return C
    elseif domain == A.domain && codomain ⊆ A.codomain
        @inbounds for α ∈ allindices(codomain)
            view(C, α, :) .= view(A, α, :)
        end
        return C
    elseif domain ⊆ A.domain && codomain ⊆ A.codomain
        @inbounds for β ∈ allindices(domain), α ∈ allindices(codomain)
            C[α,β] = A[α,β]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(A.domain ∩ domain), α ∈ allindices(A.codomain ∩ codomain)
            C[α,β] = A[α,β]
        end
        return C
    end
end

## cartesian space

function project(a::Sequence{<:CartesianSpace}, space::CartesianSpace)
    nb_cartesian_product(a.space) == nb_cartesian_product(space) || return throw(DimensionMismatch)
    CoefType = eltype(a)
    c = Sequence(space, Vector{CoefType}(undef, dimension(space)))
    if a.space == space
        @. c.coefficients = a.coefficients
        return c
    else
        @inbounds for i ∈ 1:nb_cartesian_product(space)
            component(c, i).coefficients .= project(component(a, i), space[i]).coefficients
        end
        return c
    end
end

function project(A::Operator{<:CartesianSpace,<:CartesianSpace}, domain::CartesianSpace, codomain::CartesianSpace)
    nb_cartesian_product(A.domain) == nb_cartesian_product(domain) && nb_cartesian_product(A.codomain) == nb_cartesian_product(codomain) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    else
        @inbounds for j ∈ 1:nb_cartesian_product(domain), i ∈ 1:nb_cartesian_product(codomain)
            component(C, i, j).coefficients .= project(component(A, i, j), domain[j], codomain[i]).coefficients
        end
        return C
    end
end

function project(A::Operator{<:CartesianSpace,<:VectorSpace}, domain::CartesianSpace, codomain::VectorSpace)
    nb_cartesian_product(A.domain) == nb_cartesian_product(domain) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    else
        @inbounds for j ∈ 1:nb_cartesian_product(domain)
            component(C, j).coefficients .= project(component(A, j), domain[j], codomain).coefficients
        end
        return C
    end
end

function project(A::Operator{<:VectorSpace,<:CartesianSpace}, domain::VectorSpace, codomain::CartesianSpace)
    nb_cartesian_product(A.codomain) == nb_cartesian_product(codomain) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    else
        @inbounds for i ∈ 1:nb_cartesian_product(codomain)
            component(C, i).coefficients .= project(component(A, i), domain, codomain[i]).coefficients
        end
        return C
    end
end
