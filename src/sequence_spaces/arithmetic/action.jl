## fallback methods

(A::Operator)(b) = *(A, b)

function Base.:*(A::Operator, b::Sequence)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, Vector{CoefType}(undef, dimension(A.codomain)))
    if A.domain == b.space
        mul!(c.coefficients, A.coefficients, b.coefficients)
        return c
    elseif A.domain ⊆ b.space
        c.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(A.domain), α ∈ allindices(A.codomain)
            c[α] += A[α,β]*b[β]
        end
        return c
    elseif b.space ⊆ A.domain
        c.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(b.space), α ∈ allindices(A.codomain)
            c[α] += A[α,β]*b[β]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(A.domain ∩ b.space), α ∈ allindices(A.codomain)
            c[α] += A[α,β]*b[β]
        end
        return c
    end
end

function Base.:\(A::Operator, b::Sequence)
    if b.space == A.codomain
        return Sequence(A.domain, \(A.coefficients, b.coefficients))
    elseif b.space ⊆ A.codomain
        return \(A, project(b, A.codomain))
    elseif A.codomain ⊆ b.space
        return \(project(A, A.domain, b.space), b)
    else
        space = addition_range(b.space, A.codomain)
        return \(project(A, A.domain, space), project(b, space))
    end
end

## cartesian space

function Base.:*(A::Operator{<:CartesianSpace,<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    nb_cartesian_product(A.domain) == nb_cartesian_product(b.space) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, zeros(CoefType, dimension(A.codomain)))
    @inbounds for j ∈ 1:nb_cartesian_product(A.domain)
        bⱼ = component(b, j)
        @inbounds for i ∈ 1:nb_cartesian_product(A.codomain)
            component(c, i).coefficients .+= *(component(A, i, j), bⱼ).coefficients
        end
    end
    return c
end

function Base.:*(A::Operator{<:CartesianSpace,<:VectorSpace}, b::Sequence{<:CartesianSpace})
    nb_cartesian_product(A.domain) == nb_cartesian_product(b.space) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, Vector{CoefType}(undef, dimension(A.codomain)))
    c.coefficients .= *(component(A, 1), component(b, 1)).coefficients
    @inbounds for j ∈ 2:nb_cartesian_product(A.domain)
        c.coefficients .+= *(component(A, j), component(b, j)).coefficients
    end
    return c
end

function Base.:*(A::Operator{<:VectorSpace,<:CartesianSpace}, b::Sequence{<:VectorSpace})
    nb_cartesian_product(A.codomain) == nb_cartesian_product(b.space) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.domain, Vector{CoefType}(undef, dimension(A.codomain)))
    @inbounds for i ∈ 1:nb_cartesian_product(A.codomain)
        component(c, i).coefficients .= *(component(A, i), b).coefficients
    end
    return c
end
