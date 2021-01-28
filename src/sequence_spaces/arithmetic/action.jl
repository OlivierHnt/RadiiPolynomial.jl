(A::Operator)(b) = *(A, b)

## parameter space

function Base.:*(A::Operator{<:ParameterSpace,<:SingleSpace}, b::Sequence{<:ParameterSpace})
    A.domain == b.space || return throw(DimensionMismatch)
    return Sequence(A.codomain, *(A.coefficients, b.coefficients))
end

function Base.:\(A::Operator{<:SingleSpace,<:ParameterSpace}, b::Sequence{<:ParameterSpace})
    A.codomain == b.space || return throw(DimensionMismatch)
    return Sequence(A.domain, \(A.coefficients, b.coefficients))
end

## sequence space

function Base.:*(A::Operator{<:SequenceSpace,<:SingleSpace}, b::Sequence{<:SequenceSpace})
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, Vector{CoefType}(undef, dimension(A.codomain)))
    if A.domain == b.space
        mul!(c.coefficients, A.coefficients, b.coefficients)
        return c
    elseif A.domain ⊆ b.space
        c.coefficients .= zero(CoefType)
        indices_α = allindices(A.codomain)
        @inbounds for β ∈ allindices(A.domain), α ∈ indices_α
            c[α] += A[α,β]*b[β]
        end
        return c
    elseif b.space ⊆ A.domain
        c.coefficients .= zero(CoefType)
        indices_α = allindices(A.codomain)
        @inbounds for β ∈ allindices(b.space), α ∈ indices_α
            c[α] += A[α,β]*b[β]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        indices_α = allindices(A.codomain)
        @inbounds for β ∈ ∩(allindices(A.domain), allindices(b.space)), α ∈ indices_α
            c[α] += A[α,β]*b[β]
        end
        return c
    end
end

function Base.:\(A::Operator{<:SingleSpace,<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    if b.space == A.codomain
        return Sequence(A.domain, \(A.coefficients, b.coefficients))
    elseif b.space ⊆ A.codomain
        return \(A, project(b, A.codomain))
    elseif A.codomain ⊆ b.space
        return \(project(A, A.domain, b.space), b)
    else
        space = +(b.space, A.codomain)
        return \(project(A, A.domain, space), project(b, space))
    end
end

## cartesian space

function Base.:*(A::Operator{<:CartesianSpace,<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    length(A.domain.spaces) == length(b.space.spaces) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, zeros(CoefType, dimension(A.codomain)))
    foreach((Aⱼ, bⱼ) -> c .+= *(Aⱼ, bⱼ), eachcol(A), eachcomponent(b))
    return c
end

function Base.:*(A::Operator{<:CartesianSpace,<:SingleSpace}, b::Sequence{<:CartesianSpace})
    length(A.domain.spaces) == length(b.space.spaces) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(A.codomain, zeros(CoefType, dimension(A.codomain)))
    foreach((Aⱼ, bⱼ) -> c .+= *(Aⱼ, bⱼ), eachcomponent(A), eachcomponent(b))
    return c
end

Base.:*(A::Operator{<:SingleSpace,<:CartesianSpace}, b::Sequence{<:SingleSpace}) =
    Sequence(A.codomain, vec(map(Aⱼ -> *(Aⱼ, b), eachcomponent(A))))

function Base.:\(A::Operator{<:CartesianSpace,<:CartesianSpace}, b::Sequence{<:CartesianSpace})
    A.domain == A.codomain == b.space || return throw(DimensionMismatch)
    return Sequence(A.domain, \(A.coefficients, b.coefficients))
end
