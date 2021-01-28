function project(a::Sequence{<:SequenceSpace}, space::SequenceSpace)
    CoefType = eltype(a)
    c = Sequence(space, Vector{CoefType}(undef, dimension(space)))
    if a.space == space
        @. c.coefficients = a.coefficients
        return c
    elseif a.space ⊆ space
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(a.space)
            c[α] = a[α]
        end
        return c
    elseif space ⊆ a.space
        @inbounds for α ∈ allindices(space)
            c[α] = a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ ∩(allindices(a.space), allindices(space))
            c[α] = a[α]
        end
        return c
    end
end

#

function project(a::Sequence{ParameterSpace}, space::ParameterSpace)
    a.space == space || return throw(DimensionMismatch)
    return copy(a)
end

#

function project(a::Sequence{<:CartesianSpace}, space::CartesianSpace)
    length(a.space.spaces) == length(space.spaces) || return throw(DimensionMismatch)
    CoefType = eltype(a)
    c = Sequence(space, Vector{CoefType}(undef, dimension(space)))
    if a.space == space
        @. c.coefficients = a.coefficients
        return c
    else
        foreach((cᵢ, aᵢ, sᵢ) -> cᵢ .= project(aᵢ, sᵢ), eachcomponent(c), eachcomponent(a), space.spaces)
        return c
    end
end

##

function project(A::Operator{<:SequenceSpace,<:SequenceSpace}, domain::SequenceSpace, codomain::SequenceSpace)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    elseif A.domain ⊆ domain && A.codomain == codomain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(A.domain)
            C[:,α] = A[:,α]
        end
        return c
    elseif domain ⊆ A.domain && A.codomain == codomain
        @inbounds for α ∈ allindices(domain)
            C[:,α] = A[:,α]
        end
        return C
    elseif A.domain == domain && A.codomain ⊆ codomain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(A.codomain)
            C[α,:] = A[α,:]
        end
        return C
    elseif domain == A.domain && codomain ⊆ A.codomain
        @inbounds for α ∈ allindices(codomain)
            C[α,:] = A[α,:]
        end
        return C
    elseif A.domain ⊆ domain && A.codomain ⊆ codomain
        C.coefficients .= zero(CoefType)
        indices_α = allindices(A.codomain)
        @inbounds for β ∈ allindices(A.domain), α ∈ indices_α
            C[α,β] = A[α,β]
        end
        return C
    elseif domain ⊆ A.domain && codomain ⊆ A.codomain
        indices_α = allindices(codomain)
        @inbounds for β ∈ allindices(domain), α ∈ indices_α
            C[α,β] = A[α,β]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_α = ∩(allindices(A.codomain), allindices(codomain))
        @inbounds for β ∈ ∩(allindices(A.domain), allindices(domain)), α ∈ indices_α
            C[α,β] = A[α,β]
        end
        return C
    end
end

#

function project(A::Operator{ParameterSpace,ParameterSpace}, domain::ParameterSpace, codomain::ParameterSpace)
    A.domain == domain && A.codomain == codomain || return throw(DimensionMismatch)
    return copy(A)
end

function project(A::Operator{ParameterSpace,<:SequenceSpace}, domain::ParameterSpace, codomain::SequenceSpace)
    A.domain == domain || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    elseif A.codomain ⊆ codomain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(A.codomain)
            C[α,:] = A[α,:]
        end
        return C
    elseif codomain ⊆ A.codomain
        @inbounds for α ∈ allindices(codomain)
            C[α,:] = A[α,:]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_α = ∩(allindices(A.codomain), allindices(codomain))
        @inbounds for β ∈ allindices(domain), α ∈ indices_α
            C[α,β] = A[α,β]
        end
        return C
    end
end

function project(A::Operator{<:SequenceSpace,ParameterSpace}, domain::SequenceSpace, codomain::ParameterSpace)
    A.codomain == codomain || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain
        @. C.coefficients = A.coefficients
        return C
    elseif A.domain ⊆ domain
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ allindices(A.domain)
            C[:,α] = A[:,α]
        end
        return c
    elseif domain ⊆ A.domain
        @inbounds for α ∈ allindices(domain)
            C[:,α] = A[:,α]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_α = allindices(codomain)
        @inbounds for β ∈ ∩(allindices(A.domain), allindices(domain)), α ∈ indices_α
            C[α,β] = A[α,β]
        end
        return C
    end
end

#

function project(A::Operator{<:CartesianSpace,<:CartesianSpace}, domain::CartesianSpace, codomain::CartesianSpace)
    length(A.domain.spaces) == length(domain.spaces) && length(A.codomain.spaces) == length(codomain.spaces) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, domᵢ, codomᵢ) -> Cᵢ .= project(Aᵢ, domᵢ, codomᵢ), eachcomponent(C), eachcomponent(A), domain.spaces, codomain.spaces)
        return C
    end
end

function project(A::Operator{<:CartesianSpace,<:SingleSpace}, domain::CartesianSpace, codomain::SingleSpace)
    length(A.domain.spaces) == length(domain.spaces) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, domᵢ) -> Cᵢ .= project(Aᵢ, domᵢ, codomain), eachcomponent(C), eachcomponent(A), domain.spaces)
        return C
    end
end

function project(A::Operator{<:SingleSpace,<:CartesianSpace}, domain::SingleSpace, codomain::CartesianSpace)
    length(A.codomain.spaces) == length(codomain.spaces) || return throw(DimensionMismatch)
    CoefType = eltype(A)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == domain && A.codomain == codomain
        @. C.coefficients = A.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, codomᵢ) -> Cᵢ .= project(Aᵢ, domain, codomᵢ), eachcomponent(C), eachcomponent(A), codomain.spaces)
        return C
    end
end
