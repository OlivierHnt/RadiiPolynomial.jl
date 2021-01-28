## fallback methods

Base.:+(A::Operator) = Operator(A.domain, A.codomain, +(A.coefficients))
Base.:-(A::Operator) = Operator(A.domain, A.codomain, -(A.coefficients))

Base.:∘(A::Operator, B::Operator) = *(A, B)
function Base.:*(A::Operator, B::Operator)
    @assert A.domain == B.codomain
    return Operator(B.domain, A.codomain, *(A.coefficients, B.coefficients))
end

function Base.:^(A::Operator, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(A)
    n == 1 && return copy(A)
    n == 2 && return *(A, A)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        A *= A
    end
    C = A
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            A *= A
        end
        C *= A
    end
    return C
end

function Base.:\(A::Operator, B::Operator)
    @assert A.codomain == B.codomain
    return Operator(B.domain, A.domain, \(A.coefficients, B.coefficients))
end

Base.inv(A::Operator) = Operator(A.codomain, A.domain, inv(A.coefficients))

Base.:*(A::Operator, b) = Operator(A.domain, A.codomain, *(A.coefficients, b))
Base.:*(b, A::Operator) = Operator(A.domain, A.codomain, *(b, A.coefficients))

Base.:/(A::Operator, b) = Operator(A.domain, A.codomain, /(A.coefficients, b))
Base.:/(b, A::Operator) = Operator(A.domain, A.codomain, /(b, A.coefficients))

Base.:\(A::Operator, b) = Operator(A.domain, A.codomain, \(A.coefficients, b))
Base.:\(b, A::Operator) = Operator(A.domain, A.codomain, \(b, A.coefficients))

## sequence space

function Base.:+(A::Operator{<:SequenceSpace,<:SequenceSpace}, B::Operator{<:SequenceSpace,<:SequenceSpace})
    domain, codomain = +(A.domain, B.domain), +(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = B.coefficients
        indices = allindices(A.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(B.domain)
        @inbounds view(C, :, indices) .+= view(B, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = B.coefficients
        indices = allindices(A.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(B.codomain)
        @inbounds view(C, indices, :) .+= view(B, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = B.coefficients
        indices_domain = allindices(A.domain)
        indices_codomain = allindices(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices_domain = allindices(B.domain)
        indices_codomain = allindices(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(B, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = allindices(A.domain)
        indices_A_codomain = allindices(A.codomain)
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = allindices(B.domain)
        indices_B_codomain = allindices(B.codomain)
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] += B[α,β]
        end
        return C
    end
end

function Base.:-(A::Operator{<:SequenceSpace,<:SequenceSpace}, B::Operator{<:SequenceSpace,<:SequenceSpace})
    domain, codomain = +(A.domain, B.domain), +(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = -B.coefficients
        indices = allindices(A.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(B.domain)
        @inbounds view(C, :, indices) .-= view(B, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = -B.coefficients
        indices = allindices(A.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(B.codomain)
        @inbounds view(C, indices, :) .-= view(B, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = -B.coefficients
        indices_domain = allindices(A.domain)
        indices_codomain = allindices(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices_domain = allindices(B.domain)
        indices_codomain = allindices(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .-= view(B, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = allindices(A.domain)
        indices_A_codomain = allindices(A.codomain)
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = allindices(B.domain)
        indices_B_codomain = allindices(B.codomain)
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] -= B[α,β]
        end
        return C
    end
end

function Base.:*(A::Operator{<:SequenceSpace,<:SequenceSpace}, B::Operator{<:SequenceSpace,<:SequenceSpace})
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(B.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(B.domain)))
    if A.domain == B.codomain
        mul!(C.coefficients, A.coefficients, B.coefficients)
        return C
    elseif A.domain ⊆ B.codomain
        @inbounds mul!(C.coefficients, A.coefficients, view(B, allindices(A.domain), :))
        return C
    elseif B.codomain ⊆ A.domain
        @inbounds mul!(C.coefficients, view(A, :, allindices(B.codomain)), B.coefficients)
        return C
    else
        indices = ∩(allindices(A.domain), allindices(B.codomain))
        @inbounds mul!(C.coefficients, view(A, :, indices), view(B, indices, :))
        return C
    end
end

#

function +̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, B::Operator{<:SequenceSpace,<:SequenceSpace})
    domain, codomain = +̄(A.domain, B.domain), +̄(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(A.domain)
        @inbounds view(C, :, indices) .+= view(B, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = B.coefficients
        indices = allindices(B.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(A.codomain)
        @inbounds view(C, indices, :) .+= view(B, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = B.coefficients
        indices = allindices(B.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices_domain = allindices(A.domain)
        indices_codomain = allindices(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(B, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = B.coefficients
        indices_domain = allindices(B.domain)
        indices_codomain = allindices(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = allindices(+̄(A.domain, domain))
        indices_A_codomain = allindices(+̄(A.codomain, codomain))
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = allindices(+̄(B.domain, domain))
        indices_B_codomain = allindices(+̄(B.codomain, codomain))
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] += B[α,β]
        end
        return C
    end
end

function -̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, B::Operator{<:SequenceSpace,<:SequenceSpace})
    domain, codomain = +̄(A.domain, B.domain), +̄(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(A.domain)
        @inbounds view(C, :, indices) .-= view(B, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = -B.coefficients
        indices = allindices(B.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices = allindices(A.codomain)
        @inbounds view(C, indices, :) .-= view(B, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = -B.coefficients
        indices = allindices(B.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices_domain = allindices(A.domain)
        indices_codomain = allindices(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .-= view(B, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = -B.coefficients
        indices_domain = allindices(B.domain)
        indices_codomain = allindices(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = allindices(+̄(A.domain, domain))
        indices_A_codomain = allindices(+̄(A.codomain, codomain))
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = allindices(+̄(B.domain, domain))
        indices_B_codomain = allindices(+̄(B.codomain, codomain))
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] -= B[α,β]
        end
        return C
    end
end

function +̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, b)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[α,α] += b
    end
    return C
end

+̄(b, A::Operator{<:SequenceSpace,<:SequenceSpace}) = +̄(A, b)

function -̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, b)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[α,α] -= b
    end
    return C
end

function -̄(b, A::Operator{<:SequenceSpace,<:SequenceSpace})
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[α,α] += b
    end
    return C
end

+̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling) = +̄(A, J.λ)
+̄(J::UniformScaling, A::Operator{<:SequenceSpace,<:SequenceSpace}) = +̄(J.λ, A)

-̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling) = -̄(A, J.λ)
-̄(J::UniformScaling, A::Operator{<:SequenceSpace,<:SequenceSpace}) = -̄(J.λ, A)

## cartesian space

function Base.:+(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = +(A.domain, B.domain), +(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ + Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function Base.:-(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = +(A.domain, B.domain), +(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ - Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

#

function +̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = +̄(A.domain, B.domain), +̄(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ +̄ Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function -̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
    domain, codomain = +̄(A.domain, B.domain), +̄(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    else
        foreach((Cᵢ, Aᵢ, Bᵢ) -> Cᵢ.coefficients .= (Aᵢ -̄ Bᵢ).coefficients, eachcomponent(C), eachcomponent(A), eachcomponent(B))
        return C
    end
end

function +̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, b)
    length(A.domain.spaces) == length(A.codomain.spaces) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] += b
    end
    return C
end

function +̄(b, A::Operator{<:CartesianSpace,<:CartesianSpace})
    length(A.domain.spaces) == length(A.codomain.spaces) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] += b
    end
    return C
end

function -̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, b)
    length(A.domain.spaces) == length(A.codomain.spaces) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] -= b
    end
    return C
end

function -̄(b, A::Operator{<:CartesianSpace,<:CartesianSpace})
    length(A.domain.spaces) == length(A.codomain.spaces) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] += b
    end
    return C
end

+̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling) = +̄(A, J.λ)
+̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) = +̄(J.λ, A)

-̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling) = -̄(A, J.λ)
-̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) = -̄(J.λ, A)
