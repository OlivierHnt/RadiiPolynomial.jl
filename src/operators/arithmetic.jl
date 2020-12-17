## arithmetic operations +, - between operators

Base.:+(A::Operator) = Operator(A.domain, A.codomain, +(A.coefficients))
Base.:-(A::Operator) = Operator(A.domain, A.codomain, -(A.coefficients))

function Base.:+(A::Operator, B::Operator)
    domain, codomain = A.domain ∪ B.domain, A.codomain ∪ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = B.coefficients
        indices = eachindex(A.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .+= view(B, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = B.coefficients
        indices = eachindex(A.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(B.codomain)
        @inbounds view(C, indices, :) .+= view(B, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = B.coefficients
        indices_domain = eachindex(A.domain)
        indices_codomain = eachindex(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices_domain = eachindex(B.domain)
        indices_codomain = eachindex(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(B, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = eachindex(A.domain)
        indices_A_codomain = eachindex(A.codomain)
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = eachindex(B.domain)
        indices_B_codomain = eachindex(B.codomain)
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] += B[α,β]
        end
        return C
    end
end

function Base.:-(A::Operator, B::Operator)
    domain, codomain = A.domain ∪ B.domain, A.codomain ∪ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = -B.coefficients
        indices = eachindex(A.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .-= view(B, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = -B.coefficients
        indices = eachindex(A.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(B.codomain)
        @inbounds view(C, indices, :) .-= view(B, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = -B.coefficients
        indices_domain = eachindex(A.domain)
        indices_codomain = eachindex(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices_domain = eachindex(B.domain)
        indices_codomain = eachindex(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .-= view(B, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = eachindex(A.domain)
        indices_A_codomain = eachindex(A.codomain)
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = eachindex(B.domain)
        indices_B_codomain = eachindex(B.codomain)
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] -= B[α,β]
        end
        return C
    end
end

## composition

Base.:∘(A::Operator, B::Operator) = *(A, B)

function Base.:*(A::Operator, B::Operator)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(B.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(B.domain)))
    if A.domain == B.codomain
        mul!(C.coefficients, A.coefficients, B.coefficients)
        return C
    elseif A.domain ⊆ B.codomain
        @inbounds mul!(C.coefficients, A.coefficients, view(B, eachindex(A.domain), :))
        return C
    elseif B.codomain ⊆ A.domain
        @inbounds mul!(C.coefficients, view(A, :, eachindex(B.codomain)), B.coefficients)
        return C
    else
        indices = eachindex(A.domain ∩ B.codomain)
        @inbounds mul!(C.coefficients, view(A, :, indices), view(B, indices, :))
        return C
    end
end

function Base.:^(A::Operator, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
    n == 0 && return one(A)
    n == 1 && return copy(A)
    return _power_by_squaring(A, n)
end

function _power_by_squaring(A::Operator, n::Int)
    n == 2 && return *(A, A)
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

## inverse

Base.inv(A::Operator) = Operator(A.codomain, A.domain, inv(A.coefficients))

## arithmetic operations *, /, \ with field elements

Base.:*(A::Operator, b) = Operator(A.domain, A.codomain, *(A.coefficients, b))
Base.:*(b, A::Operator) = Operator(A.domain, A.codomain, *(b, A.coefficients))

Base.:/(A::Operator, b) = Operator(A.domain, A.codomain, /(A.coefficients, b))
Base.:/(b, A::Operator) = Operator(A.domain, A.codomain, /(b, A.coefficients))

Base.:\(A::Operator, b) = Operator(A.domain, A.codomain, \(A.coefficients, b))
Base.:\(b, A::Operator) = Operator(A.domain, A.codomain, \(b, A.coefficients))

## arithmetic operations +̄, -̄ between operators

function +̄(A::Operator, B::Operator)
    domain, codomain = A.domain ∪̄ B.domain, A.codomain ∪̄ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(A.domain)
        @inbounds view(C, :, indices) .+= view(B, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = B.coefficients
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(A.codomain)
        @inbounds view(C, indices, :) .+= view(B, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = B.coefficients
        indices = eachindex(B.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices_domain = eachindex(A.domain)
        indices_codomain = eachindex(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(B, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = B.coefficients
        indices_domain = eachindex(B.domain)
        indices_codomain = eachindex(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = eachindex(A.domain ∪̄ domain)
        indices_A_codomain = eachindex(A.codomain ∪̄ codomain)
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = eachindex(B.domain ∪̄ domain)
        indices_B_codomain = eachindex(B.codomain ∪̄ codomain)
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] += B[α,β]
        end
        return C
    end
end

function -̄(A::Operator, B::Operator)
    domain, codomain = A.domain ∪̄ B.domain, A.codomain ∪̄ B.codomain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, length(codomain), length(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(A.domain)
        @inbounds view(C, :, indices) .-= view(B, :, indices)
        return C
    elseif B.domain ⊆ A.domain && A.codomain == B.codomain
        @. C.coefficients = -B.coefficients
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .+= view(A, :, indices)
        return C
    elseif A.domain == B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices = eachindex(A.codomain)
        @inbounds view(C, indices, :) .-= view(B, indices, :)
        return C
    elseif A.domain == B.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = -B.coefficients
        indices = eachindex(B.codomain)
        @inbounds view(C, indices, :) .+= view(A, indices, :)
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices_domain = eachindex(A.domain)
        indices_codomain = eachindex(A.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .-= view(B, indices_codomain, indices_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = -B.coefficients
        indices_domain = eachindex(B.domain)
        indices_codomain = eachindex(B.codomain)
        @inbounds view(C, indices_codomain, indices_domain) .+= view(A, indices_codomain, indices_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        indices_A_domain = eachindex(A.domain ∪̄ domain)
        indices_A_codomain = eachindex(A.codomain ∪̄ codomain)
        @inbounds for β ∈ indices_A_domain, α ∈ indices_A_codomain
            C[α,β] = A[α,β]
        end
        indices_B_domain = eachindex(B.domain ∪̄ domain)
        indices_B_codomain = eachindex(B.codomain ∪̄ codomain)
        @inbounds for β ∈ indices_B_domain, α ∈ indices_B_codomain
            C[α,β] -= B[α,β]
        end
        return C
    end
end

## arithmetic operations +̄, -̄ with field elements

function +̄(A::Operator, b)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ eachindex(A.domain ∩ A.codomain)
        C[α,α] += b
    end
    return C
end

+̄(b, A::Operator) = +̄(A, b)

function -̄(A::Operator, b)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ eachindex(A.domain ∩ A.codomain)
        C[α,α] -= b
    end
    return C
end

function -̄(b, A::Operator)
    CoefType = promote_type(eltype(A), typeof(b))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, length(A.codomain), length(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds for α ∈ eachindex(A.domain ∩ A.codomain)
        C[α,α] += b
    end
    return C
end

## arithmetic operations +̄, -̄ with uniform scaling operator

+̄(A::Operator, J::UniformScaling) = +̄(A, J.λ)
+̄(J::UniformScaling, A::Operator) = +̄(J.λ, A)

-̄(A::Operator, J::UniformScaling) = -̄(A, J.λ)
-̄(J::UniformScaling, A::Operator) = -̄(J.λ, A)

## some mixed operations involing functionals and operators

Base.:∘(A::Functional, B::Operator) = *(A, B)

function Base.:*(A::Functional, B::Operator)
    @assert A.domain == B.codomain # NOTE: should be relaxed
    CoefType = promote_type(eltype(A), eltype(B))
    len = length(B.domain)
    C = Functional(B.domain, Vector{CoefType}(undef, len))
    @inbounds for i ∈ 1:len
        C.coefficients[i] = mapreduce(*, +, A.coefficients, view(B.coefficients, :, i))
    end
    return C
end

Base.:∘(b::Sequence, A::Functional) = *(b, A)

function Base.:*(b::Sequence, A::Functional)
    CoefType = promote_type(eltype(b), eltype(A))
    C = Operator(A.domain, b.space, Matrix{CoefType}(undef, length(b.space), length(A.domain)))
    indices_domain = eachindex(A.domain)
    indices_space = eachindex(b.space)
    @inbounds for β ∈ indices_domain, α ∈ indices_space
        C[α,β] = b[α]*A[β]
    end
    return C
end

Base.:+(B::Operator, A::Functional) = +(A, B)

function Base.:+(A::Functional, B::Operator)
    domain = A.domain ∪ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, B.codomain, Matrix{CoefType}(undef, length(B.codomain), length(domain)))
    idx₀ = _constant_index(B.codomain)
    if A.domain == B.domain
        @. C.coefficients = B.coefficients
        @inbounds view(C, idx₀, :) .+= A.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = B.coefficients
        @inbounds view(C, idx₀, eachindex(A.domain)) .+= A.coefficients
        return C
    elseif B.domain ⊆ A.domain
        C.coefficients .= zero(CoefType)
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .= view(B, :, indices)
        @inbounds view(C, idx₀, :) .+= A.coefficients
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(B.domain)
            view(C, :, α) .= view(B, :, α)
        end
        @inbounds for α ∈ eachindex(A.domain)
            C[idx₀,α] += A[α]
        end
        return C
    end
end

function Base.:-(A::Functional, B::Operator)
    domain = A.domain ∪ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, B.codomain, Matrix{CoefType}(undef, length(B.codomain), length(domain)))
    idx₀ = _constant_index(B.codomain)
    if A.domain == B.domain
        @. C.coefficients = -B.coefficients
        @inbounds view(C, idx₀, :) .+= A.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = -B.coefficients
        @inbounds view(C, idx₀, eachindex(A.domain)) .+= A.coefficients
        return C
    elseif B.domain ⊆ A.domain
        C.coefficients .= zero(CoefType)
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .= .-view(B, :, indices)
        @inbounds view(C, idx₀, :) .+= A.coefficients
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(B.domain)
            view(C, :, α) .= .-view(B, :, α)
        end
        @inbounds for α ∈ eachindex(A.domain)
            C[idx₀,α] += A[α]
        end
        return C
    end
end

function Base.:-(B::Operator, A::Functional)
    domain = A.domain ∪ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, B.codomain, Matrix{CoefType}(undef, length(B.codomain), length(domain)))
    idx₀ = _constant_index(B.codomain)
    if A.domain == B.domain
        @. C.coefficients = B.coefficients
        @inbounds view(C, idx₀, :) .-= A.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = B.coefficients
        @inbounds view(C, idx₀, eachindex(A.domain)) .-= A.coefficients
        return C
    elseif B.domain ⊆ A.domain
        C.coefficients .= zero(CoefType)
        indices = eachindex(B.domain)
        @inbounds view(C, :, indices) .= view(B, :, indices)
        @inbounds view(C, idx₀, :) .-= A.coefficients
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(B.domain)
            view(C, :, α) .= view(B, :, α)
        end
        @inbounds for α ∈ eachindex(A.domain)
            C[idx₀,α] -= A[α]
        end
        return C
    end
end
