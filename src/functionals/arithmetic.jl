## arithmetic operations +, - between functionals

Base.:+(A::Functional) = Functional(A.domain, +(A.coefficients))
Base.:-(A::Functional) = Functional(A.domain, -(A.coefficients))

function Base.:+(A::Functional, B::Functional)
    domain = A.domain ∪ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Functional(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = B.coefficients
        @inbounds for α ∈ eachindex(A.domain)
            C[α] += A[α]
        end
        return C
    elseif B.domain ⊆ A.domain
        @. C.coefficients = A.coefficients
        @inbounds for α ∈ eachindex(B.domain)
            C[α] += B[α]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.domain)
            C[α] = A[α]
        end
        @inbounds for α ∈ eachindex(B.domain)
            C[α] += B[α]
        end
        return C
    end
end

function Base.:-(A::Functional, B::Functional)
    domain = A.domain ∪ B.domain
    NewType = promote_type(eltype(A), eltype(B))
    C = Functional(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = -B.coefficients
        @inbounds for α ∈ eachindex(A.domain)
            C[α] += A[α]
        end
        return C
    elseif B.domain ⊆ A.domain
        @. C.coefficients = A.coefficients
        @inbounds for α ∈ eachindex(B.domain)
            C[α] -= B[α]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.domain)
            C[α] = A[α]
        end
        @inbounds for α ∈ eachindex(B.domain)
            C[α] -= B[α]
        end
        return C
    end
end

## arithmetic operations +, -, *, /, \ with field elements

function Base.:+(A::Functional, b)
    CoefType = promote_type(eltype(a), typeof(b))
    C = Functional(A.domain, Vector{CoefType}(undef, length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds C[_constant_index(A.domain)] += b
    return C
end

Base.:+(b, A::Functional) = +(A, b)

function Base.:-(A::Functional, b)
    CoefType = promote_type(eltype(a), typeof(b))
    C = Functional(A.domain, Vector{CoefType}(undef, length(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds C[_constant_index(A.domain)] -= b
    return C
end

function Base.:-(b, A::Functional)
    CoefType = promote_type(eltype(a), typeof(b))
    C = Functional(A.domain, Vector{CoefType}(undef, length(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds C[_constant_index(A.domain)] += b
    return C
end

Base.:*(A::Functional, b) = Functional(A.domain, *(A.coefficients, b))
Base.:*(b, A::Functional) = Functional(A.domain, *(b, A.coefficients))

Base.:/(A::Functional, b) = Functional(A.domain, /(A.coefficients, b))
Base.:\(b, A::Functional) = Functional(A.domain, \(b, A.coefficients))

## arithmetic operations +̄, -̄ between functionals

function +̄(A::Functional, B::Functional)
    domain = A.domain ∪̄ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Functional(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = A.coefficients
        @inbounds for α ∈ eachindex(A.domain)
            C[α] += B[α]
        end
        return C
    elseif B.domain ⊆ A.domain
        @. C.coefficients = B.coefficients
        @inbounds for α ∈ eachindex(B.domain)
            C[α] += A[α]
        end
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(A.domain ∪̄ domain)
            C[α] = A[α]
        end
        @inbounds for α ∈ eachindex(B.domain ∪̄ domain)
            C[α] += B[α]
        end
        return C
    end
end

function -̄(A::Functional, B::Functional)
    domain = A.domain ∪̄ B.domain
    CoefType = promote_type(eltype(A), eltype(B))
    C = Functional(domain, Vector{CoefType}(undef, length(domain)))
    if A.domain == B.domain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain
        @. C.coefficients = A.coefficients
        @inbounds for α ∈ eachindex(A.domain)
            C[α] -= B[α]
        end
        return C
    elseif B.domain ⊆ A.domain
        @. C.coefficients = -B.coefficients
        @inbounds for α ∈ eachindex(B.domain)
            C[α] += A[α]
        end
        return C
    else
        C.coefficients .= zero(NewType)
        @inbounds for α ∈ eachindex(A.domain ∪̄ domain)
            C[α] = A[α]
        end
        @inbounds for α ∈ eachindex(B.domain ∪̄ domain)
            C[α] += B[α]
        end
        return C
    end
end
