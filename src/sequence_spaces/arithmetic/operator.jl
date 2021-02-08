## fallback methods

Base.:+(A::Operator) = Operator(A.domain, A.codomain, +(A.coefficients))
Base.:-(A::Operator) = Operator(A.domain, A.codomain, -(A.coefficients))

Base.:*(A::Operator, b) = Operator(A.domain, A.codomain, *(A.coefficients, b))
Base.:*(b, A::Operator) = Operator(A.domain, A.codomain, *(b, A.coefficients))

Base.:/(A::Operator, b) = Operator(A.domain, A.codomain, /(A.coefficients, b))
Base.:/(b, A::Operator) = Operator(A.domain, A.codomain, /(b, A.coefficients))

Base.:\(A::Operator, b) = Operator(A.domain, A.codomain, \(A.coefficients, b))
Base.:\(b, A::Operator) = Operator(A.domain, A.codomain, \(b, A.coefficients))

Base.:∘(A::Operator, B::Operator) = *(A, B)

function Base.:*(A::Operator, B::Operator)
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
        indices = allindices(A.domain ∩ B.codomain)
        @inbounds mul!(C.coefficients, view(A, :, indices), view(B, indices, :))
        return C
    end
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
    A.codomain == B.codomain || return throw(DimensionMismatch)
    return Operator(B.domain, A.domain, \(A.coefficients, B.coefficients))
end

Base.inv(A::Operator) = Operator(A.codomain, A.domain, inv(A.coefficients))

function Base.:+(A::Operator, B::Operator)
    domain, codomain = addition_range(A.domain, B.domain), addition_range(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = B.coefficients
        indices_A_domain = allindices(A.domain)
        indices_A_codomain = allindices(A.codomain)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .+= view(A, indices_A_codomain, indices_A_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices_B_domain = allindices(B.domain)
        indices_B_codomain = allindices(B.codomain)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .+= view(B, indices_B_codomain, indices_B_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(A.domain), α ∈ allindices(A.codomain)
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(B.domain), α ∈ allindices(B.codomain)
            C[α,β] += B[α,β]
        end
        return C
    end
end

function Base.:-(A::Operator, B::Operator)
    domain, codomain = addition_range(A.domain, B.domain), addition_range(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = -B.coefficients
        indices_A_domain = allindices(A.domain)
        indices_A_codomain = allindices(A.codomain)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .+= view(A, indices_A_codomain, indices_A_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = A.coefficients
        indices_B_domain = allindices(B.domain)
        indices_B_codomain = allindices(B.codomain)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .-= view(B, indices_B_codomain, indices_B_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(A.domain), α ∈ allindices(A.codomain)
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(B.domain), α ∈ allindices(B.codomain)
            C[α,β] -= B[α,β]
        end
        return C
    end
end

function +̄(A::Operator, B::Operator)
    domain, codomain = addition_bar_range(A.domain, B.domain), addition_bar_range(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients + B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices_A_domain = allindices(A.domain)
        indices_A_codomain = allindices(A.codomain)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .+= view(B, indices_A_codomain, indices_A_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = B.coefficients
        indices_B_domain = allindices(B.domain)
        indices_B_codomain = allindices(B.codomain)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .+= view(A, indices_B_codomain, indices_B_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(addition_bar_range(A.domain, domain)), α ∈ allindices(addition_bar_range(A.codomain, codomain))
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(addition_bar_range(B.domain, domain)), α ∈ allindices(addition_bar_range(B.codomain, codomain))
            C[α,β] += B[α,β]
        end
        return C
    end
end

function -̄(A::Operator, B::Operator)
    domain, codomain = addition_bar_range(A.domain, B.domain), addition_bar_range(A.codomain, B.codomain)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    if A.domain == B.domain && A.codomain == B.codomain
        @. C.coefficients = A.coefficients - B.coefficients
        return C
    elseif A.domain ⊆ B.domain && A.codomain ⊆ B.codomain
        @. C.coefficients = A.coefficients
        indices_A_domain = allindices(A.domain)
        indices_A_codomain = allindices(A.codomain)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .-= view(B, indices_A_codomain, indices_A_domain)
        return C
    elseif B.domain ⊆ A.domain && B.codomain ⊆ A.codomain
        @. C.coefficients = -B.coefficients
        indices_B_domain = allindices(B.domain)
        indices_B_codomain = allindices(B.codomain)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .+= view(A, indices_B_codomain, indices_B_domain)
        return C
    else
        C.coefficients .= zero(CoefType)
        @inbounds for β ∈ allindices(addition_bar_range(A.domain, domain)), α ∈ allindices(addition_bar_range(A.codomain, codomain))
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(addition_bar_range(B.domain, domain)), α ∈ allindices(addition_bar_range(B.codomain, codomain))
            C[α,β] -= B[α,β]
        end
        return C
    end
end

## sequence space

function +̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[α,α] += J.λ
    end
    return C
end

+̄(J::UniformScaling, A::Operator{<:SequenceSpace,<:SequenceSpace}) = +̄(A, J)

function -̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[α,α] -= J.λ
    end
    return C
end

function -̄(J::UniformScaling, A::Operator{<:SequenceSpace,<:SequenceSpace})
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds for α ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[α,α] += J.λ
    end
    return C
end

## cartesian space

for (f, f̄) ∈ ((:+, :+̄), (:-, :-̄))
    @eval begin
        function Base.$f(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
            domain, codomain = addition_range(A.domain, B.domain), addition_range(A.codomain, B.codomain)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
            if A.domain == B.domain && A.codomain == B.codomain
                @. C.coefficients = $f(A.coefficients, B.coefficients)
                return C
            else
                @inbounds for j ∈ 1:nb_cartesian_product(domain), i ∈ 1:nb_cartesian_product(codomain)
                    component(C, i, j).coefficients .= $f(component(A, i, j), component(B, i, j)).coefficients
                end
                return C
            end
        end

        function Base.$f(A::Operator{<:CartesianSpace,<:VectorSpace}, B::Operator{<:CartesianSpace,<:VectorSpace})
            domain, codomain = addition_range(A.domain, B.domain), addition_range(A.codomain, B.codomain)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
            if A.domain == B.domain && A.codomain == B.codomain
                @. C.coefficients = $f(A.coefficients, B.coefficients)
                return C
            else
                @inbounds for j ∈ 1:nb_cartesian_product(domain)
                    component(C, j).coefficients .= $f(component(A, j), component(B, j)).coefficients
                end
                return C
            end
        end

        function Base.$f(A::Operator{<:VectorSpace,<:CartesianSpace}, B::Operator{<:VectorSpace,<:CartesianSpace})
            domain, codomain = addition_range(A.domain, B.domain), addition_range(A.codomain, B.codomain)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
            if A.domain == B.domain && A.codomain == B.codomain
                @. C.coefficients = $f(A.coefficients, B.coefficients)
                return C
            else
                @inbounds for i ∈ 1:nb_cartesian_product(codomain)
                    component(C, i).coefficients .= $f(component(A, i), component(B, i)).coefficients
                end
                return C
            end
        end

        function $f̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
            domain, codomain = addition_bar_range(A.domain, B.domain), addition_bar_range(A.codomain, B.codomain)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
            if A.domain == B.domain && A.codomain == B.codomain
                @. C.coefficients = $f(A.coefficients, B.coefficients)
                return C
            else
                @inbounds for j ∈ 1:nb_cartesian_product(domain), i ∈ 1:nb_cartesian_product(codomain)
                    component(C, i, j).coefficients .= $f̄(component(A, i, j), component(B, i, j)).coefficients
                end
                return C
            end
        end

        function $f̄(A::Operator{<:CartesianSpace,<:VectorSpace}, B::Operator{<:CartesianSpace,<:VectorSpace})
            domain, codomain = addition_bar_range(A.domain, B.domain), addition_bar_range(A.codomain, B.codomain)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
            if A.domain == B.domain && A.codomain == B.codomain
                @. C.coefficients = $f(A.coefficients, B.coefficients)
                return C
            else
                @inbounds for j ∈ 1:nb_cartesian_product(domain)
                    component(C, j).coefficients .= $f̄(component(A, j), component(B, j)).coefficients
                end
                return C
            end
        end

        function $f̄(A::Operator{<:VectorSpace,<:CartesianSpace}, B::Operator{<:VectorSpace,<:CartesianSpace})
            domain, codomain = addition_bar_range(A.domain, B.domain), addition_bar_range(A.codomain, B.codomain)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
            if A.domain == B.domain && A.codomain == B.codomain
                @. C.coefficients = $f(A.coefficients, B.coefficients)
                return C
            else
                @inbounds for i ∈ 1:nb_cartesian_product(codomain)
                    component(C, i).coefficients .= $f̄(component(A, i), component(B, i)).coefficients
                end
                return C
            end
        end
    end
end

function +̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    nb_cartesian_product(A.domain) == nb_cartesian_product(A.codomain) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] += J.λ
    end
    return C
end

+̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) = +̄(A, J::UniformScaling)

function -̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    nb_cartesian_product(A.domain) == nb_cartesian_product(A.codomain) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] -= J.λ
    end
    return C
end

function -̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace})
    nb_cartesian_product(A.domain) == nb_cartesian_product(A.codomain) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(A.domain, A.codomain, Matrix{CoefType}(undef, dimension(A.codomain), dimension(A.domain)))
    @. C.coefficients = -A.coefficients
    @inbounds for i ∈ ∩(allindices(A.domain), allindices(A.codomain))
        C[i,i] += J.λ
    end
    return C
end
