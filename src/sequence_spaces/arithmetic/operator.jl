## fallback methods

Base.:+(A::Operator) = Operator(domain(A), codomain(A), +(coefficients(A)))
Base.:-(A::Operator) = Operator(domain(A), codomain(A), -(coefficients(A)))

Base.:*(A::Operator, b) = Operator(domain(A), codomain(A), *(coefficients(A), b))
Base.:*(b, A::Operator) = Operator(domain(A), codomain(A), *(b, coefficients(A)))

Base.:/(A::Operator, b) = Operator(domain(A), codomain(A), /(coefficients(A), b))
Base.:/(b, A::Operator) = Operator(domain(A), codomain(A), /(b, coefficients(A)))

Base.:\(A::Operator, b) = Operator(domain(A), codomain(A), \(coefficients(A), b))
Base.:\(b, A::Operator) = Operator(domain(A), codomain(A), \(b, coefficients(A)))

Base.:∘(A::Operator, B::Operator) = *(A, B)

function Base.:*(A::Operator, B::Operator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(domain_B, codomain_A, Matrix{CoefType}(undef, dimension(codomain_A), dimension(domain_B)))
    if domain_A == codomain_B
        mul!(coefficients(C), coefficients(A), coefficients(B))
    elseif domain_A ⊆ codomain_B
        @inbounds mul!(coefficients(C), coefficients(A), view(B, allindices(domain_A), :))
    elseif codomain_B ⊆ domain_A
        @inbounds mul!(coefficients(C), view(A, :, allindices(codomain_B)), coefficients(B))
    else
        indices = allindices(domain_A ∩ codomain_B)
        @inbounds mul!(coefficients(C), view(A, :, indices), view(B, indices, :))
    end
    return C
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
    codomain(A) == codomain(B) || return throw(DimensionMismatch)
    return Operator(domain(B), domain(A), \(coefficients(A), coefficients(B)))
end

Base.inv(A::Operator) = Operator(codomain(A), domain(A), inv(coefficients(A)))

function Base.:+(A::Operator, B::Operator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    new_domain, new_codomain = addition_range(domain_A, domain_B), addition_range(codomain_A, codomain_B)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
    if domain_A == domain_B && codomain_A == codomain_B
        coefficients(C) .= coefficients(A) .+ coefficients(B)
    elseif domain_A ⊆ domain_B && codomain_A ⊆ codomain_B
        coefficients(C) .= coefficients(B)
        indices_A_domain = allindices(domain_A)
        indices_A_codomain = allindices(codomain_A)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .+= view(A, indices_A_codomain, indices_A_domain)
    elseif domain_B ⊆ domain_A && codomain_B ⊆ codomain_A
        coefficients(C) .= coefficients(A)
        indices_B_domain = allindices(domain_B)
        indices_B_codomain = allindices(codomain_B)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .+= view(B, indices_B_codomain, indices_B_domain)
    else
        coefficients(C) .= zero(CoefType)
        @inbounds for β ∈ allindices(domain_A), α ∈ allindices(codomain_A)
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(domain_B), α ∈ allindices(codomain_B)
            C[α,β] += B[α,β]
        end
    end
    return C
end

function Base.:-(A::Operator, B::Operator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    new_domain, new_codomain = addition_range(domain_A, domain_B), addition_range(codomain_A, codomain_B)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
    if domain_A == domain_B && codomain_A == codomain_B
        coefficients(C) .= coefficients(A) .- coefficients(B)
    elseif domain_A ⊆ domain_B && codomain_A ⊆ codomain_B
        coefficients(C) .= (-).(coefficients(B))
        indices_A_domain = allindices(domain_A)
        indices_A_codomain = allindices(codomain_A)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .+= view(A, indices_A_codomain, indices_A_domain)
    elseif domain_B ⊆ domain_A && codomain_B ⊆ codomain_A
        coefficients(C) .= coefficients(A)
        indices_B_domain = allindices(domain_B)
        indices_B_codomain = allindices(codomain_B)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .-= view(B, indices_B_codomain, indices_B_domain)
    else
        coefficients(C) .= zero(CoefType)
        @inbounds for β ∈ allindices(domain_A), α ∈ allindices(codomain_A)
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(domain_B), α ∈ allindices(codomain_B)
            C[α,β] -= B[α,β]
        end
    end
    return C
end

function +̄(A::Operator, B::Operator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    new_domain, new_codomain = addition_bar_range(domain_A, domain_B), addition_bar_range(codomain_A, codomain_B)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
    if domain_A == domain_B && codomain_A == codomain_B
        coefficients(C) .= coefficients(A) .+ coefficients(B)
    elseif domain_A ⊆ domain_B && codomain_A ⊆ codomain_B
        coefficients(C) .= coefficients(A)
        indices_A_domain = allindices(domain_A)
        indices_A_codomain = allindices(codomain_A)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .+= view(B, indices_A_codomain, indices_A_domain)
    elseif domain_B ⊆ domain_A && codomain_B ⊆ codomain_A
        coefficients(C) .= coefficients(B)
        indices_B_domain = allindices(domain_B)
        indices_B_codomain = allindices(codomain_B)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .+= view(A, indices_B_codomain, indices_B_domain)
    else
        coefficients(C) .= zero(CoefType)
        @inbounds for β ∈ allindices(addition_bar_range(domain_A, domain)), α ∈ allindices(addition_bar_range(codomain_A, codomain))
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(addition_bar_range(domain_B, domain)), α ∈ allindices(addition_bar_range(codomain_B, codomain))
            C[α,β] += B[α,β]
        end
    end
    return C
end

function -̄(A::Operator, B::Operator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    new_domain, new_codomain = addition_bar_range(domain_A, domain_B), addition_bar_range(codomain_A, codomain_B)
    CoefType = promote_type(eltype(A), eltype(B))
    C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
    if domain_A == domain_B && codomain_A == codomain_B
        coefficients(C) .= coefficients(A) .- coefficients(B)
    elseif domain_A ⊆ domain_B && codomain_A ⊆ codomain_B
        coefficients(C) .= coefficients(A)
        indices_A_domain = allindices(domain_A)
        indices_A_codomain = allindices(codomain_A)
        @inbounds view(C, indices_A_codomain, indices_A_domain) .-= view(B, indices_A_codomain, indices_A_domain)
    elseif domain_B ⊆ domain_A && codomain_B ⊆ codomain_A
        coefficients(C) .= (-).(coefficients(B))
        indices_B_domain = allindices(domain_B)
        indices_B_codomain = allindices(codomain_B)
        @inbounds view(C, indices_B_codomain, indices_B_domain) .+= view(A, indices_B_codomain, indices_B_domain)
    else
        coefficients(C) .= zero(CoefType)
        @inbounds for β ∈ allindices(addition_bar_range(domain_A, domain)), α ∈ allindices(addition_bar_range(codomain_A, codomain))
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ allindices(addition_bar_range(domain_B, domain)), α ∈ allindices(addition_bar_range(codomain_B, codomain))
            C[α,β] -= B[α,β]
        end
    end
    return C
end

## sequence space

function +̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling)
    domain_A = domain(A)
    codomain_A = codomain(A)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    @inbounds for α ∈ ∩(allindices(domain_A), allindices(codomain_A))
        C[α,α] += J.λ
    end
    return C
end

+̄(J::UniformScaling, A::Operator{<:SequenceSpace,<:SequenceSpace}) = +̄(A, J)

function -̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling)
    domain_A = domain(A)
    codomain_A = codomain(A)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    @inbounds for α ∈ ∩(allindices(domain_A), allindices(codomain_A))
        C[α,α] -= J.λ
    end
    return C
end

function -̄(J::UniformScaling, A::Operator{<:SequenceSpace,<:SequenceSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= (-).(coefficients(A))
    @inbounds for α ∈ ∩(allindices(domain_A), allindices(codomain_A))
        C[α,α] += J.λ
    end
    return C
end

## cartesian space

for (f, f̄) ∈ ((:+, :+̄), (:-, :-̄))
    @eval begin
        function Base.$f(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = addition_range(domain_A, domain_B), addition_range(codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            if domain_A == domain_B && codomain_A == codomain_B
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nb_cartesian_product(new_domain), i ∈ 1:nb_cartesian_product(new_codomain)
                    coefficients(component(C, i, j)) .= coefficients($f(component(A, i, j), component(B, i, j)))
                end
            end
            return C
        end

        function Base.$f(A::Operator{<:CartesianSpace,<:VectorSpace}, B::Operator{<:CartesianSpace,<:VectorSpace})
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = addition_range(domain_A, domain_B), addition_range(codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            if domain_A == domain_B && codomain_A == codomain_B
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nb_cartesian_product(new_domain)
                    coefficients(component(C, j)) .= coefficients($f(component(A, j), component(B, j)))
                end
            end
            return C
        end

        function Base.$f(A::Operator{<:VectorSpace,<:CartesianSpace}, B::Operator{<:VectorSpace,<:CartesianSpace})
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = addition_range(domain_A, domain_B), addition_range(codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            if domain_A == domain_B && codomain_A == codomain_B
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for i ∈ 1:nb_cartesian_product(new_codomain)
                    coefficients(component(C, i)) .= coefficients($f(component(A, i), component(B, i)))
                end
            end
            return C
        end

        function $f̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, B::Operator{<:CartesianSpace,<:CartesianSpace})
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = addition_bar_range(domain_A, domain_B), addition_bar_range(codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            if domain_A == domain_B && codomain_A == codomain_B
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nb_cartesian_product(new_domain), i ∈ 1:nb_cartesian_product(new_codomain)
                    coefficients(component(C, i, j)) .= coefficients($f̄(component(A, i, j), component(B, i, j)))
                end
            end
            return C
        end

        function $f̄(A::Operator{<:CartesianSpace,<:VectorSpace}, B::Operator{<:CartesianSpace,<:VectorSpace})
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = addition_bar_range(domain_A, domain_B), addition_bar_range(codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            if domain_A == domain_B && codomain_A == codomain_B
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nb_cartesian_product(new_domain)
                    coefficients(component(C, j)) .= coefficients($f̄(component(A, j), component(B, j)))
                end
            end
            return C
        end

        function $f̄(A::Operator{<:VectorSpace,<:CartesianSpace}, B::Operator{<:VectorSpace,<:CartesianSpace})
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = addition_bar_range(domain_A, domain_B), addition_bar_range(codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = Operator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            if domain_A == domain_B && codomain_A == codomain_B
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for i ∈ 1:nb_cartesian_product(new_codomain)
                    coefficients(component(C, i)) .= coefficients($f̄(component(A, i), component(B, i)))
                end
            end
            return C
        end
    end
end

function +̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    domain_A = domain(A)
    codomain_A = codomain(A)
    nb_cartesian_product(domain_A) == nb_cartesian_product(codomain_A) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    @inbounds for i ∈ ∩(allindices(domain_A), allindices(codomain_A))
        C[i,i] += J.λ
    end
    return C
end

+̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace}) = +̄(A, J::UniformScaling)

function -̄(A::Operator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    domain_A = domain(A)
    codomain_A = codomain(A)
    nb_cartesian_product(domain_A) == nb_cartesian_product(codomain_A) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    @inbounds for i ∈ ∩(allindices(domain_A), allindices(codomain_A))
        C[i,i] -= J.λ
    end
    return C
end

function -̄(J::UniformScaling, A::Operator{<:CartesianSpace,<:CartesianSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    nb_cartesian_product(domain_A) == nb_cartesian_product(codomain_A) || return throw(DimensionMismatch)
    CoefType = promote_type(eltype(A), eltype(J))
    C = Operator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= (-).(coefficients(A))
    @inbounds for i ∈ ∩(allindices(domain_A), allindices(codomain_A))
        C[i,i] += J.λ
    end
    return C
end
