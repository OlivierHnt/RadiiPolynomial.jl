Base.:+(A::AbstractLinearOperator) = A
Base.:-(A::AbstractLinearOperator) = Negate(A)
Base.:-(J::UniformScalingOperator) = UniformScalingOperator(-J.λ)
Base.:-(A::Negate) = A.A
# Base.:-(A::Add) = -A.A + (-A.B)

Base.:+(A::AbstractLinearOperator, B::AbstractLinearOperator) = Add(A, B)
Base.:-(A::AbstractLinearOperator, B::AbstractLinearOperator) = A + (-B)

Base.:+(A::AbstractLinearOperator, J::UniformScaling) = A + UniformScalingOperator(J)
Base.:+(J::UniformScaling, A::AbstractLinearOperator) = UniformScalingOperator(J) + A
Base.:-(A::AbstractLinearOperator, J::UniformScaling) = A - UniformScalingOperator(J)
Base.:-(J::UniformScaling, A::AbstractLinearOperator) = UniformScalingOperator(J) - A

Base.:*(A::AbstractLinearOperator, B::AbstractLinearOperator) = A ∘ B
Base.:*(J::UniformScalingOperator, A::LinearOperator) = J.λ * A
Base.:*(A::LinearOperator, J::UniformScalingOperator) = A * J.λ
Base.:*(J::UniformScaling, A::AbstractLinearOperator) = UniformScalingOperator(J) * A
Base.:*(A::AbstractLinearOperator, J::UniformScaling) = A * UniformScalingOperator(J)
Base.:*(S::AbstractLinearOperator, a::Number) = S * UniformScalingOperator(a)
Base.:*(a::Number, S::AbstractLinearOperator) = UniformScalingOperator(a) * S





add!(C::LinearOperator, S₁::AbstractLinearOperator, S₂::AbstractLinearOperator) = add!(C, project(S₁, domain(C), codomain(C), eltype(C)), S₂)
add!(C::LinearOperator, S::AbstractLinearOperator, A::LinearOperator) = add!(C, project(S, domain(C), codomain(C), eltype(C)), A)
add!(C::LinearOperator, A::LinearOperator, S::AbstractLinearOperator) = add!(C, A, project(S, domain(C), codomain(C), eltype(C)))

sub!(C::LinearOperator, S₁::AbstractLinearOperator, S₂::AbstractLinearOperator) = sub!(C, project(S₁, domain(C), codomain(C), eltype(C)), S₂)
sub!(C::LinearOperator, S::AbstractLinearOperator, A::LinearOperator) = sub!(C, project(S, domain(C), codomain(C), eltype(C)), A)
sub!(C::LinearOperator, A::LinearOperator, S::AbstractLinearOperator) = sub!(C, A, project(S, domain(C), codomain(C), eltype(C)))

radd!(A::LinearOperator, S::AbstractLinearOperator) = radd!(A, project(S, domain(A), codomain(A), eltype(A)))
rsub!(A::LinearOperator, S::AbstractLinearOperator) = rsub!(A, project(S, domain(A), codomain(A), eltype(A)))

ladd!(S::AbstractLinearOperator, A::LinearOperator) = ladd!(project(S, domain(A), codomain(A), eltype(A)), A)
lsub!(S::AbstractLinearOperator, A::LinearOperator) = lsub!(project(S, domain(A), codomain(A), eltype(A)), A)

function Base.:*(A::LinearOperator, S::AbstractLinearOperator)
    domain(S, domain(A)) isa EmptySpace && return ComposedOperator(A, S)
    return A * (Projection(domain(A), eltype(A)) * S)
end
Base.:*(S::AbstractLinearOperator, A::LinearOperator) = (S * Projection(codomain(A), eltype(A))) * A

function mul!(C::LinearOperator, S₁::AbstractLinearOperator, S₂::AbstractLinearOperator, α::Number, β::Number)
    domain_C = domain(C)
    return mul!(C, S₁, project(S₂, domain_C, codomain(S₂, domain_C), eltype(C)), α, β)
end
mul!(C::LinearOperator, S::AbstractLinearOperator, A::LinearOperator, α::Number, β::Number) =
    mul!(C, project(S, codomain(A), codomain(C), eltype(C)), A, α, β)
mul!(C::LinearOperator, A::LinearOperator, S::AbstractLinearOperator, α::Number, β::Number) =
    mul!(C, A, project(S, domain(C), domain(A), eltype(C)), α, β)











# methods for `LinearOperator`

Base.:+(A::LinearOperator) = LinearOperator(domain(A), codomain(A), +(coefficients(A)))
Base.:-(A::LinearOperator) = LinearOperator(domain(A), codomain(A), -(coefficients(A)))

Base.:*(A::LinearOperator, b::Number) = LinearOperator(domain(A), codomain(A), *(coefficients(A), b))
Base.:*(b::Number, A::LinearOperator) = LinearOperator(domain(A), codomain(A), *(b, coefficients(A)))

Base.:/(A::LinearOperator, b::Number) = LinearOperator(domain(A), codomain(A), /(coefficients(A), b))
Base.:\(b::Number, A::LinearOperator) = LinearOperator(domain(A), codomain(A), \(b, coefficients(A)))

rmul!(A::LinearOperator, b::Number) =  LinearOperator(domain(A), codomain(A), rmul!(coefficients(A), b))
lmul!(b::Number, A::LinearOperator) = LinearOperator(domain(A), codomain(A), lmul!(b, coefficients(A)))

rdiv!(A::LinearOperator, b::Number) = LinearOperator(domain(A), codomain(A), rdiv!(coefficients(A), b))
ldiv!(b::Number, A::LinearOperator) = LinearOperator(domain(A), codomain(A), ldiv!(b, coefficients(A)))

function Base.:*(A::LinearOperator, B::LinearOperator)
    pdomA, pcodomB = _promote_space(domain(A), codomain(B))
    _iscompatible(pdomA, pcodomB) || return throw(ArgumentError("spaces must be compatible: A has domain $(domain(A)), B has codomain $(codomain(B))"))
    C = zeros(promote_type(eltype(A), eltype(B)), domain(B), codomain(A))
    _mul!(C, LinearOperator(pdomA, codomain(A), coefficients(A)), LinearOperator(domain(B), pcodomB, coefficients(B)), exact(true), exact(false))
    return C
end

function mul!(C::LinearOperator, A::LinearOperator, B::LinearOperator, α::Number, β::Number)
    pdomC, pdomB = _promote_space(domain(C), domain(B))
    pcodomC, pcodomA = _promote_space(codomain(C), codomain(A))
    pdomA, pcodomB = _promote_space(domain(A), codomain(B))
    _iscompatible(pdomC, pdomB) & _iscompatible(pcodomC, pcodomA) & _iscompatible(pdomA, pcodomB) || return throw(ArgumentError("spaces must be compatible"))
    _mul!(LinearOperator(pdomC, pcodomC, coefficients(C)),
          LinearOperator(pdomA, pcodomA, coefficients(A)),
          LinearOperator(pdomB, pcodomB, coefficients(B)), α, β)
    return C
end

function _mul!(C::LinearOperator, A::LinearOperator, B::LinearOperator, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B
        if domain_B == domain_C && codomain_A == codomain_C
            mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
        else
            if iszero(β)
                coefficients(C) .= zero(eltype(C))
            elseif !isone(β)
                coefficients(C) .*= β
            end
            inds_domain = indices(domain_B ∩ domain_C)
            inds_codomain = indices(codomain_A ∩ codomain_C)
            @inbounds mul!(view(C, inds_codomain, inds_domain), view(A, inds_codomain, :), view(B, :, inds_domain), α, exact(true))
        end
    else
        inds_mult = indices(domain_A ∩ codomain_B)
        if domain_B == domain_C && codomain_A == codomain_C
            @inbounds mul!(coefficients(C), view(A, :, inds_mult), view(B, inds_mult, :), α, β)
        else
            if iszero(β)
                coefficients(C) .= zero(eltype(C))
            elseif !isone(β)
                coefficients(C) .*= β
            end
            inds_domain = indices(domain_B ∩ domain_C)
            inds_codomain = indices(codomain_A ∩ codomain_C)
            @inbounds mul!(view(C, inds_codomain, inds_domain), view(A, inds_codomain, inds_mult), view(B, inds_mult, inds_domain), α, exact(true))
        end
    end
    return C
end

function rmul!(A::LinearOperator, B::LinearOperator)
    pdomA, pcodomB = _promote_space(domain(A), codomain(B))
    _iscompatible(pdomA, pcodomB) || return throw(ArgumentError("spaces must be compatible: A has domain $(domain(A)), B has codomain $(codomain(B))"))
    return LinearOperator(domain(B), codomain(A), rmul!(coefficients(A), coefficients(B)))
end

function lmul!(A::LinearOperator, B::LinearOperator)
    pdomA, pcodomB = _promote_space(domain(A), codomain(B))
    _iscompatible(pdomA, pcodomB) || return throw(ArgumentError("spaces must be compatible: A has domain $(domain(A)), B has codomain $(codomain(B))"))
    return LinearOperator(domain(B), codomain(A), lmul!(coefficients(A), coefficients(B)))
end

function Base.:^(A::LinearOperator, n::Integer)
    if n < 0
        return ^(inv(A), -n)
    elseif n == 0
        return one(A)
    elseif n == 2
        return *(A, A) # TODO: implement `_sqr` function to improve accuracy with intervals
    else
        A_ = LinearOperator(domain(A), codomain(A), Matrix{eltype(A)}(undef, size(A)))
        coefficients(A_) .= coefficients(A)
        if n == 1
            return A_
        else # power by squaring
            t = trailing_zeros(n) + 1
            n >>= t
            while (t -= 1) > 0
                A_ *= A_
            end
            C = A_
            while n > 0
                t = trailing_zeros(n) + 1
                n >>= t
                while (t -= 1) ≥ 0
                    A_ *= A_
                end
                C *= A_
            end
            return C
        end
    end
end

Base.inv(A::LinearOperator) = LinearOperator(codomain(A), domain(A), inv(coefficients(A)))

function Base.:/(A::LinearOperator, B::LinearOperator)
    pdomA, pdomB = _promote_space(domain(A), domain(B))
    _iscompatible(pdomA, pdomB) || return throw(ArgumentError("spaces must be compatible: A has domain $(domain(A)), B has codomain $(domain(B))"))
    return LinearOperator(codomain(B), codomain(A), /(coefficients(A), coefficients(B)))
end

function Base.:\(A::LinearOperator, B::LinearOperator)
    pcodomA, pcodomB = _promote_space(codomain(A), codomain(B))
    _iscompatible(pcodomA, pcodomB) || return throw(ArgumentError("spaces must be compatible: A has domain $(codomain(A)), B has codomain $(codomain(B))"))
    return LinearOperator(domain(B), domain(A), \(coefficients(A), coefficients(B)))
end

for (f, f!, rf!, lf!, _f!, _rf!, _lf!) ∈ ((:(Base.:+), :add!, :radd!, :ladd!, :_add!, :_radd!, :_ladd!),
        (:(Base.:-), :sub!, :rsub!, :lsub!, :_sub!, :_rsub!, :_lsub!))
    @eval begin
        function $f(A::LinearOperator, B::LinearOperator)
            pdomA, pdomB = _promote_space(domain(A), domain(B))
            pcodomA, pcodomB = _promote_space(codomain(A), codomain(B))
            C = zeros(promote_type(eltype(A), eltype(B)), codomain($f, pdomA, pdomB), codomain($f, pcodomA, pcodomB))
            $_f!(C, LinearOperator(pdomA, pcodomA, coefficients(A)), LinearOperator(pdomB, pcodomB, coefficients(B)))
            return C
        end

        function $f!(C::LinearOperator, A::LinearOperator, B::LinearOperator)
            pdomA, pdomB = _promote_space(domain(A), domain(B))
            pcodomA, pcodomB = _promote_space(codomain(A), codomain(B))
            domop, codomop = codomain($f, pdomA, pdomB), codomain($f, pcodomA, pcodomB)
            pdomC, pdomop = _promote_space(domain(C), domop)
            pcodomC, pcodomop = _promote_space(codomain(C), codomop)
            _iscompatible(pdomC, pdomop) & _iscompatible(pcodomC, pcodomop) || return throw(ArgumentError("spaces must be compatible"))
            $_f!(LinearOperator(pdomC, pcodomC, coefficients(C)), LinearOperator(pdomA, pcodomA, coefficients(A)), LinearOperator(pdomB, pcodomB, coefficients(B)))
            return C
        end

        function $rf!(A::LinearOperator, B::LinearOperator)
            pdomA, pdomB = _promote_space(domain(A), domain(B))
            pcodomA, pcodomB = _promote_space(codomain(A), codomain(B))
            _iscompatible(pdomA, pdomB) & _iscompatible(pcodomA, pcodomB) || return throw(ArgumentError("spaces must be compatible"))
            $_rf!(LinearOperator(pdomA, pcodomA, coefficients(A)), LinearOperator(pdomB, pcodomB, coefficients(B)))
            return A
        end

        function $lf!(A::LinearOperator, B::LinearOperator)
            pdomA, pdomB = _promote_space(domain(A), domain(B))
            pcodomA, pcodomB = _promote_space(codomain(A), codomain(B))
            _iscompatible(pdomA, pdomB) & _iscompatible(pcodomA, pcodomB) || return throw(ArgumentError("spaces must be compatible"))
            $_lf!(LinearOperator(pdomA, pcodomA, coefficients(A)), LinearOperator(pdomB, pcodomB, coefficients(B)))
            return B
        end
    end
end

function _add!(C::LinearOperator, A::LinearOperator, B::LinearOperator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == domain_B == domain_C && codomain_A == codomain_B == codomain_C
        coefficients(C) .= coefficients(A) .+ coefficients(B)
    elseif domain_A == domain_C && codomain_A == codomain_C
        coefficients(C) .= coefficients(A)
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] += getcoefficient(B, (codomain_C, α), (domain_C, β))
        end
    elseif domain_B == domain_C && codomain_B == codomain_C
        coefficients(C) .= coefficients(B)
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] += getcoefficient(A, (codomain_C, α), (domain_C, β))
        end
    else
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] = getcoefficient(A, (codomain_C, α), (domain_C, β)) + getcoefficient(B, (codomain_C, α), (domain_C, β))
        end
    end
    return C
end

function _sub!(C::LinearOperator, A::LinearOperator, B::LinearOperator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == domain_B == domain_C && codomain_A == codomain_B == codomain_C
        coefficients(C) .= coefficients(A) .- coefficients(B)
    elseif domain_A == domain_C && codomain_A == codomain_C
        coefficients(C) .= coefficients(A)
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] -= getcoefficient(B, (codomain_C, α), (domain_C, β))
        end
    elseif domain_B == domain_C && codomain_B == codomain_C
        coefficients(C) .= (-).(coefficients(B))
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] += getcoefficient(A, (codomain_C, α), (domain_C, β))
        end
    else
        @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
            C[α,β] = getcoefficient(A, (codomain_C, α), (domain_C, β)) - getcoefficient(B, (codomain_C, α), (domain_C, β))
        end
    end
    return C
end

function _radd!(A::LinearOperator, B::LinearOperator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    if domain_A == domain_B && codomain_A == codomain_B
        coefficients(A) .+= coefficients(B)
    else
        @inbounds for β ∈ indices(domain_A), α ∈ indices(codomain_A)
            A[α,β] += getcoefficient(B, (codomain_A, α), (domain_A, β))
        end
    end
    return A
end

_ladd!(A::LinearOperator, B::LinearOperator) = _radd!(B, A)

function _rsub!(A::LinearOperator, B::LinearOperator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    if domain_A == domain_B && codomain_A == codomain_B
        coefficients(A) .-= coefficients(B)
    else
        @inbounds for β ∈ indices(domain_A), α ∈ indices(codomain_A)
            A[α,β] -= getcoefficient(B, (codomain_A, α), (domain_A, β))
        end
    end
    return A
end

function _lsub!(A::LinearOperator, B::LinearOperator)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    B_ = coefficients(B)
    if domain_A == domain_B && codomain_A == codomain_B
        B_ .= coefficients(A) .- B_
    else
        B_ .= (-).(B_)
        @inbounds for β ∈ indices(domain_B), α ∈ indices(codomain_B)
            B[α,β] += getcoefficient(A, (codomain_B, α), (domain_B, β))
        end
    end
    return B
end

# Cartesian spaces

function _mul!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:CartesianSpace,<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        l = nspaces(domain_A)
        n = nspaces(codomain_A)
        m = nspaces(domain_B)
        if iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !isone(β)
            coefficients(C) .*= β
        end
        for j ∈ 1:m
            @inbounds for k ∈ 1:l
                Bₖⱼ = block(B, k, j)
                @inbounds for i ∈ 1:n
                    _mul!(block(C, i, j), block(A, i, k), Bₖⱼ, α, exact(true))
                end
            end
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, A::LinearOperator{<:VectorSpace,<:CartesianSpace}, B::LinearOperator{<:CartesianSpace,<:VectorSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        n = nspaces(codomain(A))
        m = nspaces(domain(B))
        @inbounds for j ∈ 1:m
            Bⱼ = block(B, j)
            @inbounds for i ∈ 1:n
                _mul!(block(C, i, j), block(A, i), Bⱼ, α, β)
            end
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:CartesianSpace,<:VectorSpace}, A::LinearOperator{<:CartesianSpace,<:VectorSpace}, B::LinearOperator{<:CartesianSpace,<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        l = nspaces(domain_A)
        m = nspaces(domain_B)
        if iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !isone(β)
            coefficients(C) .*= β
        end
        @inbounds for j ∈ 1:m
            Cⱼ = block(C, j)
            @inbounds for k ∈ 1:l
                _mul!(Cⱼ, block(A, k), block(B, k, j), α, exact(true))
            end
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:CartesianSpace,<:VectorSpace}, A::LinearOperator{<:VectorSpace,<:VectorSpace}, B::LinearOperator{<:CartesianSpace,<:VectorSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        @inbounds for j ∈ 1:nspaces(domain_B)
            _mul!(block(C, j), A, block(B, j), α, β)
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:VectorSpace,<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        l = nspaces(domain_A)
        n = nspaces(codomain_A)
        if iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !isone(β)
            coefficients(C) .*= β
        end
        @inbounds for k ∈ 1:l
            Bₖ = block(B, k)
            @inbounds for i ∈ 1:n
                _mul!(block(C, i), block(A, i, k), Bₖ, α, exact(true))
            end
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:VectorSpace,<:CartesianSpace}, A::LinearOperator{<:VectorSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:VectorSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        @inbounds for i ∈ 1:nspaces(codomain_A)
            _mul!(block(C, i), block(A, i), B, α, β)
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:VectorSpace,<:VectorSpace}, A::LinearOperator{<:CartesianSpace,<:VectorSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B
        if domain_B == domain_C && codomain_A == codomain_C
            mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
        else
            if iszero(β)
                coefficients(C) .= zero(eltype(C))
            elseif !isone(β)
                coefficients(C) .*= β
            end
            inds_domain = indices(domain_B ∩ domain_C)
            inds_codomain = indices(codomain_A ∩ codomain_C)
            @inbounds mul!(view(C, inds_codomain, inds_domain), view(A, inds_codomain, :), view(B, :, inds_domain), α, exact(true))
        end
    else
        if iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !isone(β)
            coefficients(C) .*= β
        end
        @inbounds for k ∈ 1:nspaces(domain_A)
            _mul!(C, block(A, k), block(B, k), α, exact(true))
        end
    end
    return C
end

for (f, _f!, _rf!, _lf!) ∈ ((:+, :_add!, :_radd!, :_ladd!), (:-, :_sub!, :_rsub!, :_lsub!))
    @eval begin
        function $_f!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:CartesianSpace,<:CartesianSpace})
            if domain(A) == domain(B) && codomain(A) == codomain(B)
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nspaces(domain(C)), i ∈ 1:nspaces(codomain(C))
                    $_f!(block(C, i, j), block(A, i, j), block(B, i, j))
                end
            end
            return C
        end

        function $_f!(C::LinearOperator{<:CartesianSpace,<:VectorSpace}, A::LinearOperator{<:CartesianSpace,<:VectorSpace}, B::LinearOperator{<:CartesianSpace,<:VectorSpace})
            if domain(A) == domain(B) && codomain(A) == codomain(B)
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nspaces(domain(C))
                    $_f!(block(C, j), block(A, j), block(B, j))
                end
            end
            return C
        end

        function $_f!(C::LinearOperator{<:VectorSpace,<:CartesianSpace}, A::LinearOperator{<:VectorSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace})
            if domain(A) == domain(B) && codomain(A) == codomain(B)
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for i ∈ 1:nspaces(codomain(C))
                    $_f!(block(C, i), block(A, i), block(B, i))
                end
            end
            return C
        end

        function $_rf!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:CartesianSpace,<:CartesianSpace})
            domain_A = domain(A)
            codomain_A = codomain(A)
            if domain_A == domain(B) && codomain_A == codomain(B)
                A_ = coefficients(A)
                A_ .= ($f).(A_, coefficients(B))
            else
                @inbounds for j ∈ 1:nspaces(domain_A), i ∈ 1:nspaces(codomain_A)
                    $_rf!(block(A, i, j), block(B, i, j))
                end
            end
            return A
        end

        function $_rf!(A::LinearOperator{<:CartesianSpace,<:VectorSpace}, B::LinearOperator{<:CartesianSpace,<:VectorSpace})
            domain_A = domain(A)
            if domain_A == domain(B) && codomain(A) == codomain(B)
                A_ = coefficients(A)
                A_ .= ($f).(A_, coefficients(B))
            else
                @inbounds for j ∈ 1:nspaces(domain_A)
                    $_rf!(block(A, j), block(B, j))
                end
            end
            return A
        end

        function $_rf!(A::LinearOperator{<:VectorSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace})
            codomain_A = codomain(A)
            if domain(A) == domain(B) && codomain_A == codomain(B)
                A_ = coefficients(A)
                A_ .= ($f).(A_, coefficients(B))
            else
                @inbounds for i ∈ 1:nspaces(codomain_A)
                    $_rf!(block(A, i), block(B, i))
                end
            end
            return A
        end

        function $_lf!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:CartesianSpace,<:CartesianSpace})
            domain_A = domain(A)
            codomain_A = codomain(A)
            if domain_A == domain(B) && codomain_A == codomain(B)
                B_ = coefficients(B)
                B_ .= ($f).(coefficients(A), B_)
            else
                @inbounds for j ∈ 1:nspaces(domain_A), i ∈ 1:nspaces(codomain_A)
                    $_lf!(block(A, i, j), block(B, i, j))
                end
            end
            return B
        end

        function $_lf!(A::LinearOperator{<:CartesianSpace,<:VectorSpace}, B::LinearOperator{<:CartesianSpace,<:VectorSpace})
            domain_A = domain(A)
            if domain_A == domain(B) && codomain(A) == codomain(B)
                B_ = coefficients(B)
                B_ .= ($f).(coefficients(A), B_)
            else
                @inbounds for j ∈ 1:nspaces(domain_A)
                    $_lf!(block(A, j), block(B, j))
                end
            end
            return B
        end

        function $_lf!(A::LinearOperator{<:VectorSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace})
            codomain_A = codomain(A)
            if domain(A) == domain(B) && codomain_A == codomain(B)
                B_ = coefficients(B)
                B_ .= ($f).(coefficients(A), B_)
            else
                @inbounds for i ∈ 1:nspaces(codomain_A)
                    $_lf!(block(A, i), block(B, i))
                end
            end
            return B
        end
    end
end

#

radd!(A::LinearOperator, J::UniformScaling) = radd!(A, UniformScalingOperator(J))
ladd!(J::UniformScaling, A::LinearOperator) = ladd!(UniformScalingOperator(J), A)
rsub!(A::LinearOperator, J::UniformScaling) = rsub!(A, UniformScalingOperator(J))
lsub!(J::UniformScaling, A::LinearOperator) = lsub!(UniformScalingOperator(J), A)

#

function radd!(A::LinearOperator, J::UniformScalingOperator)
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, A has codomain $codomain_A"))
    _radd!(A, J)
    return A
end
ladd!(J::UniformScalingOperator, A::LinearOperator) = radd!(A, J)
rsub!(A::LinearOperator, J::UniformScalingOperator) = radd!(A, -J)
function lsub!(J::UniformScalingOperator, A::LinearOperator)
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, A has codomain $codomain_A"))
    A_ = coefficients(A)
    A_ .= (-).(A_)
    _radd!(A, J)
    return A
end

function _radd!(A::LinearOperator{ScalarSpace,ScalarSpace}, J::UniformScalingOperator)
    @inbounds A[1,1] += J.λ
    return A
end

function _radd!(A::LinearOperator{<:SequenceSpace,<:SequenceSpace}, J::UniformScalingOperator)
    domain_A = domain(A)
    codomain_A = codomain(A)
    if domain_A == codomain_A
        A_ = coefficients(A)
        @inbounds for i ∈ axes(A_, 1)
            A_[i,i] += J.λ
        end
    else
        @inbounds for α ∈ indices(domain_A ∩ codomain_A)
            A[α,α] += J.λ
        end
    end
    return A
end

function _radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::UniformScalingOperator)
    domain_A = domain(A)
    if domain_A == codomain(A)
        A_ = coefficients(A)
        @inbounds for i ∈ axes(A_, 1)
            A_[i,i] += J.λ
        end
    else
        @inbounds for i ∈ 1:nspaces(domain_A)
            _radd!(block(A, i, i), J)
        end
    end
    return A
end



#

Base.:*(A::LinearOperator, B::Matrix) = A * (Projection(domain(A), eltype(A)) * B)
Base.:*(B::Matrix, A::LinearOperator) = (B * Projection(codomain(A), eltype(A))) * A

Base.:*(A::LinearOperator, B::LinearAlgebra.Diagonal) = A * Matrix(B)
Base.:*(B::LinearAlgebra.Diagonal, A::LinearOperator) = Matrix(B) * A

#

radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling}) = radd!(A, LinearAlgebra.Diagonal(UniformScalingOperator.(J.diag)))
ladd!(J::LinearAlgebra.Diagonal{<:UniformScaling}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = ladd!(LinearAlgebra.Diagonal(UniformScalingOperator.(J.diag)), A)
rsub!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling}) = rsub!(A, LinearAlgebra.Diagonal(UniformScalingOperator.(J.diag)))
lsub!(J::LinearAlgebra.Diagonal{<:UniformScaling}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = lsub!(LinearAlgebra.Diagonal(UniformScalingOperator.(J.diag)), A)

#

function radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScalingOperator})
    domain_A = domain(A)
    _iscompatible(domain_A, codomain(A)) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    _radd!(A, J)
    return A
end
ladd!(J::LinearAlgebra.Diagonal{<:UniformScalingOperator}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = radd!(A, J)
rsub!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScalingOperator}) = radd!(A, -J)
function lsub!(J::LinearAlgebra.Diagonal{<:UniformScalingOperator}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace})
    domain_A = domain(A)
    _iscompatible(domain_A, codomain(A)) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    A_ = coefficients(A)
    A_ .= (-).(A_)
    _radd!(A, J)
    return A
end

function _radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScalingOperator})
    k = 0
    @inbounds for i ∈ 1:nspaces(domain(A))
        Aᵢ = block(A, i, i)
        domain_Aᵢ = domain(Aᵢ)
        if domain_Aᵢ isa CartesianSpace
            k_ = k + 1
            k += _deep_nspaces(domain_Aᵢ)
            _radd!(Aᵢ, LinearAlgebra.Diagonal(view(J.diag, k_:k)))
        else
            k += 1
            _radd!(Aᵢ, J.diag[k])
        end
    end
    return A
end
