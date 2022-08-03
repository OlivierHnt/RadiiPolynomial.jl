# fallback methods

(A::LinearOperator)(b::Sequence) = *(A, b)

function Base.:*(A::LinearOperator, b::Sequence)
    domain_A, codomain_A = domain(A), codomain(A)
    space_b = space(b)
    _iscompatible(domain_A, space_b) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, b has space $space_b"))
    CoefType = promote_type(eltype(A), eltype(b))
    c = Sequence(codomain_A, Vector{CoefType}(undef, dimension(codomain_A)))
    _mul!(c, A, b, true, false)
    return c
end

function mul!(c::Sequence, A::LinearOperator, b::Sequence, α::Number, β::Number)
    _iscompatible(space(c), codomain(A)) & _iscompatible(domain(A), space(b)) || return throw(ArgumentError("spaces must be compatible"))
    _mul!(c, A, b, α, β)
    return c
end

function _mul!(c::Sequence, A::LinearOperator, b::Sequence, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    space_b = space(b)
    space_c = space(c)
    if domain_A == space_b
        if codomain_A == space_c
            __mul!(coefficients(c), coefficients(A), coefficients(b), α, β)
        else
            if iszero(β)
                coefficients(c) .= zero(eltype(c))
            elseif !isone(β)
                coefficients(c) .*= β
            end
            inds_space = indices(codomain_A ∩ space_c)
            @inbounds __mul!(view(c, inds_space), view(A, inds_space, :), coefficients(b), α, true)
        end
    else
        inds_mult = indices(domain_A ∩ space_b)
        if codomain_A == space_c
            @inbounds __mul!(coefficients(c), view(A, :, inds_mult), view(b, inds_mult), α, β)
        else
            if iszero(β)
                coefficients(c) .= zero(eltype(c))
            elseif !isone(β)
                coefficients(c) .*= β
            end
            inds_space = indices(codomain_A ∩ space_c)
            @inbounds __mul!(view(c, inds_space), view(A, inds_space, inds_mult), view(b, inds_mult), α, true)
        end
    end
    return c
end

function Base.:\(A::LinearOperator, b::Sequence)
    codomain_A = codomain(A)
    space_b = space(b)
    _iscompatible(codomain_A, space_b) || return throw(ArgumentError("spaces must be compatible: A has codomain $codomain_A, b has space $space_b"))
    return Sequence(domain(A), \(coefficients(A), coefficients(b)))
end

# Cartesian spaces

function _mul!(c::Sequence{<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, b::Sequence{<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    space_b = space(b)
    space_c = space(c)
    if domain_A == space_b && codomain_A == space_c
        __mul!(coefficients(c), coefficients(A), coefficients(b), α, β)
    else
        m = nspaces(domain_A)
        n = nspaces(codomain_A)
        if iszero(β)
            coefficients(c) .= zero(eltype(c))
        elseif !isone(β)
            coefficients(c) .*= β
        end
        @inbounds for j ∈ 1:m
            bⱼ = component(b, j)
            @inbounds for i ∈ 1:n
                _mul!(component(c, i), component(A, i, j), bⱼ, α, true)
            end
        end
    end
    return c
end

function _mul!(c::Sequence{<:CartesianSpace}, A::LinearOperator{<:VectorSpace,<:CartesianSpace}, b::Sequence{<:VectorSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    space_b = space(b)
    space_c = space(c)
    if domain_A == space_b && codomain_A == space_c
        __mul!(coefficients(c), coefficients(A), coefficients(b), α, β)
    else
        @inbounds for i ∈ 1:nspaces(codomain_A)
            _mul!(component(c, i), component(A, i), b, α, β)
        end
    end
    return c
end

function _mul!(c::Sequence{<:VectorSpace}, A::LinearOperator{<:CartesianSpace,<:VectorSpace}, b::Sequence{<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    space_b = space(b)
    space_c = space(c)
    if domain_A == space_b
        if codomain_A == space_c
            __mul!(coefficients(c), coefficients(A), coefficients(b), α, β)
        else
            if iszero(β)
                coefficients(c) .= zero(eltype(c))
            elseif !isone(β)
                coefficients(c) .*= β
            end
            inds_space = indices(codomain_A ∩ space_c)
            @inbounds __mul!(view(c, inds_space), view(A, inds_space, :), coefficients(b), α, true)
        end
    else
        if iszero(β)
            coefficients(c) .= zero(eltype(c))
        elseif !isone(β)
            coefficients(c) .*= β
        end
        @inbounds for j ∈ 1:nspaces(domain_A)
            _mul!(c, component(A, j), component(b, j), α, true)
        end
    end
    return c
end
