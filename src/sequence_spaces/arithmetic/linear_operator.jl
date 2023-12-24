# fallback methods

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
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    _iscompatible(domain_A, codomain_B) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, B has codomain $codomain_B"))
    CoefType = promote_type(eltype(A), eltype(B))
    C = LinearOperator(domain_B, codomain_A, Matrix{CoefType}(undef, dimension(codomain_A), dimension(domain_B)))
    _mul!(C, A, B, _safe_convert(real(CoefType), true), _safe_convert(real(CoefType), false))
    return C
end

function mul!(C::LinearOperator, A::LinearOperator, B::LinearOperator, α::Number, β::Number)
    _iscompatible(domain(C), domain(B)) & _iscompatible(codomain(C), codomain(A)) & _iscompatible(domain(A), codomain(B)) || return throw(ArgumentError("spaces must be compatible"))
    _mul!(C, A, B, α, β)
    return C
end

function _mul!(C::LinearOperator, A::LinearOperator, B::LinearOperator, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B
        if domain_B == domain_C && codomain_A == codomain_C
            __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
        else
            if _safe_iszero(β)
                coefficients(C) .= zero(eltype(C))
            elseif !_safe_isone(β)
                coefficients(C) .*= β
            end
            inds_domain = indices(domain_B ∩ domain_C)
            inds_codomain = indices(codomain_A ∩ codomain_C)
            @inbounds __mul!(view(C, inds_codomain, inds_domain), view(A, inds_codomain, :), view(B, :, inds_domain), α, _safe_convert(real(eltype(C)), true))
        end
    else
        inds_mult = indices(domain_A ∩ codomain_B)
        if domain_B == domain_C && codomain_A == codomain_C
            @inbounds __mul!(coefficients(C), view(A, :, inds_mult), view(B, inds_mult, :), α, β)
        else
            if _safe_iszero(β)
                coefficients(C) .= zero(eltype(C))
            elseif !_safe_isone(β)
                coefficients(C) .*= β
            end
            inds_domain = indices(domain_B ∩ domain_C)
            inds_codomain = indices(codomain_A ∩ codomain_C)
            @inbounds __mul!(view(C, inds_codomain, inds_domain), view(A, inds_codomain, inds_mult), view(B, inds_mult, inds_domain), α, _safe_convert(real(eltype(C)), true))
        end
    end
    return C
end

function rmul!(A::LinearOperator, B::LinearOperator)
    domain_A = domain(A)
    codomain_B = codomain(B)
    _iscompatible(domain_A, codomain_B) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, B has codomain $codomain_B"))
    return LinearOperator(domain(B), codomain(A), rmul!(coefficients(A), coefficients(B)))
end

function lmul!(A::LinearOperator, B::LinearOperator)
    domain_A = domain(A)
    codomain_B = codomain(B)
    _iscompatible(domain_A, codomain_B) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, B has codomain $codomain_B"))
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
    domain_A = domain(A)
    domain_B = domain(B)
    _iscompatible(domain_A, domain_B) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, B has domain $domain_B"))
    return LinearOperator(codomain(B), codomain(A), /(coefficients(A), coefficients(B)))
end

function Base.:\(A::LinearOperator, B::LinearOperator)
    codomain_A = codomain(A)
    codomain_B = codomain(B)
    _iscompatible(codomain_A, codomain_B) || return throw(ArgumentError("spaces must be compatible: A has codomain $codomain_A, B has codomain $codomain_B"))
    return LinearOperator(domain(B), domain(A), \(coefficients(A), coefficients(B)))
end

for (f, f!, rf!, lf!, _f!, _rf!, _lf!) ∈ ((:(Base.:+), :add!, :radd!, :ladd!, :_add!, :_radd!, :_ladd!),
        (:(Base.:-), :sub!, :rsub!, :lsub!, :_sub!, :_rsub!, :_lsub!))
    @eval begin
        function $f(A::LinearOperator, B::LinearOperator)
            domain_A, codomain_A = domain(A), codomain(A)
            domain_B, codomain_B = domain(B), codomain(B)
            new_domain, new_codomain = image($f, domain_A, domain_B), image($f, codomain_A, codomain_B)
            CoefType = promote_type(eltype(A), eltype(B))
            C = LinearOperator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
            $_f!(C, A, B)
            return C
        end

        function $f!(C::LinearOperator, A::LinearOperator, B::LinearOperator)
            _iscompatible(domain(C), image($f, domain(A), domain(B))) & _iscompatible(codomain(C), image($f, codomain(A), codomain(B))) || return throw(ArgumentError("spaces must be compatible"))
            $_f!(C, A, B)
            return C
        end

        function $rf!(A::LinearOperator, B::LinearOperator)
            domain_A = domain(A)
            codomain_A = codomain(A)
            _iscompatible(domain_A, image($f, domain_A, domain(B))) & _iscompatible(codomain_A, image($f, codomain_A, codomain(B))) || return throw(ArgumentError("spaces must be compatible"))
            $_rf!(A, B)
            return A
        end

        function $lf!(A::LinearOperator, B::LinearOperator)
            domain_B = domain(B)
            codomain_B = codomain(B)
            _iscompatible(domain_B, image($f, domain(A), domain_B)) & _iscompatible(codomain_B, image($f, codomain(A), codomain_B)) || return throw(ArgumentError("spaces must be compatible"))
            $_lf!(A, B)
            return B
        end
    end
end

for (f, _f!) ∈ ((:add_bar, :_add!), (:sub_bar, :_sub!))
    @eval function $f(A::LinearOperator, B::LinearOperator)
        domain_A, codomain_A = domain(A), codomain(A)
        domain_B, codomain_B = domain(B), codomain(B)
        new_domain, new_codomain = image($f, domain_A, domain_B), image($f, codomain_A, codomain_B)
        CoefType = promote_type(eltype(A), eltype(B))
        C = LinearOperator(new_domain, new_codomain, Matrix{CoefType}(undef, dimension(new_codomain), dimension(new_domain)))
        $_f!(C, A, B)
        return C
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
        @inbounds for β ∈ indices(domain_B ∩ domain_C), α ∈ indices(codomain_B ∩ codomain_C)
            C[α,β] += B[α,β]
        end
    elseif domain_B == domain_C && codomain_B == codomain_C
        coefficients(C) .= coefficients(B)
        @inbounds for β ∈ indices(domain_A ∩ domain_C), α ∈ indices(codomain_A ∩ codomain_C)
            C[α,β] += A[α,β]
        end
    else
        coefficients(C) .= zero(eltype(C))
        @inbounds for β ∈ indices(domain_A ∩ domain_C), α ∈ indices(codomain_A ∩ codomain_C)
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ indices(domain_B ∩ domain_C), α ∈ indices(codomain_B ∩ codomain_C)
            C[α,β] += B[α,β]
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
        @inbounds for β ∈ indices(domain_B ∩ domain_C), α ∈ indices(codomain_B ∩ codomain_C)
            C[α,β] -= B[α,β]
        end
    elseif domain_B == domain_C && codomain_B == codomain_C
        coefficients(C) .= (-).(coefficients(B))
        @inbounds for β ∈ indices(domain_A ∩ domain_C), α ∈ indices(codomain_A ∩ codomain_C)
            C[α,β] += A[α,β]
        end
    else
        coefficients(C) .= zero(eltype(C))
        @inbounds for β ∈ indices(domain_A ∩ domain_C), α ∈ indices(codomain_A ∩ codomain_C)
            C[α,β] = A[α,β]
        end
        @inbounds for β ∈ indices(domain_B ∩ domain_C), α ∈ indices(codomain_B ∩ codomain_C)
            C[α,β] -= B[α,β]
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
        @inbounds for β ∈ indices(domain_A ∩ domain_B), α ∈ indices(codomain_A ∩ codomain_B)
            A[α,β] += B[α,β]
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
        @inbounds for β ∈ indices(domain_A ∩ domain_B), α ∈ indices(codomain_A ∩ codomain_B)
            A[α,β] -= B[α,β]
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
        @inbounds for β ∈ indices(domain_A ∩ domain_B), α ∈ indices(codomain_A ∩ codomain_B)
            B[α,β] += A[α,β]
        end
    end
    return B
end

# Parameter space

Base.:+(A::LinearOperator{ParameterSpace,ParameterSpace}, J::UniformScaling) =
    @inbounds LinearOperator(domain(A), codomain(A), [A[1,1] + J.λ;;])
Base.:+(J::UniformScaling, A::LinearOperator{ParameterSpace,ParameterSpace}) =
    @inbounds LinearOperator(domain(A), codomain(A), [J.λ + A[1,1];;])
Base.:-(A::LinearOperator{ParameterSpace,ParameterSpace}, J::UniformScaling) =
    @inbounds LinearOperator(domain(A), codomain(A), [A[1,1] - J.λ;;])
Base.:-(J::UniformScaling, A::LinearOperator{ParameterSpace,ParameterSpace}) =
    @inbounds LinearOperator(domain(A), codomain(A), [J.λ - A[1,1];;])

radd!(A::LinearOperator{ParameterSpace,ParameterSpace}, J::UniformScaling) = _radd!(A, J)
ladd!(J::UniformScaling, A::LinearOperator{ParameterSpace,ParameterSpace}) = _radd!(A, J)
rsub!(A::LinearOperator{ParameterSpace,ParameterSpace}, J::UniformScaling) = _radd!(A, -J)
function lsub!(J::UniformScaling, A::LinearOperator{ParameterSpace,ParameterSpace})
    @inbounds A[1,1] = J.λ - A[1,1]
    return A
end

function _radd!(A::LinearOperator{ParameterSpace,ParameterSpace}, J::UniformScaling)
    @inbounds A[1,1] += J.λ
    return A
end

# Sequence spaces

function Base.:+(A::LinearOperator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling)
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    _radd!(C, J)
    return C
end
Base.:+(J::UniformScaling, A::LinearOperator{<:SequenceSpace,<:SequenceSpace}) = +(A, J)
Base.:-(A::LinearOperator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling) = +(A, -J)
function Base.:-(J::UniformScaling, A::LinearOperator{<:SequenceSpace,<:SequenceSpace})
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= (-).(coefficients(A))
    _radd!(C, J)
    return C
end

radd!(A::LinearOperator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling) = _radd!(A, J)
ladd!(J::UniformScaling, A::LinearOperator{<:SequenceSpace,<:SequenceSpace}) = _radd!(A, J)
rsub!(A::LinearOperator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling) = _radd!(A, -J)
function lsub!(J::UniformScaling, A::LinearOperator{<:SequenceSpace,<:SequenceSpace})
    A_ = coefficients(A)
    A_ .= (-).(A_)
    _radd!(A, J)
    return A
end

function _radd!(A::LinearOperator{<:SequenceSpace,<:SequenceSpace}, J::UniformScaling)
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

# Cartesian spaces

function _mul!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:CartesianSpace,<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        l = nspaces(domain_A)
        n = nspaces(codomain_A)
        m = nspaces(domain_B)
        if _safe_iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !_safe_isone(β)
            coefficients(C) .*= β
        end
        for j ∈ 1:m
            @inbounds for k ∈ 1:l
                Bₖⱼ = component(B, k, j)
                @inbounds for i ∈ 1:n
                    _mul!(component(C, i, j), component(A, i, k), Bₖⱼ, α, _safe_convert(real(eltype(C)), true))
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
        __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        n = nspaces(codomain(A))
        m = nspaces(domain(B))
        @inbounds for j ∈ 1:m
            Bⱼ = component(B, j)
            @inbounds for i ∈ 1:n
                _mul!(component(C, i, j), component(A, i), Bⱼ, α, β)
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
        __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        l = nspaces(domain_A)
        m = nspaces(domain_B)
        if _safe_iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !_safe_isone(β)
            coefficients(C) .*= β
        end
        @inbounds for j ∈ 1:m
            Cⱼ = component(C, j)
            @inbounds for k ∈ 1:l
                _mul!(Cⱼ, component(A, k), component(B, k, j), α, _safe_convert(real(eltype(C)), true))
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
        __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        @inbounds for j ∈ 1:nspaces(domain_B)
            _mul!(component(C, j), A, component(B, j), α, β)
        end
    end
    return C
end

function _mul!(C::LinearOperator{<:VectorSpace,<:CartesianSpace}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace}, α::Number, β::Number)
    domain_A, codomain_A = domain(A), codomain(A)
    domain_B, codomain_B = domain(B), codomain(B)
    domain_C, codomain_C = domain(C), codomain(C)
    if domain_A == codomain_B && domain_B == domain_C && codomain_A == codomain_C
        __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        l = nspaces(domain_A)
        n = nspaces(codomain_A)
        if _safe_iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !_safe_isone(β)
            coefficients(C) .*= β
        end
        @inbounds for k ∈ 1:l
            Bₖ = component(B, k)
            @inbounds for i ∈ 1:n
                _mul!(component(C, i), component(A, i, k), Bₖ, α, _safe_convert(real(eltype(C)), true))
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
        __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
    else
        @inbounds for i ∈ 1:nspaces(codomain_A)
            _mul!(component(C, i), component(A, i), B, α, β)
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
            __mul!(coefficients(C), coefficients(A), coefficients(B), α, β)
        else
            if _safe_iszero(β)
                coefficients(C) .= zero(eltype(C))
            elseif !_safe_isone(β)
                coefficients(C) .*= β
            end
            inds_domain = indices(domain_B ∩ domain_C)
            inds_codomain = indices(codomain_A ∩ codomain_C)
            @inbounds __mul!(view(C, inds_codomain, inds_domain), view(A, inds_codomain, :), view(B, :, inds_domain), α, _safe_convert(real(eltype(C)), true))
        end
    else
        if _safe_iszero(β)
            coefficients(C) .= zero(eltype(C))
        elseif !_safe_isone(β)
            coefficients(C) .*= β
        end
        @inbounds for k ∈ 1:nspaces(domain_A)
            _mul!(C, component(A, k), component(B, k), α, _safe_convert(real(eltype(C)), true))
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
                    $_f!(component(C, i, j), component(A, i, j), component(B, i, j))
                end
            end
            return C
        end

        function $_f!(C::LinearOperator{<:CartesianSpace,<:VectorSpace}, A::LinearOperator{<:CartesianSpace,<:VectorSpace}, B::LinearOperator{<:CartesianSpace,<:VectorSpace})
            if domain(A) == domain(B) && codomain(A) == codomain(B)
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for j ∈ 1:nspaces(domain(C))
                    $_f!(component(C, j), component(A, j), component(B, j))
                end
            end
            return C
        end

        function $_f!(C::LinearOperator{<:VectorSpace,<:CartesianSpace}, A::LinearOperator{<:VectorSpace,<:CartesianSpace}, B::LinearOperator{<:VectorSpace,<:CartesianSpace})
            if domain(A) == domain(B) && codomain(A) == codomain(B)
                coefficients(C) .= ($f).(coefficients(A), coefficients(B))
            else
                @inbounds for i ∈ 1:nspaces(codomain(C))
                    $_f!(component(C, i), component(A, i), component(B, i))
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
                    $_rf!(component(A, i, j), component(B, i, j))
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
                    $_rf!(component(A, j), component(B, j))
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
                    $_rf!(component(A, i), component(B, i))
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
                    $_lf!(component(A, i, j), component(B, i, j))
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
                    $_lf!(component(A, j), component(B, j))
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
                    $_lf!(component(A, i), component(B, i))
                end
            end
            return B
        end
    end
end

#

function Base.:+(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, A has codomain $codomain_A"))
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    _radd!(C, J)
    return C
end
Base.:+(J::UniformScaling, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = +(A, J)
Base.:-(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling) = +(A, -J)
function Base.:-(J::UniformScaling, A::LinearOperator{<:CartesianSpace,<:CartesianSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, A has codomain $codomain_A"))
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= (-).(coefficients(A))
    _radd!(C, J)
    return C
end

function radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, A has codomain $codomain_A"))
    _radd!(A, J)
    return A
end
ladd!(J::UniformScaling, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = radd!(A, J)
rsub!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling) = radd!(A, -J)
function lsub!(J::UniformScaling, A::LinearOperator{<:CartesianSpace,<:CartesianSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) || return throw(ArgumentError("spaces must be compatible: A has domain $domain_A, A has codomain $codomain_A"))
    A_ = coefficients(A)
    A_ .= (-).(A_)
    _radd!(A, J)
    return A
end

function _radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::UniformScaling)
    domain_A = domain(A)
    if domain_A == codomain(A)
        A_ = coefficients(A)
        @inbounds for i ∈ axes(A_, 1)
            A_[i,i] += J.λ
        end
    else
        @inbounds for i ∈ 1:nspaces(domain_A)
            _radd!(component(A, i, i), J)
        end
    end
    return A
end

#

function Base.:+(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling})
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    _radd!(C, J)
    return C
end
Base.:+(J::LinearAlgebra.Diagonal{<:UniformScaling}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = +(A, J)
function Base.:-(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling})
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= coefficients(A)
    _rsub!(C, J)
    return C
end
function Base.:-(J::LinearAlgebra.Diagonal{<:UniformScaling}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace})
    domain_A = domain(A)
    codomain_A = codomain(A)
    _iscompatible(domain_A, codomain_A) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    CoefType = promote_type(eltype(A), eltype(J))
    C = LinearOperator(domain_A, codomain_A, Matrix{CoefType}(undef, size(A)))
    coefficients(C) .= (-).(coefficients(A))
    _radd!(C, J)
    return C
end

function radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling})
    domain_A = domain(A)
    _iscompatible(domain_A, codomain(A)) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    _radd!(A, J)
    return A
end
ladd!(J::LinearAlgebra.Diagonal{<:UniformScaling}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace}) = radd!(A, J)
function rsub!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling})
    domain_A = domain(A)
    _iscompatible(domain_A, codomain(A)) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    _rsub!(A, J)
    return A
end
function lsub!(J::LinearAlgebra.Diagonal{<:UniformScaling}, A::LinearOperator{<:CartesianSpace,<:CartesianSpace})
    domain_A = domain(A)
    _iscompatible(domain_A, codomain(A)) & (_deep_nspaces(domain_A) == length(J.diag)) || return throw(ArgumentError("spaces must be compatible"))
    A_ = coefficients(A)
    A_ .= (-).(A_)
    _radd!(A, J)
    return A
end

function _radd!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling})
    k = 0
    @inbounds for i ∈ 1:nspaces(domain(A))
        Aᵢ = component(A, i, i)
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
function _rsub!(A::LinearOperator{<:CartesianSpace,<:CartesianSpace}, J::LinearAlgebra.Diagonal{<:UniformScaling})
    k = 0
    @inbounds for i ∈ 1:nspaces(domain(A))
        Aᵢ = component(A, i, i)
        domain_Aᵢ = domain(Aᵢ)
        if domain_Aᵢ isa CartesianSpace
            k_ = k + 1
            k += _deep_nspaces(domain_Aᵢ)
            _rsub!(Aᵢ, LinearAlgebra.Diagonal(view(J.diag, k_:k)))
        else
            k += 1
            _radd!(Aᵢ, -J.diag[k])
        end
    end
    return A
end
