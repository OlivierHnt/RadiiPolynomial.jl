struct Evaluation{T}
    value :: T
end

## arithmetic

for (f̄, f) ∈ ((:+̄, :+), (:-̄, :-))
    @eval begin
        function $f̄(ℰ::Evaluation{T}, A::Operator{Taylor,ParameterSpace}) where {T}
            CoefType = promote_type(T, eltype(A))
            C = Operator(domain(A), ParameterSpace(), Matrix{CoefType}(undef, size(A)))
            xⁱ = one(CoefType)
            @inbounds C[1,0] = $f(xⁱ, A[1,0])
            @inbounds for i ∈ 1:order(domain(A))
                xⁱ *= ℰ.value
                C[1,i] = $f(xⁱ, A[1,i])
            end
            return C
        end

        function $f̄(A::Operator{Taylor,ParameterSpace}, ℰ::Evaluation{T}) where {T}
            CoefType = promote_type(T, eltype(A))
            C = Operator(domain(A), ParameterSpace(), Matrix{CoefType}(undef, size(A)))
            xⁱ = one(CoefType)
            @inbounds C[1,0] = $f(A[1,0], xⁱ)
            @inbounds for i ∈ 1:order(domain(A))
                xⁱ *= ℰ.value
                C[1,i] = $f(A[1,i], xⁱ)
            end
            return C
        end

        function $f̄(ℰ::Evaluation, A::Operator{<:Fourier,ParameterSpace})
            eiωx = cis(frequency(domain(A))*ℰ.value)
            CoefType = promote_type(typeof(eiωx), eltype(A))
            C = Operator(domain(A), ParameterSpace(), Matrix{CoefType}(undef, size(A)))
            eiωxj = one(CoefType)
            @inbounds C[1,0] = $f(eiωxj, A[1,0])
            @inbounds for j ∈ 1:order(domain(A))
                eiωxj *= eiωx
                C[1,-j] = $f(conj(eiωxj), A[1,-j])
                C[1,j] = $f(eiωxj, A[1,j])
            end
            return C
        end

        function $f̄(A::Operator{<:Fourier,ParameterSpace}, ℰ::Evaluation)
            eiωx = cis(frequency(domain(A))*ℰ.value)
            CoefType = promote_type(typeof(eiωx), eltype(A))
            C = Operator(domain(A), ParameterSpace(), Matrix{CoefType}(undef, size(A)))
            eiωxj = one(CoefType)
            @inbounds C[1,0] = $f(A[1,0], eiωxj)
            @inbounds for j ∈ 1:order(domain(A))
                eiωxj *= eiωx
                C[1,-j] = $f(A[1,-j], conj(eiωxj))
                C[1,j] = $f(A[1,j], eiωxj)
            end
            return C
        end

        function $f̄(ℰ::Evaluation{T}, A::Operator{Chebyshev,ParameterSpace}) where {T}
            CoefType = promote_type(T, eltype(A))
            C = Operator(domain(A), ParameterSpace(), Matrix{CoefType}(undef, size(A)))
            ord = order(domain(A))
            @inbounds C[1,0] = one(CoefType)
            if ord > 0
                x2 = 2ℰ.value
                @inbounds C[1,1] = x2
                if ord > 1
                    @inbounds C[1,2] = x2*C[1,1] - 2C[1,0]
                    @inbounds for j ∈ 3:ord
                        C[1,j] = x2*C[1,j-1] - C[1,j-2]
                    end
                end
            end
            @. C.coefficients .= $f(C.coefficients, A.coefficients)
            return C
        end

        function $f̄(A::Operator{Chebyshev,ParameterSpace}, ℰ::Evaluation{T}) where {T}
            CoefType = promote_type(T, eltype(A))
            C = Operator(domain(A), ParameterSpace(), Matrix{CoefType}(undef, size(A)))
            ord = order(domain(A))
            @inbounds C[1,0] = one(CoefType)
            if ord > 0
                x2 = 2ℰ.value
                @inbounds C[1,1] = x2
                if ord > 1
                    @inbounds C[1,2] = x2*C[1,1] - 2C[1,0]
                    @inbounds for j ∈ 3:ord
                        C[1,j] = x2*C[1,j-1] - C[1,j-2]
                    end
                end
            end
            @. C.coefficients .= $f(A.coefficients, C.coefficients)
            return C
        end
    end
end

function +̄(ℰ::Evaluation{T}, A::Operator{Taylor,<:SequenceSpace}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    idx = _constant_index(codomain(A))
    xⁱ = one(CoefType)
    @inbounds C[idx,0] += xⁱ
    @inbounds for i ∈ 1:order(domain(A))
        xⁱ *= ℰ.value
        C[idx,i] += xⁱ
    end
    return C
end

+̄(A::Operator{Taylor,<:SequenceSpace}, ℰ::Evaluation{T}) where {T} = +̄(ℰ, A)

function -̄(ℰ::Evaluation{T}, A::Operator{Taylor,<:SequenceSpace}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = -A.coefficients
    idx = _constant_index(codomain(A))
    xⁱ = one(CoefType)
    @inbounds C[idx,0] += xⁱ
    @inbounds for i ∈ 1:order(domain(A))
        xⁱ *= ℰ.value
        C[idx,i] += xⁱ
    end
    return C
end

function -̄(A::Operator{Taylor,<:SequenceSpace}, ℰ::Evaluation{T}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    idx = _constant_index(codomain(A))
    xⁱ = one(CoefType)
    @inbounds C[idx,0] -= xⁱ
    @inbounds for i ∈ 1:order(domain(A))
        xⁱ *= ℰ.value
        C[idx,i] -= xⁱ
    end
    return C
end

function +̄(ℰ::Evaluation, A::Operator{<:Fourier,<:SequenceSpace})
    eiωx = cis(frequency(domain(A))*ℰ.value)
    CoefType = promote_type(typeof(eiωx), eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    idx = _constant_index(codomain(A))
    eiωxj = one(CoefType)
    @inbounds C[idx,0] += eiωxj
    @inbounds for j ∈ 1:order(domain(A))
        eiωxj *= eiωx
        C[idx,-j] += conj(eiωxj)
        C[idx,j] += eiωxj
    end
    return C
end

+̄(A::Operator{<:Fourier,<:SequenceSpace}, ℰ::Evaluation) = +̄(ℰ, A)

function -̄(ℰ::Evaluation, A::Operator{<:Fourier,<:SequenceSpace})
    eiωx = cis(frequency(domain(A))*ℰ.value)
    CoefType = promote_type(typeof(eiωx), eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = -A.coefficients
    idx = _constant_index(codomain(A))
    eiωxj = one(CoefType)
    @inbounds C[idx,0] += eiωxj
    @inbounds for j ∈ 1:order(domain(A))
        eiωxj *= eiωx
        C[idx,-j] += conj(eiωxj)
        C[idx,j] += eiωxj
    end
    return C
end

function -̄(A::Operator{<:Fourier,<:SequenceSpace}, ℰ::Evaluation)
    eiωx = cis(frequency(domain(A))*ℰ.value)
    CoefType = promote_type(typeof(eiωx), eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    idx = _constant_index(codomain(A))
    eiωxj = one(CoefType)
    @inbounds C[idx,0] -= eiωxj
    @inbounds for j ∈ 1:order(domain(A))
        eiωxj *= eiωx
        C[idx,-j] -= conj(eiωxj)
        C[idx,j] -= eiωxj
    end
    return C
end

function +̄(ℰ::Evaluation{T}, A::Operator{Chebyshev,<:SequenceSpace}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    idx = _constant_index(codomain(A))
    ord = order(domain(A))
    @inbounds C[idx,0] = one(CoefType)
    if ord > 0
        x2 = 2ℰ.value
        @inbounds C[idx,1] = x2
        if ord > 1
            @inbounds C[idx,2] = x2*C[idx,1] - 2C[idx,0]
            @inbounds for j ∈ 3:ord
                C[idx,j] = x2*C[idx,j-1] - C[idx,j-2]
            end
        end
    end
    @inbounds view(C, idx, :) .+= view(A, idx, :)
    return C
end

+̄(A::Operator{Chebyshev,<:SequenceSpace}, ℰ::Evaluation{T}) where {T} = +̄(ℰ, A)

function -̄(ℰ::Evaluation{T}, A::Operator{Chebyshev,<:SequenceSpace}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = -A.coefficients
    idx = _constant_index(codomain(A))
    ord = order(domain(A))
    @inbounds C[idx,0] = one(CoefType)
    if ord > 0
        x2 = 2ℰ.value
        @inbounds C[idx,1] = x2
        if ord > 1
            @inbounds C[idx,2] = x2*C[idx,1] - 2C[idx,0]
            @inbounds for j ∈ 3:ord
                C[idx,j] = x2*C[idx,j-1] - C[idx,j-2]
            end
        end
    end
    @inbounds view(C, idx, :) .-= view(A, idx, :)
    return C
end

function -̄(A::Operator{Chebyshev,<:SequenceSpace}, ℰ::Evaluation{T}) where {T}
    CoefType = promote_type(T, eltype(A))
    C = Operator(domain(A), codomain(A), Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    idx = _constant_index(codomain(A))
    ord = order(domain(A))
    @inbounds C[idx,0] = -one(CoefType)
    if ord > 0
        x2 = 2ℰ.value
        @inbounds C[idx,1] = -x2
        if ord > 1
            @inbounds C[idx,2] = x2*C[idx,1] - 2C[idx,0]
            @inbounds for j ∈ 3:ord
                C[idx,j] = x2*C[idx,j-1] - C[idx,j-2]
            end
        end
    end
    @inbounds view(C, idx, :) .+= view(A, idx, :)
    return C
end

#

function project(ℰ::Evaluation{T}, domain::Taylor) where {T}
    A = Operator(domain, ParameterSpace(), Matrix{T}(undef, 1, dimension(domain)))
    xⁱ = one(T)
    @inbounds A[1,0] = xⁱ
    @inbounds for i ∈ 1:order(domain)
        xⁱ *= ℰ.value
        A[1,i] = xⁱ
    end
    return A
end

function project(ℰ::Evaluation, domain::Fourier)
    eiωx = cis(frequency(domain)*ℰ.value)
    CoefType = typeof(eiωx)
    A = Operator(domain, ParameterSpace(), Matrix{CoefType}(undef, 1, dimension(domain)))
    eiωxj = one(CoefType)
    @inbounds A[1,0] = eiωxj
    @inbounds for j ∈ 1:order(domain)
        eiωxj *= eiωx
        A[1,-j] = conj(eiωxj)
        A[1,j] = eiωxj
    end
    return A
end

function project(ℰ::Evaluation{T}, domain::Chebyshev) where {T}
    A = Operator(domain, ParameterSpace(), Matrix{T}(undef, 1, dimension(domain)))
    ord = order(domain)
    @inbounds A[1,0] = one(T)
    ord == 0 && return A
    x2 = 2ℰ.value
    @inbounds A[1,1] = x2
    ord == 1 && return A
    @inbounds A[1,2] = x2*A[1,1] - 2A[1,0]
    @inbounds for j ∈ 3:ord
        A[1,j] = x2*A[1,j-1] - A[1,j-2]
    end
    return A
end

## action

(ℰ::Evaluation)(a::Sequence) = *(ℰ, a)
# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(ℰ::Evaluation, a::Sequence) = evaluate(a, ℰ.value)

(a::Sequence{<:UnivariateSpace})(x) = evaluate(a, x)
(a::Sequence{<:TensorSpace})(x; dims=:) = evaluate(a, x, dims)
(a::Sequence{<:TensorSpace})(x...; dims=:) = evaluate(a, x, dims)

function evaluate(a::Sequence{Taylor}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    iszero(x) && return @inbounds convert(CoefType, a[0])
    ord = order(a)
    @inbounds s = convert(CoefType, a[ord])
    @inbounds for i ∈ ord-1:-1:0
        s = s * x + a[i]
    end
    return s
end

function evaluate(a::Sequence{<:Fourier}, x)
    ord = order(a)
    eiωx = cis(frequency(a)*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(eltype(a), typeof(eiωx))
    @inbounds s₊ = convert(CoefType, a[ord])
    @inbounds s₋ = convert(CoefType, a[-ord])
    @inbounds for j ∈ ord-1:-1:1
        s₊ = s₊ * eiωx + a[j]
        s₋ = s₋ * eiωx_conj + a[-j]
    end
    return @inbounds s₋ * eiωx_conj + a[0] + s₊ * eiωx
end

function evaluate(a::Sequence{Chebyshev}, x)
    CoefType = promote_type(eltype(a), typeof(x))
    ord = order(a)
    x2 = 2x
    result = s = zero(CoefType)
    @inbounds t = convert(CoefType, 2a[ord])
    @inbounds for i ∈ ord-1:-1:1
        result = t
        t = x2 * t - s + 2a[i]
        s = result
    end
    @inbounds result = x * t - s + a[0]
    return result
end

function evaluate(a::Sequence{<:TensorSpace}, x, dims=:)
    space_a = space(a)
    A = reshape(coefficients(a), dimensions(space_a))
    return _evaluate(space_a, dims, A, x)
end

_evaluate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, A::AbstractArray{T,N}, x) where {N,T} =
    Sequence(TensorSpace((space[1:dims-1]..., space[dims+1:N]...)), vec(_evaluate(space[dims], Val(dims), A, x)))
function _evaluate(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, dims::Int, A::AbstractArray{T,2}, x) where {T}
    space_ = dims == 1 ? space[2] : space[1]
    return Sequence(space_, vec(_evaluate(space[dims], Val(dims), A, x)))
end
_evaluate(space, ::Colon, A, x) = _apply_evaluate(space, A, x)[1]

_apply_evaluate(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, x::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _evaluate(space[1], Val(N₂-N₁+1), _apply_evaluate(Base.tail(space), A, Base.tail(x)), x[1])

_apply_evaluate(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, x::NTuple{2,Any}) where {T,N} =
    @inbounds _evaluate(space[1], Val(N-1), _evaluate(space[2], Val(N), A, x[2]), x[1])

function _evaluate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    iszero(x) && return @inbounds convert(Array{CoefType,N}, selectdim(A, D, 1:1))
    ord = order(space)
    @inbounds s = convert(Array{CoefType,N}, selectdim(A, D, ord+1:ord+1))
    @inbounds for i ∈ ord-1:-1:0
        Aᵢ = selectdim(A, D, i+1:i+1)
        @. s = s * x + Aᵢ
    end
    return s
end

function _evaluate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, x) where {D,T,N}
    ord = order(space)
    eiωx = cis(frequency(space)*x)
    eiωx_conj = conj(eiωx)
    CoefType = promote_type(T, typeof(eiωx))
    @inbounds s₊ = convert(Array{CoefType,N}, selectdim(A, D, 2ord+1:2ord+1))
    @inbounds s₋ = convert(Array{CoefType,N}, selectdim(A, D, 1:1))
    @inbounds for j ∈ ord-1:-1:1
        Aⱼ = selectdim(A, D, ord+1+j:ord+1+j)
        A₋ⱼ = selectdim(A, D, ord+1-j:ord+1-j)
        @. s₊ = s₊ * eiωx + Aⱼ
        @. s₋ = s₋ * eiωx_conj + A₋ⱼ
    end
    @inbounds A₀ = selectdim(A, D, ord+1:ord+1)
    @. s₊ = s₋ * eiωx_conj + A₀ + s₊ * eiωx
    return s₊
end

function _evaluate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, x::S) where {D,T,N,S}
    CoefType = promote_type(T, S)
    ord = order(space)
    x2 = 2x
    result = Array{CoefType,N}(undef, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
    s = zeros(CoefType, axes(result))
    t = similar(result)
    @inbounds t .= 2 .* selectdim(A, D, ord+1:ord+1)
    @inbounds for i ∈ ord-1:-1:1
        Aᵢ = selectdim(A, D, i+1:i+1)
        result .= t
        @. t = x2 * t - s + 2Aᵢ
        s .= result
    end
    @inbounds A₀ = selectdim(A, D, 1:1)
    @. result = x * t - s + A₀
    return result
end
