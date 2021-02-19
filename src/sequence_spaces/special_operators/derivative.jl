struct Derivative
    n :: Int
end

Base.:*(𝒟₁::Derivative, 𝒟₂::Derivative) = Derivative(𝒟₁.n + 𝒟₂.n)
Base.:^(𝒟::Derivative, n::Int) = Derivative(𝒟.n * n)

## arithmetic

+̄(𝒟::Derivative, A::Operator) = +(project(𝒟, domain(A), codomain(A), eltype(A)), A)
+̄(A::Operator, 𝒟::Derivative) = +(A, project(𝒟, domain(A), codomain(A), eltype(A)))
-̄(𝒟::Derivative, A::Operator) = -(project(𝒟, domain(A), codomain(A), eltype(A)), A)
-̄(A::Operator, 𝒟::Derivative) = -(A, project(𝒟, domain(A), codomain(A), eltype(A)))
*̄(𝒟::Derivative, A::Operator) = *(project(𝒟, codomain(A), codomain(A), eltype(A)), A)
*̄(A::Operator, 𝒟::Derivative) = *(A, project(𝒟, domain(A), domain(A), eltype(A)))

#

function project(𝒟::Derivative, domain::Taylor, codomain::Taylor, ::Type{T}=Float64) where {T}
    A = Operator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    @inbounds for i ∈ 0:min(order(domain)-𝒟.n, order(codomain))
        A[i,i+𝒟.n] = prod(i+1:i+𝒟.n)
    end
    return A
end

function project(𝒟::Derivative, domain::Fourier{T}, codomain::Fourier{S}, ::Type{R}=complex(promote_type(T, S))) where {T,S,R}
    frequency(domain) == frequency(codomain) || return throw(DomainError)
    A = Operator(domain, codomain, zeros(R, dimension(codomain), dimension(domain)))
    iⁿωⁿ = convert(R, (im*frequency(domain))^𝒟.n)
    if isodd(𝒟.n)
        @inbounds for j ∈ 1:min(order(domain), order(codomain))
            iⁿωⁿjⁿ = iⁿωⁿ*j^𝒟.n
            A[-j,-j] = -iⁿωⁿjⁿ
            A[j,j] = iⁿωⁿjⁿ
        end
        return A
    else
        @inbounds for j ∈ 1:min(order(domain), order(codomain))
            iⁿωⁿjⁿ = iⁿωⁿ*j^𝒟.n
            A[-j,-j] = iⁿωⁿjⁿ
            A[j,j] = iⁿωⁿjⁿ
        end
        return A
    end
end

function project(𝒟::Derivative, domain::Chebyshev, codomain::Chebyshev, ::Type{T}=Float64) where {T}
    @assert 𝒟.n == 1 # TODO: lift restriction
    A = Operator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    order_domain = order(domain)
    order_codomain = order(codomain)
    @inbounds for i ∈ 1:2:order_domain
        A[0,i] = 2i
    end
    for j ∈ 2:2:order_domain-1
        @inbounds for i ∈ 1:2:min(j-1, order_codomain)
            A[i,j] = 2j
        end
        @inbounds for i ∈ 2:2:min(j, order_codomain)
            A[i,j+1] = 2(j+1)
        end
    end
    if iseven(order_domain)
        @inbounds for i ∈ 1:2:min(order_domain-1, order_codomain)
            A[i,order_domain] = 2order_domain
        end
    end
    return A
end

## action

(𝒟::Derivative)(a::Sequence) = *(𝒟, a)
# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(𝒟::Derivative, a::Sequence) = differentiate(a, 𝒟.n)

function derivative_range(s::Taylor, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    s.order < n && return Taylor(0)
    return Taylor(s.order-n)
end

function derivative_range(s::Fourier, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    return s
end

function derivative_range(s::Chebyshev, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    s.order < n && return Chebyshev(0)
    return Chebyshev(s.order-n)
end

derivative_range(s::TensorSpace{<:NTuple{N,UnivariateSpace}}, dim::Int, n::Int=1) where {N} =
    TensorSpace((s.spaces[1:dim-1]..., derivative_range(s.spaces[dim], n), s.spaces[dim+1:N]...))

function differentiate(a::Sequence{Taylor}, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    CoefType = eltype(a)
    ord = order(a)
    ord < n && return Sequence(Taylor(0), [zero(CoefType)])
    c = Sequence(Taylor(ord-n), Vector{CoefType}(undef, ord-n+1))
    @inbounds c[0] = factorial(n) * a[n]
    @inbounds for i ∈ n+1:ord
        c[i-n] = prod(i-n+1:i)*a[i]
    end
    return c
end

function differentiate(a::Sequence{<:Fourier}, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    iⁿωⁿ = (im*frequency(a))^n
    CoefType = promote_type(eltype(a), typeof(iⁿωⁿ))
    c = Sequence(space(a), Vector{CoefType}(undef, dimension(space(a))))
    @inbounds c[0] = zero(CoefType)
    if isodd(n)
        @inbounds for j ∈ 1:order(a)
            iⁿωⁿjⁿ = iⁿωⁿ*j^n
            c[j] = iⁿωⁿjⁿ*a[j]
            c[-j] = -iⁿωⁿjⁿ*a[-j]
        end
        return c
    else
        @inbounds for j ∈ 1:order(a)
            iⁿωⁿjⁿ = iⁿωⁿ*j^n
            c[j] = iⁿωⁿjⁿ*a[j]
            c[-j] = iⁿωⁿjⁿ*a[-j]
        end
        return c
    end
end

function differentiate(a::Sequence{Chebyshev}, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    n > 1 && return differentiate(differentiate(a, n-1), 1)
    CoefType = eltype(a)
    ord = order(a)
    ord == 0 && return Sequence(space(a), [zero(CoefType)])
    c = Sequence(Chebyshev(ord-1), Vector{CoefType}(undef, ord))
    @inbounds c[0] = zero(CoefType)
    @inbounds for i ∈ 1:2:ord
        c[0] += 2i*a[i]
    end
    @inbounds for i ∈ 1:ord-1
        c[i] = zero(CoefType)
        @inbounds for j ∈ i+1:2:ord
            c[i] += j*a[j]
        end
        c[i] *= 2
    end
    return c
end

function differentiate(a::Sequence{<:TensorSpace}, dim::Int, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    space_a = space(a)
    A = reshape(coefficients(a), dimensions(space_a))
    return Sequence(derivative_range(space_a, dim, n), vec(_differentiate(space_a[dim], Val(dim), A, n)))
end

function _differentiate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    ord = order(space)
    ord < n && return zeros(T, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
    C = Array{T,N}(undef, ntuple(i -> i == D ? ord-n+1 : size(A, i), Val(N)))
    @inbounds selectdim(C, D, 1) .= factorial(n) .* selectdim(A, D, n+1)
    @inbounds for i ∈ n+1:ord
        selectdim(C, D, i-n+1) .= prod(i-n+1:i) .* selectdim(A, D, i+1)
    end
    return C
end

function _differentiate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    iⁿωⁿ = (im*frequency(space))^n
    CoefType = promote_type(T, typeof(iⁿωⁿ))
    C = Array{CoefType,N}(undef, size(A))
    ord = order(space)
    @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
    if isodd(n)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ = iⁿωⁿ*j^n
            selectdim(C, D, ord+1+j) .= iⁿωⁿjⁿ .* selectdim(A, D, ord+1+j)
            selectdim(C, D, ord+1-j) .= -iⁿωⁿjⁿ .* selectdim(A, D, ord+1-j)
        end
        return C
    else
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ = iⁿωⁿ*j^n
            selectdim(C, D, ord+1+j) .= iⁿωⁿjⁿ .* selectdim(A, D, ord+1+j)
            selectdim(C, D, ord+1-j) .= -iⁿωⁿjⁿ .* selectdim(A, D, ord+1-j)
        end
        return C
    end
end

function _differentiate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    @assert n == 1 # TODO: lift restriction
    ord = order(space)
    C = Array{T,N}(undef, ntuple(i -> i == D ? ord : size(A, i), Val(N)))
    @inbounds C₀ = selectdim(C, D, 1)
    C₀ .= zero(T)
    @inbounds for i ∈ 1:2:ord
        C₀ .+= (2i) .* selectdim(A, D, i+1)
    end
    @inbounds for i ∈ 1:ord-1
        Cᵢ = selectdim(C, D, i+1)
        Cᵢ .= zero(T)
        @inbounds for j ∈ i+1:2:ord
            Cᵢ .+= j .* selectdim(A, D, j+1)
        end
        @. Cᵢ *= 2
    end
    return C
end
