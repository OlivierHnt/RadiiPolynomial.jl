struct Integral
    n :: Int
end

Base.:*(ℐ₁::Integral, ℐ₂::Integral) = Integral(ℐ₁.n + ℐ₂.n)
Base.:^(ℐ::Integral, n::Int) = Integral(ℐ.n * n)

## arithmetic

+̄(ℐ::Integral, A::Operator) = +(project(ℐ, domain(A), codomain(A), eltype(A)), A)
+̄(A::Operator, ℐ::Integral) = +(A, project(ℐ, domain(A), codomain(A), eltype(A)))
-̄(ℐ::Integral, A::Operator) = -(project(ℐ, domain(A), codomain(A), eltype(A)), A)
-̄(A::Operator, ℐ::Integral) = -(A, project(ℐ, domain(A), codomain(A), eltype(A)))
*̄(ℐ::Integral, A::Operator) = *(project(ℐ, codomain(A), codomain(A), eltype(A)), A)
*̄(A::Operator, ℐ::Integral) = *(A, project(ℐ, domain(A), domain(A), eltype(A)))

#

function project(ℐ::Integral, domain::Taylor, codomain::Taylor, ::Type{T}=Float64) where {T}
    A = Operator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    @inbounds for i ∈ 0:min(order(domain), order(codomain)-ℐ.n)
        A[i+ℐ.n,i] = one(T)/(prod(i+1:i+ℐ.n))
    end
    return A
end

function project(ℐ::Integral, domain::Fourier{T}, codomain::Fourier{S}, ::Type{R}=complex(float(promote_type(T, S)))) where {T,S,R}
    frequency(domain) == frequency(codomain) || return throw(DomainError)
    A = Operator(domain, codomain, zeros(R, dimension(codomain), dimension(domain)))
    iⁿω⁻ⁿ = convert(R, (im*inv(frequency(domain)))^ℐ.n)
    if isodd(ℐ.n)
        @inbounds for j ∈ 1:min(order(domain), order(codomain))
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^ℐ.n
            A[-j,-j] = iⁿω⁻ⁿj⁻ⁿ
            A[j,j] = -iⁿω⁻ⁿj⁻ⁿ
        end
        return A
    else
        @inbounds for j ∈ 1:min(order(domain), order(codomain))
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^ℐ.n
            A[-j,-j] = iⁿω⁻ⁿj⁻ⁿ
            A[j,j] = iⁿω⁻ⁿj⁻ⁿ
        end
        return A
    end
end

function project(ℐ::Integral, domain::Chebyshev, codomain::Chebyshev, ::Type{T}=Float64) where {T}
    @assert ℐ.n == 1 # TODO: lift restriction
    A = Operator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    order_domain = order(domain)
    order_codomain = order(codomain)
    # first two columns
    @inbounds A[0,0] = one(T)
    if order_domain ≥ 1
        @inbounds A[0,1] = -one(T)/2
        if order_codomain ≥ 2
            @inbounds A[2,1] = one(T)/4
        end
    end
    if order_codomain ≥ 1
        @inbounds A[1,0] = one(T)/2
    end
    # first row
    @inbounds for i ∈ 2:2:order_domain-1
        A[0,i] = 2/(1-i^2)
        A[0,i+1] = 2/(i*(i+2))
    end
    if iseven(order_domain)
        @inbounds A[0,order_domain] = 2/(1-order_domain^2)
    end
    # remaining
    @inbounds for i ∈ 2:min(order_domain, order_codomain+1)
        A[i-1,i] = -inv(2(i-1))
    end
    @inbounds for i ∈ 2:min(order_domain, order_codomain-1)
        A[i+1,i] = inv(2(i+1))
    end
    #
    return A
end

## action

(ℐ::Integral)(a::Sequence) = *(ℐ, a)
# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(ℐ::Integral, a::Sequence) = integrate(a, ℐ.n)

function integral_range(s::Taylor, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    return Taylor(s.order+n)
end

function integral_range(s::Fourier, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    return s
end

function integral_range(s::Chebyshev, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    return Chebyshev(s.order+n)
end

integral_range(s::TensorSpace{<:NTuple{N,UnivariateSpace}}, dim::Int, n::Int=1) where {N} =
    TensorSpace((s.spaces[1:dim-1]..., integral_range(s.spaces[dim], n), s.spaces[dim+1:N]...))

function integrate(a::Sequence{Taylor}, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    CoefType = typeof(zero(eltype(a))/1)
    ord = order(a)
    c = Sequence(Taylor(ord+n), Vector{CoefType}(undef, ord+n+1))
    if n == 1
        @inbounds c[0] = zero(CoefType)
        @inbounds c[1] = a[0]
        @inbounds for i ∈ 1:ord
            c[i+1] = a[i] / (i+1)
        end
        return c
    else
        @inbounds view(c, 0:n-1) .= zero(CoefType)
        @inbounds c[n] = a[0] / factorial(n)
        @inbounds for i ∈ 1:ord
            c[i+n] = a[i] / prod(i+1:i+n)
        end
        return c
    end
end

function integrate(a::Sequence{<:Fourier}, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    @assert iszero(a[0])
    iⁿω⁻ⁿ = (im*inv(frequency(a)))^n
    CoefType = promote_type(eltype(a), typeof(iⁿω⁻ⁿ))
    c = Sequence(space(a), Vector{CoefType}(undef, dimension(space(a))))
    @inbounds c[0] = zero(CoefType)
    if isodd(n)
        @inbounds for j ∈ 1:order(a)
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^n
            c[j] = -iⁿω⁻ⁿj⁻ⁿ*a[j]
            c[-j] = iⁿω⁻ⁿj⁻ⁿ*a[-j]
        end
        return c
    else
        @inbounds for j ∈ 1:order(a)
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^n
            c[j] = iⁿω⁻ⁿj⁻ⁿ*a[j]
            c[-j] = iⁿω⁻ⁿj⁻ⁿ*a[-j]
        end
        return c
    end
end

function integrate(a::Sequence{Chebyshev}, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    n > 1 && return integrate(integrate(a, n-1), 1)
    @inbounds a₀ = a[0]/1
    CoefType = typeof(a₀)
    ord = order(a)
    c = Sequence(Chebyshev(ord+1), Vector{CoefType}(undef, ord+2))
    if ord == 0
        @inbounds c[0] = a₀
        @inbounds c[1] = a₀/2
        return c
    elseif ord == 1
        @inbounds c[0] = a₀ - a[1]/2
        @inbounds c[1] = a₀/2
        @inbounds c[2] = a[1]/4
        return c
    else
        @inbounds c[0] = zero(CoefType)
        @inbounds for i ∈ 2:2:ord-1
            c[0] -= a[i] / (i^2-1)
            c[0] += a[i+1] / ((i+1)^2-1)
        end
        if iseven(ord)
            @inbounds c[0] -= a[ord] / (ord^2-1)
        end
        @inbounds c[0] = 2c[0] + a₀ - a[1]/2
        @inbounds for i ∈ 1:ord-1
            c[i] = (a[i-1] - a[i+1]) / (2i)
        end
        @inbounds c[ord] = a[ord-1] / (2ord)
        @inbounds c[ord+1] = a[ord] / (2(ord+1))
        return c
    end
end

function integrate(a::Sequence{<:TensorSpace}, dim::Int, n::Int=1)
    n > 0 || return throw(DomainError(n, "integrate is only defined for strictly positive integers"))
    space_a = space(a)
    A = reshape(coefficients(a), dimensions(space_a))
    return Sequence(integral_range(space_a, dim, n), vec(_integrate(space_a[dim], Val(dim), A, n)))
end

function _integrate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    CoefType = typeof(zero(T)/1)
    ord = order(space)
    C = Array{CoefType,N}(undef, ntuple(i -> i == D ? ord+n+1 : size(A, i), Val(N)))
    if n == 1
        @inbounds selectdim(C, D, 1) .= zero(CoefType)
        @inbounds selectdim(C, D, 2) .= selectdim(A, D, 1)
        @inbounds for i ∈ 1:ord
            selectdim(C, D, i+2) .= selectdim(A, D, i+1) ./ (i+1)
        end
        return C
    else
        @inbounds selectdim(C, D, 1:n) .= zero(CoefType)
        @inbounds selectdim(C, D, n+1) .= selectdim(A, D, 1) ./ factorial(n)
        @inbounds for i ∈ 1:ord
            selectdim(C, D, i+n+1) .= selectdim(A, D, i+1) ./ prod(i+1:i+n)
        end
        return C
    end
end

function _integrate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    ord = order(space)
    @assert iszero(selectdim(A, D, ord+1))
    iⁿω⁻ⁿ = (im*inv(frequency(space)))^n
    CoefType = promote_type(T, typeof(iⁿω⁻ⁿ))
    C = Array{CoefType,N}(undef, size(A))
    @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
    if isodd(n)
        @inbounds for j ∈ 1:ord
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^n
            selectdim(C, D, ord+1+j) .= -iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, D, ord+1+j)
            selectdim(C, D, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, D, ord+1-j)
        end
        return C
    else
        @inbounds for j ∈ 1:ord
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^n
            selectdim(C, D, ord+1+j) .= iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, D, ord+1+j)
            selectdim(C, D, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, D, ord+1-j)
        end
        return C
    end
end

function _integrate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    @assert n == 1 # TODO: lift restriction
    @inbounds A₀ = selectdim(A, D, 1)
    CoefType = typeof(zero(T)/1)
    ord = order(space)
    C = Array{CoefType,N}(undef, ntuple(i -> i == D ? ord+2 : size(A, i), Val(N)))
    @inbounds C₀ = selectdim(C, D, 1)
    @inbounds C₁ = selectdim(C, D, 2)
    if ord == 0
        @. C₀ = A₀
        @. C₁ = A₀ / 2
        return C
    elseif ord == 1
        @inbounds A₁ = selectdim(A, D, 2)
        @. C₀ = A₀ - A₁ / 2
        @. C₁ = A₀ / 2
        @inbounds selectdim(C, D, 3) .= A₁ ./ 4
        return C
    else
        C₀ .= zero(CoefType)
        @inbounds for i ∈ 2:2:ord-1
            C₀ .-= selectdim(A, D, i+1) ./ (i^2-1)
            C₀ .+= selectdim(A, D, i+2) ./ ((i+1)^2-1)
        end
        if iseven(ord)
            @inbounds C₀ .-= selectdim(A, D, ord+1) / (ord^2-1)
        end
        @inbounds C₀ .= 2 .* C₀ .+ A₀ .- selectdim(A, D, 2) ./ 2
        @inbounds C₁ .= (A₀ .- selectdim(A, D, 3)) ./ 2
        @inbounds for i ∈ 2:ord-1
            selectdim(C, D, i+1) .= (selectdim(A, D, i) .- selectdim(A, D, i+2)) ./ (2i)
        end
        @inbounds selectdim(C, D, ord+1) .= selectdim(A, D, ord) ./ (2ord)
        @inbounds selectdim(C, D, ord+2) .= selectdim(A, D, ord+1) ./ (2(ord+1))
        return C
    end
end
