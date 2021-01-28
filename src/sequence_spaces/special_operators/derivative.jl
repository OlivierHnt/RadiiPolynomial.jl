struct Derivative
    n :: Int
end

(𝒟::Derivative)(a) = *(𝒟, a)

# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(𝒟::Derivative, a::Sequence) = differentiate(a, 𝒟.n)
# signature needed to resolve ambiguity due to *(b, A::Operator)
Base.:*(𝒟::Derivative, A::Operator) = differentiate(A, 𝒟.n)

Base.:*(𝒟₁::Derivative, 𝒟₂::Derivative) = Derivative(𝒟₁.n + 𝒟₂.n)
Base.:^(𝒟::Derivative, n::Int) = Derivative(𝒟.n * n)

+̄(𝒟::Derivative, A::Operator{<:SequenceSpace,<:SequenceSpace}) = +(Operator(domain(A), codomain(A), 𝒟, eltype(A)), A)
+̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, 𝒟::Derivative) = +(A, Operator(domain(A), codomain(A), 𝒟, eltype(A)))
-̄(𝒟::Derivative, A::Operator{<:SequenceSpace,<:SequenceSpace}) = -(Operator(domain(A), codomain(A), 𝒟, eltype(A)), A)
-̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, 𝒟::Derivative) = -(A, Operator(domain(A), codomain(A), 𝒟, eltype(A)))
*̄(𝒟::Derivative, A::Operator{<:SequenceSpace,<:SequenceSpace}) = *(Operator(codomain(A), codomain(A), 𝒟, eltype(A)), A)
*̄(A::Operator{<:SequenceSpace,<:SequenceSpace}, 𝒟::Derivative) = *(A, Operator(domain(A), domain(A), 𝒟, eltype(A)))

# sequence space

function Operator(domain::Taylor, codomain::Taylor, 𝒟::Derivative, ::Type{T}=Float64) where {T}
    A = Operator(domain, codomain, Matrix{T}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ 0:min(order(domain)-𝒟.n, order(codomain))
        A[i,i+𝒟.n] = prod(i+1:i+𝒟.n)
    end
    return A
end

function Operator(domain::Fourier{T}, codomain::Fourier{S}, 𝒟::Derivative, ::Type{R}=complex(promote_type(T, S))) where {T,S,R}
    @assert frequency(domain) == frequency(codomain)
    A = Operator(domain, codomain, Matrix{R}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(R)
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

function Operator(domain::Chebyshev, codomain::Chebyshev, 𝒟::Derivative, ::Type{T}=Float64) where {T}
    @assert 𝒟.n == 1 # TODO: lift restriction
    A = Operator(domain, codomain, Matrix{T}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(T)
    @inbounds for i ∈ 1:2:domain.order
        A[0,i] = 2i
    end
    for j ∈ 2:2:domain.order-1
        @inbounds for i ∈ 1:2:min(j-1, codomain.order)
            A[i,j] = 2j
        end
        @inbounds for i ∈ 2:2:min(j, codomain.order)
            A[i,j+1] = 2(j+1)
        end
    end
    if iseven(domain.order)
        @inbounds for i ∈ 1:2:min(domain.order-1, codomain.order)
            A[i,domain.order] = 2domain.order
        end
    end
    return A
end

# cartesian space

# TODO: fix type instability due to cat
Operator(domain::CartesianSpace, codomain::CartesianSpace, 𝒟::Derivative) =
    Operator(domain, codomain, mapreduce((s₁, s₂) -> Operator(s₁, s₂, 𝒟).coefficients, (x, y) -> cat(x, y; dims=(1,2)),  domain.spaces, codomain.spaces))

##

# sequence space

function differentiate(s::Taylor, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    s.order < n && return Taylor(0)
    return Taylor(s.order-n)
end

function differentiate(s::Fourier, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    return s
end

function differentiate(s::Chebyshev, n::Int=1)
    n < 1 && return throw(DomainError(n, "^ is only defined for strictly positive integers"))
    s.order < n && return Chebyshev(0)
    return Chebyshev(s.order-n)
end

differentiate(s::TensorSpace{<:NTuple{N,UnivariateSpace}}, dim::Int, n::Int=1) where {N} =
    TensorSpace(tuple(s.spaces[1:dim-1]..., differentiate(s.spaces[dim], n), s.spaces[dim+1:N]...))

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
    A = reshape(a.coefficients, dimensions(a.space))
    return Sequence(differentiate(a.space, dim, n), vec(_differentiate(a.space[dim], Val(dim), A, n)))
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

# cartesian space

differentiate(s::CartesianSpace{<:NTuple{N,UnivariateSpace}}, n::Int=1) where {N} =
    CartesianSpace(map(sᵢ -> differentiate(sᵢ, n), s.spaces))

differentiate(s::CartesianSpace{<:Tuple{N₁,TensorSpace{<:NTuple{N₂,UnivariateSpace}}}}, dim::Int, n::Int=1) where {N₁,N₂} =
    CartesianSpace(map(sᵢ -> differentiate(sᵢ, dim, n), s.spaces))

# TODO: fix type instability

function differentiate(a::Sequence{CartesianSpace{T}}, n::Int=1) where {N,T<:NTuple{N,UnivariateSpace}}
    space = differentiate(a.space, n)
    return Sequence(space, mapreduce(aᵢ -> differentiate(aᵢ, n).coefficients, vcat, eachcomponent(a)))
end

function differentiate(a::Sequence{CartesianSpace{T}}, dim::Int, n::Int=1) where {N₁,N₂,T<:Tuple{N₁,TensorSpace{<:NTuple{N₂,UnivariateSpace}}}}
    space = differentiate(a.space, dim, n)
    return Sequence(space, mapreduce(aᵢ -> differentiate(aᵢ, dim, n).coefficients, vcat, eachcomponent(a)))
end
