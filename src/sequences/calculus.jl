## differentiation

function differentiate(a::Sequence{Taylor}, n::Int=1)
    @assert n ≥ 1
    CoefType = eltype(a)
    ord = order(a)
    ord < n && return Sequence(Taylor(0), [zero(CoefType)])
    c = Sequence(Taylor(ord-n), Vector{CoefType}(undef, ord-n+1))
    @inbounds for i ∈ n:ord
        c[i-n] = i*a[i]
    end
    return c
end

function differentiate(a::Sequence{<:Fourier}, n::Int=1)
    @assert n ≥ 1
    iⁿωⁿ = (im*frequency(a))^n
    CoefType = promote_type(eltype(a), typeof(iⁿωⁿ))
    c = Sequence(space(a), Vector{CoefType}(undef, length(space(a))))
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
    @assert n ≥ 1
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

function differentiate(a::Sequence{<:TensorSpace}, dims, n=1)
    A = reshape(a.coefficients, size(a.space))
    return _differentiate(a.space, dims, A, n)
end

function _differentiate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, A, n::Int) where {N}
    @assert n ≥ 1
    return Sequence(derivative_range(space, dims, n), vec(_differentiate(space[dims], Val(dims), A, n)))
end

_differentiate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, ::Colon, A, n::Int) where {N} =
    map(i -> _differentiate(space, i, A, n), 1:N)

_differentiate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, ::Colon, A, n::NTuple{N,Int}) where {N} =
    map(i -> _differentiate(space, i, A, n[i]), 1:N)

function _differentiate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    ord = order(space)
    ord < n && return zeros(T, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
    C = Array{T}(undef, ntuple(i -> i == D ? ord-n+1 : size(A, i), Val(N)))
    @inbounds for i ∈ n:ord
        _selectdim(space, C, D, i-n) .= i .* _selectdim(space, A, D, i)
    end
    return C
end

function _differentiate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    iⁿωⁿ = (im*frequency(space))^n
    CoefType = promote_type(T, typeof(iⁿωⁿ))
    C = Array{CoefType}(undef, size(A))
    @inbounds _selectdim(space, C, D, 0) .= zero(CoefType)
    if isodd(n)
        @inbounds for j ∈ 1:order(space)
            iⁿωⁿjⁿ = iⁿωⁿ*j^n
            _selectdim(space, C, D, j) .= iⁿωⁿjⁿ .* _selectdim(space, A, D, j)
            _selectdim(space, C, D, -j) .= -iⁿωⁿjⁿ .* _selectdim(space, A, D, -j)
        end
        return C
    else
        @inbounds for j ∈ 1:order(space)
            iⁿωⁿjⁿ = iⁿωⁿ*j^n
            _selectdim(space, C, D, j) .= iⁿωⁿjⁿ .* _selectdim(space, A, D, j)
            _selectdim(space, C, D, -j) .= -iⁿωⁿjⁿ .* _selectdim(space, A, D, -j)
        end
        return C
    end
end

function _differentiate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    @assert n == 1 # TODO: lift restriction
    ord = order(space)
    C = Array{T}(undef, ntuple(i -> i == D ? ord : size(A, i), Val(N)))
    @inbounds C₀ = _selectdim(space, C, D, 0)
    C₀ .= zero(T)
    @inbounds for i ∈ 1:2:ord
        C₀ .+= (2i) .* _selectdim(space, A, D, i)
    end
    @inbounds for i ∈ 1:ord-1
        Cᵢ = _selectdim(space, C, D, i)
        Cᵢ .= zero(T)
        @inbounds for j ∈ i+1:2:ord
            Cᵢ .+= j .* _selectdim(space, A, D, j)
        end
        @. Cᵢ *= 2
    end
    return C
end

## integration

function integrate(a::Sequence{Taylor}, n::Int=1)
    @assert n ≥ 1
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
    @assert n ≥ 1
    @assert iszero(a[0])
    iⁿω⁻ⁿ = (im*inv(frequency(a)))^n
    CoefType = promote_type(eltype(a), typeof(iⁿω⁻ⁿ))
    c = Sequence(space(a), Vector{CoefType}(undef, length(space(a))))
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
    @assert n ≥ 1
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

function integrate(a::Sequence{<:TensorSpace}, dims, n=1)
    A = reshape(a.coefficients, size(a.space))
    return _integrate(a.space, dims, A, n)
end

function _integrate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, dims::Int, A, n::Int) where {N}
    @assert n ≥ 1
    return Sequence(integral_range(space, dims, n), vec(_integrate(space[dims], Val(dims), A, n)))
end

_integrate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, ::Colon, A, n::Int) where {N} =
    map(i -> _integrate(space, i, A, n), 1:N)

_integrate(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, ::Colon, A, n::NTuple{N,Int}) where {N} =
    map(i -> _integrate(space, i, A, n[i]), 1:N)

function _integrate(space::Taylor, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    CoefType = typeof(zero(T)/1)
    ord = order(space)
    C = Array{CoefType}(undef, ntuple(i -> i == D ? ord+n+1 : size(A, i), Val(N)))
    if n == 1
        @inbounds _selectdim(space, C, D, 0) .= zero(CoefType)
        @inbounds _selectdim(space, C, D, 1) .= _selectdim(space, A, D, 0)
        @inbounds for i ∈ 1:ord
            _selectdim(space, C, D, i+1) .= _selectdim(space, A, D, i) ./ (i+1)
        end
        return C
    else
        @inbounds _selectdim(space, C, D, 0:n-1) .= zero(CoefType)
        @inbounds _selectdim(space, C, D, n) .= _selectdim(space, A, D, 0) ./ factorial(n)
        @inbounds for i ∈ 1:ord
            _selectdim(space, C, D, i+n) .= _selectdim(space, A, D, i) ./ prod(i+1:i+n)
        end
        return C
    end
end

function _integrate(space::Fourier, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    @assert iszero(_selectdim(space, A, D, 0))
    iⁿω⁻ⁿ = (im*inv(frequency(space)))^n
    CoefType = promote_type(T, typeof(iⁿω⁻ⁿ))
    C = Array{CoefType}(undef, size(A))
    @inbounds _selectdim(space, C, D, 0) .= zero(CoefType)
    if isodd(n)
        @inbounds for j ∈ 1:order(space)
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^n
            _selectdim(space, C, D, j) .= -iⁿω⁻ⁿj⁻ⁿ .* _selectdim(space, A, D, j)
            _selectdim(space, C, D, -j) .= iⁿω⁻ⁿj⁻ⁿ .* _selectdim(space, A, D, -j)
        end
        return C
    else
        @inbounds for j ∈ 1:order(space)
            iⁿω⁻ⁿj⁻ⁿ = iⁿω⁻ⁿ/j^n
            _selectdim(space, C, D, j) .= iⁿω⁻ⁿj⁻ⁿ .* _selectdim(space, A, D, j)
            _selectdim(space, C, D, -j) .= iⁿω⁻ⁿj⁻ⁿ .* _selectdim(space, A, D, -j)
        end
        return C
    end
end

function _integrate(space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}, n) where {D,T,N}
    @assert n == 1 # TODO: lift restriction
    @inbounds A₀ = _selectdim(space, A, D, 0)
    CoefType = typeof(zero(T)/1)
    ord = order(space)
    C = Array{CoefType}(undef, ntuple(i -> i == D ? ord+2 : size(A, i), Val(N)))
    @inbounds C₀ = _selectdim(space, C, D, 0)
    @inbounds C₁ = _selectdim(space, C, D, 1)
    if ord == 0
        @. C₀ = A₀
        @. C₁ = A₀ / 2
        return C
    elseif ord == 1
        @inbounds A₁ = _selectdim(space, A, D, 1)
        @. C₀ = A₀ - A₁ / 2
        @. C₁ = A₀ / 2
        @inbounds _selectdim(space, C, D, 2) .= A₁ ./ 4
        return C
    else
        C₀ .= zero(CoefType)
        @inbounds for i ∈ 2:2:ord-1
            C₀ .-= _selectdim(space, A, D, i) ./ (i^2-1)
            C₀ .+= _selectdim(space, A, D, i+1) ./ ((i+1)^2-1)
        end
        if iseven(ord)
            @inbounds C₀ .-= _selectdim(space, A, D, ord) / (ord^2-1)
        end
        @inbounds C₀ .= 2 .* C₀ .+ A₀ .- _selectdim(space, A, D, 1) ./ 2
        @inbounds C₁ .= (A₀ .- _selectdim(space, A, D, 2)) ./ 2
        @inbounds for i ∈ 2:ord-1
            _selectdim(space, C, D, i) .= (_selectdim(space, A, D, i-1) .- _selectdim(space, A, D, i+1)) ./ (2i)
        end
        @inbounds _selectdim(space, C, D, ord) .= _selectdim(space, A, D, ord-1) ./ (2ord)
        @inbounds _selectdim(space, C, D, ord+1) .= _selectdim(space, A, D, ord) ./ (2(ord+1))
        return C
    end
end
