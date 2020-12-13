## arithmetic operations +, - between sequences

Base.:+(a::Sequence) = Sequence(a.space, +(a.coefficients))
Base.:-(a::Sequence) = Sequence(a.space, -(a.coefficients))

function Base.:+(a::Sequence, b::Sequence)
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients + b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = b.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] += a[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] += b[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space)
            c[α] += b[α]
        end
        return c
    end
end

function Base.:-(a::Sequence, b::Sequence)
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients - b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = -b.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] += a[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] -= b[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space)
            c[α] -= b[α]
        end
        return c
    end
end

## arithmetic operation * between sequences

function Base.:*(a::Sequence, b::Sequence)
    (eltype(a) <: Real || eltype(b) <: Real) && return real(*(complex(a), complex(b)))
    npow2 = size_fft(a.space, b.space)
    C = _mul!(_fft_pow2(a, npow2), _fft_pow2(b, npow2))
    space = multiplication_range(a.space, b.space)
    return _ifft_pow2!(space, C)
end

function Base.:*(a::Sequence, b::Sequence, c::Sequence...)
    (eltype(a) <: Real || eltype(b) <: Real || any(cᵢ -> eltype(cᵢ) <: Real, c)) && return real(*(complex(a), complex(b), map(complex, c)...))
    npow2 = size_fft(a.space, b.space, map(cᵢ -> cᵢ.space, c)...)
    C = mapreduce(cᵢ -> _fft_pow2(cᵢ, npow2), _mul!, c; init = _mul!(_fft_pow2(a, npow2), _fft_pow2(b, npow2)))
    space = mapreduce(cᵢ -> cᵢ.space, multiplication_range, c; init = multiplication_range(a.space, b.space))
    return _ifft_pow2!(space, C)
end

_mul!(A::Array{T,N}, B::Array{T,N}) where {T<:Complex,N} = @. A *= B

## arithmetic operation ^

function Base.:^(a::Sequence, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    return _pow(a, n)
end

function _pow(a::Sequence, n::Int)
    eltype(a) <: Real && return real(_pow(complex(a), n))
    npow2 = size_fft(a.space, n)
    C = _power_by_squaring!(_fft_pow2(a, npow2), n)
    space = pow_range(a.space, n)
    return _ifft_pow2!(space, C)
end

function _power_by_squaring!(A::Array{T,N}, n::Int) where {T<:Complex,N}
    n == 2 && return @. A = A ^ 2
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        @. A = A ^ 2
    end
    C = copy(A)
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            @. A = A ^ 2
        end
        @. C *= A
    end
    return C
end

## arithmetic operations +, -, *, /, \ with field elements

function Base.:+(a::Sequence, b)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = a.coefficients
    @inbounds c[_constant_index(a.space)] += b
    return c
end

Base.:+(b, a::Sequence) = +(a, b)

function Base.:-(a::Sequence, b)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = a.coefficients
    @inbounds c[_constant_index(a.space)] -= b
    return c
end

function Base.:-(b, a::Sequence)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = -a.coefficients
    @inbounds c[_constant_index(a.space)] += b
    return c
end

Base.:*(a::Sequence, b) = Sequence(a.space, *(a.coefficients, b))
Base.:*(b, a::Sequence) = Sequence(a.space, *(b, a.coefficients))

Base.:/(a::Sequence, b) = Sequence(a.space, /(a.coefficients, b))
Base.:\(b, a::Sequence) = Sequence(a.space, \(b, a.coefficients))

## arithmetic operations +̄, -̄ between sequences

function +̄(a::Sequence, b::Sequence)
    space = a.space ∪̄ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients + b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] += b[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = b.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] += a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space ∪̄ space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space ∪̄ space)
            c[α] += b[α]
        end
        return c
    end
end

function -̄(a::Sequence{T₁,S₁}, b::Sequence{T₂,S₂}) where {T₁<:SequenceSpace,S₁,T₂<:SequenceSpace,S₂}
    space = a.space ∪̄ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients - b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] -= b[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = -b.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] += a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space ∪̄ space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space ∪̄ space)
            c[α] += b[α]
        end
        return c
    end
end

## arithmetic operation *̄ between sequences

*̄(a::Sequence, b::Sequence) = project(a * b, a.space ∪̄ b.space)
*̄(a::Sequence, b::Sequence, c::Sequence...) = project(*(a, b, c...), mapreduce(cᵢ -> cᵢ.space, ∪̄, c; init = a.space ∪̄ b.space))

## arithmetic operation ^̄

^̄(a::Sequence, n::Integer) = project(a ^ n, a.space)
