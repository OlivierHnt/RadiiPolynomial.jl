# convolution based on fft routines
# NOTE: for multiply and power with Chebyshev sequences of the form (a_0, a_1, ...) must be rescaled to (a_0, 2a_1, ...)

function multiply(a::Sequence, b::Sequence)
    (eltype(a) <: Real || eltype(b) <: Real) && return real(multiply(complex(a), complex(b)))
    npow2 = fft_size(a.space, b.space)
    C = _mul!(_fft_pow2(a, npow2), _fft_pow2(b, npow2))
    space = multiplication_range(a.space, b.space)
    return _ifft_pow2!(space, C)
end

function multiply(a::Sequence, b::Sequence, c::Sequence...)
    (eltype(a) <: Real || eltype(b) <: Real || any(cᵢ -> eltype(cᵢ) <: Real, c)) && return real(multiply(complex(a), complex(b), map(complex, c)...))
    npow2 = fft_size(a.space, b.space, map(cᵢ -> cᵢ.space, c)...)
    C = mapreduce(cᵢ -> _fft_pow2(cᵢ, npow2), _mul!, c; init = _mul!(_fft_pow2(a, npow2), _fft_pow2(b, npow2)))
    space = mapreduce(cᵢ -> cᵢ.space, multiplication_range, c; init = multiplication_range(a.space, b.space))
    return _ifft_pow2!(space, C)
end

_mul!(A::Array{T,N}, B::Array{T,N}) where {T<:Complex,N} = @. A *= B

# power based on fft routines

function power(a::Sequence, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    return _pow(a, n)
end

function _pow(a::Sequence, n::Int)
    eltype(a) <: Real && return real(_pow(complex(a), n))
    npow2 = fft_size(a.space, n)
    C = _power_by_squaring!(_fft_pow2(a, npow2), n)
    space = multiplication_range(a.space, n)
    return _ifft_pow2!(space, C)
end

function _power_by_squaring!(A::Array{T,N}, n::Int) where {T<:Complex,N}
    n == 2 && return @. A *= A
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        @. A *= A
    end
    C = copy(A)
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            @. A *= A
        end
        @. C *= A
    end
    return C
end

## NOTE: perhaps promote to float type in _extension functions for Chebyshev? -> to allow fft on Chebyshev sequence of integers
# NOTE:
# function f(x)
#     s = max( fft_size(x[1].space, x[2].space, x[3].space), fft_size(x[1].space, x[3].space, x[3].space) )
#     fft(x[1], s), fft(x[2], s), fft(x[3], s)
#
#     x[1]*x[2]*x[3]
#     x[1]*x[3]*x[3]
#     x[1]*x[2]
# end

## user facing functions contain a check for safety

_size_dfs(space::Taylor) = length(space)
_size_dfs(space::Fourier) = length(space)
_size_dfs(space::Chebyshev) = 2space.order+1

fft_size(s₁::UnivariateSpace, s₂::UnivariateSpace) =
    nextpow(2, _size_dfs(s₁) + _size_dfs(s₂))

fft_size(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    map((s₁_, s₂_) -> nextpow(2, _size_dfs(s₁_) + _size_dfs(s₂_)), s₁.spaces, s₂.spaces)

fft_size(s₁::UnivariateSpace, s₂::UnivariateSpace, s₃::UnivariateSpace...) =
    nextpow(2, _size_dfs(s₁) + _size_dfs(s₂) + mapreduce(s₃_ -> _size_dfs(s₃_), +, s₃))

fft_size(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₃::TensorSpace{<:NTuple{N,UnivariateSpace}}...) where {N} =
    ntuple(i -> nextpow(2, _size_dfs(s₁[i]) + _size_dfs(s₂[i]) + mapreduce(s₃_ -> _size_dfs(s₃_[i]), +, s₃)), Val(N))

fft_size(space::UnivariateSpace, n::Integer) =
    nextpow(2, n * _size_dfs(space))

fft_size(space::TensorSpace, n::Integer) =
    map(sᵢ -> nextpow(2, n * _size_dfs(sᵢ)), space.spaces)

function fft(a::Sequence{T}, n::Int) where {T<:UnivariateSpace}
    @assert ispow2(n) && _size_dfs(a.space) ≤ n
    eltype(a) <: Real && return fft(complex(a), n)
    return _fft_pow2(a, n)
end

function fft(a::Sequence{TensorSpace{T}}, n::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
    @assert all(ispow2, n) && all(i -> _size_dfs(a.space[i]) ≤ n[i], 1:N)
    eltype(a) <: Real && return fft(complex(a), n)
    return _fft_pow2(a, n)
end

function ifft!(space::UnivariateSpace, A::Vector{T}) where {T}
    @assert ispow2(length(A)) && length(space) ≤ length(A)
    return _ifft_pow2!(space, A)
end

function ifft!(space::TensorSpace{<:NTuple{N,UnivariateSpace}}, A::Array{T,N}) where {T,N}
    @assert all(ispow2, size(A)) && all(i -> size(space, i) ≤ size(A, i), 1:N)
    return _ifft_pow2!(space, A)
end

## to discrete fourier series

function _fft_pow2(a::Sequence{Taylor}, n::Int)
    B = zeros(eltype(a), n)
    @inbounds B[eachindex(a.coefficients)] .= a.coefficients
    return _fft_pow2!(B)
end

function _fft_pow2(a::Sequence{<:Fourier}, n::Int)
    B = zeros(eltype(a), n)
    @inbounds B[eachindex(a.coefficients)] .= a.coefficients
    return _fft_pow2!(B)
end

function _fft_pow2(a::Sequence{Chebyshev}, n::Int)
    B = zeros(eltype(a), n)
    len = length(a.space)
    @inbounds B[len] = a[0]
    @inbounds for i ∈ 1:order(a.space)
        aᵢ½ = a[i]/2
        B[len-i] = aᵢ½
        B[len+i] = aᵢ½
    end
    return _fft_pow2!(B)
end

function _fft_pow2(a::Sequence{TensorSpace{T}}, n::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
    B = zeros(eltype(a), n)
    A = reshape(a.coefficients, size(a.space))
    @inbounds B[axes(A)...] .= A
    _apply_extension!(a.space, B)
    return _fft_pow2!(B)
end

_apply_extension!(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::Array{T,N₂}) where {N₁,T,N₂} =
    @inbounds _extension!(space[1], Val(N₂-N₁+1), _apply_extension!(Base.tail(space), A))

_apply_extension!(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::Array{T,N}) where {T,N} =
    @inbounds _extension!(space[1], Val(N-1), _extension!(space[2], Val(N), A))

_extension!(space::Taylor, ::Val, A) = A

_extension!(space::Fourier, ::Val, A) = A

function _extension!(space::Chebyshev, ::Val{D}, A::Array{T,N}) where {D,T,N}
    n = length(space)
    n == 1 && return A
    n2 = n÷2
    @inbounds A_last_view = selectdim(A, D, n)
    A_last_view ./= 2
    @inbounds A_shift_last_view = selectdim(A, D, 2n-1)
    @inbounds A_first_view = selectdim(A, D, 1)
    A_shift_last_view .= A_last_view
    A_last_view .= A_first_view
    A_first_view .= A_shift_last_view
    @inbounds for i ∈ 2:n2
        A_last_view = selectdim(A, D, n+1-i)
        A_last_view ./= 2
        A_shift_last_view = selectdim(A, D, 2n-i)
        A_first_view = selectdim(A, D, i)
        A_first_view ./= 2
        A_shift_first_view = selectdim(A, D, n-1+i)
        A_shift_last_view .= A_last_view
        A_shift_first_view .= A_first_view
        A_last_view .= A_shift_first_view
        A_first_view .= A_shift_last_view
    end
    n % 2 == 0 && return A
    @inbounds A_last_view = selectdim(A, D, n-n2)
    A_last_view ./= 2
    @inbounds A_shift_last_view = selectdim(A, D, 2n-n2-1)
    A_shift_last_view .= A_last_view
    return A
end

## conversion discrete fourier series to series

_reduction_eachindex(space::Taylor) = 1:length(space)
_reduction_eachindex(space::Fourier) = 1:length(space)
_reduction_eachindex(space::Chebyshev) = (n = length(space); return n:2n-1)
_reduction_axes(space::TensorSpace) = map(_reduction_eachindex, space.spaces)

function _ifft_pow2!(space::Taylor, A::Vector)
    _ifft_pow2!(A)
    return Sequence(space, resize!(A, length(space)))
end

function _ifft_pow2!(space::Fourier, A::Vector)
    _ifft_pow2!(A)
    return Sequence(space, resize!(A, length(space)))
end

function _ifft_pow2!(space::Chebyshev, A::Vector)
    _ifft_pow2!(A)
    len = length(space)
    resize!(A, len)
    reverse!(A)
    @inbounds @. A[2:len] *= 2
    return Sequence(space, A)
end

function _ifft_pow2!(space::TensorSpace, A::Array)
    _ifft_pow2!(A)
    axes_ = _reduction_axes(space)
    @inbounds B = A[axes_...]
    _apply_reduction!(space, B)
    return Sequence(space, vec(B))
end

_apply_reduction!(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::Array{T,N₂}) where {N₁,T,N₂} =
    @inbounds _reduction!(space[1], Val(N₂-N₁+1), _apply_reduction!(Base.tail(space), A))

_apply_reduction!(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::Array{T,N}) where {T,N} =
    @inbounds _reduction!(space[1], Val(N-1), _reduction!(space[2], Val(N), A))

_reduction!(space::Taylor, ::Val, A) = A

_reduction!(space::Fourier, ::Val, A) = A

function _reduction!(space::Chebyshev, ::Val{D}, A::Array{T,N}) where {D,T,N}
    @inbounds for i ∈ 2:length(space)
        selectdim(A, D, i) .*= 2
    end
    return A
end

## FFT routines

function _fft_pow2!(a::AbstractArray{T,N}) where {T<:Complex,N}
    for a_slice ∈ eachslice(a; dims = 1)
        _fft_pow2!(a_slice)
    end
    a_ = reshape(a, size(a, 1), :)
    for a_col ∈ eachcol(a_)
        _fft_pow2!(a_col)
    end
    return a
end

function _ifft_pow2!(a::AbstractArray{T,N}) where {T<:Complex,N}
    for a_slice ∈ eachslice(a; dims = 1)
        _ifft_pow2!(a_slice)
    end
    a_ = reshape(a, size(a, 1), :)
    for a_col ∈ eachcol(a_)
        _ifft_pow2!(a_col)
    end
    return a
end

function _fft_pow2!(a::AbstractVector{Complex{T}}) where {T}
    bitreverse!(a)
    iπ = im*convert(T, π)
    n = length(a)
    N = 2
    while N ≤ n
        N½ = N÷2
        ω_ = exp(-iπ/N½)
        for k ∈ 1:N:n
            ω = one(Complex{T})
            @inbounds for j ∈ k:k+N½-1
                j′ = j+N½
                aj′_ω = a[j′]*ω
                a[j′] = a[j] - aj′_ω
                a[j] = a[j] + aj′_ω
                ω *= ω_
            end
        end
        N <<= 1
    end
    return a
end

function _fft_pow2!(a::AbstractVector{Complex{T}}) where {T<:Interval}
    bitreverse!(a)
    iπ = im*@interval(π)
    n = length(a)
    N = 2
    while N ≤ n
        N½= N÷2
        iπ_N½ = iπ/N½
        for k ∈ 1:N:n
            @inbounds for (count,j) ∈ enumerate(k:k+N½-1)
                j′ = j+N½
                ω_aj′ = a[j′]*exp(-iπ_N½*(count-1))
                a[j′] = a[j] - ω_aj′
                a[j] = a[j] + ω_aj′
            end
        end
        N <<= 1
    end
    return a
end

function _ifft_pow2!(a::AbstractVector{T}) where {T<:Complex}
    conj!(_fft_pow2!(conj!(a)))
    n = length(a)
    @inbounds for i ∈ 1:n
        a[i] /= n
    end
    return a
end

function bitreverse!(a::AbstractVector)
    n = length(a)
    n½ = n÷2
    j = 1
    for i ∈ 1:n-1
        if i < j
            @inbounds a[j], a[i] = a[i], a[j]
        end
        k = n½
        while 2 ≤ k < j
            j -= k
            k ÷= 2
        end
        j += k
    end
    return a
end
