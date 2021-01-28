# function multiply(a::Sequence{<:UnivariateSpace}, b::Sequence{<:UnivariateSpace})
#     npow2 = fft_length(a.space, b.space)
#     C = _mul!(_sequence2dfs(a, npow2), _sequence2dfs(b, npow2))
#     space = *(a.space, b.space)
#     CoefType = promote_type(eltype(a), eltype(b))
#     return _dfs2sequence!(C, space, CoefType)
# end
#
# function multiply(a::Sequence{<:UnivariateSpace}, b::Sequence{<:UnivariateSpace}, c::Sequence{<:UnivariateSpace}...)
#     npow2 = fft_length(a.space, b.space, map(cᵢ -> cᵢ.space, c)...)
#     C = mapreduce(cᵢ -> _sequence2dfs(cᵢ, npow2), _mul!, c; init = _mul!(_sequence2dfs(a, npow2), _sequence2dfs(b, npow2)))
#     space = mapreduce(cᵢ -> cᵢ.space, *, c; init = *(a.space, b.space))
#     CoefType = mapreduce(eltype, promote_type, c; init = promote_type(eltype(a), eltype(b)))
#     return _dfs2sequence!(C, space, CoefType)
# end
#
# function multiply(a::Sequence{<:TensorSpace}, b::Sequence{<:TensorSpace})
#     npow2 = fft_size(a.space, b.space)
#     C = _mul!(_sequence2dfs(a, npow2), _sequence2dfs(b, npow2))
#     space = *(a.space, b.space)
#     CoefType = promote_type(eltype(a), eltype(b))
#     return _dfs2sequence!(C, space, CoefType)
# end
#
# function multiply(a::Sequence{<:TensorSpace}, b::Sequence{<:TensorSpace}, c::Sequence{<:TensorSpace}...)
#     npow2 = fft_size(a.space, b.space, map(cᵢ -> cᵢ.space, c)...)
#     C = mapreduce(cᵢ -> _sequence2dfs(cᵢ, npow2), _mul!, c; init = _mul!(_sequence2dfs(a, npow2), _sequence2dfs(b, npow2)))
#     space = mapreduce(cᵢ -> cᵢ.space, *, c; init = *(a.space, b.space))
#     CoefType = mapreduce(eltype, promote_type, c; init = promote_type(eltype(a), eltype(b)))
#     return _dfs2sequence!(C, space, CoefType)
# end
#
# _mul!(A, B) = @. A *= B
#
# function power(a::Sequence{<:UnivariateSpace}, n::Int)
#     n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
#     n == 0 && return one(a)
#     n == 1 && return copy(a)
#     npow2 = fft_length(a.space, n)
#     C = _power_by_squaring!(_sequence2dfs(a, npow2), n)
#     space = ^(a.space, n)
#     return _dfs2sequence!(C, space, eltype(a))
# end
#
# function power(a::Sequence{<:TensorSpace}, n::Int)
#     n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
#     n == 0 && return one(a)
#     n == 1 && return copy(a)
#     npow2 = fft_size(a.space, n)
#     C = _power_by_squaring!(_sequence2dfs(a, npow2), n)
#     space = ^(a.space, n)
#     return _dfs2sequence!(C, space, eltype(a))
# end
#
# function _power_by_squaring!(A, n)
#     n == 2 && return _mul!(A, A)
#     t = trailing_zeros(n) + 1
#     n >>= t
#     while (t -= 1) > 0
#         _mul!(A, A)
#     end
#     C = copy(A)
#     while n > 0
#         t = trailing_zeros(n) + 1
#         n >>= t
#         while (t -= 1) ≥ 0
#             _mul!(A, A)
#         end
#         _mul!(C, A)
#     end
#     return C
# end

## user facing functions contain a check for safety

fft_length(s₁::UnivariateSpace, s₂::UnivariateSpace) =
    nextpow(2, dfs_dimension(s₁) + dfs_dimension(s₂))

fft_length(s₁::UnivariateSpace, s₂::UnivariateSpace, s₃::UnivariateSpace...) =
    nextpow(2, dfs_dimension(s₁) + dfs_dimension(s₂) + mapreduce(dfs_dimension, +, s₃))

fft_length(space::UnivariateSpace, n::Int) =
    nextpow(2, n * dfs_dimension(space))

fft_size(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    map(fft_length, s₁.spaces, s₂.spaces)

fft_size(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₃::TensorSpace{<:NTuple{N,UnivariateSpace}}...) where {N} =
    ntuple(i -> nextpow(2, dfs_dimension(s₁[i]) + dfs_dimension(s₂[i]) + mapreduce(s₃_ -> dfs_dimension(s₃_[i]), +, s₃)), Val(N))

fft_size(space::TensorSpace, n::Int) =
    map(sᵢ -> fft_length(dfs_dimension(sᵢ), n), space.spaces)

dfs_dimension(space::Taylor) = space.order+1
dfs_dimension(space::Fourier) = 2space.order+1
dfs_dimension(space::Chebyshev) = 2space.order+1
dfs_dimensions(space::TensorSpace, i::Int) = dfs_dimension(space[i])
dfs_dimensions(space::TensorSpace) = map(dfs_dimension, space.spaces)

function dfs(a::Sequence{<:UnivariateSpace}, n::Int)
    @assert ispow2(n) && dfs_dimension(a.space) ≤ n
    return _sequence2dfs(a, n)
end

function dfs(a::Sequence{TensorSpace{T}}, n::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
    @assert all(ispow2, n) && all(i -> dfs_dimensions(a.space, i) ≤ n[i], 1:N)
    return _sequence2dfs(a, n)
end

function idfs!(A::Vector{T}, space::UnivariateSpace) where {T}
    @assert ispow2(length(A)) && dimension(space) ≤ length(A)
    return _dfs2sequence!(A, space, T)
end

function idfs!(A::Array{T,N}, space::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {T,N}
    @assert all(ispow2, size(A)) && all(i -> dimensions(space, i) ≤ size(A, i), 1:N)
    return _dfs2sequence!(A, space, T)
end

## sequence to discrete fourier series

function _sequence2dfs(a::Sequence{Taylor}, n::Int)
    B = zeros(complex(eltype(a)), n)
    @inbounds view(B, eachindex(a.coefficients)) .= a.coefficients
    return _fft_pow2!(B)
end

function _sequence2dfs(a::Sequence{<:Fourier}, n::Int)
    C = zeros(complex(eltype(a)), n)
    @inbounds view(C, eachindex(a.coefficients)) .= a.coefficients
    return _fft_pow2!(C)
end

function _sequence2dfs(a::Sequence{Chebyshev}, n::Int)
    C = zeros(complex(eltype(a)), n)
    len = length(a.coefficients)
    @inbounds C[len] = a[0]
    @inbounds for i ∈ 1:order(a)
        C[len-i] = a[i]
        C[len+i] = a[i]
    end
    return _fft_pow2!(C)
end

function _sequence2dfs(a::Sequence{TensorSpace{T}}, n::NTuple{N,Int}) where {N,T<:NTuple{N,UnivariateSpace}}
    C = zeros(complex(eltype(a)), n)
    A = reshape(a.coefficients, dimensions(a.space))
    @inbounds view(C, axes(A)...) .= A
    _apply_sequence2dfs!(C, a.space)
    return _fft_pow2!(C)
end

_apply_sequence2dfs!(A::Array{T,N₂}, space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}) where {N₁,T,N₂} =
    @inbounds _sequence2dfs!(_apply_sequence2dfs!(A, Base.tail(space)), space[1], Val(N₂-N₁+1))

_apply_sequence2dfs!(A::Array{T,N}, space::TensorSpace{<:NTuple{2,UnivariateSpace}}) where {T,N} =
    @inbounds _sequence2dfs!(_sequence2dfs!(A, space[2], Val(N)), space[1], Val(N-1))

_sequence2dfs!(A, space::Taylor, ::Val) = A

_sequence2dfs!(A, space::Fourier, ::Val) = A

function _extension!(A, space::Chebyshev, ::Val{D}) where {D}
    n = dimension(space)
    n == 1 && return A
    n2 = n÷2
    @inbounds A₁ = selectdim(A, D, 1)
    @inbounds Aₙ = selectdim(A, D, n)
    @inbounds A₂ₙ₋₁ = selectdim(A, D, 2n-1)
    A₂ₙ₋₁ .= Aₙ
    Aₙ .= A₁
    A₁ .= A₂ₙ₋₁
    @inbounds for i ∈ 2:n2
        Aᵢ = selectdim(A, D, i)
        Aₙ₊₁₋ᵢ = selectdim(A, D, n+1-i)
        Aₙ₋₁₊ᵢ = selectdim(A, D, n-1+i)
        A₂ₙ₋ᵢ = selectdim(A, D, 2n-i)
        A₂ₙ₋ᵢ .= Aₙ₊₁₋ᵢ
        Aₙ₋₁₊ᵢ .= Aᵢ
        Aₙ₊₁₋ᵢ .= Aₙ₋₁₊ᵢ
        Aᵢ .= A₂ₙ₋ᵢ
    end
    n % 2 == 0 && return A
    @inbounds selectdim(A, D, n-n2) .= selectdim(A, D, 2n-n2-1)
    return A
end

## discrete fourier series to sequence

function _dfs2sequence!(A::Vector{T}, space::Taylor, ::Type{T}) where {T}
    _ifft_pow2!(A)
    return Sequence(space, resize!(A, dimension(space)))
end

function _dfs2sequence!(A::Vector{T}, space::Fourier, ::Type{T}) where {T}
    _ifft_pow2!(A)
    return Sequence(space, resize!(A, dimension(space)))
end

function _dfs2sequence!(A::Vector{T}, space::Chebyshev, ::Type{T}) where {T}
    _ifft_pow2!(A)
    return Sequence(space, reverse!(resize!(A, dimension(space))))
end

function _dfs2sequence!(A::Vector{T}, space::UnivariateSpace, ::Type{S}) where {T,S}
    _ifft_pow2!(A)
    v = Vector{S}(undef, dimension(space))
    @inbounds v .= view(A, _dfs2sequence_eachindex(space))
    return Sequence(space, v)
end

function _dfs2sequence!(A::Array{T}, space::TensorSpace, ::Type{S}) where {T,S}
    _ifft_pow2!(A)
    @inbounds B = A[_dfs2sequence_axes(space)...]
    return Sequence(space, vec(B))
end

_dfs2sequence_eachindex(space::Taylor) = 1:dimension(space)
_dfs2sequence_eachindex(space::Fourier) = 1:dimension(space)
_dfs2sequence_eachindex(space::Chebyshev) = (n = dimension(space); return n:2n-1)
_dfs2sequence_axes(space::TensorSpace) = map(_dfs2sequence_eachindex, space.spaces)

## FFT routines

function _fft_pow2!(a::AbstractArray)
    for a_slice ∈ eachslice(a; dims = 1)
        _fft_pow2!(a_slice)
    end
    a_ = reshape(a, size(a, 1), :)
    for a_col ∈ eachcol(a_)
        _fft_pow2!(a_col)
    end
    return a
end

function _ifft_pow2!(a::AbstractArray)
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
    iπ = im*convert(T, π)
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

function _ifft_pow2!(a::AbstractVector)
    conj!(_fft_pow2!(conj!(a)))
    a ./= length(a)
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
