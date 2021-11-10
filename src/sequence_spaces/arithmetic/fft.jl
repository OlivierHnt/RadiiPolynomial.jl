# dimension of discrete Fourier series

fft_size(s₁::BaseSpace, s₂::BaseSpace) = (_fft_length(s₁, s₂),)

fft_size(s₁::BaseSpace, s₂::BaseSpace, s₃::BaseSpace...) = (_fft_length(s₁, s₂, s₃...),)

fft_size(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    map(_fft_length, spaces(s₁), spaces(s₂))

fft_size(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}, s₃::TensorSpace{<:NTuple{N,BaseSpace}}...) where {N} =
    ntuple(i -> nextpow(2, _dfs_dimension(s₁, i) + _dfs_dimension(s₂, i) + mapreduce(s₃_ -> _dfs_dimension(s₃_, i), +, s₃)), Val(N))

fft_size(s::BaseSpace, n::Int) = (_fft_length(s, n),)

fft_size(s::TensorSpace, n::Int) = map(sᵢ -> _fft_length(sᵢ, n), spaces(s))

_fft_length(s₁::BaseSpace, s₂::BaseSpace) =
    nextpow(2, _dfs_dimension(s₁) + _dfs_dimension(s₂))

_fft_length(s₁::BaseSpace, s₂::BaseSpace, s₃::BaseSpace...) =
    nextpow(2, _dfs_dimension(s₁) + _dfs_dimension(s₂) + mapreduce(_dfs_dimension, +, s₃))

_fft_length(s::BaseSpace, n::Int) = nextpow(2, n * _dfs_dimension(s))

#

_dfs_dimensions(s::TensorSpace) = map(_dfs_dimension, spaces(s))

_dfs_dimension(s::TensorSpace, i::Int) = _dfs_dimension(s[i])

# Taylor
_dfs_dimension(s::Taylor) = dimension(s)
# Fourier
_dfs_dimension(s::Fourier) = dimension(s)
# Chebyshev
_dfs_dimension(s::Chebyshev) = 2order(s)+1

# sequence to discrete Fourier series

fft(a::Sequence{<:BaseSpace}, n::Tuple{Int}) = @inbounds fft(a, n[1])

function fft(a::Sequence{<:BaseSpace}, n::Int)
    space_a = space(a)
    ispow2(n) & (_dfs_dimension(space_a) ≤ n) || return throw(DimensionMismatch)
    CoefType = complex(eltype(a))
    C = zeros(CoefType, n)
    A = coefficients(a)
    @inbounds view(C, eachindex(A)) .= A
    _preprocess_data_sequence2dfs!(space_a, C)
    return _fft_pow2!(C)
end

function fft!(C::AbstractVector, a::Sequence{<:BaseSpace})
    space_a = space(a)
    n = length(C)
    ispow2(n) & (_dfs_dimension(space_a) ≤ n) || return throw(DimensionMismatch)
    C .= zero(eltype(C))
    A = coefficients(a)
    @inbounds view(C, eachindex(A)) .= A
    _preprocess_data_sequence2dfs!(space_a, C)
    return _fft_pow2!(C)
end

function fft(a::Sequence{TensorSpace{T}}, n::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace}}
    space_a = space(a)
    all(ispow2, n) & all(i -> _dfs_dimension(space_a, i) ≤ n[i], 1:N) || return throw(DimensionMismatch)
    CoefType = complex(eltype(a))
    C = zeros(CoefType, n)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    @inbounds view(C, axes(A)...) .= A
    _apply_preprocess_data_sequence2dfs!(space_a, C)
    return _fft_pow2!(C)
end

function fft!(C::AbstractArray{T,N}, a::Sequence{TensorSpace{S}}) where {T,N,S<:NTuple{N,BaseSpace}}
    space_a = space(a)
    n = size(C)
    all(ispow2, n) & all(i -> _dfs_dimension(space_a, i) ≤ n[i], 1:N) || return throw(DimensionMismatch)
    C .= zero(T)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    @inbounds view(C, axes(A)...) .= A
    _apply_preprocess_data_sequence2dfs!(space_a, C)
    return _fft_pow2!(C)
end

_apply_preprocess_data_sequence2dfs!(space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _preprocess_data_sequence2dfs!(space[1], Val(N₂-N₁+1), _apply_preprocess_data_sequence2dfs!(Base.tail(space), A))

_apply_preprocess_data_sequence2dfs!(space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _preprocess_data_sequence2dfs!(space[1], Val(N), A)

# Taylor

_preprocess_data_sequence2dfs!(::Taylor, A) = A

_preprocess_data_sequence2dfs!(::Taylor, ::Val, A) = A

# Fourier

_preprocess_data_sequence2dfs!(::Fourier, A) = A

_preprocess_data_sequence2dfs!(::Fourier, ::Val, A) = A

# Chebyshev

function _preprocess_data_sequence2dfs!(space::Chebyshev, A)
    len = length(A)
    @inbounds for i ∈ 1:order(space)
        A[len+1-i] = A[i+1]
    end
    return A
end

function _preprocess_data_sequence2dfs!(space::Chebyshev, ::Val{D}, A) where {D}
    len = size(A, D)
    @inbounds for i ∈ 1:order(space)
        selectdim(A, D, len+1-i) .= selectdim(A, D, i+1)
    end
    return A
end

# discrete Fourier series to sequence

function ifft!(c::Sequence{<:BaseSpace}, A::AbstractVector)
    space_c = space(c)
    n = length(A)
    ispow2(n) & (dimension(space_c) ≤ n) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    @inbounds coefficients(c) .= view(A, 1:dimension(space_c))
    return c
end

function rifft!(c::Sequence{<:BaseSpace}, A::AbstractVector)
    space_c = space(c)
    n = length(A)
    ispow2(n) & (dimension(space_c) ≤ n) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    @inbounds coefficients(c) .= real.(view(A, 1:dimension(space_c)))
    return c
end

function ifft!(c::Sequence{TensorSpace{T}}, A::AbstractArray{S,N}) where {N,T<:NTuple{N,BaseSpace},S}
    space_c = space(c)
    all(i -> ispow2(size(A, i)) & (dimension(space_c, i) ≤ size(A, i)), 1:N) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    @inbounds C .= view(A, map(sᵢ -> 1:dimension(sᵢ), spaces(space_c))...)
    return c
end

function rifft!(c::Sequence{TensorSpace{T}}, A::AbstractArray{S,N}) where {N,T<:NTuple{N,BaseSpace},S}
    space_c = space(c)
    all(i -> ispow2(size(A, i)) & (dimension(space_c, i) ≤ size(A, i)), 1:N) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    @inbounds C .= real.(view(A, map(sᵢ -> 1:dimension(sᵢ), spaces(space_c))...))
    return c
end

# FFT routines

function _fft_pow2!(a::AbstractArray{<:Complex})
    sz = size(a, 1)
    a_ = _no_alloc_reshape(a, (sz, length(a)÷sz))
    for a_row ∈ eachrow(a_)
        _fft_pow2!(a_row)
    end
    for a_col ∈ eachcol(a_)
        _fft_pow2!(a_col)
    end
    return a
end

function _ifft_pow2!(a::AbstractArray{<:Complex})
    sz = size(a, 1)
    a_ = _no_alloc_reshape(a, (sz, length(a)÷sz))
    for a_row ∈ eachrow(a_)
        _ifft_pow2!(a_row)
    end
    for a_col ∈ eachcol(a_)
        _ifft_pow2!(a_col)
    end
    return a
end

function _fft_pow2!(a::AbstractVector{Complex{T}}) where {T}
    _bitreverse!(a)
    n = length(a)
    N = 2
    while N ≤ n
        N½ = N÷2
        ω_ = cispi(-one(T)/N½)
        for k ∈ 1:N:n
            ω = one(Complex{T})
            @inbounds for j ∈ k:k+N½-1
                j′ = j + N½
                aj′_ω = a[j′] * ω
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
    _bitreverse!(a)
    π_ = convert(T, π)
    n = length(a)
    Ω = Vector{Complex{T}}(undef, n÷2)
    N = 2
    @inbounds while N ≤ n
        N½ = N÷2
        π2N⁻¹ = π_/N½
        view(Ω, 1:N½) .= cis.((-π2N⁻¹) .* (0:N½-1))
        for k ∈ 1:N:n
            @inbounds for (i, j) ∈ enumerate(k:k+N½-1)
                j′ = j + N½
                aj′_ω = a[j′] * Ω[i]
                a[j′] = a[j] - aj′_ω
                a[j] = a[j] + aj′_ω
            end
        end
        N <<= 1
    end
    return a
end

function _ifft_pow2!(a::AbstractVector{<:Complex})
    conj!(_fft_pow2!(conj!(a)))
    a ./= length(a)
    return a
end

function _bitreverse!(a::AbstractVector)
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
