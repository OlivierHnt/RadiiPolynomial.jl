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
    _is_fft_size_compatible(n, space_a) || return throw(DimensionMismatch)
    CoefType = complex(eltype(a))
    C = zeros(CoefType, n)
    A = coefficients(a)
    @inbounds view(C, eachindex(A)) .= A
    _preprocess!(C, space_a)
    return _fft_pow2!(C)
end

function fft!(C::AbstractVector, a::Sequence{<:BaseSpace})
    l = length(C)
    Base.OneTo(l) == eachindex(C) || return throw(ArgumentError("offset vectors are not supported"))
    space_a = space(a)
    _is_fft_size_compatible(l, space_a) || return throw(DimensionMismatch)
    C .= zero(eltype(C))
    A = coefficients(a)
    @inbounds view(C, eachindex(A)) .= A
    _preprocess!(C, space_a)
    return _fft_pow2!(C)
end

function fft(a::Sequence{TensorSpace{T}}, n::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace}}
    space_a = space(a)
    _is_fft_size_compatible(n, space_a) || return throw(DimensionMismatch)
    CoefType = complex(eltype(a))
    C = zeros(CoefType, n)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    @inbounds view(C, axes(A)...) .= A
    _apply_preprocess!(C, space_a)
    return _fft_pow2!(C)
end

function fft!(C::AbstractArray{T,N}, a::Sequence{TensorSpace{S}}) where {T,N,S<:NTuple{N,BaseSpace}}
    sz = size(C)
    Base.OneTo.(sz) == axes(C) || return throw(ArgumentError("offset arrays are not supported"))
    space_a = space(a)
    _is_fft_size_compatible(sz, space_a) || return throw(DimensionMismatch)
    C .= zero(T)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    @inbounds view(C, axes(A)...) .= A
    _apply_preprocess!(C, space_a)
    return _fft_pow2!(C)
end

_is_fft_size_compatible(n, s) = ispow2(n) & (_dfs_dimension(s) ≤ n)
_is_fft_size_compatible(n::Tuple, s) = @inbounds _is_fft_size_compatible(n[1], s[1]) & _is_fft_size_compatible(Base.tail(n), Base.tail(s))
_is_fft_size_compatible(n::Tuple{Int}, s) = @inbounds _is_fft_size_compatible(n[1], s[1])

_apply_preprocess!(C::AbstractArray{T,N₁}, space::TensorSpace{<:NTuple{N₂,BaseSpace}}) where {T,N₁,N₂} =
    @inbounds _preprocess!(_apply_preprocess!(C, Base.tail(space)), space[1], Val(N₁-N₂+1))

_apply_preprocess!(C::AbstractArray{T,N}, space::TensorSpace{<:Tuple{BaseSpace}}) where {T,N} =
    @inbounds _preprocess!(C, space[1], Val(N))

# Taylor

_preprocess!(C::AbstractVector, ::Taylor) = C

_preprocess!(C::AbstractArray, ::Taylor, ::Val) = C

# Fourier

function _preprocess!(C::AbstractVector, space::Fourier)
    circshift!(C, copy(C), -order(space))
    return C
end

function _preprocess!(C::AbstractArray{T,N}, space::Fourier, ::Val{D}) where {T,N,D}
    ord = order(space)
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, -ord, 0), Val(N)))
    return C
end

# Chebyshev

function _preprocess!(C::AbstractVector, space::Chebyshev)
    len = length(C)
    @inbounds for i ∈ 2:order(space)+1
        C[len+2-i] = C[i]
    end
    return C
end

function _preprocess!(C::AbstractArray, space::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    @inbounds for i ∈ 2:order(space)+1
        selectdim(C, D, len+2-i) .= selectdim(C, D, i)
    end
    return C
end

# discrete Fourier series to sequence

function ifft!(A::AbstractVector{T}, space::BaseSpace) where {T}
    l = length(A)
    Base.OneTo(l) == eachindex(A) || return throw(ArgumentError("offset vectors are not supported"))
    _is_ifft_size_compatible(l, space) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _postprocess!(A, space)
    C = Vector{complex(T)}(undef, dimension(space))
    @inbounds C .= view(A, eachindex(C))
    return Sequence(space, C)
end

function rifft!(A::AbstractVector{T}, space::BaseSpace) where {T}
    l = length(A)
    Base.OneTo(l) == eachindex(A) || return throw(ArgumentError("offset vectors are not supported"))
    _is_ifft_size_compatible(l, space) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _postprocess!(A, space)
    C = Vector{real(T)}(undef, dimension(space))
    @inbounds C .= real.(view(A, eachindex(C)))
    return Sequence(space, C)
end

function ifft!(c::Sequence{<:BaseSpace}, A::AbstractVector)
    l = length(A)
    Base.OneTo(l) == eachindex(A) || return throw(ArgumentError("offset vectors are not supported"))
    space_c = space(c)
    _is_ifft_size_compatible(l, space_c) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _postprocess!(A, space_c)
    C = coefficients(c)
    @inbounds C .= view(A, eachindex(C))
    return c
end

function rifft!(c::Sequence{<:BaseSpace}, A::AbstractVector)
    l = length(A)
    Base.OneTo(l) == eachindex(A) || return throw(ArgumentError("offset vectors are not supported"))
    space_c = space(c)
    _is_ifft_size_compatible(l, space_c) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _postprocess!(A, space_c)
    C = coefficients(c)
    @inbounds C .= real.(view(A, eachindex(C)))
    return c
end

function ifft!(A::AbstractArray{T,N}, space::TensorSpace{<:NTuple{N,BaseSpace}}) where {T,N}
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    _is_ifft_size_compatible(sz, space) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _apply_postprocess!(A, space)
    c = Sequence(space, Vector{complex(T)}(undef, dimension(space)))
    C = _no_alloc_reshape(coefficients(c), dimensions(space))
    @inbounds C .= view(A, axes(C)...)
    return c
end

function rifft!(A::AbstractArray{T,N}, space::TensorSpace{<:NTuple{N,BaseSpace}}) where {T,N}
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    _is_ifft_size_compatible(sz, space) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _apply_postprocess!(A, space)
    c = Sequence(space, Vector{real(T)}(undef, dimension(space)))
    C = _no_alloc_reshape(coefficients(c), dimensions(space))
    @inbounds C .= real.(view(A, axes(C)...))
    return c
end

function ifft!(c::Sequence{TensorSpace{T}}, A::AbstractArray{S,N}) where {N,T<:NTuple{N,BaseSpace},S}
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    space_c = space(c)
    _is_ifft_size_compatible(sz, space_c) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _apply_postprocess!(A, space_c)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    @inbounds C .= view(A, axes(C)...)
    return c
end

function rifft!(c::Sequence{TensorSpace{T}}, A::AbstractArray{S,N}) where {N,T<:NTuple{N,BaseSpace},S}
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    space_c = space(c)
    _is_ifft_size_compatible(sz, space_c) || return throw(DimensionMismatch)
    _ifft_pow2!(A)
    _apply_postprocess!(A, space_c)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    @inbounds C .= real.(view(A, axes(C)...))
    return c
end

_is_ifft_size_compatible(n, s) = ispow2(n) & (dimension(s) ≤ n)
_is_ifft_size_compatible(n::Tuple, s) = @inbounds _is_ifft_size_compatible(n[1], s[1]) & _is_ifft_size_compatible(Base.tail(n), Base.tail(s))
_is_ifft_size_compatible(n::Tuple{Int}, s) = @inbounds _is_ifft_size_compatible(n[1], s[1])

_apply_postprocess!(C::AbstractArray{T,N₁}, space::TensorSpace{<:NTuple{N₂,BaseSpace}}) where {T,N₁,N₂} =
    @inbounds _postprocess!(_apply_postprocess!(C, Base.tail(space)), space[1], Val(N₁-N₂+1))

_apply_postprocess!(C::AbstractArray{T,N}, space::TensorSpace{<:Tuple{BaseSpace}}) where {T,N} =
    @inbounds _postprocess!(C, space[1], Val(N))

# Taylor

_postprocess!(C, ::Taylor) = C

_postprocess!(C, ::Taylor, ::Val) = C

# Fourier

function _postprocess!(C::AbstractVector, space::Fourier)
    circshift!(C, copy(C), order(space))
    return C
end

function _postprocess!(C::AbstractArray{T,N}, space::Fourier, ::Val{D}) where {T,N,D}
    ord = order(space)
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, ord, 0), Val(N)))
    return C
end

# Chebyshev

_postprocess!(C, ::Chebyshev) = C

_postprocess!(C, ::Chebyshev, ::Val) = C

# FFT routines

function _fft_pow2!(a::AbstractArray{<:Complex})
    @inbounds for i ∈ axes(a, 1)
        _fft_pow2!(selectdim(a, 1, i))
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_pow2!(a_col)
    end
    return a
end

_fft_pow2!(a::AbstractArray{<:Complex{<:Interval}}) = _fft_pow2!(a, Vector{eltype(a)}(undef, maximum(size(a))÷2))

function _fft_pow2!(a::AbstractArray{<:Complex{<:Interval}}, Ω)
    @inbounds for i ∈ axes(a, 1)
        _fft_pow2!(selectdim(a, 1, i), Ω)
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_pow2!(a_col, Ω)
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

_fft_pow2!(a::AbstractVector{<:Complex{<:Interval}}) = _fft_pow2!(a, Vector{eltype(a)}(undef, length(a)÷2))

function _fft_pow2!(a::AbstractVector{Complex{T}}, Ω) where {T<:Interval}
    _bitreverse!(a)
    π_ = convert(T, π)
    n = length(a)
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

function _ifft_pow2!(a::AbstractArray{<:Complex})
    @inbounds for i ∈ axes(a, 1)
        _ifft_pow2!(selectdim(a, 1, i))
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _ifft_pow2!(a_col)
    end
    return a
end

_ifft_pow2!(a::AbstractArray{<:Complex{<:Interval}}) = _ifft_pow2!(a, Vector{eltype(a)}(undef, maximum(size(a))÷2))

function _ifft_pow2!(a::AbstractArray{<:Complex{<:Interval}}, Ω)
    @inbounds for i ∈ axes(a, 1)
        _ifft_pow2!(selectdim(a, 1, i), Ω)
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _ifft_pow2!(a_col, Ω)
    end
    return a
end

function _ifft_pow2!(a::AbstractVector{<:Complex})
    conj!(_fft_pow2!(conj!(a)))
    a ./= length(a)
    return a
end

function _ifft_pow2!(a::AbstractVector{<:Complex{<:Interval}}, Ω)
    conj!(_fft_pow2!(conj!(a), Ω))
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
