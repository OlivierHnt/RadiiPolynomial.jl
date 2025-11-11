_call_ifft!(C, s, ::Type{<:Real}) = rifft!(C, s)
_call_ifft!(C, s, ::Type) = ifft!(C, s)

_apply!(f!, C::AbstractArray{T,N₁}, space::TensorSpace{<:NTuple{N₂,BaseSpace}}) where {T,N₁,N₂} =
    @inbounds f!(_apply!(f!, C, Base.tail(space)), space[1], Val(N₁-N₂+1))
_apply!(f!, C::AbstractArray{T,N}, space::TensorSpace{<:Tuple{BaseSpace}}) where {T,N} =
    @inbounds f!(C, space[1], Val(N))
_apply!(f!, C::AbstractVector, space::BaseSpace) = f!(C, space)



# dimension for DFT and FFT

fft_size(s::TensorSpace) = map(sᵢ -> fft_size(sᵢ), spaces(s))
fft_size(s::BaseSpace) = nextpow(2, _dft_dimension(s))

_dft_dimension(s::BaseSpace) = 2order(s)+1
_dft_dimension(s::Chebyshev) = 2order(s)+!ispow2(order(s))
_dft_dimension(s::CosFourier) = 2order(s)+!ispow2(order(s))



# sequence to grid

fft(a::Sequence{<:BaseSpace}, n::Integer=fft_size(space(a))) =
    fft!(zeros(complex(float(eltype(a))), n), a)
fft(a::Sequence{TensorSpace{T}}, n::NTuple{N,Integer}=fft_size(space(a))) where {N,T<:NTuple{N,BaseSpace}} =
    fft!(zeros(complex(float(eltype(a))), n), a)

function fft!(C::AbstractArray, a::Sequence{<:SequenceSpace})
    sz = size(C)
    Base.OneTo.(sz) == axes(C) || return throw(ArgumentError("offset arrays are not supported"))
    space_a = space(a)
    _is_fft_size_compatible(sz, space_a) || return throw(DimensionMismatch)
    C .= zero(eltype(C))
    A = _no_alloc_reshape(a)
    @inbounds view(C, axes(A)...) .= A
    _apply!(_preprocess!, C, space_a)
    return _fft_pow2!(C)
end

_is_fft_size_compatible(n::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds _is_fft_size_compatible(n[1], s[1]) & _is_fft_size_compatible(Base.tail(n), Base.tail(s))
_is_fft_size_compatible(n::Tuple{Integer}, s::TensorSpace{<:Tuple{BaseSpace}}) = @inbounds _is_fft_size_compatible(n[1], s[1])
_is_fft_size_compatible(n::Tuple{Integer}, s::BaseSpace) = @inbounds _is_fft_size_compatible(n[1], s)
_is_fft_size_compatible(n::Integer, s::BaseSpace) = ispow2(n) & (_dft_dimension(s) ≤ n)

# Taylor

function _preprocess!(C::AbstractVector, space::Taylor)
    len = length(C)
    ord = order(space)
    @inbounds view(C, len:-1:len+1-ord) .= view(C, 2:ord+1)
    @inbounds view(C, 2:ord+1) .= zero(eltype(C))
    return C
end

function _preprocess!(C::AbstractArray, space::Taylor, ::Val{D}) where {D}
    len = size(C, D)
    ord = order(space)
    @inbounds selectdim(C, D, len:-1:len+1-ord) .= selectdim(C, D, 2:ord+1)
    @inbounds selectdim(C, D, 2:ord+1) .= zero(eltype(C))
    return C
end

# Fourier

function _preprocess!(C::AbstractVector, space::Fourier)
    @inbounds reverse!(view(C, 1:dimension(space)))
    circshift!(C, copy(C), -order(space))
    return C
end

function _preprocess!(C::AbstractArray{T,N}, space::Fourier, ::Val{D}) where {T,N,D}
    ord = order(space)
    @inbounds reverse!(selectdim(C, D, 1:dimension(space)); dims = D)
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, -ord, 0), Val(N)))
    return C
end

# Chebyshev

function _preprocess!(C::AbstractVector, space::Chebyshev)
    len = length(C)
    ord = order(space)
    @inbounds view(C, len:-1:len+1-ord) .= view(C, 2:ord+1)
    if len != 1
        @inbounds C[len÷2+1] *= exact(2)
    end
    return C
end

function _preprocess!(C::AbstractArray, space::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    ord = order(space)
    @inbounds selectdim(C, D, len:-1:len+1-ord) .= selectdim(C, D, 2:ord+1)
    if len != 1
        @inbounds selectdim(C, D, len÷2+1) .*= exact(2)
    end
    return C
end

# CosFourier

_preprocess!(C::AbstractVector, space::CosFourier) = _preprocess!(C, Chebyshev(order(space)))
_preprocess!(C::AbstractArray, space::CosFourier, ::Val{D}) where {D} = _preprocess!(C, Chebyshev(order(space)), Val(D))

# SinFourier

function _preprocess!(C::AbstractVector, space::SinFourier)
    len = length(C)
    ord = order(space)
    @inbounds view(C, 2:ord+1) .= view(C, 1:ord) .* complex(exact(false), exact(true))
    @inbounds C[1] = zero(eltype(C))
    @inbounds view(C, len:-1:len+1-ord) .= .- view(C, 2:ord+1)
    return C
end

function _preprocess!(C::AbstractArray, space::SinFourier, ::Val{D}) where {D}
    len = size(C, D)
    ord = order(space)
    @inbounds selectdim(C, D, 2:ord+1) .= selectdim(C, D, 1:ord) .* complex(exact(false), exact(true))
    @inbounds selectdim(C, D, 1) .= zero(eltype(C))
    @inbounds selectdim(C, D, len:-1:len+1-ord) .= .- selectdim(C, D, 2:ord+1)
    return C
end



# grid to sequence

ifft(A::AbstractArray, space::SequenceSpace) = ifft!(complex.(A), space) # complex copy
rifft(A::AbstractArray, space::SequenceSpace) = rifft!(complex.(A), space) # complex copy

ifft!(A::AbstractArray, space::SequenceSpace) = ifft!(zeros(complex(float(eltype(A))), space), A)
rifft!(A::AbstractArray, space::SequenceSpace) = rifft!(zeros(real(float(eltype(A))), space), A)

ifft!(c::Sequence{<:SequenceSpace}, A::AbstractArray) = _ifft!(c, A, identity)
rifft!(c::Sequence{<:SequenceSpace}, A::AbstractArray) = _ifft!(c, A, real)

function _ifft!(c::Sequence{<:SequenceSpace}, A::AbstractArray, f::Union{typeof(identity),typeof(real)})
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    all(ispow2, sz) || return throw(ArgumentError("all sizes must be a power of 2"))
    _ifft_pow2!(A)
    _apply!(_postprocess!, A, space(c))
    C = _no_alloc_reshape(c)
    C .= zero(eltype(c))
    inds_C, inds_A = _ifft_get_index(sz, space(c))
    @inbounds view(C, inds_C...) .= f.(view(A, inds_A...))
    return c
end

function _ifft_get_index(n::NTuple{N,Integer}, space::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    v = map(_ifft_get_index, n, spaces(space))
    return ntuple(i -> v[i][1], Val(N)), ntuple(i -> v[i][2], Val(N))
end
_ifft_get_index(n::Tuple{Integer}, space::BaseSpace) = @inbounds map(tuple, _ifft_get_index(n[1], space))

# Taylor

_ifft_get_index(n::Integer, space::Taylor) = 1:min(n, dimension(space)), 1:min(n, dimension(space))

function _postprocess!(C::AbstractVector, ::Taylor)
    circshift!(C, copy(C), -1)
    reverse!(C)
    return C
end

function _postprocess!(C::AbstractArray{T,N}, ::Taylor, ::Val{D}) where {T,N,D}
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, -1, 0), Val(N)))
    reverse!(C; dims = D)
    return C
end

# Fourier

function _ifft_get_index(n::Integer, space::Fourier)
    ord_C = order(space)
    ord_A = n÷2
    ord_A ≤ ord_C && return ord_C+1-ord_A:ord_C+ord_A, 1:n # accomodate for the Nquilst frequency
    return 1:2ord_C+1, ord_A+1-ord_C:ord_A+1+ord_C
end

function _postprocess!(C::AbstractVector, ::Fourier)
    # C has length 2^n
    ord = length(C)÷2
    circshift!(C, copy(C), ord-1)
    reverse!(C)
    return C
end

function _postprocess!(C::AbstractArray{T,N}, ::Fourier, ::Val{D}) where {T,N,D}
    ord = size(C, D)÷2
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, ord-1, 0), Val(N)))
    reverse!(C; dims = D)
    return C
end

# Chebyshev

_ifft_get_index(n::Integer, space::Chebyshev) = 1:min(n÷2+1, dimension(space)), 1:min(n÷2+1, dimension(space))

function _postprocess!(C::AbstractVector, ::Chebyshev)
    len = length(C)
    if len != 1
        @inbounds C[len÷2+1] /= exact(2) # Nquilst frequency
    end
    return C
end

function _postprocess!(C::AbstractArray, ::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    if len != 1
        @inbounds selectdim(C, D, len÷2+1) ./= exact(2) # Nquilst frequency
    end
    return C
end

# CosFourier

_ifft_get_index(n::Integer, space::CosFourier) = _ifft_get_index(n, Chebyshev(order(space)))

_postprocess!(C::AbstractVector, space::CosFourier) = _postprocess!(C, Chebyshev(order(space)))

_postprocess!(C::AbstractArray, space::CosFourier, ::Val{D}) where {D} = _postprocess!(C, Chebyshev(order(space)), Val(D))

# SinFourier

_ifft_get_index(n::Integer, space::SinFourier) = 1:min(n÷2, dimension(space)), 1:min(n÷2, dimension(space))

function _postprocess!(C::AbstractVector, ::SinFourier)
    ord = length(C) ÷ 2
    @inbounds view(C, 1:ord) .= -complex(exact(false), exact(true)) .* view(C, 2:ord+1)
    return C
end

function _postprocess!(C::AbstractArray, ::SinFourier, ::Val{D}) where {D}
    ord = size(C, D) ÷ 2
    @inbounds selectdim(C, D, 1:ord) .= -complex(exact(false), exact(true)) .* selectdim(C, D, 2:ord+1)
    return C
end



# FFT routines

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

const roots_of_unity = Dict{Tuple{Int,Int},Complex{Interval{Float64}}}()

N_fft = 2^16
for k ∈ 0:N_fft-1
    rat = -k//N_fft
    get!(roots_of_unity, (numerator(rat), denominator(rat))) do
        return setprecision(256) do
            return cispi(interval(BigFloat, rat))
        end
    end
end


#

function _ifft_pow2!(a::AbstractVector{<:Complex})
    conj!(_fft_pow2!(conj!(a)))
    a ./= exact(length(a))
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



function _ifft_pow2!(a::AbstractVector{Complex{Interval{Float64}}})
    conj!(_fft_pow2!(conj!(a)))
    a ./= exact(length(a))
    return a
end

function _ifft_pow2!(a::AbstractArray{Complex{Interval{Float64}}})
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

function _ifft_pow2!(a::AbstractVector{<:Complex{<:Interval}}, Ω)
    conj!(_fft_pow2!(conj!(a), Ω))
    a ./= exact(length(a))
    return a
end

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

#

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



function _fft_pow2!(a::AbstractArray{Complex{Interval{Float64}}})
    @inbounds for i ∈ axes(a, 1)
        _fft_pow2!(selectdim(a, 1, i))
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_pow2!(a_col)
    end
    return a
end

function _fft_pow2!(a::AbstractVector{Complex{Interval{Float64}}})
    _bitreverse!(a)
    n = length(a)
    N = 2
    @inbounds while N ≤ n
        N½ = N÷2
        for k ∈ 1:N:n
            @inbounds for j ∈ k:k+N½-1
                j′ = j + N½
                rat = (k-j)//N½
                ω = get!(roots_of_unity, (numerator(rat), denominator(rat))) do
                    return setprecision(256) do
                        return cispi(interval(BigFloat, rat))
                    end
                end
                aj′_ω = a[j′] * ω
                a[j′] = a[j] - aj′_ω
                a[j] = a[j] + aj′_ω
            end
        end
        N <<= 1
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

function _fft_pow2!(a::AbstractVector{Complex{Interval{T}}}, Ω) where {T}
    _bitreverse!(a)
    n = length(a)
    N = 2
    @inbounds while N ≤ n
        N½ = N÷2
        view(Ω, 1:N½) .= cispi.(interval.(T, (0:-1:1-N½) .// N½))
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
