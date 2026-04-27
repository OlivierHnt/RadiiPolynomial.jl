_call_to_seq!(C, s, ::Type{<:Real}) = real(to_seq!(zeros(float(eltype(C)), s), C))
_call_to_seq!(C, s, ::Type) = to_seq!(C, s)



# helper function

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

fft_size(s::SymmetricSpace) = fft_size(desymmetrize(s))



# sequence to grid
# uses the backward (unnormalized inverse) FFT: Y[j] = Σₖ C[k] e^{+2πi kj/N}

to_grid(a::Sequence{<:SequenceSpace}, n=fft_size(space(a))) = to_grid!(zeros(complex(float(eltype(a))), n), a)

to_grid!(C::AbstractArray, a::Sequence{<:SymmetricSpace}) = to_grid!(C, Projection(desymmetrize(space(a))) * a)

function to_grid!(C::AbstractArray, a::Sequence{<:NoSymSpace})
    sz = size(C)
    Base.OneTo.(sz) == axes(C) || return throw(ArgumentError("offset arrays are not supported"))
    space_a = space(a)
    _is_fft_size_compatible(sz, space_a) || return throw(DimensionMismatch)
    C .= zero(eltype(C))
    A = _no_alloc_reshape(a)
    @inbounds view(C, axes(A)...) .= A
    _apply!(_preprocess_to_grid!, C, space_a)
    return _bfft_pow2!(C)
end

_is_fft_size_compatible(n::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds _is_fft_size_compatible(n[1], s[1]) & _is_fft_size_compatible(Base.tail(n), Base.tail(s))
_is_fft_size_compatible(n::Tuple{Integer}, s::TensorSpace{<:Tuple{BaseSpace}}) = @inbounds _is_fft_size_compatible(n[1], s[1])
_is_fft_size_compatible(n::Tuple{Integer}, s::BaseSpace) = @inbounds _is_fft_size_compatible(n[1], s)
_is_fft_size_compatible(n::Integer, s::BaseSpace) = ispow2(n) & (_dft_dimension(s) ≤ n)

# Taylor: coefficients already in standard DFT order

_preprocess_to_grid!(C::AbstractVector, ::Taylor) = C
_preprocess_to_grid!(C::AbstractArray, ::Taylor, ::Val) = C

# Fourier: move zero-frequency from center to position 1

function _preprocess_to_grid!(C::AbstractVector, space::Fourier)
    circshift!(C, copy(C), -order(space))
    return C
end

function _preprocess_to_grid!(C::AbstractArray{T,N}, space::Fourier, ::Val{D}) where {T,N,D}
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, -order(space), 0), Val(N)))
    return C
end

# Chebyshev: mirror

function _preprocess_to_grid!(C::AbstractVector, space::Chebyshev)
    len = length(C)
    ord = order(space)
    @inbounds view(C, len:-1:len+1-ord) .= view(C, 2:ord+1)
    if len != 1
        @inbounds C[len÷2+1] *= exact(2)
    end
    return C
end

function _preprocess_to_grid!(C::AbstractArray, space::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    ord = order(space)
    @inbounds selectdim(C, D, len:-1:len+1-ord) .= selectdim(C, D, 2:ord+1)
    if len != 1
        @inbounds selectdim(C, D, len÷2+1) .*= exact(2)
    end
    return C
end



# grid to sequence
# uses the forward FFT: X[k] = Σⱼ x[j] e^{-2πi kj/N}, then divides by N

# function interpolation

to_seq(a::Sequence, s::SequenceSpace) = to_seq!(to_grid(a, fft_size(space(a))), s)

function to_seq(f::Function, s::SequenceSpace)
    N = fft_size(s)
    C = [complex(f(_node(s, j, N)...)) for j ∈ CartesianIndices(Base.UnitRange.(0, Tuple(N) .- 1))]
    return to_seq!(C, s)
end

_node(s::TensorSpace, j, N) = map((sᵢ, jᵢ, Nᵢ) -> _node(sᵢ, jᵢ, Nᵢ), spaces(s), Tuple(j), N)
_node(::Taylor, j, N) = cispi(2j[1]/N)
_node(s::Fourier, j, N) = 2π/frequency(s)*j[1]/N
_node(::Chebyshev, j, N) = cospi(2j[1]/N)
_node(s::SymmetricSpace, j, N) = _node(desymmetrize(s), j, N)

#

to_seq(A::AbstractArray, space::SequenceSpace) = to_seq!(complex.(A), space) # complex copy

to_seq!(A::AbstractArray, space::SequenceSpace) = to_seq!(zeros(complex(float(eltype(A))), space), A)

to_seq!(c::Sequence{<:SymmetricSpace}, A::AbstractArray) = project!(c, to_seq!(Projection(desymmetrize(space(c))) * c, A))

function to_seq!(c::Sequence{<:NoSymSpace}, A::AbstractArray)
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    all(ispow2, sz) || return throw(ArgumentError("all sizes must be a power of 2"))
    _fft_pow2!(A)
    A ./= exact(prod(sz))
    _apply!(_postprocess_to_seq!, A, space(c))
    C = _no_alloc_reshape(c)
    C .= zero(eltype(c))
    inds_C, inds_A = _fft_get_index(sz, space(c))
    @inbounds view(C, inds_C...) .= view(A, inds_A...)
    return c
end

function _fft_get_index(n::NTuple{N,Integer}, space::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    v = map(_fft_get_index, n, spaces(space))
    return ntuple(i -> v[i][1], Val(N)), ntuple(i -> v[i][2], Val(N))
end
_fft_get_index(n::Tuple{Integer}, space::BaseSpace) = @inbounds map(tuple, _fft_get_index(n[1], space))

# Taylor: DFT output already in coefficient order

_fft_get_index(n::Integer, space::Taylor) = 1:min(n, dimension(space)), 1:min(n, dimension(space))

_postprocess_to_seq!(C::AbstractVector, ::Taylor) = C
_postprocess_to_seq!(C::AbstractArray, ::Taylor, ::Val) = C

# Fourier: move zero-frequency from position 1 to center

function _fft_get_index(n::Integer, space::Fourier)
    ord_C = order(space)
    ord_A = n÷2
    ord_A ≤ ord_C && return ord_C+1-ord_A:ord_C+ord_A, 1:n
    return 1:2ord_C+1, ord_A+1-ord_C:ord_A+1+ord_C
end

function _postprocess_to_seq!(C::AbstractVector, ::Fourier)
    circshift!(C, copy(C), length(C)÷2)
    return C
end

function _postprocess_to_seq!(C::AbstractArray{T,N}, ::Fourier, ::Val{D}) where {T,N,D}
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, size(C, D)÷2, 0), Val(N)))
    return C
end

# Chebyshev: halve the Nyquist frequency

_fft_get_index(n::Integer, space::Chebyshev) = 1:min(n÷2+1, dimension(space)), 1:min(n÷2+1, dimension(space))

function _postprocess_to_seq!(C::AbstractVector, ::Chebyshev)
    len = length(C)
    if len != 1
        @inbounds C[len÷2+1] /= exact(2) # Nyquist frequency
    end
    return C
end

function _postprocess_to_seq!(C::AbstractArray, ::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    if len != 1
        @inbounds selectdim(C, D, len÷2+1) ./= exact(2) # Nyquist frequency
    end
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


# Backward (unnormalized inverse) FFT: Y[j] = Σₖ x[k] e^{+2πi kj/N}

_bfft_pow2!(a::AbstractArray{<:Complex}) = conj!(_fft_pow2!(conj!(a)))


# Forward FFT: X[k] = Σⱼ x[j] e^{-2πi kj/N}

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



function _fft_pow2!(a::AbstractArray{Complex{Interval{T}}}) where {T<:Union{Float16,Float32,Float64}}
    @inbounds for i ∈ axes(a, 1)
        _fft_pow2!(selectdim(a, 1, i))
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_pow2!(a_col)
    end
    return a
end

function _fft_pow2!(a::AbstractVector{Complex{Interval{T}}}) where {T<:Union{Float16,Float32,Float64}}
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



function _fft_pow2!(a::AbstractArray{<:Complex{<:Interval}})
    Ω = Vector{eltype(a)}(undef, maximum(size(a))÷2)
    @inbounds for i ∈ axes(a, 1)
        _fft_pow2!(selectdim(a, 1, i), Ω)
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_pow2!(a_col, Ω)
    end
    return a
end

_fft_pow2!(a::AbstractVector{<:Complex{<:Interval}}) = _fft_pow2!(a, Vector{eltype(a)}(undef, maximum(size(a))÷2))

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
