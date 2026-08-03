_call_to_seq!(C, s, ::Type{<:Real}) = real(to_seq!(zeros(float(eltype(C)), s), C))
_call_to_seq!(C, s, ::Type) = to_seq!(C, s)



# helper function

_apply!(f!, C::AbstractArray{T,N₁}, space::TensorSpace{<:NTuple{N₂,BaseSpace}}) where {T,N₁,N₂} =
    @inbounds f!(_apply!(f!, C, Base.tail(space)), space[1], Val(N₁-N₂+1))
_apply!(f!, C::AbstractArray{T,N}, space::TensorSpace{<:Tuple{BaseSpace}}) where {T,N} =
    @inbounds f!(C, space[1], Val(N))
_apply!(f!, C::AbstractVector, space::BaseSpace) = f!(C, space)



# dimension for DFT and FFT

fft_size(s::TensorSpace) = map(_fft_size, spaces(s))
fft_size(s::BaseSpace) = (_fft_size(s),)
fft_size(s::SymmetricSpace) = fft_size(desymmetrize(s))

_fft_size(s::BaseSpace) = nextpow(2, _dft_dimension(s))

_dft_dimension(s::BaseSpace) = 2order(s)+1
_dft_dimension(s::Chebyshev) = 2order(s)+!ispow2(order(s))

_is_fft_size_compatible(n::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds _is_fft_size_compatible(n[1], s[1]) & _is_fft_size_compatible(Base.tail(n), Base.tail(s))
_is_fft_size_compatible(n::Tuple{Integer}, s::TensorSpace{<:Tuple{BaseSpace}}) = @inbounds _is_fft_size_compatible(n[1], s[1])
_is_fft_size_compatible(n::Tuple{Integer}, s::BaseSpace) = @inbounds _is_fft_size_compatible(n[1], s)
_is_fft_size_compatible(n::Integer, s::BaseSpace) = ispow2(n) & (_dft_dimension(s) ≤ n)

# dimension of sampling grid

grid_size(s::TensorSpace) = map(_grid_size, spaces(s))
grid_size(s::BaseSpace) = (_grid_size(s),)
grid_size(s::SymmetricSpace) = grid_size(desymmetrize(s))

_grid_size(s::BaseSpace) = _fft_size(s)
_grid_size(s::Chebyshev) = _fft_size(s)÷2+1

_is_grid_size_compatible(n::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    @inbounds _is_grid_size_compatible(n[1], s[1]) & _is_grid_size_compatible(Base.tail(n), Base.tail(s))
_is_grid_size_compatible(n::Tuple{Integer}, s::TensorSpace{<:Tuple{BaseSpace}}) = @inbounds _is_grid_size_compatible(n[1], s[1])
_is_grid_size_compatible(n::Tuple{Integer}, s::BaseSpace) = @inbounds _is_grid_size_compatible(n[1], s)
_is_grid_size_compatible(n::Integer, s::BaseSpace) = _is_fft_size_compatible(n, s)
_is_grid_size_compatible(n::Integer, s::Chebyshev) = _is_fft_size_compatible(_full_fft_size(n, s), s)

# recover fft size from sampling grid

_full_fft_size(sz::Tuple{Vararg{Integer}}, s::SymmetricSpace) = _full_fft_size(sz, desymmetrize(s))
_full_fft_size(sz::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = map(_full_fft_size, sz, spaces(s))
_full_fft_size(sz::Tuple{Integer}, s::BaseSpace) = (_full_fft_size(sz[1], s),)
_full_fft_size(sz::Tuple{Vararg{Integer}}, ::SequenceSpace) = sz
_full_fft_size(m::Integer, ::BaseSpace) = m
_full_fft_size(m::Integer, ::Chebyshev) = ifelse((m > 2) & ispow2(m-1), 2*(m-1), m)



# sequence to grid
# uses the backward (unnormalized inverse) FFT: Y[j] = Σₖ C[k] e^{+2πi kj/N}

to_grid(a::Sequence{<:SequenceSpace}, m::Integer) = to_grid(a, (m,))
to_grid(a::Sequence{<:SequenceSpace}, m::NTuple{D,Integer} = fft_size(space(a))) where {D} =
    to_grid!(_grid_buffer(eltype(a), last(_lead_inner(space(a), Val(D))), m), a)

to_grid!(C::AbstractArray{<:Number}, a::Sequence{<:SymmetricSpace}) = to_grid!(C, Projection(desymmetrize(space(a))) * a)

function to_grid!(x_grid::AbstractArray{<:Sequence,D}, a::Sequence{<:SequenceSpace}) where {D}
    s_lead, inner = _lead_inner(space(a), Val(D))
    all(x -> space(x) == inner, x_grid) || return throw(ArgumentError("the grid elements must have space $inner"))
    C = _no_alloc_reshape(coefficients(a), (dimension(s_lead), dimension(inner)))
    return _fill_grid!(x_grid, C, s_lead)
end
function _fill_grid!(x_grid::AbstractArray{<:Any,D}, C::AbstractMatrix, s_lead::NoSymSpace) where {D}
    cache = Array{_grid_eltype(x_grid),D}(undef, size(x_grid))
    @inbounds for j ∈ axes(C, 2)
        to_grid!(cache, Sequence(s_lead, view(C, :, j)))
        @inbounds for (i, x) ∈ enumerate(x_grid)
            coefficients(x)[j] = cache[i]
        end
    end
    return x_grid
end

function to_grid!(C::AbstractArray{<:Number}, a::Sequence{<:NoSymSpace})
    sz = size(C)
    Base.OneTo.(sz) == axes(C) || return throw(ArgumentError("offset arrays are not supported"))
    space_a = space(a)
    full_sz = _full_fft_size(sz, space_a)
    if sz != full_sz # user grid with folded Chebyshev axes: transform in a full-size buffer, then fold
        _is_grid_size_compatible(sz, space_a) || return throw(DimensionMismatch("the grid size must be compatible with the space: size is $sz, space is $space_a"))
        C_full = to_grid!(zeros(eltype(C), full_sz), a)
        C .= view(C_full, map(n -> 1:n, sz)...)
        return C
    end
    _is_fft_size_compatible(sz, space_a) || return throw(DimensionMismatch("the grid size must be compatible with the space: size is $sz, space is $space_a"))
    C .= zero(eltype(C))
    A = _no_alloc_reshape(a)
    @inbounds view(C, axes(A)...) .= A
    _apply!(_preprocess_to_grid!, C, space_a)
    return _bfft_pow2!(C)
end

#--

_grid_buffer(::Type{T}, ::ScalarSpace, m::Tuple{Vararg{Integer}}) where {T} =
    zeros(complex(float(T)), m)
_grid_buffer(::Type{T}, inner::SequenceSpace, m::Tuple{Vararg{Integer}}) where {T} =
    [Sequence(inner, Vector{complex(float(T))}(undef, dimension(inner))) for _ ∈ CartesianIndices(m)]

# the grid may be an abstractly typed container (e.g. `Matrix{Sequence}`), so the
# coefficient type is derived from the elements rather than from `eltype(eltype(x_grid))`
_grid_eltype(x_grid) = mapreduce(eltype, promote_type, x_grid)

function _lead_inner(s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Val{D}) where {N,D}
    0 < D < N || return throw(ArgumentError("the grid must have at least one axis and at most one axis per factor of $s"))
    a, b = ntuple(i -> s[i], Val(D)), ntuple(i -> s[D+i], Val(N-D))
    return _maybe_tensorspace(a), _maybe_tensorspace(b)
end
function _lead_inner(s::SymmetricSpace{<:TensorSpace{<:NTuple{N,BaseSpace}}}, ::Val{D}) where {N,D}
    s_lead, rest = _lead_inner(desymmetrize(s), Val(D))
    return s_lead, SymmetricSpace(rest, _restrict(symmetry(s), Val(D)))
end
_lead_inner(s::Union{BaseSpace,SymmetricSpace{<:BaseSpace}}, ::Val{1}) = (s, ScalarSpace())
_lead_inner(s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Val{N}) where {N} = (s, ScalarSpace())
_lead_inner(s::SymmetricSpace{<:TensorSpace{<:NTuple{N,BaseSpace}}}, ::Val{N}) where {N} = (s, ScalarSpace())
_lead_inner(s::SequenceSpace, ::Val) =
    throw(ArgumentError("the grid must have at least one axis and at most one axis per factor of $s"))
_maybe_tensorspace(t::Tuple{BaseSpace}) = @inbounds t[1]
_maybe_tensorspace(t::Tuple{Vararg{BaseSpace}}) = TensorSpace(t)
function _restrict(G::Group{N,T}, ::Val{D}) where {N,T,D}
    # restrict a symmetry group acting trivially on the first `D` indices to the
    # trailing indices (inverse of `⊗` with a `NoSymSpace`)
    els = Set{GroupElement{N-D,T}}()
    for g ∈ elements(G)
        A = g.index_action.matrix
        ϕ = g.coef_action.phase
        (all(A[i,j] == (i == j) for i ∈ 1:D, j ∈ 1:N) &&
         all(iszero(A[i,j]) for i ∈ D+1:N, j ∈ 1:D) &&
         all(iszero(ϕ[i]) for i ∈ 1:D)) ||
            return throw(ArgumentError("the symmetry group does not act trivially on the leading $D factors"))
        push!(els, GroupElement(
            IndexAction(StaticArrays.SMatrix{N-D,N-D,Int}(view(A, D+1:N, D+1:N))),
            CoefAction(g.coef_action.amplitude, StaticArrays.SVector{N-D,Rational{Int}}(view(ϕ, D+1:N)))))
    end
    return unsafe_group!(els)
end

#--

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
    C = [complex(f(_node(s, j, _node_size(s, N))...)) for j ∈ CartesianIndices(Base.UnitRange.(0, N .- 1))]
    return to_seq!(C, s)
end

_node_size(::BaseSpace, N::Tuple{Integer}) = @inbounds N[1]
_node_size(::TensorSpace, N::Tuple{Vararg{Integer}}) = N
_node_size(s::SymmetricSpace, N::Tuple{Vararg{Integer}}) = _node_size(desymmetrize(s), N)

_node(s::TensorSpace, j, N) = map((sᵢ, jᵢ, Nᵢ) -> _node(sᵢ, jᵢ, Nᵢ), spaces(s), Tuple(j), N)
_node(::Taylor, j, N) = cispi(2j[1]/N)
_node(s::Fourier, j, N) = 2π/frequency(s)*j[1]/N
_node(::Chebyshev, j, N) = cospi(2j[1]/N)
_node(s::SymmetricSpace, j, N) = _node(desymmetrize(s), j, N)

#

to_seq(A::AbstractArray, space::SequenceSpace) =
    to_seq!(_unfold_grid(A, _full_fft_size(size(A), space)), space)
to_seq(x_grid::AbstractArray{<:Sequence}, s::SequenceSpace) = to_seq!(_seq_buffer(x_grid, s), x_grid, s)
to_seq(::AbstractArray{<:Sequence}, s::SymmetricSpace) = throw(ArgumentError(_grid_factors_message(s)))
# `to_grid` splits the factors of the grid off with `_restrict`, which is only
# defined when the symmetry group acts trivially on them
_grid_factors_message(s::SymmetricSpace) = "the factors of the grid must not be symmetric, got $s"

to_seq!(A::AbstractArray, space::SequenceSpace) = to_seq!(zeros(complex(float(eltype(A))), space), A)
to_seq!(c::Sequence{<:SymmetricSpace}, A::AbstractArray) = project!(c, to_seq!(zeros(eltype(c), desymmetrize(space(c))), A))

function to_seq!(c::Sequence, x_grid::AbstractArray{<:Sequence,D}, s::NoSymSpace) where {D}
    _check_grid_axes(s, Val(D))
    inner = space(first(x_grid))
    all(x -> space(x) == inner, x_grid) || return throw(ArgumentError("all sequences must have the same space"))
    space(c) == _combine(s, inner) || return throw(ArgumentError("the destination must have space $(_combine(s, inner))"))
    C = _no_alloc_reshape(coefficients(c), (dimension(s), dimension(inner)))
    _fill_seq!(C, x_grid, s)
    return c
end
function _fill_seq!(C::AbstractMatrix, x_grid::AbstractArray{<:Any,D}, s::NoSymSpace) where {D}
    cache = Array{complex(float(_grid_eltype(x_grid))),D}(undef, size(x_grid))
    @inbounds for j ∈ axes(C, 2)
        for (i, x) ∈ enumerate(x_grid)
            cache[i] = coefficients(x)[j]
        end
        C[:,j] .= coefficients(to_seq(cache, s))
    end
    return C
end

function to_seq!(c::Sequence{<:NoSymSpace}, A::AbstractArray)
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    space_c = space(c)
    full_sz = _full_fft_size(sz, space_c)
    if sz != full_sz # user grid with folded Chebyshev axes: unfold, then transform
        _is_grid_size_compatible(sz, space_c) || return throw(DimensionMismatch("the grid size must be compatible with the space: size is $sz, space is $space_c"))
        return to_seq!(c, _unfold_grid(A, full_sz))
    end
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

#--

function _unfold_grid(A::AbstractArray{T,N}, full_sz::NTuple{N,Integer}) where {T,N}
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    C = zeros(complex(float(T)), full_sz)
    @inbounds view(C, map(n -> 1:n, sz)...) .= A
    for d ∈ 1:N
        m, n = sz[d], full_sz[d]
        if m < n
            @inbounds selectdim(C, d, m+1:n) .= selectdim(C, d, m-1:-1:2)
        end
    end
    return C
end

_seq_buffer(x_grid::AbstractArray{<:Sequence}, s::NoSymSpace) =
    zeros(complex(float(_grid_eltype(x_grid))), _combine(s, space(first(x_grid))))

_check_grid_axes(::BaseSpace, ::Val{1}) = nothing
_check_grid_axes(::TensorSpace{<:NTuple{D,BaseSpace}}, ::Val{D}) where {D} = nothing
_check_grid_axes(s::SequenceSpace, ::Val) =
    throw(ArgumentError("the grid must have one axis per factor of $s"))

_combine(s::SequenceSpace, ::ScalarSpace) = s
_combine(s::SequenceSpace, inner::SequenceSpace) = s ⊗ inner

#--

# Taylor: DFT output already in coefficient order

_fft_get_index(n::Integer, space::Taylor) = 1:min(n, dimension(space)), 1:min(n, dimension(space))

_postprocess_to_seq!(C::AbstractVector, ::Taylor) = C
_postprocess_to_seq!(C::AbstractArray, ::Taylor, ::Val) = C

# Fourier: move zero-frequency from position 1 to center

function _fft_get_index(n::Integer, space::Fourier)
    ord_C = order(space)
    ord_A = n÷2
    ord_A ≤ ord_C && return ord_C+1-ord_A:ord_C+(n == 1)+ord_A, 1:n
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

const FFT_ALGORITHM = Ref(:interval) # default

function set_fft_algorithm(algo::Symbol)
    algo ∉ (:interval, :apriori_bound) && return throw(ArgumentError("algorithm must be :interval or :apriori_bound"))
    FFT_ALGORITHM[] = algo
    return algo
end

#-

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

# twiddle factors e^{-iπ j/len} for j = 0, ..., len-1

#= NOTE

A transform of length `n` runs `log2(n)` stages, the `k`-th combining elements
`2^(k-1)` apart. The last stage needs `n÷2` distinct factors and every earlier
one an evenly spaced subset of them, so a single table of that length serves
them all.

Every angle is a dyadic rational, hence exact, so building the table at the
working precision is as tight as building it higher and rounding.

Tables are memoised on element type, precision and length,
the precision belonging to the key because `BigFloat` sets it at runtime.
=#

struct RootsOfUnity{T<:AbstractFloat}
    interval :: Vector{Complex{Interval{T}}}
    mid :: Vector{Complex{T}}
    radius :: T
end

function RootsOfUnity{T}(len::Integer) where {T<:AbstractFloat}
    W = [cispi(interval(T, -j//len)) for j ∈ 0:len-1]
    ρ = maximum(w -> sup(abs(w - mid(w))), W)
    return RootsOfUnity{T}(W, mid.(W), ρ)
end

const roots_of_unity = Dict{Tuple{DataType,Int,Int},RootsOfUnity}() # (type, precision, length)
const roots_of_unity_lock = ReentrantLock()

function _roots_of_unity(::Type{T}, len::Integer) where {T<:AbstractFloat}
    key = (T, precision(T), len)
    return lock(roots_of_unity_lock) do
        return get!(() -> RootsOfUnity{T}(len), roots_of_unity, key)::RootsOfUnity{T}
    end
end

_twiddle_table(::Type{Complex{Interval{T}}}, len::Integer) where {T<:AbstractFloat} = _roots_of_unity(T, len).interval
_twiddle_table(::Type{Complex{T}}, len::Integer) where {T<:AbstractFloat} = _roots_of_unity(T, len).mid

# Backward (unnormalized inverse) FFT: Y[j] = Σₖ x[k] e^{+2πi kj/N}

_bfft_pow2!(a::AbstractArray{<:Complex}) = conj!(_fft_pow2!(conj!(a)))

# Forward FFT: X[k] = Σⱼ x[j] e^{-2πi kj/N}

_fft_pow2!(a::AbstractArray{<:Complex{<:AbstractFloat}}) = _fft_pow2_table!(a)

function _fft_pow2!(a::AbstractArray{Complex{Interval{T}}}) where {T<:AbstractFloat}
    FFT_ALGORITHM[] === :apriori_bound && return _fft_apriori_bound!(a)
    return _fft_pow2_table!(a)
end

function _fft_pow2_table!(a::AbstractArray)
    @inbounds for i ∈ axes(a, 1)
        _fft_pow2_table!(selectdim(a, 1, i))
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_pow2_table!(a_col)
    end
    return a
end

_fft_pow2_table!(a::AbstractVector) =
    _fft_pow2_table!(a, _twiddle_table(eltype(a), max(length(a)÷2, 1)))

function _fft_pow2_table!(a::AbstractVector, W::AbstractVector)
    _bitreverse!(a)
    n = length(a)
    len = length(W)
    N = 2
    while N ≤ n
        N½ = N÷2
        stride = len ÷ N½
        for k ∈ 1:N:n
            @inbounds for (i, j) ∈ enumerate(k:k+N½-1)
                j′ = j + N½
                aj′_ω = a[j′] * W[(i-1)*stride+1]
                a[j′] = a[j] - aj′_ω
                a[j] = a[j] + aj′_ω
            end
        end
        N <<= 1
    end
    return a
end

#-

_modulus(x::T, y::T) where {T<:AbstractFloat} = sup(abs(complex(interval(x), interval(y))))

function _fft_apriori_bound!(a::AbstractArray{Complex{Interval{T}}}) where {T<:AbstractFloat}
    C = Array{Complex{T}}(undef, size(a))
    eʳ = eⁱ = eˢ = zero(T)
    Mʳ = Mⁱ = Mˢ = zero(T)
    @inbounds for i ∈ eachindex(a)
        re, im = real(a[i]), imag(a[i])
        mʳ, mⁱ = mid(re), mid(im)
        rʳ, rⁱ = radius(re), radius(im)
        C[i] = complex(mʳ, mⁱ)
        eʳ = max(eʳ, rʳ) ; eⁱ = max(eⁱ, rⁱ) ; eˢ = max(eˢ, rʳ + rⁱ)
        Mʳ = max(Mʳ, abs(mʳ)) ; Mⁱ = max(Mⁱ, abs(mⁱ)) ; Mˢ = max(Mˢ, abs(mʳ) + abs(mⁱ))
    end
    e₀ = min(nextfloat(eˢ), _modulus(eʳ, eⁱ)) # bounds |a[i] - C[i]|
    M₀ = min(nextfloat(Mˢ), _modulus(Mʳ, Mⁱ)) # bounds |C[i]|
    _fft_pow2_table!(C)
    ρ = maximum(n -> _roots_of_unity(T, max(n÷2, 1)).radius, size(a))
    e = _fft_error_bound(interval(M₀) + interval(e₀), interval(e₀), interval(ρ), sum(trailing_zeros, size(a)))
    @inbounds for i ∈ eachindex(a)
        a[i] = interval(C[i], e; format = :midpoint)
    end
    return a
end

function _fft_error_bound(M::Interval{T}, e::Interval{T}, ρ::Interval{T}, nstages::Integer) where {T<:AbstractFloat}
    # a stage maps a bound `e` on the absolute error and a bound `M` on the magnitude to
    #
    #     Et = γ(M+e)(1+ρ) + e(1+ρ) + Mρ     error of `t = a[j′] * ω`
    #     Tt = (M+e)(1+ρ)(1+γ)               magnitude of `t`
    #     e  ← e + Et + u(M + e + Tt)        error of `a[j] ± t`
    #     M  ← (M + Tt)(1+u)
    #
    u = interval(eps(T))/interval(2) # unit roundoff
    γ = sqrt(interval(T, 5)) * u # Brent-Percival-Zimmermann bound
    for _ ∈ 1:nstages
        Et = γ*(M+e)*(interval(1)+ρ) + e*(interval(1)+ρ) + M*ρ
        Tt = (M+e)*(interval(1)+ρ)*(interval(1)+γ)
        e = e + Et + u*(M + e + Tt)
        M = (M + Tt)*(interval(1)+u)
    end
    return sup(e)
end
