# the grid comes fresh from `to_grid` at the call site, so it is fed to `to_coef!`
_call_to_coef!(C, s, ::Type{<:Real}) = real(to_coef!(_coef_buffer(C, s), C))
_call_to_coef!(C, s, ::Type) = to_coef!(_coef_buffer(C, s), C)



# helper function

_apply!(f!, C::AbstractArray{T,N₁}, space::TensorSpace{<:NTuple{N₂,BaseSpace}}) where {T,N₁,N₂} =
    @inbounds f!(_apply!(f!, C, Base.tail(space)), space[1], Val(N₁-N₂+1))
_apply!(f!, C::AbstractArray{T,N}, space::TensorSpace{<:Tuple{BaseSpace}}) where {T,N} =
    @inbounds f!(C, space[1], Val(N))
_apply!(f!, C::AbstractVector, space::BaseSpace) = f!(C, space)



# dimension for FFT

"""
    fft_size(s::SequenceSpace)

Return the size of the discrete transform underlying `s`, one entry per factor. For `Chebyshev` this counts the *mirrored* grid, so it is larger than [`grid_size`](@ref); elsewhere the two agree.

Prefer [`grid_size`](@ref) when choosing how finely to sample: it is the number of nodes actually required.

# Examples

```jldoctest
julia> fft_size(Chebyshev(2)), grid_size(Chebyshev(2))
((4,), (3,))
```

See also: [`grid_size`](@ref), [`to_grid`](@ref) and [`to_coef`](@ref).
"""
fft_size(s::TensorSpace) = map(_fft_size, spaces(s))
fft_size(s::BaseSpace) = (_fft_size(s),)
fft_size(s::SymmetricSpace) = fft_size(desymmetrize(s))

_fft_size(s::Taylor) = order(s)+1 # TODO: really?
_fft_size(s::Fourier) = 2order(s)+1
_fft_size(s::Chebyshev) = max(2order(s), 1) # the coefficients are mirrored

# dimension of sampling grid

"""
    grid_size(s::SequenceSpace)

Return, as a tuple with one entry per factor of `s`, the number of sampling nodes that determines `s` exactly. In other words, the smallest grid size on which [`to_grid`](@ref) and [`to_coef`](@ref) are inverse to one another.

The nodes are:
- the roots of unity for `Taylor`,
- the equispaced points of the period for `Fourier`,
- the Chebyshev-Lobatto points for `Chebyshev`, ordered from ``x = 1`` down to ``x = -1``.

# Examples

```jldoctest
julia> grid_size(Taylor(2)), grid_size(Fourier(2, 1.0)), grid_size(Chebyshev(2))
((3,), (5,), (3,))
```

See also: [`to_grid`](@ref), [`to_coef`](@ref) and [`fft_size`](@ref).
"""
grid_size(s::TensorSpace) = map(_grid_size, spaces(s))
grid_size(s::BaseSpace) = (_grid_size(s),)
grid_size(s::SymmetricSpace) = grid_size(desymmetrize(s))

_grid_size(s::BaseSpace) = _fft_size(s)
_grid_size(s::Chebyshev) = _fft_size(s)÷2+1 # the mirrored nodes are dropped

fast_grid_size(s::SequenceSpace) = fast_grid_size(grid_size(s), s)
fast_grid_size(sz::Tuple{Vararg{Integer}}, s::SymmetricSpace) = fast_grid_size(sz, desymmetrize(s))
fast_grid_size(sz::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = map(_fast_grid_size, sz, spaces(s))
fast_grid_size(sz::Tuple{Integer}, s::BaseSpace) = (_fast_grid_size(sz[1], s),)

_fast_grid_size(m::Integer, ::BaseSpace) = nextpow(2, m)
_fast_grid_size(m::Integer, ::Chebyshev) = m == 1 ? 1 : nextpow(2, m-1)+1 # the mirror is what must be a power of two



# recover fft size from sampling grid

_full_fft_size(sz::Tuple{Vararg{Integer}}, s::SymmetricSpace) = _full_fft_size(sz, desymmetrize(s))
_full_fft_size(sz::NTuple{N,Integer}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = map(_full_fft_size, sz, spaces(s))
_full_fft_size(sz::Tuple{Integer}, s::BaseSpace) = (_full_fft_size(sz[1], s),)
_full_fft_size(sz::Tuple{Vararg{Integer}}, ::SequenceSpace) = sz
_full_fft_size(m::Integer, ::BaseSpace) = m
_full_fft_size(m::Integer, ::Chebyshev) = max(2*(m-1), 1) # the nodes unfold onto their mirror image

# a grid fits a space whenever the transform it unfolds to is large enough

function _check_grid_size(sz::Tuple{Vararg{Integer}}, s::SequenceSpace)
    all(map(≤, fft_size(s), _full_fft_size(sz, s))) ||
        throw(DimensionMismatch("the grid size must be compatible with the space: size is $sz, space is $s"))
    return nothing
end



# sequence to grid
# uses the backward (unnormalized inverse) FFT: Y[j] = Σₖ C[k] e^{+2πi kj/N}

"""
    to_grid(a::Sequence, m = grid_size(space(a)))

Evaluate `a` at the sampling nodes of its space and return the array of values.

`m` is a tuple of grid sizes, one per discretized axis; an `Integer` is accepted as shorthand for a single axis. Any size from [`grid_size`](@ref) upwards is allowed.

Giving fewer sizes than `space(a)` has factors discretizes only the leading factors and returns a grid of `Sequence`s on the remaining ones.

See also: [`to_coef`](@ref), [`grid_size`](@ref) and [`to_grid!`](@ref).
"""
to_grid(a::Sequence{<:SequenceSpace}, m::Integer) = to_grid(a, (m,))
to_grid(a::Sequence{<:SequenceSpace}, m::NTuple{D,Integer} = grid_size(space(a))) where {D} =
    to_grid!(_grid_buffer(eltype(a), last(_lead_inner(space(a), Val(D))), m), a)

"""
    to_grid!(x_grid, a::Sequence)

In-place version of [`to_grid`](@ref), writing the sampled values into `x_grid`.

See also: [`to_grid`](@ref) and [`to_coef!`](@ref).
"""
function to_grid!(x_grid::AbstractArray{<:Sequence,D}, a::Sequence{<:SequenceSpace}) where {D}
    s_lead, inner = _lead_inner(space(a), Val(D))
    all(x -> space(x) == inner, x_grid) || return throw(ArgumentError("the grid elements must have space $inner"))
    _check_grid_size(size(x_grid), s_lead)
    C = _no_alloc_reshape(coefficients(a), (dimension(s_lead), dimension(inner)))
    return _fill_grid!(x_grid, C, s_lead)
end
function _fill_grid!(x_grid::AbstractArray{<:Any,D}, C::AbstractMatrix, s_lead::NoSymSpace) where {D}
    sz = size(x_grid)
    cache = Array{_grid_eltype(x_grid),D}(undef, _full_fft_size(sz, s_lead))
    nodes = view(cache, map(n -> 1:n, sz)...)
    @inbounds for j ∈ axes(C, 2)
        _to_grid!(cache, Sequence(s_lead, view(C, :, j)))
        for (i, x) ∈ enumerate(x_grid)
            coefficients(x)[j] = nodes[i]
        end
    end
    return x_grid
end

to_grid!(C::AbstractArray{<:Number}, a::Sequence{<:SymmetricSpace}) = to_grid!(C, Projection(desymmetrize(space(a))) * a)
function to_grid!(C::AbstractArray{<:Number}, a::Sequence{<:NoSymSpace})
    sz = size(C)
    Base.OneTo.(sz) == axes(C) || return throw(ArgumentError("offset arrays are not supported"))
    space_a = space(a)
    _check_grid_size(sz, space_a)
    full_sz = _full_fft_size(sz, space_a)
    sz == full_sz && return _to_grid!(C, a)
    # mirrored axes: transform in a full-size buffer, then keep the nodes
    C .= view(_to_grid!(zeros(eltype(C), full_sz), a), map(n -> 1:n, sz)...)
    return C
end
function _to_grid!(C::AbstractArray, a::Sequence)
    # `C` is sized as the transform, not as the grid
    C .= zero(eltype(C))
    A = _no_alloc_reshape(a)
    @inbounds view(C, axes(A)...) .= A
    _apply!(_preprocess_to_grid!, C, space(a))
    return _bfft!(C)
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
    # on the tightest grid the top mode is its own mirror image
    2ord == len && @inbounds C[ord+1] *= exact(2)
    return C
end

function _preprocess_to_grid!(C::AbstractArray, space::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    ord = order(space)
    @inbounds selectdim(C, D, len:-1:len+1-ord) .= selectdim(C, D, 2:ord+1)
    2ord == len && @inbounds selectdim(C, D, ord+1) .*= exact(2)
    return C
end



# grid to sequence
# uses the forward FFT: X[k] = Σⱼ x[j] e^{-2πi kj/N}, then divides by N

# function interpolation

"""
    to_coef(f::Function, s::SequenceSpace)
    to_coef(a::Sequence, s::SequenceSpace)
    to_coef(x_grid::AbstractArray, s::SequenceSpace)

Interpolate onto `s`, returning a [`Sequence`](@ref).

A grid of `Sequence`s is also accepted, in which case only the leading factors are interpolated.

See also: [`to_grid`](@ref), [`grid_size`](@ref) and [`to_coef!`](@ref).
"""
to_coef(a::Sequence, s::SequenceSpace) = to_coef(to_grid(a), s)

function to_coef(f::Function, s::SequenceSpace)
    m = grid_size(s)
    N = _full_fft_size(m, s)
    C = [complex(f(_node(s, j, _node_size(s, N))...)) for j ∈ CartesianIndices(Base.UnitRange.(0, m .- 1))]
    return to_coef(C, s)
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

to_coef(x_grid::AbstractArray, s::SequenceSpace) = to_coef!(_coef_buffer(x_grid, s), _maybe_copy_grid(x_grid))
_maybe_copy_grid(A::AbstractArray{<:Number}) = _unfold_grid(A, size(A)) # in-place, copy needed
_maybe_copy_grid(x_grid::AbstractArray) = x_grid # only read from, no copy needed

"""
    to_coef!(c::Sequence, x_grid)

In-place version of [`to_coef`](@ref), writing the interpolant into `c` and potentially also using `x_grid` as a buffer.

See also: [`to_coef`](@ref) and [`to_grid!`](@ref).
"""
function to_coef!(c::Sequence, x_grid::AbstractArray{<:Sequence,D}) where {D}
    s_lead, inner = _lead_inner(space(c), Val(D))
    all(x -> space(x) == inner, x_grid) || return throw(ArgumentError("the grid elements must have space $inner"))
    C = _no_alloc_reshape(coefficients(c), (dimension(s_lead), dimension(inner)))
    _fill_coef!(C, x_grid, s_lead)
    return c
end
function _fill_coef!(C::AbstractMatrix, x_grid::AbstractArray{<:Any,D}, s_lead::NoSymSpace) where {D}
    sz = size(x_grid)
    cache = Array{complex(float(_grid_eltype(x_grid))),D}(undef, _full_fft_size(sz, s_lead))
    @inbounds nodes = view(cache, map(n -> 1:n, sz)...)
    @inbounds for j ∈ axes(C, 2)
        for (i, x) ∈ enumerate(x_grid)
            nodes[i] = coefficients(x)[j]
        end
        _to_coef!(Sequence(s_lead, view(C, :, j)), _mirror_grid!(cache, sz))
    end
    return C
end

to_coef!(c::Sequence{<:SymmetricSpace}, A::AbstractArray{<:Number}) =
    project!(c, to_coef!(zeros(eltype(c), desymmetrize(space(c))), A))
function to_coef!(c::Sequence{<:NoSymSpace}, A::AbstractArray{<:Number})
    sz = size(A)
    Base.OneTo.(sz) == axes(A) || return throw(ArgumentError("offset arrays are not supported"))
    full_sz = _full_fft_size(sz, space(c))
    return _to_coef!(c, sz == full_sz ? A : _unfold_grid(A, full_sz)) # mirrored axes need more room
end
function _to_coef!(c::Sequence, A::AbstractArray)
    # `A` is sized as the transform, not as the grid
    sz = size(A)
    _fft!(A)
    A ./= exact(prod(sz))
    _apply!(_postprocess_to_coef!, A, space(c))
    C = _no_alloc_reshape(c)
    C .= zero(eltype(c))
    inds_C, inds_A = _fft_get_index(sz, space(c))
    @inbounds view(C, inds_C...) .= view(A, inds_A...)
    return c
end
function _fft_get_index(n::NTuple{N,Integer}, space::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    v = map(_fft_get_index, n, spaces(space))
    return @inbounds ntuple(i -> v[i][1], Val(N)), ntuple(i -> v[i][2], Val(N))
end
_fft_get_index(n::Tuple{Integer}, space::BaseSpace) = @inbounds map(tuple, _fft_get_index(n[1], space))
_fft_get_index(n::Integer, space::BaseSpace) = 1:min(n, dimension(space)), 1:min(n, dimension(space))

#--

function _unfold_grid(A::AbstractArray{T,N}, full_sz::NTuple{N,Integer}) where {T,N}
    C = zeros(complex(float(T)), full_sz)
    view(C, map(n -> 1:n, size(A))...) .= A
    return _mirror_grid!(C, size(A))
end

function _mirror_grid!(C::AbstractArray{<:Any,N}, sz::NTuple{N,Integer}) where {N}
    for d ∈ 1:N
        m, n = sz[d], size(C, d)
        if m < n
            selectdim(C, d, m+1:n) .= selectdim(C, d, m-1:-1:2)
        end
    end
    return C
end

_coef_buffer(x_grid::AbstractArray{<:Number}, s::SequenceSpace) =
    zeros(complex(float(eltype(x_grid))), s)
_coef_buffer(x_grid::AbstractArray{<:Sequence}, s::NoSymSpace) =
    zeros(complex(float(_grid_eltype(x_grid))), _combine(s, space(first(x_grid))))

_combine(s::SequenceSpace, ::ScalarSpace) = s
_combine(s::SequenceSpace, inner::SequenceSpace) = s ⊗ inner

#--

# Taylor: DFT output already in coefficient order

_postprocess_to_coef!(C::AbstractVector, ::Taylor) = C
_postprocess_to_coef!(C::AbstractArray, ::Taylor, ::Val) = C

# Fourier: move zero-frequency from position 1 to center

function _fft_get_index(n::Integer, space::Fourier)
    ord_C = order(space)
    ord_A = n÷2
    ord_A ≤ ord_C && return ord_C+1-ord_A:ord_C+isodd(n)+ord_A, 1:n # every mode fits
    return 1:2ord_C+1, ord_A+1-ord_C:ord_A+1+ord_C
end

function _postprocess_to_coef!(C::AbstractVector, ::Fourier)
    circshift!(C, copy(C), length(C)÷2)
    return C
end

function _postprocess_to_coef!(C::AbstractArray{T,N}, ::Fourier, ::Val{D}) where {T,N,D}
    circshift!(C, copy(C), ntuple(i -> ifelse(i == D, size(C, D)÷2, 0), Val(N)))
    return C
end

# Chebyshev: halve the Nyquist frequency, which carries both mirror images

_fft_get_index(n::Integer, space::Chebyshev) = 1:min(n÷2+1, dimension(space)), 1:min(n÷2+1, dimension(space))

function _postprocess_to_coef!(C::AbstractVector, ::Chebyshev)
    len = length(C)
    iseven(len) && @inbounds C[len÷2+1] /= exact(2)
    return C
end

function _postprocess_to_coef!(C::AbstractArray, ::Chebyshev, ::Val{D}) where {D}
    len = size(C, D)
    iseven(len) && @inbounds selectdim(C, D, len÷2+1) ./= exact(2)
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

# twiddle factors e^{-iπ j/len} for j = 0, ..., len-1

#= NOTE

A radix-2 transform of length `n` runs `log2(n)` stages, the `k`-th combining
elements `2^(k-1)` apart. The last stage needs `n÷2` distinct factors and every
earlier one an evenly spaced subset of them, so a single table of that length
serves them all. The other routines read the same table by half-turns, hence the
angles `-πj/len` rather than `-2πj/len`.

For a power of two every angle is a dyadic rational, hence exact, so building the
table at the working precision is as tight as building it higher and rounding.

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

# ω_n^t = e^{-2πi t/n}, read off the half-turn table of length `n`

_root(W::AbstractVector, t::Integer, n::Integer) = 2t < n ? @inbounds(W[2t+1]) : @inbounds(-W[2t-n+1])

# Backward (unnormalized inverse) FFT: Y[j] = Σₖ x[k] e^{+2πi kj/N}

_bfft!(a::AbstractArray{<:Complex}) = conj!(_fft!(conj!(a)))

# Forward FFT: X[k] = Σⱼ x[j] e^{-2πi kj/N}

_fft!(a::AbstractArray{<:Complex}) = _fft_table!(a)

function _fft_table!(a::AbstractArray)
    @inbounds for i ∈ axes(a, 1)
        _fft_table!(selectdim(a, 1, i))
    end
    n = size(a, 1)
    for a_col ∈ eachcol(_no_alloc_reshape(a, (n, length(a)÷n)))
        _fft_table!(a_col)
    end
    return a
end

#= NOTE

Every length is supported: a power of two runs the radix-2 routine, a composite
length `n = r m` is decimated into `r` transforms of length `m` (Cooley-Tukey),
and a prime length is summed term by term or, once that gets too costly, turned
into a convolution of power-of-two length (Bluestein).
=#

function _fft_table!(a::AbstractVector)
    n = length(a)
    ispow2(n) && return _fft_radix2!(a)
    r = _smallest_prime_factor(n)
    r < n && return _fft_decimate!(a, r)
    n ≤ 32 && return _fft_naive!(a) # beyond that length Bluestein is faster, at a slightly wider enclosure
    return _fft_bluestein!(a)
end

function _smallest_prime_factor(n::Integer)
    r = 2
    while r*r ≤ n
        n % r == 0 && return r
        r += 1
    end
    return n
end

_fft_radix2!(a::AbstractVector) =
    _fft_radix2!(a, _twiddle_table(eltype(a), max(length(a)÷2, 1)))

function _fft_radix2!(a::AbstractVector, W::AbstractVector)
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

# X[k] = Σ_q ω_n^{qk} Y_q[k mod m], with Y_q the transform of x[q], x[q+r], ...

function _fft_decimate!(a::AbstractVector, r::Integer)
    n = length(a)
    m = n ÷ r
    W = _twiddle_table(eltype(a), n)
    Y = [_fft!(a[q:r:n]) for q ∈ 1:r]
    @inbounds for k ∈ 0:n-1
        i = k % m + 1
        aₖ = Y[1][i]
        for q ∈ 1:r-1
            aₖ += _root(W, (q*k) % n, n) * Y[q+1][i]
        end
        a[k+1] = aₖ
    end
    return a
end

function _fft_naive!(a::AbstractVector)
    n = length(a)
    W = _twiddle_table(eltype(a), n)
    x = copy(a)
    @inbounds for k ∈ 0:n-1
        aₖ = x[1]
        for j ∈ 1:n-1
            aₖ += x[j+1] * _root(W, (j*k) % n, n)
        end
        a[k+1] = aₖ
    end
    return a
end

#-

#= NOTE

Bluestein's algorithm turns a transform of prime length `n` into a convolution of
power-of-two length: since `2jk = j² + k² - (k-j)²`,

    X[k] = wₖ Σⱼ (x[j] wⱼ) conj(w_{k-j}),   wₘ = e^{-iπ m²/n},

whose kernel `conj(w)` is even and gets padded to a length `2^p ≥ 2n-1`. The
angles are rationals of denominator `n`, with `m²` reduced modulo `2n` to keep
them small; the transformed kernel is memoised alongside the chirp.
=#

struct Chirp{T<:AbstractFloat}
    interval :: Vector{Complex{Interval{T}}}
    mid :: Vector{Complex{T}}
    kernel_interval :: Vector{Complex{Interval{T}}}
    kernel_mid :: Vector{Complex{T}}
end

function Chirp{T}(n::Integer) where {T<:AbstractFloat}
    w = [cispi(interval(T, -((m*m) % (2n))//n)) for m ∈ 0:n-1]
    B = zeros(Complex{Interval{T}}, nextpow(2, 2n-1))
    @inbounds for m ∈ 0:n-1
        B[m+1] = conj(w[m+1])
        m > 0 && (B[end+1-m] = B[m+1])
    end
    return Chirp{T}(w, mid.(w), _fft!(copy(B)), _fft!(mid.(B)))
end

const chirps = Dict{Tuple{DataType,Int,Int},Chirp}() # (type, precision, length)
const chirps_lock = ReentrantLock()

function _chirp(::Type{T}, n::Integer) where {T<:AbstractFloat}
    key = (T, precision(T), n)
    return lock(chirps_lock) do
        return get!(() -> Chirp{T}(n), chirps, key)::Chirp{T}
    end
end

_chirp_table(::Type{Complex{Interval{T}}}, n::Integer) where {T<:AbstractFloat} =
    (c = _chirp(T, n); (c.interval, c.kernel_interval))
_chirp_table(::Type{Complex{T}}, n::Integer) where {T<:AbstractFloat} =
    (c = _chirp(T, n); (c.mid, c.kernel_mid))

function _fft_bluestein!(a::AbstractVector)
    n = length(a)
    w, B̂ = _chirp_table(eltype(a), n)
    u = zeros(eltype(a), length(B̂))
    @inbounds u[1:n] .= a .* w
    _fft!(u)
    u .*= B̂
    _bfft!(u)
    @inbounds a .= w .* view(u, 1:n) ./ exact(length(B̂))
    return a
end
