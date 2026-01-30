const CONV_ALGORITHM = Ref(:sum) # default

function set_conv_algorithm(algo::Symbol)
    algo ∉ (:fft, :sum) && return throw(ArgumentError("algorithm must be :fft or :sum"))
    CONV_ALGORITHM[] = algo
    return algo
end

# multiplication

"""
    *(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})

Compute the discrete convolution (associated with `space(a)` and `space(b)`) of
`a` and `b`.

See also: [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function Base.:*(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = codomain(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    if CONV_ALGORITHM[] === :fft
        A = fft(a, fft_size(space_c))
        B = fft(b, fft_size(space_c))
        C = A .* B
        c = _call_ifft!(C, space_c, CoefType)
        _enforce_zeros!(c, a, b)
        banach_rounding!(c, a, b)
    else # CONV_ALGORITHM[] === :sum
        c = Sequence(space_c, zeros(CoefType, dimension(space_c)))
        _add_mul!(c, a, b, convert(real(CoefType), exact(true)))
    end
    return c
end

function mul(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X)
    space_c = codomain(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    if CONV_ALGORITHM[] === :fft
        A = fft(a, fft_size(space_c))
        B = fft(b, fft_size(space_c))
        C = A .* B
        c = _call_ifft!(C, space_c, CoefType)
        _enforce_zeros!(c, a, b)
        banach_rounding!(c, a, b, X)
    else # CONV_ALGORITHM[] === :sum
        c = Sequence(space_c, zeros(CoefType, dimension(space_c)))
        _add_mul!(c, a, b, convert(real(CoefType), exact(true)))
    end
    return c
end

function mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = codomain(mul_bar, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    if CONV_ALGORITHM[] === :fft
        full_space = codomain(*, space(a), space(b))
        A = fft(a, fft_size(full_space))
        B = fft(b, fft_size(full_space))
        C = A .* B
        c = _call_ifft!(C, space_c, CoefType)
        _enforce_zeros!(c, a, b)
        banach_rounding!(c, a, b)
    else # CONV_ALGORITHM[] === :sum
        c = Sequence(space_c, zeros(CoefType, dimension(space_c)))
        _add_mul!(c, a, b, convert(real(CoefType), exact(true)))
    end
    return c
end

function Base.:*(a::InfiniteSequence, b::InfiniteSequence)
    X = banachspace(a) ∩ banachspace(b)
    full_c = mul(sequence(a), sequence(b), X)
    c = project(full_c, space(a) ∪ space(b))
    @inbounds view(full_c, indices(space(c))) .= zero(eltype(c))
    return InfiniteSequence(c, norm(full_c, X) +
            sequence_norm(a) * sequence_error(b) +
            sequence_norm(b) * sequence_error(a) +
            sequence_error(a) * sequence_error(b),
            X)
end

Base.:*(a::InfiniteSequence, b::Sequence) = a * InfiniteSequence(b, banachspace(a))
Base.:*(a::Sequence, b::InfiniteSequence) = InfiniteSequence(a, banachspace(b)) * b

#-
_to_interval(::Type{T}, x) where {T<:Union{Interval,Complex{<:Interval}}} = interval(zero(T), x; format = :midpoint)

_to_interval(::Type{T}, _) where {T} = zero(T)

function banach_rounding_order(bound::T, X::Ell1{GeometricWeight{T}}) where {T<:AbstractFloat}
    (rate(weight(X)) ≤ 1) | isinf(bound) && return typemax(Int)
    v = bound/eps(T)
    v ≤ 1 && return 0
    order = log(v)/log(rate(weight(X)))
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

function banach_rounding_order(bound::T,  X::Ell1{AlgebraicWeight{T}}) where {T<:AbstractFloat}
    (rate(weight(X)) == 0) | isinf(bound) && return typemax(Int)
    v = bound/eps(T)
    v ≤ 1 && return 0
    order = exp(log(v)/rate(weight(X)))-1
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

for T ∈ (:GeometricWeight, :AlgebraicWeight)
    @eval begin
        function banach_rounding_order(bound_::Real, X::Ell1{<:$T})
            bound, r = promote(float(sup(bound_)), float(sup(rate(weight(X)))))
            return banach_rounding_order(bound, Ell1($T(r)))
        end
    end
end

function banach_rounding_order(bound_::Real, X::Ell1{<:Tuple})
    bound = sup(bound_)
    return map(wᵢ -> banach_rounding_order(bound, Ell1(wᵢ)), weight(X))
end

#

function banach_rounding!(c::Sequence, a::Sequence, b::Sequence)
    X = Ell1(weight(a)) ∩ Ell1(weight(b))
    bound = norm(a, X) * norm(b, X)
    return banach_rounding!(c, bound, X, banach_rounding_order(bound, X))
end

function banach_rounding!(c::Sequence, a::Sequence, n::Integer)
    X = Ell1(weight(a))
    bound = norm(a, X)^n
    return banach_rounding!(c, bound, X, banach_rounding_order(bound, X))
end

function banach_rounding!(c::Sequence, a::Sequence, b::Sequence, X::Ell1)
    bound = norm(a, X) * norm(b, X)
    return banach_rounding!(c, bound, X, banach_rounding_order(bound, X))
end

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{S}}, bound::Real, X::Ell1, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace},S}
    (inf(bound) ≥ 0) & all(≥(0), rounding_order) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    M = typemax(Int)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((i, ord) -> ifelse(ord == M, 0//1, ifelse(ord == 0, 1//1, abs(i) // max(1, ord))), +, α, rounding_order) ≥ 1
            μᵅ = bound / _getindex(weight(X), space_a, α)
            a[α] = _to_interval(S, sup(μᵅ))
        end
    end
    return a
end

# Taylor

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weight(X)))
        μⁱ = bound / _getindex(weight(X), space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            a[i] = _to_interval(T, sup(μⁱ))
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(weight(X), space_a, i)
        a[i] = _to_interval(T, sup(μⁱ))
    end
    return a
end

# Fourier

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weight(X)))
        μⁱ = bound / _getindex(weight(X), space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            x = _to_interval(T, sup(μⁱ))
            a[i] = x
            a[-i] = x
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(weight(X), space_a, i)
        x = _to_interval(T, sup(μⁱ))
        a[i] = x
        a[-i] = x
    end
    return a
end

# Chebyshev

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weight(X)))
        μⁱ = bound / _getindex(weight(X), space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            a[i] = _to_interval(T, sup(μⁱ))
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T}
    (inf(bound) ≥ 0) & (rounding_order ≥ 0) || return throw(DomainError((bound, rounding_order), "the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(weight(X), space_a, i)
        a[i] = _to_interval(T, sup(μⁱ))
    end
    return a
end
#-

#-
_enforce_zeros!(c::Sequence{<:BaseSpace}, a, b) =
    _enforce_zeros!(coefficients(c), coefficients(a), coefficients(b), space(c), space(a), space(b))

_enforce_zeros!(c::Sequence{<:TensorSpace}, a, b) =
    _enforce_zeros!(_no_alloc_reshape(coefficients(c), dimensions(space(c))),
                    _no_alloc_reshape(coefficients(a), dimensions(space(a))),
                    _no_alloc_reshape(coefficients(b), dimensions(space(b))),
                    space(c), space(a), space(b))

_enforce_zeros!(C::AbstractArray{T,N₁}, A, B, space_c::TensorSpace{<:NTuple{N₂,BaseSpace}}, space_a, space_b) where {T,N₁,N₂} =
    @inbounds _enforce_zeros!(_enforce_zeros!(C, A, B, space_c[1], space_a[1], space_b[1], Val(N₁ - N₂ + 1)), A, B, Base.tail(space_c), Base.tail(space_a), Base.tail(space_b))
_enforce_zeros!(C::AbstractArray{T,N}, A, B::AbstractArray, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a, space_b) where {T,N} =
    @inbounds _enforce_zeros!(C, A, B, space_c[1], space_a[1], space_b[1], Val(N))

function _enforce_zeros!(C, A, B, sc::BaseSpace, sa, sb)
    amin, amax = _nonzero_bounds(A, sa)
    bmin, bmax = _nonzero_bounds(B, sb)
    cmin, _ = _get_order_mul(sc, amin, bmin)
    _, cmax = _get_order_mul(sc, amax, bmax)
    CoefType = eltype(C)
    for i ∈ 1:length(C)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            C[i] = zero(CoefType)
        end
    end
    return C
end
function _enforce_zeros!(C, A, B, sc::BaseSpace, sa, sb, ::Val{D}) where {D}
    amin, amax = _nonzero_bounds(A, sa, Val(D))
    bmin, bmax = _nonzero_bounds(B, sb, Val(D))
    cmin, _ = _get_order_mul(sc, amin, bmin)
    _, cmax = _get_order_mul(sc, amax, bmax)
    CoefType = eltype(C)
    @inbounds for i ∈ 1:size(C, D)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            selectdim(C, D, i) .= zero(CoefType)
        end
    end
    return C
end

function _nonzero_bounds(C, s)
    first_idx = 0
    last_idx  = 0
    found = false
    @inbounds for i ∈ 1:length(C)
        if !iszero(C[i])
            if !found
                first_idx = i
                found = true
            end
            last_idx = i
        end
    end
    !found && return 0, -1 # all zeros
    return _index_to_math(s, first_idx), _index_to_math(s, last_idx)
end
function _nonzero_bounds(C, s, ::Val{D}) where {D}
    first_idx = 0
    last_idx  = 0
    found = false
    @inbounds for i ∈ 1:size(C, D)
        if any(!iszero, selectdim(C, D, i))
            if !found
                first_idx = i
                found = true
            end
            last_idx = i
        end
    end
    !found && return 0, -1 # all zeros
    return _index_to_math(s, first_idx), _index_to_math(s, last_idx)
end

_get_order_mul(::Taylor, i, j) = (i+j, i+j)
_index_to_math(::Taylor, j) = j - 1

_get_order_mul(::Fourier, i, j) = (i+j, i+j)
_index_to_math(s::Fourier, j) = j - (div(dimension(s), 2) + 1)

_get_order_mul(::Chebyshev, i, j) = (abs(i-j), i+j)
_index_to_math(::Chebyshev, j) = j - 1

_get_order_mul(::CosFourier, i, j) = (abs(i-j), i+j)
_index_to_math(::CosFourier, j) = j - 1

_get_order_mul(::SinFourier, i, j) = (abs(i-j), i+j)
_index_to_math(::SinFourier, j) = j
#-

function _add_mul!(c::Sequence{<:SequenceSpace}, a, b, α)
    for k ∈ indices(space(c))
        for j ∈ _convolution_indices(space(a), space(b), k)
            c[k] += getcoefficient(a, _extract_valid_index(desymmetrize(space(a)), k, j)) * getcoefficient(b, _extract_valid_index(desymmetrize(space(b)), j))
        end
        c[k] *= α
    end
    return c
end
function _add_mul!(c::Sequence{<:BaseSpace}, a, b, α)
    _add_mul!(coefficients(c), coefficients(a), coefficients(b), α, space(c), space(a), space(b))
    return c
end
function _add_mul!(c::Sequence{<:TensorSpace}, a, b, α)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    B = _no_alloc_reshape(coefficients(b), dimensions(space_b))
    _add_mul!(C, A, B, α, space_c, space_a, space_b)
    return c
end

function _add_mul!(C, A, B, α, space_c::BaseSpace, space_a::BaseSpace, space_b::BaseSpace)
    for i ∈ indices(space_c)
        __convolution!(C, A, B, α, space_c, space_a, space_b, i)
    end
    return C
end
function _add_mul!(C, A, B, α, space_c::TensorSpace{<:NTuple{N,BaseSpace}}, space_a::TensorSpace{<:NTuple{N,BaseSpace}}, space_b::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    remaining_space_c = Base.front(space_c)
    remaining_space_a = Base.front(space_a)
    remaining_space_b = Base.front(space_b)
    @inbounds current_space_c = space_c[N]
    @inbounds current_space_a = space_a[N]
    @inbounds current_space_b = space_b[N]
    for i ∈ indices(current_space_c)
        _convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, i)
    end
    return C
end
_add_mul!(C, A, B, α, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a::TensorSpace{<:Tuple{BaseSpace}}, space_b::TensorSpace{<:Tuple{BaseSpace}}) =
    @inbounds _add_mul!(C, A, B, α, space_c[1], space_a[1], space_b[1])

function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, i) where {T,N}
    @inbounds Cᵢ = selectdim(C, N, _findposition(i, current_space_c))
    @inbounds for j ∈ _convolution_indices(current_space_a, current_space_b, i)
        x = _inverse_symmetry_action(current_space_c, i) * _symmetry_action(current_space_a, i, j) * _symmetry_action(current_space_b, j)
        if !iszero(x)
            _add_mul!(Cᵢ,
                selectdim(A, N, _findposition(_extract_valid_index(current_space_a, i, j), current_space_a)),
                selectdim(B, N, _findposition(_extract_valid_index(current_space_b, j), current_space_b)),
                exact(x) * α, remaining_space_c, remaining_space_a, remaining_space_b)
        end
    end
    return C
end

function __convolution!(C, A, B, α, space_c, space_a, space_b, i)
    Cᵢ = zero(promote_type(eltype(A), eltype(B)))
    @inbounds @simd for j ∈ _convolution_indices(space_a, space_b, i)
        x = _inverse_symmetry_action(space_c, i) * _symmetry_action(space_a, i, j) * _symmetry_action(space_b, j)
        if !iszero(x)
            Cᵢ += exact(x) * A[_findposition(_extract_valid_index(space_a, i, j), space_a)] * B[_findposition(_extract_valid_index(space_b, j), space_b)]
        end
    end
    @inbounds C[_findposition(i, space_c)] += Cᵢ * α
    return C
end

#

codomain(::typeof(*), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> codomain(*, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

codomain(::typeof(mul_bar), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> codomain(mul_bar, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

_convolution_indices(s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    TensorIndices(map(_convolution_indices, spaces(s₁), spaces(s₂), α))

# Taylor

codomain(::typeof(*), s₁::Taylor, s₂::Taylor) = Taylor(order(s₁) + order(s₂))

codomain(::typeof(mul_bar), s₁::Taylor, s₂::Taylor) = intersect(s₁, s₂)

_convolution_indices(s₁::Taylor, s₂::Taylor, i::Int) = max(i-order(s₁), 0):min(i, order(s₂))

# Fourier

function codomain(::typeof(*), s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    ω₁ = frequency(s₁)
    ω₂ = frequency(s₂)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    R = promote_type(T, S)
    return Fourier(order(s₁) + order(s₂), convert(R, ω₁))
end

codomain(::typeof(mul_bar), s₁::Fourier, s₂::Fourier) = intersect(s₁, s₂)

_convolution_indices(s₁::Fourier, s₂::Fourier, i::Int) = intersect(i .- indices(s₁), indices(s₂))

# Chebyshev

codomain(::typeof(*), s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(order(s₁) + order(s₂))

codomain(::typeof(mul_bar), s₁::Chebyshev, s₂::Chebyshev) = intersect(s₁, s₂)

_convolution_indices(s₁::Chebyshev, s₂::Chebyshev, i::Int) = max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

# CosFourier and SinFourier

codomain(::typeof(*), s₁::CosFourier, s₂::CosFourier) = CosFourier(codomain(*, desymmetrize(s₁), desymmetrize(s₂)))
codomain(::typeof(*), s₁::SinFourier, s₂::SinFourier) = CosFourier(codomain(*, desymmetrize(s₁), desymmetrize(s₂)))
codomain(::typeof(*), s₁::CosFourier, s₂::SinFourier) = SinFourier(codomain(*, desymmetrize(s₁), desymmetrize(s₂)))
codomain(::typeof(*), s₁::SinFourier, s₂::CosFourier) = SinFourier(codomain(*, desymmetrize(s₁), desymmetrize(s₂)))

codomain(::typeof(mul_bar), s₁::CosFourier, s₂::CosFourier) = CosFourier(codomain(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))
codomain(::typeof(mul_bar), s₁::SinFourier, s₂::SinFourier) = CosFourier(codomain(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))
codomain(::typeof(mul_bar), s₁::CosFourier, s₂::SinFourier) = SinFourier(codomain(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))
codomain(::typeof(mul_bar), s₁::SinFourier, s₂::CosFourier) = SinFourier(codomain(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))

_convolution_indices(s₁::CosFourier, s₂::CosFourier, i::Int) = _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)
_convolution_indices(s₁::SinFourier, s₂::SinFourier, i::Int) = _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)
_convolution_indices(s₁::CosFourier, s₂::SinFourier, i::Int) = _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)
_convolution_indices(s₁::SinFourier, s₂::CosFourier, i::Int) = _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)

#

function codomain(::typeof(*), s₁::SymmetricSpace, s₂::SymmetricSpace)
    V = codomain(*, desymmetrize(s₁), desymmetrize(s₂))
    G = _group_product(symmetry(s₁), symmetry(s₂))
    return SymmetricSpace(V, G)
end
function _group_product(G₁::Group, G₂::Group)
    D₁ = Dict((idx_action(g), val_action(g).phase, val_action(g).conjugacy) => val_action(g).amplitude for g ∈ elements(G₁))
    D₂ = Dict((idx_action(g), val_action(g).phase, val_action(g).conjugacy) => val_action(g).amplitude for g ∈ elements(G₂))
    elems = Set{GroupElement}()
    for (key, amp₁) ∈ D₁
        if haskey(D₂, key)
            amp₂ = D₂[key]
            push!(elems, GroupElement(key[1], PhaseVal(amp₁ * amp₂, key[2], key[3])))
        end
    end
    return Group(elems)
end

_convolution_indices(s₁::SymmetricSpace, s₂::SymmetricSpace, k::Int) = _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), k)





# integer power

"""
    ^(a::Sequence{<:SequenceSpace}, n::Int)

Compute the discrete convolution (associated with `space(a)`) of `a` with itself
`n` times.

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref).
"""
function Base.:^(a::Sequence{<:SequenceSpace}, n::Integer)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    if CONV_ALGORITHM[] === :fft
        space_c = codomain(^, space(a), n)
        A = fft(a, fft_size(space_c))
        C = A .^ n
        c = _call_ifft!(C, space_c, eltype(a))
        _enforce_zeros!(c, a, n)
        banach_rounding!(c, a, n)
    else # CONV_ALGORITHM[] === :sum
        n == 2 && return _sqr(a)
        # power by squaring
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) > 0
            a = _sqr(a)
        end
        c = a
        while n > 0
            t = trailing_zeros(n) + 1
            n >>= t
            while (t -= 1) ≥ 0
                a = _sqr(a)
            end
            c = c * a
        end
    end
    return c
end

function Base.:^(a::InfiniteSequence, n::Integer)
    n < 0 && return inv(a^(-n))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return a*a
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a *= a
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a *= a
        end
        c = c * a
    end
    return c
end

function pow_bar(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _sqr_bar(a)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _sqr_bar(a)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _sqr_bar(a)
        end
        c = mul_bar(c, a)
    end
    return c
end

#-
_enforce_zeros!(c::Sequence{<:BaseSpace}, a, n::Integer) =
    _enforce_zeros!(coefficients(c), coefficients(a), space(c), space(a), n)

_enforce_zeros!(c::Sequence{<:TensorSpace}, a, n::Integer) =
    _enforce_zeros!(_no_alloc_reshape(coefficients(c), dimensions(space(c))),
                    _no_alloc_reshape(coefficients(a), dimensions(space(a))),
                    space(c), space(a), n)

_enforce_zeros!(C::AbstractArray{T,N₁}, A, space_c::TensorSpace{<:NTuple{N₂,BaseSpace}}, space_a, n) where {T,N₁,N₂} =
    @inbounds _enforce_zeros!(_enforce_zeros!(C, A, space_c[1], space_a[1], n, Val(N₁ - N₂ + 1)), A, Base.tail(space_c), Base.tail(space_a), n)
_enforce_zeros!(C::AbstractArray{T,N}, A, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a, n) where {T,N} =
    @inbounds _enforce_zeros!(C, A, space_c[1], space_a[1], n, Val(N))

function _enforce_zeros!(C, A, sc::BaseSpace, sa, n)
    amin, amax = _nonzero_bounds(A, sa)
    cmin, _ = _get_order_pow(sc, amin, n)
    _, cmax = _get_order_pow(sc, amax, n)
    CoefType = eltype(C)
    for i ∈ 1:length(C)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            C[i] = zero(CoefType)
        end
    end
    return C
end
function _enforce_zeros!(C, A, sc::BaseSpace, sa, n, ::Val{D}) where {D}
    amin, amax = _nonzero_bounds(A, sa, Val(D))
    cmin, _ = _get_order_pow(sc, amin, n)
    _, cmax = _get_order_pow(sc, amax, n)
    CoefType = eltype(C)
    @inbounds for i ∈ 1:size(C, D)
        k = _index_to_math(sc, i)
        if k < cmin || cmax < k
            selectdim(C, D, i) .= zero(CoefType)
        end
    end
    return C
end

_get_order_pow(::Taylor, i, n) = (i*n, i*n)

_get_order_pow(::Fourier, i, n) = (i*n, i*n)

_get_order_pow(::Chebyshev, i, n) = (ifelse(isodd(n), i % 2, 0), i*n)

_get_order_pow(::CosFourier, i, n) = (ifelse(isodd(n), i % 2, 0), i*n)

_get_order_pow(::SinFourier, i, n) = (ifelse(isodd(n), i, 0), i*n)
#-

function _sqr(a::Sequence{<:SequenceSpace})
    new_space = codomain(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a)
    return c
end

function _sqr_bar(a::Sequence{<:SequenceSpace})
    new_space = codomain(pow_bar, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a)
    return c
end

_add_sqr!(c::Sequence, a) = _add_mul!(c, a, a, convert(real(eltype(c)), exact(true)))

#

function codomain(::typeof(^), s::SequenceSpace, n::Integer)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    s² = codomain(*, s, s)
    n == 2 && return s²
    return codomain(*, s², codomain(^, s, n-2))
end

function codomain(::typeof(pow_bar), s::SequenceSpace, n::Integer)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    s² = codomain(mul_bar, s, s)
    n == 2 && return s²
    return codomain(mul_bar, s², codomain(pow_bar, s, n-2))
end

# Taylor

function _add_sqr!(c::Sequence{Taylor}, a)
    order_a = order(space(a))
    @inbounds a₀ = a[0]
    @inbounds c[0] += a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            cᵢ += a[i-j] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i÷2]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
        end
    end
    return c
end

# Fourier

function _add_sqr!(c::Sequence{<:Fourier}, a)
    order_a = order(space(a))
    c₀ = zero(eltype(a))
    @inbounds for j ∈ 1:order_a
        c₀ += a[j] * a[-j]
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += exact(2) * c₀ + a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = c₋ᵢ = zero(eltype(a))
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[i-j] * a[j]
            c₋ᵢ += a[j-i] * a[-j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            a_neg_i½ = a[-i½]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
            c[-i] += exact(2) * c₋ᵢ + a_neg_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
            c[-i] += exact(2) * c₋ᵢ
        end
    end
    return c
end

# Chebyshev

function _add_sqr!(c::Sequence{Chebyshev}, a)
    order_a = order(space(a))
    c₀ = zero(eltype(a))
    @inbounds for j ∈ 1:order_a
        aⱼ = a[j]
        c₀ += aⱼ ^ exact(2)
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += exact(2) * c₀ + a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[abs(i-j)] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
        end
    end
    return c
end

# CosFourier and SinFourier

function _add_sqr!(c::Sequence{<:CosFourier}, a::Sequence{<:CosFourier})
    order_a = order(space(a))
    c₀ = zero(eltype(a))
    @inbounds for j ∈ 1:order_a
        aⱼ = a[j]
        c₀ += aⱼ ^ exact(2)
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += exact(2) * c₀ + a₀ ^ exact(2)
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = zero(eltype(a))
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[abs(i-j)] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            c[i] += exact(2) * cᵢ + a_i½ ^ exact(2)
        else
            c[i] += exact(2) * cᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{CosFourier}, a::Sequence{SinFourier})
    N_a = order(space(a))

    @inbounds c[0] += sum(a[j]^exact(2) for j in 1:N_a)

    N_c = order(space(c))
    @inbounds for i in 1:N_c
        s = zero(eltype(a))
        for j in 1:N_a
            for k in 1:N_a
                if j + k == i
                    s -= a[j] * a[k]
                end
                if abs(j - k) == i
                    s += a[j] * a[k]
                end
            end
        end
        c[i] += exact(1) * s
    end

    return c
end
