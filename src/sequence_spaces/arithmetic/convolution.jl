function banach_rounding_order(weights::GeometricWeights{T}, bound::T) where {T<:AbstractFloat}
    isone(rate(weights)) | isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    order = log(bound/ϵ)/log(rate(weights))
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end
function banach_rounding_order(weights::AlgebraicWeights{T}, bound::T) where {T<:AbstractFloat}
    iszero(rate(weights)) | isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    order = exp(log(bound/ϵ)/rate(weights))-1
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

function banach_rounding_order(weights::GeometricWeights, bound::Real)
    rate_, bound_ = promote(float(rate(weights)), float(bound))
    return banach_rounding_order(GeometricWeights(rate_), bound_)
end
function banach_rounding_order(weights::AlgebraicWeights, bound::Real)
    rate_, bound_ = promote(float(rate(weights)), float(bound))
    return banach_rounding_order(AlgebraicWeights(rate_), bound_)
end

banach_rounding_order(weights::GeometricWeights{<:Interval}, bound::Interval) =
    banach_rounding_order(GeometricWeights(sup(rate(weights))), sup(bound))
banach_rounding_order(weights::AlgebraicWeights{<:Interval}, bound::Interval) =
    banach_rounding_order(AlgebraicWeights(sup(rate(weights))), sup(bound))

banach_rounding_order(weights::GeometricWeights{<:Interval}, bound::Real) =
    banach_rounding_order(GeometricWeights(sup(rate(weights))), bound)
banach_rounding_order(weights::AlgebraicWeights{<:Interval}, bound::Real) =
    banach_rounding_order(AlgebraicWeights(sup(rate(weights))), bound)

banach_rounding_order(weights::GeometricWeights, bound::Interval) =
    banach_rounding_order(weights, sup(bound))
banach_rounding_order(weights::AlgebraicWeights, bound::Interval) =
    banach_rounding_order(weights, sup(bound))

banach_rounding_order(weights::Tuple, bound::Real) =
    map(νᵢ -> banach_rounding_order(νᵢ, bound), weights)

function banach_rounding_order(weights::Tuple, bound::Interval)
    sup_bound = sup(bound)
    return map(νᵢ -> banach_rounding_order(νᵢ, sup_bound), weights)
end

#

function banach_rounding!(a, weights, bound)
    banach_rounding!(a, weights, bound, banach_rounding_order(weights, bound))
    return a
end

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{S}}, weights::NTuple{N,Weights}, bound::Real, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace},S<:Interval}
    M = typemax(Int)
    w = map(ord -> ifelse(ord == 0, 1, ord), rounding_order)
    space_a = space(a)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((wᵢ, αᵢ) -> wᵢ == M ? 0//1 : abs(αᵢ) // wᵢ, +, w, α) ≥ 1
            μᵅ = bound / _weight(space_a, weights, α)
            sup_μᵅ = sup(μᵅ)
            a[α] = Interval(-sup_μᵅ, sup_μᵅ)
        end
    end
    return a
end

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{Complex{S}}}, weights::NTuple{N,Weights}, bound::Real, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace},S<:Interval}
    M = typemax(Int)
    w = map(ord -> ifelse(ord == 0, 1, ord), rounding_order)
    space_a = space(a)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((wᵢ, αᵢ) -> wᵢ == M ? 0//1 : abs(αᵢ) // wᵢ, +, w, α) ≥ 1
            μᵅ = bound / _weight(space_a, weights, α)
            sup_μᵅ = sup(μᵅ)
            x = Interval(-sup_μᵅ, sup_μᵅ)
            a[α] = complex(x, x)
        end
    end
    return a
end

# Taylor

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, weights::GeometricWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weights))
        μⁱ = bound / _weight(space(a), weights, rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            a[i] = Interval(-sup_μⁱ, sup_μⁱ)
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, weights::AlgebraicWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _weight(space_a, weights, i))
        a[i] = Interval(-sup_μⁱ, sup_μⁱ)
    end
    return a
end

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{Complex{T}}}, weights::GeometricWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weights))
        μⁱ = bound / _weight(space(a), weights, rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            a[i] = complex(x, x)
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{Complex{T}}}, weights::AlgebraicWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _weight(space_a, weights, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = complex(x, x)
    end
    return a
end

# Fourier

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, weights::GeometricWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weights))
        μⁱ = bound / _weight(space(a), weights, rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            a[i] = x
            a[-i] = x
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, weights::AlgebraicWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _weight(space_a, weights, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = x
        a[-i] = x
    end
    return a
end

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{Complex{T}}}, weights::GeometricWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weights))
        μⁱ = bound / _weight(space(a), weights, rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            complex_x = complex(x, x)
            a[i] = complex_x
            a[-i] = complex_x
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{Complex{T}}}, weights::AlgebraicWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _weight(space_a, weights, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        complex_x = complex(x, x)
        a[i] = complex_x
        a[-i] = complex_x
    end
    return a
end

# Chebyshev

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, weights::GeometricWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weights))
        μⁱ = bound / _weight(space(a), weights, rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            a[i] = Interval(-sup_μⁱ, sup_μⁱ)
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, weights::AlgebraicWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _weight(space_a, weights, i))
        a[i] = Interval(-sup_μⁱ, sup_μⁱ)
    end
    return a
end

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{Complex{T}}}, weights::GeometricWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(weights))
        μⁱ = bound / _weight(space(a), weights, rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            a[i] = complex(x, x)
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{Complex{T}}}, weights::AlgebraicWeights, bound::Real, rounding_order::Int) where {T<:Interval}
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _weight(space_a, weights, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = complex(x, x)
    end
    return a
end

#

_inf_max_order(::BaseSpace) = typemax(Int)
_inf_max_order(::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = ntuple(i -> typemax(Int), Val(N))

function Base.:*(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_mul!(c, a, b, _inf_max_order(new_space))
    return c
end

function banach_rounding_mul(a::Sequence{<:BaseSpace}, b::Sequence{<:BaseSpace}, weights::Weights)
    X = Weightedℓ¹(weights)
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(weights, bound_ab)
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_ab, rounding_order)
    _add_mul!(c, a, b, rounding_order-1)
    return c
end

function banach_rounding_mul(a::Sequence{TensorSpace{T}}, b::Sequence{TensorSpace{S}}, weights::NTuple{N,Weights}) where {N,T<:NTuple{N,BaseSpace},S<:NTuple{N,BaseSpace}}
    X = Weightedℓ¹(weights)
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(weights, bound_ab)
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_ab, rounding_order)
    _add_mul!(c, a, b, ntuple(i -> rounding_order[i]-1, Val(N)))
    return c
end

function _add_mul!(c::Sequence{<:BaseSpace}, a, b, max_order)
    C = coefficients(c)
    A = coefficients(a)
    B = coefficients(b)
    _add_mul!(C, A, B, space(c), space(a), space(b), max_order)
    return c
end

function _add_mul!(c::Sequence{<:TensorSpace}, a, b, max_order)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    B = _no_alloc_reshape(coefficients(b), dimensions(space_b))
    _add_mul!(C, A, B, space_c, space_a, space_b, max_order, 0//1)
    return c
end

function _add_mul!(C, A, B, space_c::BaseSpace, space_a, space_b, max_order)
    ord = min(order(space_c), max_order)
    for i ∈ _convolution_indices(space_c, ord)
        _convolution!(C, A, B, space_c, space_a, space_b, i)
    end
    return C
end

function _add_mul!(C, A, B, space_c::TensorSpace{<:NTuple{N,BaseSpace}}, space_a, space_b, max_order, n) where {N}
    remaining_space_c = Base.front(space_c)
    remaining_space_a = Base.front(space_a)
    remaining_space_b = Base.front(space_b)
    remaining_max_order = Base.front(max_order)
    @inbounds current_space_c = space_c[N]
    @inbounds current_space_a = space_a[N]
    @inbounds current_space_b = space_b[N]
    @inbounds current_max_order = max_order[N]
    current_order_c = order(current_space_c)
    if current_max_order == typemax(Int)
        for i ∈ _convolution_indices(current_space_c, current_order_c)
            _convolution!(C, A, B, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i)
        end
    else
        ord = min(current_order_c, floor(Int, current_max_order * (1 - n)))
        w = ifelse(current_max_order == 0, 1, current_max_order)
        for i ∈ _convolution_indices(current_space_c, ord)
            n_ = n + abs(i) // w
            _convolution!(C, A, B, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n_, i)
        end
    end
    return C
end

function _add_mul!(C, A, B, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a, space_b, max_order, n)
    @inbounds current_space_c = space_c[1]
    @inbounds current_space_a = space_a[1]
    @inbounds current_space_b = space_b[1]
    @inbounds current_max_order = max_order[1]
    current_order_c = order(current_space_c)
    ord = current_max_order == typemax(Int) ? current_order_c : min(current_order_c, floor(Int, current_max_order * (1 - n)))
    for i ∈ _convolution_indices(current_space_c, ord)
        _convolution!(C, A, B, current_space_c, current_space_a, current_space_b, i)
    end
    return C
end

# Taylor

_convolution_indices(::Taylor, order) = 0:order

function _convolution!(C, A, B, space_c::Taylor, space_a::Taylor, space_b::Taylor, i)
    order_a = order(space_a)
    order_b = order(space_b)
    @inbounds for j ∈ max(i-order_a, 0):min(i, order_b)
        C[i+1] += A[i-j+1] * B[j+1]
    end
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, current_space_c::Taylor, current_space_a::Taylor, current_space_b::Taylor, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i) where {T,N}
    current_order_a = order(current_space_a)
    current_order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-current_order_a, 0):min(i, current_order_b)
        _add_mul!(Cᵢ, selectdim(A, N, i-j+1), selectdim(B, N, j+1), remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n)
    end
    return C
end

# Fourier

_convolution_indices(::Fourier, order) = -order:order

function _convolution!(C, A, B, space_c::Fourier, space_a::Fourier, space_b::Fourier, i)
    order_c = order(space_c)
    order_a = order(space_a)
    order_b = order(space_b)
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
        C[i+order_c+1] += A[i-j+order_a+1] * B[j+order_b+1]
    end
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, current_space_c::Fourier, current_space_a::Fourier, current_space_b::Fourier, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i) where {T,N}
    current_order_c = order(current_space_c)
    current_order_a = order(current_space_a)
    current_order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+current_order_c+1)
    @inbounds for j ∈ max(i-current_order_a, -current_order_b):min(i+current_order_a, current_order_b)
        _add_mul!(Cᵢ, selectdim(A, N, i-j+current_order_a+1), selectdim(B, N, j+current_order_b+1), remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n)
    end
    return C
end

# Chebyshev

_convolution_indices(::Chebyshev, order) = 0:order

function _convolution!(C, A, B, space_c::Chebyshev, space_a::Chebyshev, space_b::Chebyshev, i)
    order_a = order(space_a)
    order_b = order(space_b)
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
        C[i+1] += A[abs(i-j)+1] * B[abs(j)+1]
    end
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, current_space_c::Chebyshev, current_space_a::Chebyshev, current_space_b::Chebyshev, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i) where {T,N}
    current_order_a = order(current_space_a)
    current_order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-current_order_a, -current_order_b):min(i+current_order_a, current_order_b)
        _add_mul!(Cᵢ, selectdim(A, N, abs(i-j)+1), selectdim(B, N, abs(j)+1), remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n)
    end
    return C
end

#

function Base.:^(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
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
        c *= a
    end
    return c
end

function banach_rounding_pow(a::Sequence{<:SequenceSpace}, n::Int, weights)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _banach_rounding_sqr(a, weights)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _banach_rounding_sqr(a, weights)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _banach_rounding_sqr(a, weights)
        end
        c = banach_rounding_mul(c, a, weights)
    end
    return c
end

function _sqr(a::Sequence{<:SequenceSpace})
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a, _inf_max_order(new_space))
    return c
end

function _banach_rounding_sqr(a::Sequence{<:BaseSpace}, weights::Weights)
    X = Weightedℓ¹(weights)
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    rounding_order = banach_rounding_order(weights, bound_a²)
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_a², rounding_order)
    _add_sqr!(c, a, rounding_order-1)
    return c
end

function _banach_rounding_sqr(a::Sequence{TensorSpace{T}}, weights::NTuple{N,Weights}) where {N,T<:NTuple{N,BaseSpace}}
    X = Weightedℓ¹(weights)
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    rounding_order = banach_rounding_order(weights, bound_a²)
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_a², rounding_order)
    _add_sqr!(c, a, ntuple(i -> rounding_order[i]-1, Val(N)))
    return c
end

function _add_sqr!(c::Sequence{<:BaseSpace}, a, max_order)
    C = coefficients(c)
    A = coefficients(a)
    _add_sqr!(C, A, space(c), space(a), max_order)
    return c
end

function _add_sqr!(c::Sequence{<:TensorSpace}, a, max_order)
    space_c = space(c)
    space_a = space(a)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    _add_mul!(C, A, A, space_c, space_a, space_a, max_order, 0//1)
    return c
end

# Taylor

function _add_sqr!(C, A, space_c::Taylor, space_a::Taylor, max_order)
    order_a = order(space_a)
    @inbounds A₀ = A[1]
    @inbounds C[1] = A₀ * A₀
    ord = min(order(space_c), max_order)
    @inbounds for i ∈ 1:ord
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            C[i+1] += A[i-j+1] * A[j+1]
        end
        C[i+1] *= 2
        if iszero(i_odd)
            A_i½ = A[i÷2+1]
            C[i+1] += A_i½ * A_i½
        end
    end
    return C
end

# Fourier

function _add_sqr!(C, A, space_c::Fourier, space_a::Fourier, max_order)
    order_c = order(space_c)
    order_a = order(space_a)
    @inbounds for j ∈ 1:order_a
        C[order_c+1] += A[j+order_a+1] * A[-j+order_a+1]
    end
    @inbounds A₀ = A[order_a+1]
    @inbounds C[order_c+1] = 2C[order_c+1] + A₀ * A₀
    ord = min(order_c, max_order)
    @inbounds for i ∈ 1:ord
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            C[i+order_c+1] += A[i-j+order_a+1] * A[j+order_a+1]
            C[-i+order_c+1] += A[j-i+order_a+1] * A[-j+order_a+1]
        end
        C[i+order_c+1] *= 2
        C[-i+order_c+1] *= 2
        if iszero(i_odd)
            A_i½ = A[i½+order_a+1]
            A_neg_i½ = A[-i½+order_a+1]
            C[i+order_c+1] += A_i½ * A_i½
            C[-i+order_c+1] += A_neg_i½ * A_neg_i½
        end
    end
    return C
end

# Chebyshev

function _add_sqr!(C, A, space_c::Chebyshev, space_a::Chebyshev, max_order)
    order_a = order(space_a)
    @inbounds for j ∈ 1:order_a
        Aⱼ = A[j+1]
        C[1] += Aⱼ * Aⱼ
    end
    @inbounds A₀ = A[1]
    @inbounds C[1] = 2C[1] + A₀ * A₀
    ord = min(order(space_c), max_order)
    @inbounds for i ∈ 1:ord
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            C[i+1] += A[abs(i-j)+1] * A[j+1]
        end
        C[i+1] *= 2
        if iszero(i_odd)
            A_i½ = A[i½+1]
            C[i+1] += A_i½ * A_i½
        end
    end
    return C
end

#

function *̄(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(*̄, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_mul!(c, a, b, _inf_max_order(new_space))
    return c
end

function banach_rounding_mul_bar(a::Sequence{<:BaseSpace}, b::Sequence{<:BaseSpace}, weights::Weights)
    X = Weightedℓ¹(weights)
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(weights, bound_ab)
    new_space = image(*̄, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_ab, rounding_order)
    _add_mul!(c, a, b, rounding_order-1)
    return c
end

function banach_rounding_mul_bar(a::Sequence{TensorSpace{T}}, b::Sequence{TensorSpace{S}}, weights::NTuple{N,Weights}) where {N,T<:NTuple{N,BaseSpace},S<:NTuple{N,BaseSpace}}
    X = Weightedℓ¹(weights)
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(weights, bound_ab)
    new_space = image(*̄, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_ab, rounding_order)
    _add_mul!(c, a, b, ntuple(i -> rounding_order[i]-1, Val(N)))
    return c
end

#

function ^̄(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "^̄ is only defined for positive integers"))
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
        c = c *̄ a
    end
    return c
end

function banach_rounding_pow_bar(a::Sequence{<:SequenceSpace}, n::Int, weights)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _banach_rounding_sqr_bar(a, weights)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _banach_rounding_sqr_bar(a, weights)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _banach_rounding_sqr_bar(a, weights)
        end
        c = banach_rounding_mul_bar(c, a, weights)
    end
    return c
end

function _sqr_bar(a::Sequence{<:SequenceSpace})
    new_space = image(^̄, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a, _inf_max_order(new_space))
    return c
end

function _banach_rounding_sqr_bar(a::Sequence{<:BaseSpace}, weights::Weights)
    X = Weightedℓ¹(weights)
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    rounding_order = banach_rounding_order(weights, bound_a²)
    new_space = image(^̄, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_a², rounding_order)
    _add_sqr!(c, a, rounding_order-1)
    return c
end

function _banach_rounding_sqr_bar(a::Sequence{TensorSpace{T}}, weights::NTuple{N,Weights}) where {N,T<:NTuple{N,BaseSpace}}
    X = Weightedℓ¹(weights)
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    rounding_order = banach_rounding_order(weights, bound_a²)
    new_space = image(^̄, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, weights, bound_a², rounding_order)
    _add_sqr!(c, a, ntuple(i -> rounding_order[i]-1, Val(N)))
    return c
end
