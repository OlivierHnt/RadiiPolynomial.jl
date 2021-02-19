## geometric decay

function geometric_decay(a::Sequence{Taylor})
    decay = _geometric_decay(coefficients(a))
    return max(one(decay), decay)
end

function geometric_decay(a::Sequence{<:Fourier})
    @inbounds a₋ = view(a, -order(a):0)
    @inbounds a₊ = view(a, 0:order(a))
    decay = min(inv( _geometric_decay(a₋) ), _geometric_decay(a₊))
    return max(one(decay), decay)
end

function geometric_decay(a::Sequence{Chebyshev})
    decay = _geometric_decay(coefficients(a))
    return max(one(decay), decay)
end

geometric_decay(a::Sequence{TensorSpace{T}}) where {N,T<:NTuple{N,UnivariateSpace}} =
    ntuple(i -> geometric_decay(a, i), Val(N))

function geometric_decay(a::Sequence{TensorSpace{T}}, i::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    space_a = space(a)
    idx = ntuple(Val(N)) do j
        if j == i
            return allindices(space_a[j])
        else
            cst_idx = _constant_index(space_a[j])
            return cst_idx:cst_idx
        end
    end
    @inbounds A = view(a, idx)
    return geometric_decay(Sequence(space_a[i], A))
end

function _geometric_decay_abs!(A::AbstractVector{T}) where {T<:AbstractFloat}
    ϵ = eps(T)
    filter!(Aᵢ -> Aᵢ > ϵ, A)
    n = length(A)
    n < 2 && return one(T)
    log_A = log.(A)
    return exp(-_linear_regression(log_A))
end

_geometric_decay(A::AbstractVector{T}) where {T<:Interval} =
    Interval(_geometric_decay_abs!(map(Aᵢ -> abs(sup(Aᵢ)), A)))

_geometric_decay(A::AbstractVector{T}) where {T<:Complex{<:Interval}}=
    Interval(_geometric_decay_abs!(map(Aᵢ -> abs(sup(real(Aᵢ)) + im*sup(imag(Aᵢ))), A)))

_geometric_decay(A::AbstractVector) = _geometric_decay_abs!(abs.(A))

function _linear_regression(y::AbstractVector{T}) where {T<:AbstractFloat}
    # simple linear regression
    n = length(y)
    n < 2 && return zero(T)
    ȳ = sum(y)/n
    x̄ = (n+1)/2
    return 6sum(t -> (t[1]-x̄)*(t[2]-ȳ), enumerate(y))*inv(n*(6x̄*(x̄-n-1) + n*(2n+3) + 1))
end

##

function _banach_rounding_order(decay::T, bound::T) where {T<:AbstractFloat}
    isone(decay) && return typemax(Int)
    isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    x = log(bound*inv(ϵ))*inv(log(decay))
    isinf(x) && return typemax(Int)
    return ceil(Int, x)
end

_banach_rounding_order(decay::Interval, bound::Interval) =
    _banach_rounding_order(sup(decay), sup(bound))

_banach_rounding_order(decay::Tuple{Vararg{AbstractFloat}}, bound::AbstractFloat) =
    map(νᵢ -> _banach_rounding_order(νᵢ, bound), decay)

function _banach_rounding_order(decay::Tuple{Vararg{Interval}}, bound::Interval)
    sup_bound = sup(bound)
    return map(νᵢ -> _banach_rounding_order(sup(νᵢ), sup_bound), decay)
end

function _banach_rounding!(a::Sequence{Taylor}, decay, bound, max_order)
    max_order == typemax(Int) && return a
    ν⁻¹ = inv(decay)
    μⁱ = bound * pow(ν⁻¹, max_order)
    @inbounds for i ∈ max_order+1:order(a)
        μⁱ *= ν⁻¹
        sup_μⁱ = sup(μⁱ)
        a[i] = Interval(-sup_μⁱ, sup_μⁱ)
    end
    return a
end

function _banach_rounding!(a::Sequence{<:Fourier}, decay, bound, max_order)
    max_order == typemax(Int) && return a
    ν⁻¹ = inv(decay)
    μⁱ = bound * pow(ν⁻¹, max_order)
    @inbounds for i ∈ max_order+1:order(a)
        μⁱ *= ν⁻¹
        sup_μⁱ = sup(μⁱ)
        interval_μⁱ = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = interval_μⁱ
        a[-i] = interval_μⁱ
    end
    return a
end

function _banach_rounding!(a::Sequence{Chebyshev}, decay, bound, max_order)
    max_order == typemax(Int) && return a
    ν⁻¹ = inv(decay)
    μⁱ = bound * pow(ν⁻¹, max_order)
    @inbounds for i ∈ max_order+1:order(a)
        μⁱ *= ν⁻¹
        sup_μⁱ = sup(μⁱ)
        a[i] = Interval(-sup_μⁱ, sup_μⁱ)
    end
    return a
end

function _banach_rounding!(a::Sequence{TensorSpace{T}}, decay, bound, max_order) where {N,T<:NTuple{N,UnivariateSpace}}
    M = typemax(Int)
    w = map(ord -> ifelse(ord == 0, 1, ord), max_order)
    ν⁻¹ = inv.(decay)
    @inbounds for α ∈ allindices(space(a))
        if mapreduce((wᵢ, αᵢ) -> wᵢ == M ? 0//1 : abs(αᵢ) // wᵢ, +, w, α) > 1
            sup_μᵅ = sup(mapreduce((ν⁻¹ᵢ, αᵢ) -> pow(ν⁻¹ᵢ, abs(αᵢ)), *, ν⁻¹, α; init = bound))
            a[α] = Interval(-sup_μᵅ, sup_μᵅ)
        end
    end
    return a
end

##

function Base.:*(a::Sequence{<:UnivariateSpace}, b::Sequence{<:UnivariateSpace})
    decay = min(geometric_decay(a), geometric_decay(b))
    return _mul(a, b, decay)
end

function Base.:*(a::Sequence{<:TensorSpace}, b::Sequence{<:TensorSpace})
    decay = map(min, geometric_decay(a), geometric_decay(b))
    return _mul(a, b, decay)
end

function _mul(a, b, decay)
    bound_ab = norm_weighted_ℓ¹(a, decay) * norm_weighted_ℓ¹(b, decay)
    banach_rounding_order = _banach_rounding_order(decay, bound_ab)
    return _mul(a, b, decay, bound_ab, banach_rounding_order)
end

function _mul(a, b, decay, bound, banach_rounding_order)
    new_space = convolution_range(space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_mul!(c, a, b, banach_rounding_order)
    real(CoefType) <: Interval && return _banach_rounding!(c, decay, bound, banach_rounding_order)
    return c
end

function _add_mul!(c::Sequence{Taylor}, a::Sequence{Taylor}, b::Sequence{Taylor}, max_order)
    order_a = order(a)
    order_b = order(b)
    ord = min(order(c), max_order)
    Base.Threads.@threads for i ∈ 0:ord
        @inbounds for j ∈ max(i-order_a, 0):min(i, order_b)
            c[i] += a[i-j] * b[j]
        end
    end
    return c
end

function _add_mul!(c::Sequence{<:Fourier}, a::Sequence{<:Fourier}, b::Sequence{<:Fourier}, max_order)
    order_a = order(a)
    order_b = order(b)
    ord = min(order(c), max_order)
    Base.Threads.@threads for i ∈ -ord:ord
        @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
            c[i] += a[i-j] * b[j]
        end
    end
    return c
end

function _add_mul!(c::Sequence{Chebyshev}, a::Sequence{Chebyshev}, b::Sequence{Chebyshev}, max_order)
    order_a = order(a)
    order_b = order(b)
    ord = min(order(c), max_order)
    Base.Threads.@threads for i ∈ 0:ord
        @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
            c[i] += a[abs(i-j)] * b[abs(j)]
        end
    end
    return c
end

function _add_mul!(c::Sequence{TensorSpace{T}}, a, b, max_order) where {N,T<:NTuple{N,UnivariateSpace}}
    space_c = space(c)
    C = reshape(coefficients(c), dimensions(space_c))
    A = reshape(coefficients(a), dimensions(space(a)))
    B = reshape(coefficients(b), dimensions(space(b)))
    @inbounds _add_mul!(C, space_c, space_c[N], A, B, max_order, 0//1)
    return c
end

function _add_mul!(C, space::TensorSpace{Tuple{Taylor}}, current_space::Taylor, A, B, max_order, n)
    @inbounds current_max_order = max_order[1]
    order_A = size(A, 1)-1
    order_B = size(B, 1)-1
    order_C = order(current_space)
    ord = current_max_order == typemax(Int) ? order_C : min(order_C, floor(Int, current_max_order * (1 - n)))
    for i ∈ 0:ord
        @inbounds for j ∈ max(i-order_A, 0):min(i, order_B)
            C[i+1] += A[i-j+1] * B[j+1]
        end
    end
    return C
end

function _add_mul!(C, space::TensorSpace{<:NTuple{N,UnivariateSpace}}, current_space::Taylor, A, B, max_order, n) where {N}
    remaining_max_order = Base.front(max_order)
    remaining_space = Base.front(space)
    @inbounds next_space = space[N-1]
    @inbounds current_max_order = max_order[N]
    order_A = size(A, N)-1
    order_B = size(B, N)-1
    order_C = order(current_space)
    if current_max_order == typemax(Int)
        @inbounds for i ∈ 0:order_C
            Cᵢ = selectdim(C, N, i+1)
            @inbounds for j ∈ max(i-order_A, 0):min(i, order_B)
                _add_mul!(Cᵢ, remaining_space, next_space, selectdim(A, N, i-j+1), selectdim(B, N, j+1), remaining_max_order, n)
            end
        end
        return C
    else
        ord = min(order_C, floor(Int, current_max_order * (1 - n)))
        w = ifelse(current_max_order == 0, 1, current_max_order)
        @inbounds for i ∈ 0:ord
            n_ = n + i // w
            Cᵢ = selectdim(C, N, i+1)
            @inbounds for j ∈ max(i-order_A, 0):min(i, order_B)
                _add_mul!(Cᵢ, remaining_space, next_space, selectdim(A, N, i-j+1), selectdim(B, N, j+1), remaining_max_order, n_)
            end
        end
        return C
    end
end

function _add_mul!(C, space::TensorSpace{<:Tuple{Fourier}}, current_space::Fourier, A, B, max_order, n)
    @inbounds current_max_order = max_order[1]
    order_A = (size(A, 1)-1)÷2
    order_B = (size(B, 1)-1)÷2
    order_C = order(current_space)
    ord = current_max_order == typemax(Int) ? order_C : min(order_C, floor(Int, current_max_order * (1 - n)))
    for i ∈ -ord:ord
        @inbounds for j ∈ max(i-order_A, -order_B):min(i+order_A, order_B)
            C[i+order_C+1] += A[i-j+order_A+1] * B[j+order_B+1]
        end
    end
    return C
end

function _add_mul!(C, space::TensorSpace{<:NTuple{N,UnivariateSpace}}, current_space::Fourier, A, B, max_order, n) where {N}
    remaining_max_order = Base.front(max_order)
    remaining_space = Base.front(space)
    @inbounds next_space = space[N-1]
    @inbounds current_max_order = max_order[N]
    order_A = (size(A, N)-1)÷2
    order_B = (size(B, N)-1)÷2
    order_C = order(current_space)
    if current_max_order == typemax(Int)
        @inbounds for i ∈ -order_C:order_C
            Cᵢ = selectdim(C, N, i+order_C+1)
            @inbounds for j ∈ max(i-order_A, -order_B):min(i+order_A, order_B)
                _add_mul!(Cᵢ, remaining_space, next_space, selectdim(A, N, i-j+order_A+1), selectdim(B, N, j+order_B+1), remaining_max_order, n)
            end
        end
        return C
    else
        ord = min(order_C, floor(Int, current_max_order * (1 - n)))
        w = ifelse(current_max_order == 0, 1, current_max_order)
        @inbounds for i ∈ -ord:ord
            n_ = n + abs(i) // w
            Cᵢ = selectdim(C, N, i+order_C+1)
            @inbounds for j ∈ max(i-order_A, -order_B):min(i+order_A, order_B)
                _add_mul!(Cᵢ, remaining_space, next_space, selectdim(A, N, i-j+order_A+1), selectdim(B, N, j+order_B+1), remaining_max_order, n_)
            end
        end
        return C
    end
end

function _add_mul!(C, space::TensorSpace{Tuple{Chebyshev}}, current_space::Chebyshev, A, B, max_order, n)
    @inbounds current_max_order = max_order[1]
    order_A = size(A, 1)-1
    order_B = size(B, 1)-1
    order_C = order(current_space)
    ord = current_max_order == typemax(Int) ? order_C : min(order_C, floor(Int, current_max_order * (1 - n)))
    for i ∈ 0:ord
        @inbounds for j ∈ max(i-order_A, -order_B):min(i+order_A, order_B)
            C[i+1] += A[abs(i-j)+1] * B[abs(j)+1]
        end
    end
    return C
end

function _add_mul!(C, space::TensorSpace{<:NTuple{N,UnivariateSpace}}, current_space::Chebyshev, A, B, max_order, n) where {N}
    remaining_max_order = Base.front(max_order)
    remaining_space = Base.front(space)
    @inbounds next_space = space[N-1]
    @inbounds current_max_order = max_order[N]
    order_A = size(A, N)-1
    order_B = size(B, N)-1
    order_C = order(current_space)
    if current_max_order == typemax(Int)
        @inbounds for i ∈ 0:order_C
            Cᵢ = selectdim(C, N, i+1)
            @inbounds for j ∈ max(i-order_A, -order_B):min(i+order_A, order_B)
                _add_mul!(Cᵢ, remaining_space, next_space, selectdim(A, N, abs(i-j)+1), selectdim(B, N, abs(j)+1), remaining_max_order, n)
            end
        end
        return C
    else
        ord = min(order_C, floor(Int, current_max_order * (1 - n)))
        w = ifelse(current_max_order == 0, 1, current_max_order)
        @inbounds for i ∈ 0:ord
            n_ = n + i // w
            Cᵢ = selectdim(C, N, i+1)
            @inbounds for j ∈ max(i-order_A, -order_B):min(i+order_A, order_B)
                _add_mul!(Cᵢ, remaining_space, next_space, selectdim(A, N, abs(i-j)+1), selectdim(B, N, abs(j)+1), remaining_max_order, n_)
            end
        end
        return C
    end
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

_sqr(a::Sequence{<:SequenceSpace}) = _sqr(a, geometric_decay(a))

function _sqr(a, decay)
    norm_a = norm_weighted_ℓ¹(a, decay)
    bound_a² = norm_a*norm_a
    banach_rounding_order = _banach_rounding_order(decay, bound_a²)
    return _sqr(a, decay, bound_a², banach_rounding_order)
end

function _sqr(a, decay, bound, banach_rounding_order)
    space_a = space(a)
    new_space = convolution_range(space_a, space_a)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a, banach_rounding_order)
    real(CoefType) <: Interval && return _banach_rounding!(c, decay, bound, banach_rounding_order)
    return c
end

function _add_sqr!(c::Sequence{Taylor}, a::Sequence{Taylor}, max_order)
    order_a = order(a)
    @inbounds c[0] = a[0]^2
    @inbounds for i ∈ 1:min(order(c), max_order)
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            c[i] += a[i-j] * a[j]
        end
        c[i] *= 2
        i_odd == 1 && continue
        c[i] += a[i÷2]^2
    end
    return c
end

function _add_sqr!(c::Sequence{<:Fourier}, a::Sequence{<:Fourier}, max_order)
    order_a = order(a)
    @inbounds for j ∈ 1:order_a
        c[0] += a[j] * a[-j]
    end
    @inbounds c[0] = 2c[0] + a[0]^2
    @inbounds for i ∈ 1:min(order(c), max_order)
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            c[i] += a[i-j] * a[j]
            c[-i] += a[j-i] * a[-j]
        end
        c[i] *= 2
        c[-i] *= 2
        i_odd == 1 && continue
        c[i] += a[i½]^2
        c[-i] += a[-i½]^2
    end
    return c
end

function _add_sqr!(c::Sequence{Chebyshev}, a::Sequence{Chebyshev}, max_order)
    order_a = order(a)
    @inbounds for j ∈ 1:order_a
        c[0] += a[j]^2
    end
    @inbounds c[0] = 2c[0] + a[0]^2
    @inbounds for i ∈ 1:min(order(c), max_order)
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            c[i] += a[abs(i-j)] * a[j]
        end
        c[i] *= 2
        i_odd == 1 && continue
        c[i] += a[i½]^2
    end
    return c
end

function _add_sqr!(c::Sequence{TensorSpace{T}}, a, max_order) where {N,T<:NTuple{N,UnivariateSpace}}
    space_c = space(c)
    C = reshape(coefficients(c), dimensions(space_c))
    A = reshape(coefficients(a), dimensions(space(a)))
    @inbounds _add_mul!(C, space_c, space_c[N], A, A, max_order, 0//1)
    return c
end

#

function *̄(a::Sequence{<:UnivariateSpace}, b::Sequence{<:UnivariateSpace})
    decay = min(geometric_decay(a), geometric_decay(b))
    return _mul_bar(a, b, decay)
end

function *̄(a::Sequence{<:TensorSpace}, b::Sequence{<:TensorSpace})
    decay = map(min, geometric_decay(a), geometric_decay(b))
    return _mul_bar(a, b, decay)
end

function _mul_bar(a, b, decay)
    bound_ab = norm_weighted_ℓ¹(a, decay) * norm_weighted_ℓ¹(b, decay)
    banach_rounding_order = _banach_rounding_order(decay, bound_ab)
    return _mul_bar(a, b, decay, bound_ab, banach_rounding_order)
end

function _mul_bar(a, b, decay, bound, banach_rounding_order)
    new_space = convolution_bar_range(space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_mul!(c, a, b, banach_rounding_order)
    real(CoefType) <: Interval && return _banach_rounding!(c, decay, bound, banach_rounding_order)
    return c
end

#

function ^̄(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
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

_sqr_bar(a::Sequence{<:SequenceSpace}) = _sqr_bar(a, geometric_decay(a))

function _sqr_bar(a, decay)
    norm_a = norm_weighted_ℓ¹(a, decay)
    bound_a² = norm_a*norm_a
    banach_rounding_order = _banach_rounding_order(decay, bound_a²)
    return _sqr_bar(a, decay, bound_a², banach_rounding_order)
end

function _sqr_bar(a, decay, bound, banach_rounding_order)
    space_a = space(a)
    new_space = convolution_bar_range(space_a, space_a)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a, banach_rounding_order)
    real(CoefType) <: Interval && return _banach_rounding!(c, decay, bound, banach_rounding_order)
    return c
end
