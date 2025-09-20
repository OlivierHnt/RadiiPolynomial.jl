"""
    Evaluation{T<:Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}}} <: AbstractLinearOperator

Generic evaluation operator. A value of `nothing` indicates that no evaluation
should be performed.

Field:
- `value :: T`

Constructors:
- `Evaluation(::Union{Nothing,Number})`
- `Evaluation(::Tuple{Vararg{Union{Nothing,Number}}})`
- `Evaluation(value::Union{Number,Nothing}...)`: equivalent to `Evaluation(value)`

See also: [`evaluate`](@ref), [`evaluate!`](@ref),
[`project(::Evaluation, ::VectorSpace, ::VectorSpace)`](@ref) and
[`project!(::LinearOperator, ::Evaluation)`](@ref).

# Examples

```jldoctest
julia> Evaluation(1.0)
Evaluation{Float64}(1.0)

julia> Evaluation(1.0, nothing, 2.0)
Evaluation{Tuple{Float64, Nothing, Float64}}((1.0, nothing, 2.0))
```
"""
struct Evaluation{T<:Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}}} <: AbstractLinearOperator
    value :: T
    Evaluation{T}(value::T) where {T<:Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}}} = new{T}(value)
    Evaluation{Tuple{}}(::Tuple{}) = throw(ArgumentError("Evaluation is only defined for at least one Number or Nothing"))
end

Evaluation(value::T) where {T<:Union{Nothing,Number}} = Evaluation{T}(value)
Evaluation(value::T) where {T<:Tuple{Vararg{Union{Nothing,Number}}}} = Evaluation{T}(value)
Evaluation(value::Union{Number,Nothing}...) = Evaluation(value)

value(ℰ::Evaluation) = ℰ.value

"""
    *(ℰ::Evaluation, a::AbstractSequence)

Evaluate `a` at `value(ℰ)`; equivalent to `evaluate(a, value(ℰ))`.

See also: [`Evaluation`](@ref), [`(::AbstractSequence)(::Any, ::Vararg)`](@ref),
[`evaluate`](@ref) and [`evaluate!`](@ref).
"""
Base.:*(ℰ::Evaluation, a::AbstractSequence) = evaluate(a, value(ℰ))

"""
    (a::AbstractSequence)(x, y...)

Evaluate `a` at `(x, y...)`; equivalent to `evaluate(a, (x, y...))` or
`evaluate(a, x)` if `y` is not provided.

See also: [`evaluate`](@ref), [`evaluate!`](@ref), [`Evaluation`](@ref),
[`*(::Evaluation, ::AbstractSequence)`](@ref) and [`(::Evaluation)(::AbstractSequence)`](@ref).
"""
(a::AbstractSequence)(x, y...) = evaluate(a, (x, y...))
(a::AbstractSequence)(x) = evaluate(a, x)

"""
    evaluate(a::Sequence, x)

Evaluate `a` at `x`.

See also: [`(::Sequence)(::Any, ::Vararg)`](@ref), [`evaluate!`](@ref), [`Evaluation`](@ref),
[`*(::Evaluation, ::Sequence)`](@ref) and [`(::Evaluation)(::Sequence)`](@ref).
"""
function evaluate(a::Sequence, x)
    ℰ = Evaluation(x)
    space_a = space(a)
    new_space = codomain(ℰ, space_a)
    CoefType = _coeftype(ℰ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, ℰ, a)
    return _return_evaluate(c, x)
end

_return_evaluate(a::Sequence, ::Any) = a
_return_evaluate(a::Sequence{<:SequenceSpace}, ::Union{Number,Tuple{Vararg{Number}}}) = coefficients(a)[1]
_return_evaluate(a::Sequence{<:CartesianSpace}, ::Union{Number,Tuple{Vararg{Number}}}) = coefficients(a)

evaluate(a::InfiniteSequence, x) = _return_evaluate(evaluate(sequence(a), x), a)

_return_evaluate(c, a::InfiniteSequence) = interval(c, sequence_error(a); format = :midpoint)
_return_evaluate(c::Sequence, a::InfiniteSequence) = InfiniteSequence(interval.(c, sequence_error(a); format = :midpoint), sequence_error(a), banachspace(a))

"""
    evaluate!(c::Union{AbstractVector,Sequence}, a::Sequence, x)

Evaluate `a` at `x`. The result is stored in `c` by overwriting it.

See also: [`(::Sequence)(::Any, ::Vararg)`](@ref), [`evaluate`](@ref), [`Evaluation`](@ref),
[`*(::Evaluation, ::Sequence)`](@ref) and [`(::Evaluation)(::Sequence)`](@ref).
"""
function evaluate!(c::Sequence, a::Sequence, x)
    ℰ = Evaluation(x)
    space_c = space(c)
    new_space = codomain(ℰ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℰ(a) has space $new_space"))
    _apply!(c, ℰ, a)
    return c
end
function evaluate!(c::AbstractVector, a::Sequence, x)
    evaluate!(Sequence(codomain(Evaluation(x), space(a)), c), a, x)
    return c
end

# """
#     project!(C::LinearOperator, ℰ::Evaluation)

# Represent `ℰ` as a [`LinearOperator`](@ref) from `domain` to `codomain`.
# The result is stored in `C` by overwriting it.

# See also: [`project(::Evaluation, ::VectorSpace, ::VectorSpace)`](@ref) and
# [`Evaluation`](@ref).
# """
# function project!(C::LinearOperator, ℰ::Evaluation)
#     domain_C = domain(C)
#     image_domain = codomain(ℰ, domain_C)
#     codomain_C = codomain(C)
#     _iscompatible(ℰ, image_domain, codomain_C) || return throw(ArgumentError("spaces must be compatible: image of domain(C) under $ℰ is $image_domain, C has codomain $codomain_C"))
#     CoefType = eltype(C)
#     coefficients(C) .= zero(CoefType)
#     _project!(C, ℰ, _memo(domain_C, CoefType))
#     return C
# end

_iscompatible(::Evaluation, ::VectorSpace, ::VectorSpace) = false
_iscompatible(::Evaluation, s₁::SequenceSpace, s₂::SequenceSpace) = _iscompatible(s₁, s₂)
_iscompatible(::Evaluation{<:Union{Number,Tuple{Vararg{Number}}}}, ::SequenceSpace, ::ParameterSpace) = true
_iscompatible(::Evaluation{<:Union{Number,Tuple{Vararg{Number}}}}, ::SequenceSpace, ::SequenceSpace) = true
_iscompatible(ℰ::Evaluation, s₁::CartesianPower, s₂::CartesianPower) =
    (nspaces(s₁) == nspaces(s₂)) & _iscompatible(ℰ, space(s₁), space(s₂))
_iscompatible(ℰ::Evaluation, s₁::CartesianProduct{<:NTuple{N,VectorSpace}}, s₂::CartesianProduct{<:NTuple{N,VectorSpace}}) where {N} =
    @inbounds _iscompatible(ℰ, s₁[1], s₂[1]) & _iscompatible(ℰ, Base.tail(s₁), Base.tail(s₂))
_iscompatible(ℰ::Evaluation, s₁::CartesianProduct{<:Tuple{VectorSpace}}, s₂::CartesianProduct{<:Tuple{VectorSpace}}) =
    @inbounds _iscompatible(ℰ, s₁[1], s₂[1])
_iscompatible(ℰ::Evaluation, s₁::CartesianPower, s₂::CartesianProduct) =
    (nspaces(s₁) == nspaces(s₂)) & all(s₂ᵢ -> _iscompatible(ℰ, space(s₁), s₂ᵢ), spaces(s₂))
_iscompatible(ℰ::Evaluation, s₁::CartesianProduct, s₂::CartesianPower) =
    (nspaces(s₁) == nspaces(s₂)) & all(s₁ᵢ -> _iscompatible(ℰ, s₁ᵢ, space(s₂)), spaces(s₁))

# Sequence spaces

_memo(s::TensorSpace, ::Type{T}) where {T} = map(sᵢ -> _memo(sᵢ, T), spaces(s))

codomain(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((xᵢ, sᵢ) -> codomain(Evaluation(xᵢ), sᵢ), value(ℰ), spaces(s)))

_coeftype(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Evaluation(value(ℰ)[1]), s[1], T), _coeftype(Evaluation(Base.tail(value(ℰ))), Base.tail(s), T))
_coeftype(ℰ::Evaluation{<:Tuple{Union{Nothing,Number}}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Evaluation(value(ℰ)[1]), s[1], T)

function _apply!(c, ℰ::Evaluation, a::Sequence{<:TensorSpace})
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), _effective_dimensions(ℰ, space(c)))
    _apply!(C, ℰ, space_a, A)
    return c
end

_effective_dimensions(ℰ::Evaluation{<:Tuple{Nothing,Vararg{Union{Nothing,Number}}}}, space::TensorSpace) =
    (dimension(space[1]), _effective_dimensions(Evaluation(Base.tail(value(ℰ))), Base.tail(space))...)
_effective_dimensions(::Evaluation{<:Tuple{Nothing}}, space::TensorSpace) =
    (dimension(space[1]),)
_effective_dimensions(ℰ::Evaluation, space::TensorSpace) =
    _effective_dimensions(Evaluation(Base.tail(value(ℰ))), Base.tail(space))
_effective_dimensions(::Evaluation{<:Tuple{Number}}, ::TensorSpace) = ()

_apply!(C, ℰ::Evaluation, space::TensorSpace, A) =
    @inbounds _apply!(C, Evaluation(value(ℰ)[1]), space[1], _apply(Evaluation(Base.tail(value(ℰ))), Base.tail(space), A))

_apply!(C, ℰ::Evaluation, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply!(C, Evaluation(value(ℰ)[1]), space[1], A)

_apply(ℰ::Evaluation, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply(Evaluation(value(ℰ)[1]), space[1], Val(N₂-N₁+1), _apply(Evaluation(Base.tail(value(ℰ))), Base.tail(space), A))

_apply(ℰ::Evaluation, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _apply(Evaluation(value(ℰ)[1]), space[1], Val(N), A)

_project!(C::LinearOperator, ℰ::Evaluation) = _project!(C, ℰ, _memo(domain(C), eltype(C)))
function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℰ::Evaluation, memo)
    domain_C = domain(C)
    codomain_C = codomain(C)
    CoefType = eltype(C)
    @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
        C[α,β] = _getindex(ℰ, domain_C, codomain_C, CoefType, α, β, memo)
    end
    return C
end
function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℰ::Evaluation{<:Union{Number,Tuple{Vararg{Number}}}}, memo)
    domain_C = domain(C)
    codomain_C = codomain(C)
    CoefType = eltype(C)
    α = _findindex_constant(codomain_C)
    image_ℰ = codomain(ℰ, domain_C)
    α′ = _findindex_constant(image_ℰ)
    @inbounds for β ∈ indices(domain_C)
        C[α,β] = _getindex(ℰ, domain_C, image_ℰ, CoefType, α′, β, memo)
    end
    return C
end
function _project!(C::LinearOperator{<:SequenceSpace,ParameterSpace}, ℰ::Evaluation{<:Union{Number,Tuple{Vararg{Number}}}}, memo)
    domain_C = domain(C)
    CoefType = eltype(C)
    image_ℰ = codomain(ℰ, domain_C)
    α′ = _findindex_constant(image_ℰ)
    @inbounds for β ∈ indices(domain_C)
        C[1,β] = _getindex(ℰ, domain_C, image_ℰ, CoefType, α′, β, memo)
    end
    return C
end

_getindex(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}, α, β, memo) where {N,T} =
    @inbounds _getindex(Evaluation(value(ℰ)[1]), domain[1], codomain[1], T, α[1], β[1], memo[1]) * _getindex(Evaluation(Base.tail(value(ℰ))), Base.tail(domain), Base.tail(codomain), T, Base.tail(α), Base.tail(β), Base.tail(memo))
_getindex(ℰ::Evaluation{<:Tuple{Union{Nothing,Number}}}, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}, α, β, memo) where {T} =
    @inbounds _getindex(Evaluation(value(ℰ)[1]), domain[1], codomain[1], T, α[1], β[1], memo[1])

# Taylor

_memo(::Taylor, ::Type) = nothing

codomain(::Evaluation{Nothing}, s::Taylor) = s
codomain(::Evaluation, ::Taylor) = Taylor(0)

_coeftype(::Evaluation{Nothing}, ::Taylor, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, ::Taylor, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{Taylor})
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{Taylor})
    x = value(ℰ)
    if iszero(x)
        @inbounds c[0] = a[0]
    else
        ord = order(a)
        @inbounds c[0] = a[ord]
        @inbounds for i ∈ ord-1:-1:0
            c[0] = c[0] * x + a[i]
        end
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::Taylor, A)
    C .= A
    return C
end
function _apply!(C::AbstractArray{T}, ℰ::Evaluation, space::Taylor, A) where {T}
    x = value(ℰ)
    if iszero(x)
        @inbounds C .= selectdim(A, 1, 1)
    else
        ord = order(space)
        @inbounds C .= selectdim(A, 1, ord+1)
        @inbounds for i ∈ ord-1:-1:0
            C .= C .* x .+ selectdim(A, 1, i+1)
        end
    end
    return C
end

_apply(::Evaluation{Nothing}, ::Taylor, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = value(ℰ)
    CoefType = _coeftype(ℰ, space, T)
    if iszero(x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
    else
        ord = order(space)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, ord+1))
        @inbounds for i ∈ ord-1:-1:0
            C .= C .* x .+ selectdim(A, D, i+1)
        end
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::Taylor, ::Taylor, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, ::Taylor, ::Taylor, ::Type{T}, i, j, memo) where {T}
    if i == 0
        x = value(ℰ)
        if j == 0
            return one(T)
        else
            if iszero(x)
                return zero(T)
            else
                return convert(T, x ^ exact(j))
            end
        end
    else
        return zero(T)
    end
end

# Fourier

_memo(::Fourier, ::Type) = nothing

codomain(::Evaluation{Nothing}, s::Fourier) = s
codomain(::Evaluation, s::Fourier) = Fourier(0, frequency(s))

_coeftype(::Evaluation{Nothing}, ::Fourier, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, s::Fourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cis(frequency(s)*zero(T))), S)

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{<:Fourier})
    coefficients(c) .= coefficients(a)
    return c
end
_apply!(c, ℰ::Evaluation, a::Sequence{<:Fourier}) = __apply!(c, ℰ, a)
function __apply!(c, ℰ::Evaluation, a::Sequence{<:Fourier})
    x = value(ℰ)
    ord = order(a)
    @inbounds c[0] = a[0]
    if iszero(x)
        @inbounds for j ∈ 1:ord
            c[0] += a[j] + a[-j]
        end
    elseif ord > 0
        ωx = frequency(a)*x
        @inbounds for j ∈ 1:ord
            eiωxj = cis(ωx * exact(j))
            c[0] += a[j] * eiωxj + a[-j] / eiωxj
        end
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::Fourier, A)
    C .= A
    return C
end
_apply!(C::AbstractArray, ℰ::Evaluation, space::Fourier, A) = __apply!(C, ℰ, space, A)
function __apply!(C::AbstractArray, ℰ::Evaluation, space::Fourier, A)
    x = value(ℰ)
    ord = order(space)
    @inbounds C .= selectdim(A, 1, ord+1)
    if iszero(x)
        @inbounds for j ∈ 1:ord
            C .+= selectdim(A, 1, ord+1+j) .+ selectdim(A, 1, ord+1-j)
        end
    elseif ord > 0
        ωx = frequency(space)*x
        @inbounds for j ∈ 1:ord
            eiωxj = cis(ωx * exact(j))
            C .+= selectdim(A, 1, ord+1+j) .* eiωxj .+ selectdim(A, 1, ord+1-j) ./ eiωxj
        end
    end
    return C
end

_apply(::Evaluation{Nothing}, ::Fourier, ::Val, A::AbstractArray) = A
_apply(ℰ::Evaluation, space::Fourier, d::Val, A::AbstractArray) = __apply(ℰ, space, d, A, _coeftype(ℰ, space, eltype(A)))
function __apply(ℰ::Evaluation, space::Fourier, ::Val{D}, A::AbstractArray{T,N}, ::Type{CoefType}) where {D,T,N,CoefType}
    x = value(ℰ)
    ord = order(space)
    @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, ord+1))
    if iszero(x)
        @inbounds for j ∈ 1:ord
            C .+= selectdim(A, D, ord+1+j) .+ selectdim(A, D, ord+1-j)
        end
    elseif ord > 0
        ωx = frequency(space)*x
        @inbounds for j ∈ 1:ord
            eiωxj = cis(ωx * exact(j))
            C .+= selectdim(A, D, ord+1+j) .* eiωxj .+ selectdim(A, D, ord+1-j) ./ eiωxj
        end
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::Fourier, ::Fourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::Fourier, ::Fourier, ::Type{T}, i, j, memo) where {T}
    if i == 0
        x = value(ℰ)
        if j == 0 || iszero(x)
            return one(T)
        else
            return convert(T, cis(frequency(domain) * x * exact(j)))
        end
    else
        return zero(T)
    end
end

# Chebyshev

_memo(::Chebyshev, ::Type{T}) where {T} = Dict{Int,T}()

codomain(::Evaluation{Nothing}, s::Chebyshev) = s
codomain(::Evaluation, ::Chebyshev) = Chebyshev(0)

_coeftype(::Evaluation{Nothing}, ::Chebyshev, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, ::Chebyshev, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{Chebyshev})
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{Chebyshev})
    x = value(ℰ)
    ord = order(a)
    if iszero(x)
        @inbounds c[0] = a[0]
        @inbounds for i ∈ 2:2:ord
            c[0] += exact(ifelse(i%4 == 0, 2, -2)) * a[i]
        end
    elseif isone(-x)
        @inbounds c[0] = a[0]
        @inbounds for i ∈ 1:ord
            c[0] += exact(ifelse(isodd(i), -2, 2)) * a[i]
        end
    elseif isone(x)
        @inbounds c[0] = a[0]
        @inbounds for i ∈ 1:ord
            c[0] += exact(2) * a[i]
        end
    else
        if ord == 0
            @inbounds c[0] = a[0]
        elseif ord == 1
            @inbounds c[0] = a[0] + exact(2) * x * a[1]
        else
            CoefType = eltype(c)
            x2 = exact(2) * x
            s = zero(CoefType)
            @inbounds t = convert(CoefType, exact(2) * a[ord])
            @inbounds c[0] = zero(CoefType)
            @inbounds for i ∈ ord-1:-1:1
                c[0] = t
                t = x2 * t - s + exact(2) * a[i]
                s = c[0]
            end
            @inbounds c[0] = x * t - s + a[0]
        end
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::Chebyshev, A)
    C .= A
    return C
end
function _apply!(C::AbstractArray{T}, ℰ::Evaluation, space::Chebyshev, A) where {T}
    x = value(ℰ)
    ord = order(space)
    if iszero(x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 2:2:ord
            C .+= exact(ifelse(i%4 == 0, 2, -2)) .* selectdim(A, 1, i+1)
        end
    elseif isone(-x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 1:ord
            C .+= exact(ifelse(isodd(i), -2, 2)) .* selectdim(A, 1, i+1)
        end
    elseif isone(x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 1:ord
            C .+= exact(2) .* selectdim(A, 1, i+1)
        end
    else
        if ord == 0
            @inbounds C .= selectdim(A, 1, 1)
        elseif ord == 1
            @inbounds C .= selectdim(A, 1, 1) .+ (exact(2) * x) .* selectdim(A, 1, 2)
        else
            x2 = exact(2) * x
            @inbounds Aᵢ = selectdim(A, 1, ord+1)
            sz = size(Aᵢ)
            s = zeros(T, sz)
            t = Array{T}(undef, sz)
            t .= exact(2) .* Aᵢ
            @inbounds for i ∈ ord-1:-1:1
                C .= t
                t .= x2 .* t .- s .+ exact(2) .* selectdim(A, 1, i+1)
                s .= C
            end
            @inbounds C .= x .* t .- s .+ selectdim(A, 1, 1)
        end
    end
    return C
end

_apply(::Evaluation{Nothing}, ::Chebyshev, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = value(ℰ)
    CoefType = _coeftype(ℰ, space, T)
    ord = order(space)
    if iszero(x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 2:2:ord
            C .+= exact(ifelse(i%4 == 0, 2, -2)) .* selectdim(A, D, i+1)
        end
        return C
    elseif isone(-x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 1:ord
            C .+= exact(ifelse(isodd(i), -2, 2)) .* selectdim(A, D, i+1)
        end
        return C
    elseif isone(x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 1:ord
            C .+= exact(2) .* selectdim(A, D, i+1)
        end
        return C
    else
        if ord == 0
            return @inbounds convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        elseif ord == 1
            return @inbounds convert(Array{CoefType,N-1}, selectdim(A, D, 1) .+ (exact(2) * x) .* selectdim(A, D, 2))
        else
            x2 = exact(2) * x
            @inbounds Aᵢ = selectdim(A, D, ord+1)
            sz = size(Aᵢ)
            s = zeros(CoefType, sz)
            t = Array{CoefType,N-1}(undef, sz)
            t .= exact(2) .* Aᵢ
            C = Array{CoefType,N-1}(undef, sz)
            @inbounds for i ∈ ord-1:-1:1
                C .= t
                t .= x2 .* t .- s .+ exact(2) .* selectdim(A, D, i+1)
                s .= C
            end
            @inbounds C .= x .* t .- s .+ selectdim(A, D, 1)
            return C
        end
    end
end

_getindex(::Evaluation{Nothing}, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::Chebyshev, codomain::Chebyshev, ::Type{T}, i, j, memo) where {T}
    if i == 0
        if j == 0
            return one(T)
        else
            x = value(ℰ)
            if iszero(x)
                if isodd(j)
                    return zero(T)
                elseif j%4 == 0
                    return convert(T, exact(2))
                else
                    return convert(T, exact(-2))
                end
            elseif isone(-x)
                if isodd(j)
                    return convert(T, exact(-2))
                else
                    return convert(T, exact(2))
                end
            elseif isone(x)
                return convert(T, exact(2))
            else
                x2 = convert(T, exact(2) * x)
                if j == 1
                    return x2
                elseif j == 2
                    return get!(memo, j) do
                        convert(T, x2 * x2 - convert(T, exact(2)))
                    end
                else
                    return get!(memo, j) do
                        x2 * _getindex(ℰ, domain, codomain, T, i, j-1, memo) - _getindex(ℰ, domain, codomain, T, i, j-2, memo)
                    end
                end
            end
        end
    else
        return zero(T)
    end
end

# CosFourier

_memo(::CosFourier, ::Type) = nothing

codomain(::Evaluation{Nothing}, s::CosFourier) = s
codomain(::Evaluation, s::CosFourier) = CosFourier(0, frequency(s))

_coeftype(::Evaluation{Nothing}, ::CosFourier, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, s::CosFourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cos(frequency(s)*zero(T))), S)

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{<:CosFourier})
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{<:CosFourier})
    x = value(ℰ)
    ord = order(a)
    @inbounds c[0] = a[ord]
    if ord > 0
        if iszero(x)
            @inbounds for j ∈ ord-1:-1:1
                c[0] += a[j]
            end
        else
            ωx = frequency(a)*x
            @inbounds c[0] *= cos(ωx * exact(ord))
            @inbounds for j ∈ ord-1:-1:1
                c[0] += a[j] * cos(ωx * exact(j))
            end
        end
        @inbounds c[0] = exact(2) * c[0] + a[0]
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::CosFourier, A)
    C .= A
    return C
end
function _apply!(C::AbstractArray, ℰ::Evaluation, space::CosFourier, A)
    x = value(ℰ)
    ord = order(space)
    @inbounds C .= selectdim(A, 1, ord+1)
    if ord > 0
        if iszero(x)
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, 1, j+1)
            end
        else
            ωx = frequency(space)*x
            C .*= cos(ωx * exact(ord))
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, 1, j+1) .* cos(ωx * exact(j))
            end
        end
        @inbounds C .= exact(2) .* C .+ selectdim(A, 1, 1)
    end
    return C
end

_apply(::Evaluation{Nothing}, ::CosFourier, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::CosFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = value(ℰ)
    CoefType = _coeftype(ℰ, space, T)
    ord = order(space)
    @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, ord+1))
    if ord > 0
        if iszero(x)
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, D, j+1)
            end
        else
            ωx = frequency(space)*x
            C .*= cos(ωx * exact(ord))
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, D, j+1) .* cos(ωx * exact(j))
            end
        end
        @inbounds C .= exact(2) .* C .+ selectdim(A, D, 1)
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::CosFourier, ::CosFourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::CosFourier, ::CosFourier, ::Type{T}, i, j, memo) where {T}
    if i == 0
        x = value(ℰ)
        if j == 0
            return one(T)
        elseif iszero(x)
            return convert(T, exact(2))
        else
            return convert(T, exact(2) * cos(frequency(domain) * x * exact(j)))
        end
    else
        return zero(T)
    end
end

# SinFourier

_memo(::SinFourier, ::Type) = nothing

codomain(::Evaluation{Nothing}, s::SinFourier) = s
codomain(::Evaluation, s::SinFourier) = Fourier(0, frequency(s))

_coeftype(::Evaluation{Nothing}, ::SinFourier, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, s::SinFourier, ::Type{S}) where {T,S} =
    promote_type(typeof(sin(frequency(s)*zero(T))), S)

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{<:SinFourier})
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{<:SinFourier})
    x = value(ℰ)
    if iszero(x)
        @inbounds c[0] = zero(eltype(c))
    else
        ord = order(a)
        ωx = frequency(a)*x
        @inbounds c[0] = a[ord] * sin(ωx * exact(ord))
        @inbounds for j ∈ ord-1:-1:1
            c[0] += a[j] * sin(ωx * exact(j))
        end
        @inbounds c[0] *= exact(2)
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::SinFourier, A)
    C .= A
    return C
end
function _apply!(C::AbstractArray, ℰ::Evaluation, space::SinFourier, A)
    x = value(ℰ)
    if iszero(x)
        C .= zero(eltype(C))
    else
        ord = order(space)
        ωx = frequency(space)*x
        @inbounds C .= selectdim(A, 1, ord) .* sin(ωx * exact(ord))
        @inbounds for j ∈ ord-1:-1:1
            C .+= selectdim(A, 1, j) .* sin(ωx * exact(j))
        end
        C .*= exact(2)
    end
    return C
end

_apply(::Evaluation{Nothing}, ::SinFourier, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::SinFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = value(ℰ)
    CoefType = _coeftype(ℰ, space, T)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, D, ord)
    C = Array{CoefType,N-1}(undef, size(Aᵢ))
    if iszero(x)
        C .= zero(CoefType)
    else
        ωx = frequency(space)*x
        @inbounds C .= Aᵢ .* sin(ωx * exact(ord))
        @inbounds for j ∈ ord-1:-1:1
            C .+= selectdim(A, D, j) .* sin(ωx * exact(j))
        end
        C .*= exact(2)
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::SinFourier, ::SinFourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::SinFourier, ::Fourier, ::Type{T}, i, j, memo) where {T}
    x = value(ℰ)
    if i == 0 && !iszero(x)
        return convert(T, exact(2) * sin(frequency(domain) * x * exact(j)))
    else
        return zero(T)
    end
end

# Cartesian spaces

_memo(s::CartesianPower, ::Type{T}) where {T} = _memo(space(s), T)

codomain(ℰ::Evaluation, s::CartesianPower) = CartesianPower(codomain(ℰ, space(s)), nspaces(s))

_coeftype(ℰ::Evaluation, s::CartesianPower, ::Type{T}) where {T} = _coeftype(ℰ, space(s), T)

_memo(s::CartesianProduct, ::Type{T}) where {T} = map(sᵢ -> _memo(sᵢ, T), spaces(s))

codomain(ℰ::Evaluation, s::CartesianProduct) = CartesianProduct(map(sᵢ -> codomain(ℰ, sᵢ), spaces(s)))

_coeftype(ℰ::Evaluation, s::CartesianProduct, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(ℰ, s[1], T), _coeftype(ℰ, Base.tail(s), T))
_coeftype(ℰ::Evaluation, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(ℰ, s[1], T)

function _apply!(c, ℰ::Evaluation, a::Sequence{<:CartesianPower})
    @inbounds for i ∈ 1:nspaces(space(c))
        _apply!(component(c, i), ℰ, component(a, i))
    end
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{CartesianProduct{T}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _apply!(component(c, 1), ℰ, component(a, 1))
    @inbounds _apply!(component(c, 2:N), ℰ, component(a, 2:N))
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{CartesianProduct{T}}) where {T<:Tuple{VectorSpace}}
    @inbounds _apply!(component(c, 1), ℰ, component(a, 1))
    return c
end

function _project!(C::LinearOperator{<:CartesianPower,<:CartesianSpace}, ℰ::Evaluation, memo)
    @inbounds for i ∈ 1:nspaces(domain(C))
        _project!(component(C, i, i), ℰ, memo)
    end
    return C
end
function _project!(C::LinearOperator{<:CartesianProduct,<:CartesianSpace}, ℰ::Evaluation, memo)
    @inbounds for i ∈ 1:nspaces(domain(C))
        _project!(component(C, i, i), ℰ, memo[i])
    end
    return C
end
