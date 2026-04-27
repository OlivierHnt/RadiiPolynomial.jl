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

# Tensor space

domain(::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, ::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = EmptySpace()
domain(::Evaluation{<:NTuple{N,Nothing}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = s

codomain(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((xᵢ, sᵢ) -> codomain(Evaluation(xᵢ), sᵢ), value(ℰ), spaces(s)))

_coeftype(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Evaluation(value(ℰ)[1]), s[1], T), _coeftype(Evaluation(Base.tail(value(ℰ))), Base.tail(s), T))
_coeftype(ℰ::Evaluation{<:Tuple{Union{Nothing,Number}}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Evaluation(value(ℰ)[1]), s[1], T)

getcoefficient(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Evaluation(value(ℰ)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Evaluation(Base.tail(value(ℰ))), (Base.tail(codom), Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(ℰ::Evaluation{<:Tuple{Union{Nothing,Number}}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Evaluation(value(ℰ)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

#

domain(::Evaluation{<:Number}, ::ScalarSpace) = EmptySpace()
domain(::Evaluation{<:Tuple{Vararg{Number}}}, ::ScalarSpace) = EmptySpace()

# Taylor

domain(::Evaluation{Nothing}, s::Taylor) = s
domain(::Evaluation{<:Number}, ::Taylor) = EmptySpace()

codomain(::Evaluation{Nothing}, s::Taylor) = s
codomain(::Evaluation{<:Number}, ::Taylor) = Taylor(0)

_coeftype(::Evaluation{Nothing}, ::Taylor, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, ::Taylor, ::Type{S}) where {T<:Number,S} =
    promote_type(T, S)

getcoefficient(::Evaluation{Nothing}, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T} =
    ifelse(i == j, one(T), zero(T))
function getcoefficient(ℰ::Evaluation{<:Number}, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    i == 0 || return zero(T)
    j == 0 && return one(T)
    x = value(ℰ)
    _safe_iszero(x) && return zero(T)
    return convert(T, x ^ exact(j))
end

# Fourier

domain(::Evaluation{Nothing}, s::Fourier) = s
domain(::Evaluation{<:Number}, ::Fourier) = EmptySpace()

codomain(::Evaluation{<:Number}, s::Fourier) = Fourier(0, frequency(s))
codomain(::Evaluation{Nothing}, s::Fourier) = s

_coeftype(::Evaluation{Nothing}, ::Fourier, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, s::Fourier, ::Type{S}) where {T<:Number,S} =
    promote_type(typeof(cis(frequency(s)*zero(T))), S)

getcoefficient(::Evaluation{Nothing}, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T} =
    ifelse(i == j, one(T), zero(T))
function getcoefficient(ℰ::Evaluation{<:Number}, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T}
    i == 0 || return zero(T)
    x = value(ℰ)
    (j == 0) | _safe_iszero(x) && return one(T)
    return convert(T, cis(frequency(dom) * x * exact(j)))
end

# Chebyshev

domain(::Evaluation{Nothing}, s::Chebyshev) = s
domain(::Evaluation{<:Number}, ::Chebyshev) = EmptySpace()

codomain(::Evaluation{Nothing}, s::Chebyshev) = s
codomain(::Evaluation{<:Number}, ::Chebyshev) = Chebyshev(0)

_coeftype(::Evaluation{Nothing}, ::Chebyshev, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, ::Chebyshev, ::Type{S}) where {T<:Number,S} =
    promote_type(T, S)

getcoefficient(::Evaluation{Nothing}, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T} =
    ifelse(i == j, one(T), zero(T))
function getcoefficient(ℰ::Evaluation{<:Number}, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    i == 0 || return zero(T)
    j == 0 && return one(T)
    x = value(ℰ)
    if _safe_iszero(x)
        isodd(j) && return zero(T)
        j%4 == 0 && return convert(T, exact(2))
        return convert(T, exact(-2))
    elseif _safe_isone(-x)
        isodd(j) && return convert(T, exact(-2))
        return convert(T, exact(2))
    elseif _safe_isone(x)
        return convert(T, exact(2))
    else
        return convert(T, exact(2) * cos(exact(j) * acos(x)))
    end
end

# Symmetric space

domain(::Evaluation{Nothing}, s::SymmetricSpace{<:BaseSpace}) = s
domain(::Evaluation{<:NTuple{N,Nothing}}, s::SymmetricSpace{<:TensorSpace{<:NTuple{N,BaseSpace}}}) where {N} = s
domain(::Evaluation{<:Number}, s::SymmetricSpace{<:BaseSpace}) = EmptySpace()
domain(::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::SymmetricSpace{<:TensorSpace{<:NTuple{N,BaseSpace}}}) where {N} = EmptySpace()

codomain(::Evaluation{Nothing}, s::SymmetricSpace{<:BaseSpace}) = s
codomain(::Evaluation{<:NTuple{N,Nothing}}, s::SymmetricSpace{<:TensorSpace{<:NTuple{N,BaseSpace}}}) where {N} = s
codomain(ℰ::Evaluation{<:Number}, s::SymmetricSpace{<:BaseSpace}) = SymmetricSpace(codomain(ℰ, desymmetrize(s)), _sym_with_cst_coef(symmetry(s)))
codomain(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::SymmetricSpace{<:TensorSpace{<:NTuple{N,BaseSpace}}}) where {N} = SymmetricSpace(codomain(ℰ, desymmetrize(s)), _sym_with_cst_coef(symmetry(s)))

_coeftype(ℰ::Evaluation, s::SymmetricSpace, ::Type{T}) where {T} = _coeftype(ℰ, desymmetrize(s), T)





# action

Base.:*(ℰ::Evaluation, a::AbstractSequence) = evaluate(a, value(ℰ))

(a::AbstractSequence)(x::Union{Nothing,Number}, y::Union{Nothing,Number}...) = evaluate(a, (x, y...))
(a::AbstractSequence)(x::Union{Nothing,Number}) = evaluate(a, x)

function evaluate(a::Sequence, x::Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}})
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

function evaluate!(c::Sequence, a::Sequence, x::Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}})
    ℰ = Evaluation(x)
    space_c = space(c)
    new_space = codomain(ℰ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℰ(a) has space $new_space"))
    _apply!(c, ℰ, a)
    return c
end
function evaluate!(c::AbstractVector, a::Sequence, x::Union{Number,Tuple{Vararg{Number}}})
    evaluate!(Sequence(codomain(Evaluation(x), space(a)), c), a, x)
    return c
end

# Tensor space

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

# Taylor

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{Taylor})
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{Taylor})
    x = value(ℰ)
    if _safe_iszero(x)
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
    if _safe_iszero(x)
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
    if _safe_iszero(x)
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

# Fourier

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{<:Fourier})
    coefficients(c) .= coefficients(a)
    return c
end
_apply!(c, ℰ::Evaluation, a::Sequence{<:Fourier}) = __apply!(c, ℰ, a)
function __apply!(c, ℰ::Evaluation, a::Sequence{<:Fourier})
    x = value(ℰ)
    ord = order(a)
    @inbounds c[0] = a[0]
    if _safe_iszero(x)
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
    if _safe_iszero(x)
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
    if _safe_iszero(x)
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

# Chebyshev

function _apply!(c, ::Evaluation{Nothing}, a::Sequence{Chebyshev})
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c, ℰ::Evaluation, a::Sequence{Chebyshev})
    x = value(ℰ)
    ord = order(a)
    if _safe_iszero(x)
        @inbounds c[0] = a[0]
        @inbounds for i ∈ 2:2:ord
            c[0] += exact(ifelse(i%4 == 0, 2, -2)) * a[i]
        end
    elseif _safe_isone(-x)
        @inbounds c[0] = a[0]
        @inbounds for i ∈ 1:ord
            c[0] += exact(ifelse(isodd(i), -2, 2)) * a[i]
        end
    elseif _safe_isone(x)
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
    if _safe_iszero(x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 2:2:ord
            C .+= exact(ifelse(i%4 == 0, 2, -2)) .* selectdim(A, 1, i+1)
        end
    elseif _safe_isone(-x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 1:ord
            C .+= exact(ifelse(isodd(i), -2, 2)) .* selectdim(A, 1, i+1)
        end
    elseif _safe_isone(x)
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
    if _safe_iszero(x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 2:2:ord
            C .+= exact(ifelse(i%4 == 0, 2, -2)) .* selectdim(A, D, i+1)
        end
        return C
    elseif _safe_isone(-x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 1:ord
            C .+= exact(ifelse(isodd(i), -2, 2)) .* selectdim(A, D, i+1)
        end
        return C
    elseif _safe_isone(x)
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





#

evaluate(a::InfiniteSequence, x::Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}}) = _return_evaluate(evaluate(sequence(a), x), a)

_return_evaluate(c, a::InfiniteSequence) = interval(c, sequence_error(a); format = :midpoint)

_return_evaluate(c::Sequence, a::InfiniteSequence) = InfiniteSequence(interval.(c, sequence_error(a); format = :midpoint), sequence_error(a), banachspace(a))
