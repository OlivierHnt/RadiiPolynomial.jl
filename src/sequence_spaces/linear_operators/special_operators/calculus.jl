"""
    Derivative{T<:Union{Int,Tuple{Vararg{Int}}}} <: AbstractLinearOperator

Generic derivative operator.

Field:
- `order :: T`

Constructors:
- `Derivative(::Int)`
- `Derivative(::Tuple{Vararg{Int}})`
- `Derivative(order::Int...)`: equivalent to `Derivative(order)`

See also: [`differentiate`](@ref), [`differentiate!`](@ref),
[`project(::Derivative, ::VectorSpace, ::VectorSpace)`](@ref)
and [`project!(::LinearOperator, ::Derivative)`](@ref).

# Examples

```jldoctest
julia> Derivative(1)
Derivative{Int64}(1)

julia> Derivative(1, 2)
Derivative{Tuple{Int64, Int64}}((1, 2))
```
"""
struct Derivative{T<:Union{Int,Tuple{Vararg{Int}}}} <: AbstractLinearOperator
    order :: T
    function Derivative{T}(order::T) where {T<:Int}
        order < 0 && return throw(DomainError(order, "Derivative is only defined for positive integers"))
        return new{T}(order)
    end
    function Derivative{T}(order::T) where {T<:Tuple{Vararg{Int}}}
        any(n -> n < 0, order) && return throw(DomainError(order, "Derivative is only defined for positive integers"))
        return new{T}(order)
    end
    Derivative{Tuple{}}(::Tuple{}) = throw(ArgumentError("Derivative is only defined for at least one Int"))
end

Derivative(order::T) where {T<:Int} = Derivative{T}(order)
Derivative(order::T) where {T<:Tuple{Vararg{Int}}} = Derivative{T}(order)
Derivative(order::Int...) = Derivative(order)

order(𝒟::Derivative) = 𝒟.order

function _infer_domain(D::Derivative{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    s_out = map((αᵢ, sᵢ) -> _infer_domain(Derivative(αᵢ), sᵢ), order(D), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return TensorSpace(s_out)
end
_infer_domain(D::Derivative, s::Taylor) = codomain(Integral(order(D)), s)
_infer_domain(::Derivative, s::Fourier) = s
_infer_domain(D::Derivative, s::Chebyshev) = iszero(order(D)) ? s : EmptySpace() # flags an error
_infer_domain(D::Derivative, s::CosFourier) = codomain(Integral(order(D)), s)
_infer_domain(D::Derivative, s::SinFourier) = codomain(Integral(order(D)), s)
function _infer_domain(D::Derivative, s::CartesianPower)
    s_out = _infer_domain(D, space(s))
    s_out isa EmptySpace && return EmptySpace()
    return CartesianPower(s_out, nspaces(s))
end
function _infer_domain(D::Derivative, s::CartesianSpace)
    s_out = map(sᵢ -> _infer_domain(D, sᵢ), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return CartesianProduct(s_out)
end

"""
    Integral{T<:Union{Int,Tuple{Vararg{Int}}}} <: AbstractLinearOperator

Generic integral operator.

Field:
- `order :: T`

Constructors:
- `Integral(::Int)`
- `Integral(::Tuple{Vararg{Int}})`
- `Integral(order::Int...)`: equivalent to `Integral(order)`

See also: [`integrate`](@ref), [`integrate!`](@ref),
[`project(::Integral, ::VectorSpace, ::VectorSpace)`](@ref)
and [`project!(::LinearOperator, ::Integral)`](@ref).

# Examples

```jldoctest
julia> Integral(1)
Integral{Int64}(1)

julia> Integral(1, 2)
Integral{Tuple{Int64, Int64}}((1, 2))
```
"""
struct Integral{T<:Union{Int,Tuple{Vararg{Int}}}} <: AbstractLinearOperator
    order :: T
    function Integral{T}(order::T) where {T<:Int}
        order < 0 && return throw(DomainError(order, "Integral is only defined for positive integers"))
        return new{T}(order)
    end
    function Integral{T}(order::T) where {T<:Tuple{Vararg{Int}}}
        any(n -> n < 0, order) && return throw(DomainError(order, "Integral is only defined for positive integers"))
        return new{T}(order)
    end
    Integral{Tuple{}}(::Tuple{}) = throw(ArgumentError("Integral is only defined for at least one Int"))
end

Integral(order::T) where {T<:Int} = Integral{T}(order)
Integral(order::T) where {T<:Tuple{Vararg{Int}}} = Integral{T}(order)
Integral(order::Int...) = Integral(order)

order(ℐ::Integral) = ℐ.order

function _infer_domain(I::Integral{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    s_out = map((αᵢ, sᵢ) -> _infer_domain(Integral(αᵢ), sᵢ), order(I), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return TensorSpace(s_out)
end
_infer_domain(I::Integral, s::Taylor) = codomain(Derivative(order(I)), s)
_infer_domain(::Integral, s::Fourier) = s
_infer_domain(I::Integral, s::Chebyshev) = iszero(order(I)) ? s : EmptySpace() # flags an error
_infer_domain(I::Integral, s::CosFourier) = codomain(Derivative(order(I)), s)
_infer_domain(I::Integral, s::SinFourier) = codomain(Derivative(order(I)), s)
function _infer_domain(I::Integral, s::CartesianPower)
    s_out = _infer_domain(I, space(s))
    s_out isa EmptySpace && return EmptySpace()
    return CartesianPower(s_out, nspaces(s))
end
function _infer_domain(I::Integral, s::CartesianSpace)
    s_out = map(sᵢ -> _infer_domain(I, sᵢ), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return CartesianProduct(s_out)
end

"""
    *(𝒟::Derivative, a::AbstractSequence)

Compute the `order(𝒟)`-th derivative of `a`; equivalent to `differentiate(a, order(𝒟))`.

See also: [`Derivative`](@ref), [`differentiate`](@ref) and
[`differentiate!`](@ref).
"""
Base.:*(𝒟::Derivative, a::AbstractSequence) = differentiate(a, order(𝒟))

"""
    differentiate(a::AbstractSequence, α=1)

Compute the `α`-th derivative of `a`.

See also: [`differentiate!`](@ref), [`Derivative`](@ref),
[`*(::Derivative, ::Sequence)`](@ref) and [`(::Derivative)(::Sequence)`](@ref).
"""
function differentiate(a::Sequence, α=1)
    𝒟 = Derivative(α)
    space_a = space(a)
    new_space = codomain(𝒟, space_a)
    CoefType = _coeftype(𝒟, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒟, a)
    return c
end

function differentiate(a::InfiniteSequence, α=1)
    c = differentiate(sequence(a), α)
    X = banachspace(a)
    seq_err = sequence_error(a)
    iszero(seq_err) && return InfiniteSequence(c, X)
    return InfiniteSequence(c, _derivative_error(X, space(a), space(c), α) * seq_err, X)
end

_derivative_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _derivative_error(X, dom[1], codom[1], α[1]) * _derivative_error(X, Base.tail(dom), Base.tail(codom), Base.tail(α))
_derivative_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _derivative_error(X, dom[1], codom[1], α[1])

_derivative_error(X::Ell1{<:NTuple{N,Weight}}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _derivative_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1]) * _derivative_error(Ell1(Base.tail(weight(X))), Base.tail(dom), Base.tail(codom), Base.tail(α))
_derivative_error(X::Ell1{<:Tuple{Weight}}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _derivative_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1])

function _derivative_error(X::Ell1{<:GeometricWeight}, ::Taylor, ::Taylor, n::Int)
    ν = rate(weight(X))
    n == 0 && return one(ν)
    n == 1 && return ν                                                   / (ν - exact(1))^2
    n == 2 && return ν * (ν + exact(1))                                  / (ν - exact(1))^3
    n == 3 && return ν * (ν^2 + exact(4)*ν + exact(1))                   / (ν - exact(1))^4
    n == 4 && return ν * (ν + exact(1)) * (ν^2 + exact(10)*ν + exact(1)) / (ν - exact(1))^5
    return throw(DomainError) # TODO: lift restriction
end
function _derivative_error(X::Ell1{<:GeometricWeight}, ::Fourier, codom::Fourier, n::Int)
    ν = rate(weight(X))
    n == 0 && return one(ν)
    n == 1 && return frequency(codom)   * exact(2) * ν                                                               / (ν - exact(1))^2
    n == 2 && return frequency(codom)^2 * exact(2) * ν * (ν + exact(1))                                          / (ν - exact(1))^3
    n == 3 && return frequency(codom)^3 * exact(2) * ν * (ν^2 + exact(4)*ν + exact(1))                       / (ν - exact(1))^4
    n == 4 && return frequency(codom)^4 * exact(2) * ν * (ν + exact(1)) * (ν^2 + exact(10)*ν + exact(1)) / (ν - exact(1))^5
    return throw(DomainError) # TODO: lift restriction
end
function _derivative_error(::Ell1{<:GeometricWeight}, ::Chebyshev, ::Chebyshev, n::Int)
    n == 0 && return interval(1)
    return throw(DomainError) # TODO: lift restriction
end

"""
    differentiate!(c::Sequence, a::Sequence, α=1)

Compute the `α`-th derivative of `a`. The result is stored in `c` by overwriting it.

See also: [`differentiate`](@ref), [`Derivative`](@ref),
[`*(::Derivative, ::Sequence)`](@ref) and [`(::Derivative)(::Sequence)`](@ref).
"""
function differentiate!(c::Sequence, a::Sequence, α=1)
    𝒟 = Derivative(α)
    space_c = space(c)
    new_space = codomain(𝒟, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒟(a) has space $new_space"))
    _apply!(c, 𝒟, a)
    return c
end

"""
    *(ℐ::Integral, a::AbstractSequence)

Compute the `order(ℐ)`-th integral of `a`; equivalent to `integrate(a, order(ℐ))`.

See also: [`(::Integral)(::AbstractSequence)`](@ref), [`Integral`](@ref),
[`integrate`](@ref) and [`integrate!`](@ref).
"""
Base.:*(ℐ::Integral, a::AbstractSequence) = integrate(a, order(ℐ))

"""
    integrate(a::AbstractSequence, α=1)

Compute the `α`-th integral of `a`.

See also: [`integrate!`](@ref), [`Integral`](@ref),
[`*(::Integral, ::Sequence)`](@ref) and [`(::Integral)(::Sequence)`](@ref).
"""
function integrate(a::Sequence, α=1)
    ℐ = Integral(α)
    space_a = space(a)
    new_space = codomain(ℐ, space_a)
    CoefType = _coeftype(ℐ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, ℐ, a)
    return c
end

function integrate(a::InfiniteSequence, α=1)
    c = integrate(sequence(a), α)
    X = banachspace(a)
    return InfiniteSequence(c, _integral_error(X, space(a), space(c), α) * sequence_error(a), X)
end

_integral_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _integral_error(X, dom[1], codom[1], α[1]) * _integral_error(X, Base.tail(dom), Base.tail(codom), Base.tail(α))
_integral_error(X::Ell1{IdentityWeight}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _integral_error(X, dom[1], codom[1], α[1])

_integral_error(X::Ell1{<:NTuple{N,Weight}}, dom::TensorSpace{<:NTuple{N,BaseSpace}}, codom::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _integral_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1]) * _integral_error(Ell1(Base.tail(weight(X))), Base.tail(dom), Base.tail(codom), Base.tail(α))
_integral_error(X::Ell1{<:Tuple{Weight}}, dom::TensorSpace{<:Tuple{BaseSpace}}, codom::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _integral_error(Ell1(weight(X)[1]), dom[1], codom[1], α[1])

function _integral_error(X::Ell1, dom::Taylor, codom::Taylor, n::Int)
    v = __getindex(weight(X), codom, n)
    return v * _nzval(Integral(n), dom, codom, typeof(v), n, 0)
end
_integral_error(::Ell1, dom::Fourier{T}, codom::Fourier{S}, n::Int) where {T<:Real,S<:Real} =
    abs(_nzval(Integral(n), dom, codom, complex(promote_type(T, S)), 1, 1))
function _integral_error(X::Ell1, ::Chebyshev, codom::Chebyshev, n::Int)
    v = exact(1) + __getindex(weight(X), codom, n)
    n == 0 && return one(v)
    n == 1 && return v
    return throw(DomainError) # TODO: lift restriction
end

__getindex(::IdentityWeight, ::BaseSpace, ::Int) = interval(1)
__getindex(w::Weight, s::BaseSpace, n::Int) = _getindex(w, s, n)


"""
    integrate!(c::Sequence, a::Sequence, α=1)

Compute the `α`-th integral of `a`. The result is stored in `c` by overwriting it.

See also: [`integrate`](@ref), [`Integral`](@ref),
[`*(::Integral, ::Sequence)`](@ref) and [`(::Integral)(::Sequence)`](@ref).
"""
function integrate!(c::Sequence, a::Sequence, α=1)
    ℐ = Integral(α)
    space_c = space(c)
    new_space = codomain(ℐ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℐ(a) has space $new_space"))
    _apply!(c, ℐ, a)
    return c
end

for (F, f) ∈ ((:Derivative, :differentiate), (:Integral, :integrate))
    @eval begin
        Base.:*(ℱ₁::$F{Int}, ℱ₂::$F{Int}) = $F(order(ℱ₁) + order(ℱ₂))
        Base.:*(ℱ₁::$F{NTuple{N,Int}}, ℱ₂::$F{NTuple{N,Int}}) where {N} = $F(map(+, order(ℱ₁), order(ℱ₂)))

        Base.:^(ℱ::$F{Int}, n::Integer) = $F(order(ℱ) * n)
        Base.:^(ℱ::$F{<:Tuple{Vararg{Int}}}, n::Integer) = $F(map(αᵢ -> *(αᵢ, n), order(ℱ)))
        Base.:^(ℱ::$F{NTuple{N,Int}}, n::NTuple{N,Integer}) where {N} = $F(map(*, order(ℱ), n))
    end
end

# Sequence spaces

for F ∈ (:Derivative, :Integral)
    @eval begin
        codomain(ℱ::$F{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
            TensorSpace(map((αᵢ, sᵢ) -> codomain($F(αᵢ), sᵢ), order(ℱ), spaces(s)))

        _coeftype(ℱ::$F{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
            @inbounds promote_type(_coeftype($F(order(ℱ)[1]), s[1], T), _coeftype($F(Base.tail(order(ℱ))), Base.tail(s), T))
        _coeftype(ℱ::$F{Tuple{Int}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
            @inbounds _coeftype($F(order(ℱ)[1]), s[1], T)

        function _apply!(c::Sequence{<:TensorSpace}, ℱ::$F, a)
            space_a = space(a)
            A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
            C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
            _apply!(C, ℱ, space_a, A)
            return c
        end

        _apply!(C, ℱ::$F, space::TensorSpace, A) =
            @inbounds _apply!(C, $F(order(ℱ)[1]), space[1], _apply($F(Base.tail(order(ℱ))), Base.tail(space), A))

        _apply!(C, ℱ::$F, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply!(C, $F(order(ℱ)[1]), space[1], A)

        _apply(ℱ::$F, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
            @inbounds _apply($F(order(ℱ)[1]), space[1], Val(N₂-N₁+1), _apply($F(Base.tail(order(ℱ))), Base.tail(space), A))

        _apply(ℱ::$F, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
            @inbounds _apply($F(order(ℱ)[1]), space[1], Val(N), A)
    end
end

for F ∈ (:Derivative, :Integral)
    for (_f, __f) ∈ ((:_nzind_domain, :__nzind_domain), (:_nzind_codomain, :__nzind_codomain))
        @eval begin
            $_f(ℱ::$F{NTuple{N,Int}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
                TensorIndices($__f(ℱ, domain, codomain))
            $__f(ℱ::$F, domain::TensorSpace, codomain) =
                @inbounds ($_f($F(order(ℱ)[1]), domain[1], codomain[1]), $__f($F(Base.tail(order(ℱ))), Base.tail(domain), Base.tail(codomain))...)
            $__f(ℱ::$F, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain) =
                @inbounds ($_f($F(order(ℱ)[1]), domain[1], codomain[1]),)
        end
    end

    @eval begin
        function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℱ::$F)
            domain_C = domain(C)
            codomain_C = codomain(C)
            CoefType = eltype(C)
            @inbounds for (α, β) ∈ zip(_nzind_codomain(ℱ, domain_C, codomain_C), _nzind_domain(ℱ, domain_C, codomain_C))
                C[α,β] = _nzval(ℱ, domain_C, codomain_C, CoefType, α, β)
            end
            return C
        end

        _nzval(ℱ::$F{NTuple{N,Int}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}, α, β) where {N,T} =
            @inbounds _nzval($F(order(ℱ)[1]), domain[1], codomain[1], T, α[1], β[1]) * _nzval($F(Base.tail(order(ℱ))), Base.tail(domain), Base.tail(codomain), T, Base.tail(α), Base.tail(β))
        _nzval(ℱ::$F{Tuple{Int}}, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}, α, β) where {T} =
            @inbounds _nzval($F(order(ℱ)[1]), domain[1], codomain[1], T, α[1], β[1])
    end
end

# Taylor

codomain(𝒟::Derivative, s::Taylor) = Taylor(max(0, order(s)-order(𝒟)))

_coeftype(::Derivative, ::Taylor, ::Type{T}) where {T} = T

function _apply!(c::Sequence{Taylor}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        order_a = order(a)
        if order_a < n
            @inbounds c[0] = zero(eltype(c))
        elseif n == 1
            @inbounds for i ∈ 1:order_a
                c[i-1] = exact(i) * a[i]
            end
        else
            space_a = space(a)
            CoefType = eltype(c)
            @inbounds for i ∈ n:order_a
                c[i-n] = _nzval(𝒟, space_a, space_a, CoefType, i-n, i) * a[i]
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::Taylor, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    else
        ord = order(space)
        if ord < n
            C .= zero(T)
        elseif n == 1
            @inbounds for i ∈ 1:ord
                selectdim(C, 1, i) .= exact(i) .* selectdim(A, 1, i+1)
            end
        else
            @inbounds for i ∈ n:ord
                selectdim(C, 1, i-n+1) .= _nzval(𝒟, space, space, T, i-n, i) .* selectdim(A, 1, i+1)
            end
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        ord = order(space)
        if ord < n
            return zeros(CoefType, ntuple(i -> ifelse(i == D, 1, size(A, i)), Val(N)))
        elseif n == 1
            C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord, size(A, i)), Val(N)))
            @inbounds for i ∈ 1:ord
                selectdim(C, D, i) .= exact(i) .* selectdim(A, D, i+1)
            end
            return C
        else
            C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord-n+1, size(A, i)), Val(N)))
            @inbounds for i ∈ n:ord
                selectdim(C, D, i-n+1) .= _nzval(𝒟, space, space, CoefType, i-n, i) .* selectdim(A, D, i+1)
            end
            return C
        end
    end
end

_nzind_domain(𝒟::Derivative, domain::Taylor, codomain::Taylor) =
    order(𝒟):min(order(domain), order(codomain)+order(𝒟))

_nzind_codomain(𝒟::Derivative, domain::Taylor, codomain::Taylor) =
    0:min(order(domain)-order(𝒟), order(codomain))

function _nzval(𝒟::Derivative, ::Taylor, ::Taylor, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    p = one(real(T))
    for k ∈ 1:n
        p = exact(i+k) * p
    end
    return convert(T, p)
end

codomain(ℐ::Integral, s::Taylor) = Taylor(order(s)+order(ℐ))

_coeftype(::Integral, ::Taylor, ::Type{T}) where {T} = typeof(inv(one(T))*zero(T))

function _apply!(c::Sequence{Taylor}, ℐ::Integral, a)
    n = order(ℐ)
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        @inbounds c[0] = zero(eltype(c))
        @inbounds for i ∈ 0:order(a)
            c[i+1] = a[i] / exact(i+1)
        end
    else
        space_a = space(a)
        CoefType = eltype(c)
        @inbounds view(c, 0:n-1) .= zero(CoefType)
        @inbounds for i ∈ 0:order(a)
            c[i+n] = _nzval(ℐ, space_a, space_a, CoefType, i+n, i) * a[i]
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Taylor, A) where {T}
    n = order(ℐ)
    if n == 0
        C .= A
    elseif n == 1
        ord = order(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        @inbounds for i ∈ 0:ord
            selectdim(C, 1, i+2) .= selectdim(A, 1, i+1) ./ exact(i+1)
        end
    else
        ord = order(space)
        @inbounds selectdim(C, 1, 1:n) .= zero(T)
        @inbounds for i ∈ 0:ord
            selectdim(C, 1, i+n+1) .= _nzval(ℐ, space, space, T, i+n, i) .* selectdim(A, 1, i+1)
        end
    end
    return C
end

function _apply(ℐ::Integral, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(ℐ)
    CoefType = _coeftype(ℐ, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif n == 1
        ord = order(space)
        C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord+2, size(A, i)), Val(N)))
        @inbounds selectdim(C, D, 1) .= zero(CoefType)
        @inbounds for i ∈ 0:ord
            selectdim(C, D, i+2) .= selectdim(A, D, i+1) ./ exact(i+1)
        end
        return C
    else
        ord = order(space)
        C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord+n+1, size(A, i)), Val(N)))
        @inbounds selectdim(C, D, 1:n) .= zero(CoefType)
        @inbounds for i ∈ 0:ord
            selectdim(C, D, i+n+1) .= _nzval(ℐ, space, space, CoefType, i+n, i) .* selectdim(A, D, i+1)
        end
        return C
    end
end

_nzind_domain(ℐ::Integral, domain::Taylor, codomain::Taylor) =
    0:min(order(domain), order(codomain)-order(ℐ))

_nzind_codomain(ℐ::Integral, domain::Taylor, codomain::Taylor) =
    order(ℐ):min(order(domain)+order(ℐ), order(codomain))

_nzval(ℐ::Integral, s₁::Taylor, s₂::Taylor, ::Type{T}, i, j) where {T} =
    convert(T, inv(real(_nzval(Derivative(order(ℐ)), s₁, s₂, T, j, i))))

# Fourier

codomain(::Derivative, s::Fourier) = s

_coeftype(::Derivative, ::Fourier{T}, ::Type{S}) where {T,S} = complex(typeof(zero(T)*zero(S)))

function _apply!(c::Sequence{<:Fourier}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(real(eltype(a)))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        if n == 1
            @inbounds for j ∈ 1:order(c)
                ωj = ω * exact(j)
                aⱼ = a[j]
                a₋ⱼ = a[-j]
                c[j] = complex(-ωj * imag(aⱼ), ωj * real(aⱼ))
                c[-j] = complex(ωj * imag(a₋ⱼ), -ωj * real(a₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    sign_iⁿ_ωⁿjⁿ = exact(sign_iⁿ) * (ω * exact(j)) ^ exact(n)
                    aⱼ = a[j]
                    a₋ⱼ = a[-j]
                    c[j] = complex(-sign_iⁿ_ωⁿjⁿ * imag(aⱼ), sign_iⁿ_ωⁿjⁿ * real(aⱼ))
                    c[-j] = complex(sign_iⁿ_ωⁿjⁿ * imag(a₋ⱼ), -sign_iⁿ_ωⁿjⁿ * real(a₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
                    c[j] = iⁿωⁿjⁿ_real * a[j]
                    c[-j] = iⁿωⁿjⁿ_real * a[-j]
                end
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::Fourier, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, ord+1) .= zero(T)
        if n == 1
            @inbounds for j ∈ 1:ord
                ωj = ω * exact(j)
                Aⱼ = selectdim(A, 1, ord+1+j)
                A₋ⱼ = selectdim(A, 1, ord+1-j)
                selectdim(C, 1, ord+1+j) .= complex.((-ωj) .* imag.(Aⱼ), ωj .* real.(Aⱼ))
                selectdim(C, 1, ord+1-j) .= complex.(ωj .* imag.(A₋ⱼ), (-ωj) .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ωⁿjⁿ = exact(sign_iⁿ) * (ω * exact(j)) ^ exact(n)
                    Aⱼ = selectdim(A, 1, ord+1+j)
                    A₋ⱼ = selectdim(A, 1, ord+1-j)
                    selectdim(C, 1, ord+1+j) .= complex.((-sign_iⁿ_ωⁿjⁿ) .* imag.(Aⱼ), sign_iⁿ_ωⁿjⁿ .* real.(Aⱼ))
                    selectdim(C, 1, ord+1-j) .= complex.(sign_iⁿ_ωⁿjⁿ .* imag.(A₋ⱼ), (-sign_iⁿ_ωⁿjⁿ) .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
                    selectdim(C, 1, ord+1+j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, ord+1+j)
                    selectdim(C, 1, ord+1-j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, ord+1-j)
                end
            end
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        C = Array{CoefType,N}(undef, size(A))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
        if n == 1
            @inbounds for j ∈ 1:ord
                ωj = ω * exact(j)
                Aⱼ = selectdim(A, D, ord+1+j)
                A₋ⱼ = selectdim(A, D, ord+1-j)
                selectdim(C, D, ord+1+j) .= complex.((-ωj) .* imag.(Aⱼ), ωj .* real.(Aⱼ))
                selectdim(C, D, ord+1-j) .= complex.(ωj .* imag.(A₋ⱼ), (-ωj) .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ωⁿjⁿ = exact(sign_iⁿ) * (ω * exact(j)) ^ exact(n)
                    Aⱼ = selectdim(A, D, ord+1+j)
                    A₋ⱼ = selectdim(A, D, ord+1-j)
                    selectdim(C, D, ord+1+j) .= complex.((-sign_iⁿ_ωⁿjⁿ) .* imag.(Aⱼ), sign_iⁿ_ωⁿjⁿ .* real.(Aⱼ))
                    selectdim(C, D, ord+1-j) .= complex.(sign_iⁿ_ωⁿjⁿ .* imag.(A₋ⱼ), (-sign_iⁿ_ωⁿjⁿ) .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
                    selectdim(C, D, ord+1+j) .= iⁿωⁿjⁿ_real .* selectdim(A, D, ord+1+j)
                    selectdim(C, D, ord+1-j) .= iⁿωⁿjⁿ_real .* selectdim(A, D, ord+1-j)
                end
            end
        end
        return C
    end
end

function _nzind_domain(::Derivative, domain::Fourier, codomain::Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return -ord:ord
end

function _nzind_codomain(::Derivative, domain::Fourier, codomain::Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return -ord:ord
end

function _nzval(𝒟::Derivative, domain::Fourier, ::Fourier, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(real(T)) * frequency(domain) * exact(j)) ^ exact(n)
        r = n % 4
        if r == 0
            return convert(T, complex(ωⁿjⁿ, zero(ωⁿjⁿ)))
        elseif r == 1
            return convert(T, complex(zero(ωⁿjⁿ), ωⁿjⁿ))
        elseif r == 2
            return convert(T, complex(-ωⁿjⁿ, zero(ωⁿjⁿ)))
        else
            return convert(T, complex(zero(ωⁿjⁿ), -ωⁿjⁿ))
        end
    end
end

codomain(::Integral, s::Fourier) = s

_coeftype(::Integral, ::Fourier{T}, ::Type{S}) where {T,S} = complex(typeof(inv(one(real(S))*one(T))*zero(S)))

function _apply!(c::Sequence{<:Fourier}, ℐ::Integral, a)
    n = order(ℐ)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        @inbounds iszero(a[0]) || return throw(DomainError("Fourier coefficient of order zero must be zero"))
        ω = one(real(eltype(a)))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        if n == 1
            @inbounds for j ∈ 1:order(c)
                ω⁻¹j⁻¹ = inv(ω * exact(j))
                aⱼ = a[j]
                a₋ⱼ = a[-j]
                c[j] = complex(ω⁻¹j⁻¹ * imag(aⱼ), -ω⁻¹j⁻¹ * real(aⱼ))
                c[-j] = complex(-ω⁻¹j⁻¹ * imag(a₋ⱼ), ω⁻¹j⁻¹ * real(a₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    sign_iⁿ_ω⁻ⁿj⁻ⁿ = exact(sign_iⁿ) * inv(ω * exact(j)) ^ exact(n)
                    aⱼ = a[j]
                    a₋ⱼ = a[-j]
                    c[j] = complex(sign_iⁿ_ω⁻ⁿj⁻ⁿ * imag(aⱼ), -sign_iⁿ_ω⁻ⁿj⁻ⁿ * real(aⱼ))
                    c[-j] = complex(-sign_iⁿ_ω⁻ⁿj⁻ⁿ * imag(a₋ⱼ), sign_iⁿ_ω⁻ⁿj⁻ⁿ * real(a₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    iⁿω⁻ⁿj⁻ⁿ_real = exact(iⁿ_real) * inv(ω * exact(j)) ^ exact(n)
                    c[j] = iⁿω⁻ⁿj⁻ⁿ_real * a[j]
                    c[-j] = iⁿω⁻ⁿj⁻ⁿ_real * a[-j]
                end
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Fourier, A) where {T}
    n = order(ℐ)
    if n == 0
        C .= A
    else
        ord = order(space)
        @inbounds all(iszero, selectdim(A, 1, ord+1)) || return throw(DomainError("Fourier coefficients of order zero along dimension 1 must be zero"))
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, ord+1) .= zero(T)
        if n == 1
            @inbounds for j ∈ 1:ord
                ω⁻¹j⁻¹ = inv(ω * exact(j))
                Aⱼ = selectdim(A, 1, ord+1+j)
                A₋ⱼ = selectdim(A, 1, ord+1-j)
                selectdim(C, 1, ord+1+j) .= Complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
                selectdim(C, 1, ord+1-j) .= Complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ω⁻ⁿj⁻ⁿ = exact(sign_iⁿ) * inv(ω * exact(j)) ^ exact(n)
                    Aⱼ = selectdim(A, 1, ord+1+j)
                    A₋ⱼ = selectdim(A, 1, ord+1-j)
                    selectdim(C, 1, ord+1+j) .= Complex.(sign_iⁿ_ω⁻ⁿj⁻ⁿ .* imag.(Aⱼ), (-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* real.(Aⱼ))
                    selectdim(C, 1, ord+1-j) .= Complex.((-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* imag.(A₋ⱼ), sign_iⁿ_ω⁻ⁿj⁻ⁿ .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ_real = exact(iⁿ_real) * inv(ω * exact(j)) ^ exact(n)
                    selectdim(C, 1, ord+1+j) .= iⁿω⁻ⁿj⁻ⁿ_real .* selectdim(A, 1, ord+1+j)
                    selectdim(C, 1, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ_real .* selectdim(A, 1, ord+1-j)
                end
            end
        end
    end
    return C
end

function _apply(ℐ::Integral, space::Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(ℐ)
    CoefType = _coeftype(ℐ, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        ord = order(space)
        @inbounds all(iszero, selectdim(A, D, ord+1)) || return throw(DomainError("Fourier coefficient of order zero along dimension $D must be zero"))
        ω = one(real(T))*frequency(space)
        C = Array{CoefType,N}(undef, size(A))
        @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
        if n == 1
            @inbounds for j ∈ 1:ord
                ω⁻¹j⁻¹ = inv(ω * exact(j))
                Aⱼ = selectdim(A, D, ord+1+j)
                A₋ⱼ = selectdim(A, D, ord+1-j)
                selectdim(C, D, ord+1+j) .= Complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
                selectdim(C, D, ord+1-j) .= Complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ω⁻ⁿj⁻ⁿ = exact(sign_iⁿ) * inv(ω * exact(j)) ^ exact(n)
                    Aⱼ = selectdim(A, D, ord+1+j)
                    A₋ⱼ = selectdim(A, D, ord+1-j)
                    selectdim(C, D, ord+1+j) .= Complex.(sign_iⁿ_ω⁻ⁿj⁻ⁿ .* imag.(Aⱼ), (-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* real.(Aⱼ))
                    selectdim(C, D, ord+1-j) .= Complex.((-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* imag.(A₋ⱼ), sign_iⁿ_ω⁻ⁿj⁻ⁿ .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ_real = exact(iⁿ_real) * inv(ω * exact(j)) ^ exact(n)
                    selectdim(C, D, ord+1+j) .= iⁿω⁻ⁿj⁻ⁿ_real .* selectdim(A, D, ord+1+j)
                    selectdim(C, D, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ_real .* selectdim(A, D, ord+1-j)
                end
            end
        end
        return C
    end
end

function _nzind_domain(::Integral, domain::Fourier, codomain::Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return -ord:ord
end

function _nzind_codomain(::Integral, domain::Fourier, codomain::Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return -ord:ord
end

function _nzval(ℐ::Integral, domain::Fourier, ::Fourier, ::Type{T}, i, j) where {T}
    n = order(ℐ)
    if n == 0
        return one(T)
    else
        if j == 0
            return zero(T)
        else
            ω⁻ⁿj⁻ⁿ = inv(one(real(T)) * frequency(domain) * exact(j)) ^ exact(n)
            r = n % 4
            if r == 0
                return convert(T, complex(ω⁻ⁿj⁻ⁿ, zero(ω⁻ⁿj⁻ⁿ)))
            elseif r == 1
                return convert(T, complex(zero(ω⁻ⁿj⁻ⁿ), -ω⁻ⁿj⁻ⁿ))
            elseif r == 2
                return convert(T, complex(-ω⁻ⁿj⁻ⁿ, zero(ω⁻ⁿj⁻ⁿ)))
            else
                return convert(T, complex(zero(ω⁻ⁿj⁻ⁿ), ω⁻ⁿj⁻ⁿ))
            end
        end
    end
end

# Chebyshev

codomain(𝒟::Derivative, s::Chebyshev) = Chebyshev(max(0, order(s)-order(𝒟)))

_coeftype(::Derivative, ::Chebyshev, ::Type{T}) where {T} = T

function _apply!(c::Sequence{Chebyshev}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        CoefType = eltype(c)
        order_a = order(a)
        if order_a < n
            @inbounds c[0] = zero(CoefType)
        else
            @inbounds for i ∈ 0:order_a-1
                c[i] = zero(CoefType)
                @inbounds for j ∈ i+1:2:order_a
                    c[i] += exact(j) * a[j]
                end
                c[i] *= exact(2)
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::Chebyshev, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    elseif n == 1
        ord = order(space)
        if ord < n
            C .= zero(T)
        else
            @inbounds for i ∈ 0:ord-1
                Cᵢ = selectdim(C, 1, i+1)
                Cᵢ .= zero(T)
                @inbounds for j ∈ i+1:2:ord
                    Cᵢ .+= exact(2j) .* selectdim(A, 1, j+1)
                end
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

function _apply(𝒟::Derivative, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif n == 1
        ord = order(space)
        if ord < n
            return zeros(CoefType, ntuple(i -> i == D ? 1 : size(A, i), Val(N)))
        else
            C = zeros(CoefType, ntuple(i -> i == D ? ord : size(A, i), Val(N)))
            @inbounds for i ∈ 0:ord-1
                Cᵢ = selectdim(C, D, i+1)
                @inbounds for j ∈ i+1:2:ord
                    Cᵢ .+= exact(2j) .* selectdim(A, D, j+1)
                end
            end
            return C
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_domain(𝒟::Derivative, domain::Chebyshev, codomain::Chebyshev)
    if order(𝒟) == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif order(𝒟) == 1
        len = sum(j -> length((j-1)%2:2:min(j-1, order(codomain))), 1:order(domain); init = 0)
        v = Vector{Int}(undef, len)
        l = 0
        @inbounds for j ∈ 1:order(domain)
            lnext = l+length((j-1)%2:2:min(j-1, order(codomain)))
            view(v, 1+l:lnext) .= j
            l = lnext
        end
        return v
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_codomain(𝒟::Derivative, domain::Chebyshev, codomain::Chebyshev)
    if order(𝒟) == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif order(𝒟) == 1
        len = sum(j -> length((j-1)%2:2:min(j-1, order(codomain))), 1:order(domain); init = 0)
        v = Vector{Int}(undef, len)
        l = 0
        @inbounds for j ∈ 1:order(domain)
            r = (j-1)%2:2:min(j-1, order(codomain))
            lnext = l+length(r)
            view(v, 1+l:lnext) .= r
            l = lnext
        end
        return v
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzval(𝒟::Derivative, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    if n == 0
        return one(T)
    elseif n == 1
        return convert(T, exact(2j))
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

codomain(ℐ::Integral, s::Chebyshev) = Chebyshev(order(s)+order(ℐ))

_coeftype(::Integral, ::Chebyshev, ::Type{T}) where {T} = T

function _apply!(c::Sequence{Chebyshev}, ℐ::Integral, a)
    n = order(ℐ)
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        order_a = order(a)
        if order_a == 0
            @inbounds c[0] = a[0]
            @inbounds c[1] = a[0] / exact(2)
        elseif order_a == 1
            @inbounds c[0] = a[0] - a[1] / exact(2)
            @inbounds c[1] = a[0] / exact(2)
            @inbounds c[2] = a[1] / exact(4)
        else
            @inbounds c[0] = zero(eltype(c))
            @inbounds for i ∈ 2:2:order_a-1
                c[0] += a[i+1] / exact((i+1)^2-1) - a[i] / exact(i^2-1)
            end
            if iseven(order_a)
                @inbounds c[0] -= a[order_a] / exact(order_a^2-1)
            end
            @inbounds c[0] = exact(2) * c[0] + a[0] - a[1] / exact(2)
            @inbounds c[1] = (a[0] - a[2]) / exact(2)
            @inbounds for i ∈ 2:order_a-1
                c[i] = (a[i-1] - a[i+1]) / exact(2i)
            end
            @inbounds c[order_a] = a[order_a-1] / exact(2order_a)
            @inbounds c[order_a+1] = a[order_a] / exact(2(order_a+1))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Chebyshev, A) where {T}
    n = order(ℐ)
    if n == 0
        C .= A
    elseif n == 1
        ord = order(space)
        @inbounds C₀ = selectdim(C, 1, 1)
        @inbounds C₁ = selectdim(C, 1, 2)
        @inbounds A₀ = selectdim(A, 1, 1)
        if ord == 0
            C₀ .= A₀
            C₁ .= A₀ ./ exact(2)
        elseif ord == 1
            @inbounds A₁ = selectdim(A, 1, 2)
            C₀ .= A₀ .- A₁ ./ exact(2)
            C₁ .= A₀ ./ exact(2)
            @inbounds selectdim(C, 1, 3) .= A₁ ./ exact(4)
        else
            C₀ .= zero(T)
            @inbounds for i ∈ 2:2:ord-1
                C₀ .+= selectdim(A, 1, i+2) ./ exact((i+1)^2-1) .- selectdim(A, 1, i+1) ./ exact(i^2-1)
            end
            if iseven(ord)
                @inbounds C₀ .-= selectdim(A, 1, ord+1) ./ exact(ord^2-1)
            end
            @inbounds C₀ .= exact(2) .* C₀ .+ A₀ .- selectdim(A, 1, 2) ./ exact(2)
            @inbounds C₁ .= (A₀ .- selectdim(A, 1, 3)) ./ exact(2)
            @inbounds for i ∈ 2:ord-1
                selectdim(C, 1, i+1) .= (selectdim(A, 1, i) .- selectdim(A, 1, i+2)) ./ exact(2i)
            end
            @inbounds selectdim(C, 1, ord+1) .= selectdim(A, 1, ord) ./ exact(2ord)
            @inbounds selectdim(C, 1, ord+2) .= selectdim(A, 1, ord+1) ./ exact(2(ord+1))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

function _apply(ℐ::Integral, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(ℐ)
    CoefType = _coeftype(ℐ, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif n == 1
        ord = order(space)
        C = Array{CoefType,N}(undef, ntuple(i -> i == D ? ord+2 : size(A, i), Val(N)))
        @inbounds C₀ = selectdim(C, D, 1)
        @inbounds C₁ = selectdim(C, D, 2)
        @inbounds A₀ = selectdim(A, D, 1)
        if ord == 0
            C₀ .= A₀
            C₁ .= A ./ exact(2)
        elseif ord == 1
            @inbounds A₁ = selectdim(A, D, 2)
            C₀ .= A₀ .- A₁ ./ exact(2)
            C₁ .= A₀ ./ exact(2)
            @inbounds selectdim(C, D, 3) .= A₁ ./ exact(4)
        else
            C₀ .= zero(CoefType)
            @inbounds for i ∈ 2:2:ord-1
                C₀ .+= selectdim(A, D, i+2) ./ exact((i+1)^2-1) .- selectdim(A, D, i+1) ./ exact(i^2-1)
            end
            if iseven(ord)
                @inbounds C₀ .-= selectdim(A, D, ord+1) ./ exact(ord^2-1)
            end
            @inbounds C₀ .= exact(2) .* C₀ .+ A₀ .- selectdim(A, D, 2) ./ exact(2)
            @inbounds C₁ .= (A₀ .- selectdim(A, D, 3)) ./ exact(2)
            @inbounds for i ∈ 2:ord-1
                selectdim(C, D, i+1) .= (selectdim(A, D, i) .- selectdim(A, D, i+2)) ./ exact(2i)
            end
            @inbounds selectdim(C, D, ord+1) .= selectdim(A, D, ord) ./ exact(2ord)
            @inbounds selectdim(C, D, ord+2) .= selectdim(A, D, ord+1) ./ exact(2(ord+1))
        end
        return C
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_domain(ℐ::Integral, domain::Chebyshev, codomain::Chebyshev)
    if order(ℐ) == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif order(ℐ) == 1
        len = 0
        for j ∈ 0:order(domain)
            if j < 2
                len += 1 + (j+1 ≤ order(codomain))
            else
                len += (1 + (j+1 ≤ order(codomain))) + (j-1 ≤ order(codomain))
            end
        end
        v = Vector{Int}(undef, len)
        idx = 1
        for j ∈ 0:order(domain)
            if j < 2
                idx2 = idx + (j+1 ≤ order(codomain))
                view(v, idx:idx2) .= j
                idx = idx2 + 1
            else
                idx2 = (idx + (j+1 ≤ order(codomain))) + (j-1 ≤ order(codomain))
                view(v, idx:idx2) .= j
                idx = idx2 + 1
            end
        end
        return v
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_codomain(ℐ::Integral, domain::Chebyshev, codomain::Chebyshev)
    if order(ℐ) == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif order(ℐ) == 1
        len = 0
        for j ∈ 0:order(domain)
            if j < 2
                len += 1 + (j+1 ≤ order(codomain))
            else
                len += (1 + (j+1 ≤ order(codomain))) + (j-1 ≤ order(codomain))
            end
        end
        v = Vector{Int}(undef, len)
        idx = 1
        for j ∈ 0:order(domain)
            if j < 2
                if j+1 ≤ order(codomain)
                    v[idx] = 0
                    v[idx+1] = j+1
                    idx += 2
                else
                    v[idx] = 0
                    idx += 1
                end
            else
                if j+1 ≤ order(codomain)
                    v[idx] = 0
                    v[idx+1] = j-1
                    v[idx+2] = j+1
                    idx += 3
                elseif j-1 ≤ order(codomain)
                    v[idx] = 0
                    v[idx+1] = j-1
                    idx += 2
                else
                    v[idx] = 0
                    idx += 1
                end
            end
        end
        return v
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzval(ℐ::Integral, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j) where {T}
    n = order(ℐ)
    if n == 0
        return one(T)
    elseif n == 1
        if i == 0
            if j == 0
                return one(T)
            elseif j == 1
                return convert(T, -one(T) / exact(2))
            elseif iseven(j)
                return convert(T, exact(2) * one(T) / exact(1-j^2))
            else
                return convert(T, exact(2) * one(T) / exact(j^2-1))
            end
        elseif i == 1 && j == 0
            return convert(T, one(T) / exact(2))
        elseif i == 2 && j == 1
            return convert(T, one(T) / exact(4))
        else
            if i+1 == j
                return convert(T, -one(T) / exact(2i))
            else # i == j+1
                return convert(T, one(T) / exact(2i))
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# CosFourier

codomain(𝒟::Derivative, s::CosFourier) = iseven(order(𝒟)) ? s : SinFourier(desymmetrize(s))

_coeftype(::Derivative, ::CosFourier{T}, ::Type{S}) where {T,S} = typeof(zero(T)*zero(S))

function _apply!(c::Sequence{<:CosFourier}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(real(eltype(a)))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:order(c)
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            c[j] = iⁿωⁿjⁿ_real * a[j]
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::CosFourier, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    elseif iseven(n)
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, 1, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j+1)
        end
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, 1, j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j+1)
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::CosFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif iseven(n)
        C = Array{CoefType,N}(undef, size(A))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        @inbounds selectdim(C, D, 1) .= zero(CoefType)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, D, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j+1)
        end
        return C
    else
        C = Array{CoefType,N}(undef, ntuple(i -> size(A, i) - ifelse(i == D, 1, 0), Val(N)))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, D, j) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j+1)
        end
        return C
    end
end

function _nzind_domain(𝒟::Derivative, domain::CosFourier, codomain::CosFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return (order(𝒟) > 0):ord
end
function _nzind_domain(::Derivative, domain::CosFourier, codomain::SinFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzind_codomain(𝒟::Derivative, domain::CosFourier, codomain::CosFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return (order(𝒟) > 0):ord
end
function _nzind_codomain(::Derivative, domain::SinFourier, codomain::CosFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzval(𝒟::Derivative, domain::Union{CosFourier,SinFourier}, ::CosFourier, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(real(T)) * frequency(domain) * exact(j)) ^ exact(n)
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ)) # (n%4 == 0) | (n%4 == 1)
    end
end

codomain(ℐ::Integral, s::CosFourier) = iseven(order(ℐ)) ? s : SinFourier(desymmetrize(s))

# SinFourier

codomain(𝒟::Derivative, s::SinFourier) = iseven(order(𝒟)) ? s : CosFourier(desymmetrize(s))

_coeftype(::Derivative, ::SinFourier{T}, ::Type{S}) where {T,S} = typeof(zero(T)*zero(S))

function _apply!(c::Sequence{<:SinFourier}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(real(eltype(a)))*frequency(a)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:order(c)
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            c[j] = iⁿωⁿjⁿ_real * a[j]
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::SinFourier, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    elseif iseven(n)
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, 1, j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j)
        end
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, 1, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j)
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::SinFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif iseven(n)
        C = Array{CoefType,N}(undef, size(A))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, D, j) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j)
        end
        return C
    else
        C = Array{CoefType,N}(undef, ntuple(i -> size(A, i) + ifelse(i == D, 1, 0), Val(N)))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        @inbounds selectdim(C, D, 1) .= zero(CoefType)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = exact(iⁿ_real) * (ω * exact(j)) ^ exact(n)
            selectdim(C, D, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j)
        end
        return C
    end
end

function _nzind_domain(::Derivative, domain::SinFourier, codomain::Union{CosFourier,SinFourier})
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzind_codomain(::Derivative, domain::Union{CosFourier,SinFourier}, codomain::SinFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    _safe_isequal(ω₁, ω₂) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzval(𝒟::Derivative, domain::Union{CosFourier,SinFourier}, ::SinFourier, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(real(T)) * frequency(domain) * exact(j)) ^ exact(n)
        return convert(T, ifelse((n+1)%4 < 2, ωⁿjⁿ, -ωⁿjⁿ)) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
    end
end

codomain(ℐ::Integral, s::SinFourier) = iseven(order(ℐ)) ? s : CosFourier(desymmetrize(s))

#

"""
    Laplacian <: AbstractLinearOperator

Laplacian operator.
"""
struct Laplacian <: AbstractLinearOperator end

function _infer_domain(Δ::Laplacian, s::TensorSpace)
    s_out = map(sᵢ -> _infer_domain(Δ, sᵢ), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return TensorSpace(map((s1, s2) -> codomain(+, s1, s2), spaces(s), s_out))
end
_infer_domain(::Laplacian, s::BaseSpace) = _infer_domain(Derivative(2), s)
function _infer_domain(Δ::Laplacian, s::CartesianPower)
    s_out = _infer_domain(Δ, space(s))
    s_out isa EmptySpace && return EmptySpace()
    return CartesianPower(s_out, nspaces(s))
end
function _infer_domain(Δ::Laplacian, s::CartesianSpace)
    s_out = map(sᵢ -> _infer_domain(Δ, sᵢ), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return CartesianProduct(s_out)
end

"""
    *(Δ::Laplacian, a::AbstractSequence)

Compute the laplacian of `a`; equivalent to `laplacian(a)`.

See also: [`(::Laplacian)(::AbstractSequence)`](@ref), [`Laplacian`](@ref),
[`laplacian`](@ref) and [`laplacian!`](@ref).
"""
Base.:*(::Laplacian, a::AbstractSequence) = laplacian(a)

"""
    (Δ::Laplacian)(a::AbstractSequence)

Compute the laplacian `a`; equivalent to `laplacian(a)`.

See also: [`*(::Laplacian, ::AbstractSequence)`](@ref), [`Laplacian`](@ref),
[`laplacian`](@ref) and [`laplacian!`](@ref).
"""
(Δ::Laplacian)(a::AbstractSequence) = *(Δ, a)

"""
    laplacian(a::Sequence)

Compute the laplacian of `a`.

See also: [`laplacian!`](@ref), [`Laplacian`](@ref),
[`*(::Laplacian, ::Sequence)`](@ref) and [`(::Laplacian)(::Sequence)`](@ref).
"""
function laplacian(a::Sequence)
    Δ = Laplacian()
    space_a = space(a)
    new_space = codomain(Δ, space_a)
    CoefType = _coeftype(Δ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, Δ, a)
    return c
end

"""
    laplacian!(c::Sequence, a::Sequence)

Compute the laplacian `a`. The result is stored in `c` by overwriting it.

See also: [`laplacian`](@ref), [`Laplacian`](@ref),
[`*(::Laplacian, ::Sequence)`](@ref) and [`(::Laplacian)(::Sequence)`](@ref).
"""
function laplacian!(c::Sequence, a::Sequence)
    Δ = Laplacian()
    space_c = space(c)
    new_space = codomain(Δ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, Laplacian(a) has space $new_space"))
    _apply!(c, Δ, a)
    return c
end

"""
    project!(C::LinearOperator, Δ::Laplacian)

Represent `Δ` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project(::Laplacian, ::VectorSpace, ::VectorSpace)`](@ref) and
[`Laplacian`](@ref)
"""
function project!(C::LinearOperator, Δ::Laplacian)
    image_domain = codomain(Δ, domain(C))
    codomain_C = codomain(C)
    _iscompatible(image_domain, codomain_C) || return throw(ArgumentError("spaces must be compatible: codomain of domain(C) under $Δ is $image_domain, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, Δ)
    return C
end

#

codomain(::Laplacian, s::TensorSpace) = s
codomain(::Laplacian, s::TensorSpace{<:Tuple{BaseSpace}}) = TensorSpace(codomain(Derivative(2), first(s.spaces)))

codomain(::Laplacian, s::BaseSpace) = codomain(Derivative(2), s)


_coeftype(::Laplacian, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    _coeftype(Derivative(ntuple(i -> 2, Val(N))), s, T)

_coeftype(::Laplacian, s::BaseSpace, ::Type{T}) where {T} = _coeftype(Derivative(2), s, T)


function _apply!(c::Sequence{<:TensorSpace{<:NTuple{N,BaseSpace}}}, ::Laplacian, a) where {N}
    _apply!(c, Derivative(ntuple(j -> ifelse(j==1, 2, 0), Val(N))), a)
    c_ = similar(c)
    for i ∈ 2:N
        radd!(c, _apply!(c_, Derivative(ntuple(j -> ifelse(j==i, 2, 0), Val(N))), a))
    end
    return c
end

_apply!(c::Sequence{<:BaseSpace}, ::Laplacian, a) = _apply!(c, Derivative(2), a)


function _project!(C::LinearOperator{<:BaseSpace,<:BaseSpace}, ::Laplacian)
    _project!(C, Derivative(2))
    return C
end

function _project!(C::LinearOperator{<:TensorSpace{<:NTuple{N,BaseSpace}},<:TensorSpace{<:NTuple{N,BaseSpace}}}, ::Laplacian) where {N}
    CoefType = eltype(C)
    C_ = zero(C)
    for i ∈ 1:N
        radd!(C, _project!(C_, Derivative(ntuple(j -> ifelse(j==i, 2, 0), Val(N)))))
        coefficients(C_) .= zero(CoefType)
    end
    return C
end


_nzind_domain(::Laplacian, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    _nzind_domain(Derivative(ntuple(i -> 2, Val(N))), domain, codomain)

_nzind_codomain(::Laplacian, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    _nzind_codomain(Derivative(ntuple(i -> 2, Val(N))), domain, codomain)

_nzind_domain(::Laplacian, domain::BaseSpace, codomain::BaseSpace) =
    _nzind_domain(Derivative(2), domain, codomain)

_nzind_codomain(::Laplacian, domain::BaseSpace, codomain::BaseSpace) =
    _nzind_codomain(Derivative(2), domain, codomain)



# Cartesian spaces

for F ∈ (:Derivative, :Integral, :Laplacian)
    @eval begin
        codomain(ℱ::$F, s::CartesianPower) =
            CartesianPower(codomain(ℱ, space(s)), nspaces(s))

        codomain(ℱ::$F, s::CartesianProduct) =
            CartesianProduct(map(sᵢ -> codomain(ℱ, sᵢ), spaces(s)))

        _coeftype(ℱ::$F, s::CartesianPower, ::Type{T}) where {T} =
            _coeftype(ℱ, space(s), T)

        _coeftype(ℱ::$F, s::CartesianProduct, ::Type{T}) where {T} =
            @inbounds promote_type(_coeftype(ℱ, s[1], T), _coeftype(ℱ, Base.tail(s), T))
        _coeftype(ℱ::$F, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
            @inbounds _coeftype(ℱ, s[1], T)

        function _apply!(c::Sequence{<:CartesianPower}, ℱ::$F, a)
            @inbounds for i ∈ 1:nspaces(space(c))
                _apply!(component(c, i), ℱ, component(a, i))
            end
            return c
        end
        function _apply!(c::Sequence{CartesianProduct{T}}, ℱ::$F, a) where {N,T<:NTuple{N,VectorSpace}}
            @inbounds _apply!(component(c, 1), ℱ, component(a, 1))
            @inbounds _apply!(component(c, 2:N), ℱ, component(a, 2:N))
            return c
        end
        function _apply!(c::Sequence{CartesianProduct{T}}, ℱ::$F, a) where {T<:Tuple{VectorSpace}}
            @inbounds _apply!(component(c, 1), ℱ, component(a, 1))
            return c
        end

        function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ℱ::$F)
            @inbounds for i ∈ 1:nspaces(domain(C))
                _project!(component(C, i, i), ℱ)
            end
            return C
        end
    end
end
