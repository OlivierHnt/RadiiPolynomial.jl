"""
    Derivative{T<:Union{Int,Tuple{Vararg{Int}}}} <: SpecialOperator

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
struct Derivative{T<:Union{Int,Tuple{Vararg{Int}}}} <: SpecialOperator
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

"""
    Integral{T<:Union{Int,Tuple{Vararg{Int}}}} <: SpecialOperator

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
struct Integral{T<:Union{Int,Tuple{Vararg{Int}}}} <: SpecialOperator
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

"""
    *(𝒟::Derivative, a::AbstractSequence)

Compute the `order(𝒟)`-th derivative of `a`; equivalent to `differentiate(a, order(𝒟))`.

See also: [`(::Derivative)(::AbstractSequence)`](@ref), [`Derivative`](@ref),
[`differentiate`](@ref) and [`differentiate!`](@ref).
"""
Base.:*(𝒟::Derivative, a::AbstractSequence) = differentiate(a, order(𝒟))

"""
    (𝒟::Derivative)(a::AbstractSequence)

Compute the `order(𝒟)`-th derivative of `a`; equivalent to `differentiate(a, order(𝒟))`.

See also: [`*(::Derivative, ::AbstractSequence)`](@ref), [`Derivative`](@ref),
[`differentiate`](@ref) and [`differentiate!`](@ref).
"""
(𝒟::Derivative)(a::AbstractSequence) = *(𝒟, a)

"""
    differentiate(a::Sequence, α=1)

Compute the `α`-th derivative of `a`.

See also: [`differentiate!`](@ref), [`Derivative`](@ref),
[`*(::Derivative, ::Sequence)`](@ref) and [`(::Derivative)(::Sequence)`](@ref).
"""
function differentiate(a::Sequence, α=1)
    𝒟 = Derivative(α)
    space_a = space(a)
    new_space = image(𝒟, space_a)
    CoefType = _coeftype(𝒟, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒟, a)
    return c
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
    new_space = image(𝒟, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒟(a) has space $new_space"))
    _apply!(c, 𝒟, a)
    return c
end

"""
    project(𝒟::Derivative, domain::VectorSpace, codomain::VectorSpace, ::Type{T}=_coeftype(𝒟, domain, Float64))

Represent `𝒟` as a [`LinearOperator`](@ref) from `domain` to `codomain`.

See also: [`project!(::LinearOperator, ::Derivative)`](@ref) and [`Derivative`](@ref).
"""
function project(𝒟::Derivative, domain::VectorSpace, codomain::VectorSpace, ::Type{T}=_coeftype(𝒟, domain, Float64)) where {T}
    image_domain = image(𝒟, domain)
    _iscompatible(image_domain, codomain) || return throw(ArgumentError("spaces must be compatible: image of domain under $𝒟 is $image_domain, codomain is $codomain"))
    ind_domain = _findposition_nzind_domain(𝒟, domain, codomain)
    ind_codomain = _findposition_nzind_codomain(𝒟, domain, codomain)
    C = LinearOperator(domain, codomain, SparseArrays.sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), dimension(codomain), dimension(domain)))
    _project!(C, 𝒟)
    return C
end

"""
    project!(C::LinearOperator, 𝒟::Derivative)

Represent `𝒟` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project(::Derivative, ::VectorSpace, ::VectorSpace)`](@ref) and
[`Derivative`](@ref).
"""
function project!(C::LinearOperator, 𝒟::Derivative)
    image_domain = image(𝒟, domain(C))
    codomain_C = codomain(C)
    _iscompatible(image_domain, codomain_C) || return throw(ArgumentError("spaces must be compatible: image of domain(C) under $𝒟 is $image_domain, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, 𝒟)
    return C
end

"""
    *(ℐ::Integral, a::AbstractSequence)

Compute the `order(ℐ)`-th integral of `a`; equivalent to `integrate(a, order(ℐ))`.

See also: [`(::Integral)(::AbstractSequence)`](@ref), [`Integral`](@ref),
[`integrate`](@ref) and [`integrate!`](@ref).
"""
Base.:*(ℐ::Integral, a::AbstractSequence) = integrate(a, order(ℐ))

"""
    (ℐ::Integral)(a::AbstractSequence)

Compute the `order(ℐ)`-th integral of `a`; equivalent to `integrate(a, order(ℐ))`.

See also: [`*(::Integral, ::AbstractSequence)`](@ref), [`Integral`](@ref),
[`integrate`](@ref) and [`integrate!`](@ref).
"""
(ℐ::Integral)(a::AbstractSequence) = *(ℐ, a)

"""
    integrate(a::Sequence, α=1)

Compute the `α`-th integral of `a`.

See also: [`integrate!`](@ref), [`Integral`](@ref),
[`*(::Integral, ::Sequence)`](@ref) and [`(::Integral)(::Sequence)`](@ref).
"""
function integrate(a::Sequence, α=1)
    ℐ = Integral(α)
    space_a = space(a)
    new_space = image(ℐ, space_a)
    CoefType = _coeftype(ℐ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, ℐ, a)
    return c
end

"""
    integrate!(c::Sequence, a::Sequence, α=1)

Compute the `α`-th integral of `a`. The result is stored in `c` by overwriting it.

See also: [`integrate`](@ref), [`Integral`](@ref),
[`*(::Integral, ::Sequence)`](@ref) and [`(::Integral)(::Sequence)`](@ref).
"""
function integrate!(c::Sequence, a::Sequence, α=1)
    ℐ = Integral(α)
    space_c = space(c)
    new_space = image(ℐ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℐ(a) has space $new_space"))
    _apply!(c, ℐ, a)
    return c
end

"""
    project(ℐ::Integral, domain::VectorSpace, codomain::VectorSpace, ::Type{T}=_coeftype(ℐ, domain, Float64))

Represent `ℐ` as a [`LinearOperator`](@ref) from `domain` to `codomain`.

See also: [`project!(::LinearOperator, ::Integral)`](@ref) and [`Integral`](@ref).
"""
function project(ℐ::Integral, domain::VectorSpace, codomain::VectorSpace, ::Type{T}=_coeftype(ℐ, domain, Float64)) where {T}
    image_domain = image(ℐ, domain)
    _iscompatible(image_domain, codomain) || return throw(ArgumentError("spaces must be compatible: image of domain under $ℐ is $image_domain, codomain is $codomain"))
    ind_domain = _findposition_nzind_domain(ℐ, domain, codomain)
    ind_codomain = _findposition_nzind_codomain(ℐ, domain, codomain)
    C = LinearOperator(domain, codomain, SparseArrays.sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), dimension(codomain), dimension(domain)))
    _project!(C, ℐ)
    return C
end

"""
    project!(C::LinearOperator, ℐ::Integral)

Represent `ℐ` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project(::Integral, ::VectorSpace, ::VectorSpace)`](@ref) and
[`Integral`](@ref)
"""
function project!(C::LinearOperator, ℐ::Integral)
    image_domain = image(ℐ, domain(C))
    codomain_C = codomain(C)
    _iscompatible(image_domain, codomain_C) || return throw(ArgumentError("spaces must be compatible: image of domain(C) under $ℐ is $image_domain, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, ℐ)
    return C
end

for (F, f) ∈ ((:Derivative, :differentiate), (:Integral, :integrate))
    @eval begin
        Base.:*(ℱ₁::$F{Int}, ℱ₂::$F{Int}) = $F(order(ℱ₁) + order(ℱ₂))
        Base.:*(ℱ₁::$F{NTuple{N,Int}}, ℱ₂::$F{NTuple{N,Int}}) where {N} = $F(map(+, order(ℱ₁), order(ℱ₂)))

        Base.:^(ℱ::$F{Int}, n::Integer) = $F(order(ℱ) * n)
        Base.:^(ℱ::$F{<:Tuple{Vararg{Int}}}, n::Integer) = $F(map(αᵢ -> *(αᵢ, n), order(ℱ)))
        Base.:^(ℱ::$F{NTuple{N,Int}}, n::NTuple{N,Integer}) where {N} = $F(map(*, order(ℱ), n))

        _findposition_nzind_domain(ℱ::$F, domain, codomain) =
            _findposition(_nzind_domain(ℱ, domain, codomain), domain)

        _findposition_nzind_codomain(ℱ::$F, domain, codomain) =
            _findposition(_nzind_codomain(ℱ, domain, codomain), codomain)
    end
end

# Sequence spaces

for F ∈ (:Derivative, :Integral)
    @eval begin
        image(ℱ::$F{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
            TensorSpace(map((αᵢ, sᵢ) -> image($F(αᵢ), sᵢ), order(ℱ), spaces(s)))

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

image(𝒟::Derivative, s::Taylor) = Taylor(max(0, order(s)-order(𝒟)))

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
                c[i-1] = ExactReal(i) * a[i]
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
                selectdim(C, 1, i) .= ExactReal(i) .* selectdim(A, 1, i+1)
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
                selectdim(C, D, i) .= ExactReal(i) .* selectdim(A, D, i+1)
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
        p = ExactReal(i+k) * p
    end
    return convert(T, p)
end

image(ℐ::Integral, s::Taylor) = Taylor(order(s)+order(ℐ))

_coeftype(::Integral, ::Taylor, ::Type{T}) where {T} = typeof(inv(one(T))*zero(T))

function _apply!(c::Sequence{Taylor}, ℐ::Integral, a)
    n = order(ℐ)
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        @inbounds c[0] = zero(eltype(c))
        @inbounds for i ∈ 0:order(a)
            c[i+1] = a[i] / ExactReal(i+1)
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
            selectdim(C, 1, i+2) .= selectdim(A, 1, i+1) ./ ExactReal(i+1)
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
            selectdim(C, D, i+2) .= selectdim(A, D, i+1) ./ ExactReal(i+1)
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

image(::Derivative, s::Fourier) = s

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
                ωj = ω * ExactReal(j)
                aⱼ = a[j]
                a₋ⱼ = a[-j]
                c[j] = complex(-ωj * imag(aⱼ), ωj * real(aⱼ))
                c[-j] = complex(ωj * imag(a₋ⱼ), -ωj * real(a₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    sign_iⁿ_ωⁿjⁿ = ExactReal(sign_iⁿ) * (ω * ExactReal(j)) ^ ExactReal(n)
                    aⱼ = a[j]
                    a₋ⱼ = a[-j]
                    c[j] = complex(-sign_iⁿ_ωⁿjⁿ * imag(aⱼ), sign_iⁿ_ωⁿjⁿ * real(aⱼ))
                    c[-j] = complex(sign_iⁿ_ωⁿjⁿ * imag(a₋ⱼ), -sign_iⁿ_ωⁿjⁿ * real(a₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
                ωj = ω * ExactReal(j)
                Aⱼ = selectdim(A, 1, ord+1+j)
                A₋ⱼ = selectdim(A, 1, ord+1-j)
                selectdim(C, 1, ord+1+j) .= complex.((-ωj) .* imag.(Aⱼ), ωj .* real.(Aⱼ))
                selectdim(C, 1, ord+1-j) .= complex.(ωj .* imag.(A₋ⱼ), (-ωj) .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ωⁿjⁿ = ExactReal(sign_iⁿ) * (ω * ExactReal(j)) ^ ExactReal(n)
                    Aⱼ = selectdim(A, 1, ord+1+j)
                    A₋ⱼ = selectdim(A, 1, ord+1-j)
                    selectdim(C, 1, ord+1+j) .= complex.((-sign_iⁿ_ωⁿjⁿ) .* imag.(Aⱼ), sign_iⁿ_ωⁿjⁿ .* real.(Aⱼ))
                    selectdim(C, 1, ord+1-j) .= complex.(sign_iⁿ_ωⁿjⁿ .* imag.(A₋ⱼ), (-sign_iⁿ_ωⁿjⁿ) .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
                ωj = ω * ExactReal(j)
                Aⱼ = selectdim(A, D, ord+1+j)
                A₋ⱼ = selectdim(A, D, ord+1-j)
                selectdim(C, D, ord+1+j) .= complex.((-ωj) .* imag.(Aⱼ), ωj .* real.(Aⱼ))
                selectdim(C, D, ord+1-j) .= complex.(ωj .* imag.(A₋ⱼ), (-ωj) .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ωⁿjⁿ = ExactReal(sign_iⁿ) * (ω * ExactReal(j)) ^ ExactReal(n)
                    Aⱼ = selectdim(A, D, ord+1+j)
                    A₋ⱼ = selectdim(A, D, ord+1-j)
                    selectdim(C, D, ord+1+j) .= complex.((-sign_iⁿ_ωⁿjⁿ) .* imag.(Aⱼ), sign_iⁿ_ωⁿjⁿ .* real.(Aⱼ))
                    selectdim(C, D, ord+1-j) .= complex.(sign_iⁿ_ωⁿjⁿ .* imag.(A₋ⱼ), (-sign_iⁿ_ωⁿjⁿ) .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
        ωⁿjⁿ = (one(real(T)) * frequency(domain) * ExactReal(j)) ^ ExactReal(n)
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

image(::Integral, s::Fourier) = s

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
                ω⁻¹j⁻¹ = inv(ω * ExactReal(j))
                aⱼ = a[j]
                a₋ⱼ = a[-j]
                c[j] = complex(ω⁻¹j⁻¹ * imag(aⱼ), -ω⁻¹j⁻¹ * real(aⱼ))
                c[-j] = complex(-ω⁻¹j⁻¹ * imag(a₋ⱼ), ω⁻¹j⁻¹ * real(a₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    sign_iⁿ_ω⁻ⁿj⁻ⁿ = ExactReal(sign_iⁿ) * inv(ω * ExactReal(j)) ^ ExactReal(n)
                    aⱼ = a[j]
                    a₋ⱼ = a[-j]
                    c[j] = complex(sign_iⁿ_ω⁻ⁿj⁻ⁿ * imag(aⱼ), -sign_iⁿ_ω⁻ⁿj⁻ⁿ * real(aⱼ))
                    c[-j] = complex(-sign_iⁿ_ω⁻ⁿj⁻ⁿ * imag(a₋ⱼ), sign_iⁿ_ω⁻ⁿj⁻ⁿ * real(a₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    iⁿω⁻ⁿj⁻ⁿ_real = ExactReal(iⁿ_real) * inv(ω * ExactReal(j)) ^ ExactReal(n)
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
                ω⁻¹j⁻¹ = inv(ω * ExactReal(j))
                Aⱼ = selectdim(A, 1, ord+1+j)
                A₋ⱼ = selectdim(A, 1, ord+1-j)
                selectdim(C, 1, ord+1+j) .= Complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
                selectdim(C, 1, ord+1-j) .= Complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ω⁻ⁿj⁻ⁿ = ExactReal(sign_iⁿ) * inv(ω * ExactReal(j)) ^ ExactReal(n)
                    Aⱼ = selectdim(A, 1, ord+1+j)
                    A₋ⱼ = selectdim(A, 1, ord+1-j)
                    selectdim(C, 1, ord+1+j) .= Complex.(sign_iⁿ_ω⁻ⁿj⁻ⁿ .* imag.(Aⱼ), (-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* real.(Aⱼ))
                    selectdim(C, 1, ord+1-j) .= Complex.((-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* imag.(A₋ⱼ), sign_iⁿ_ω⁻ⁿj⁻ⁿ .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ_real = ExactReal(iⁿ_real) * inv(ω * ExactReal(j)) ^ ExactReal(n)
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
                ω⁻¹j⁻¹ = inv(ω * ExactReal(j))
                Aⱼ = selectdim(A, D, ord+1+j)
                A₋ⱼ = selectdim(A, D, ord+1-j)
                selectdim(C, D, ord+1+j) .= Complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
                selectdim(C, D, ord+1-j) .= Complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
            end
        else
            if isodd(n)
                sign_iⁿ = ifelse(n%4 == 1, 1, -1)
                @inbounds for j ∈ 1:ord
                    sign_iⁿ_ω⁻ⁿj⁻ⁿ = ExactReal(sign_iⁿ) * inv(ω * ExactReal(j)) ^ ExactReal(n)
                    Aⱼ = selectdim(A, D, ord+1+j)
                    A₋ⱼ = selectdim(A, D, ord+1-j)
                    selectdim(C, D, ord+1+j) .= Complex.(sign_iⁿ_ω⁻ⁿj⁻ⁿ .* imag.(Aⱼ), (-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* real.(Aⱼ))
                    selectdim(C, D, ord+1-j) .= Complex.((-sign_iⁿ_ω⁻ⁿj⁻ⁿ) .* imag.(A₋ⱼ), sign_iⁿ_ω⁻ⁿj⁻ⁿ .* real.(A₋ⱼ))
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ_real = ExactReal(iⁿ_real) * inv(ω * ExactReal(j)) ^ ExactReal(n)
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
            ω⁻ⁿj⁻ⁿ = inv(one(real(T)) * frequency(domain) * ExactReal(j)) ^ ExactReal(n)
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

image(𝒟::Derivative, s::Chebyshev) = Chebyshev(max(0, order(s)-order(𝒟)))

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
                    c[i] += ExactReal(j) * a[j]
                end
                c[i] *= ExactReal(2)
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
                    Cᵢ .+= ExactReal(2j) .* selectdim(A, 1, j+1)
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
                    Cᵢ .+= ExactReal(2j) .* selectdim(A, D, j+1)
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
        return convert(T, ExactReal(2j))
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

image(ℐ::Integral, s::Chebyshev) = Chebyshev(order(s)+order(ℐ))

_coeftype(::Integral, ::Chebyshev, ::Type{T}) where {T} = T

function _apply!(c::Sequence{Chebyshev}, ℐ::Integral, a)
    n = order(ℐ)
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        order_a = order(a)
        if order_a == 0
            @inbounds c[0] = a[0]
            @inbounds c[1] = a[0] / ExactReal(2)
        elseif order_a == 1
            @inbounds c[0] = a[0] - a[1] / ExactReal(2)
            @inbounds c[1] = a[0] / ExactReal(2)
            @inbounds c[2] = a[1] / ExactReal(4)
        else
            @inbounds c[0] = zero(eltype(c))
            @inbounds for i ∈ 2:2:order_a-1
                c[0] += a[i+1] / ExactReal((i+1)^2-1) - a[i] / ExactReal(i^2-1)
            end
            if iseven(order_a)
                @inbounds c[0] -= a[order_a] / ExactReal(order_a^2-1)
            end
            @inbounds c[0] = ExactReal(2) * c[0] + a[0] - a[1] / ExactReal(2)
            @inbounds c[1] = (a[0] - a[2]) / ExactReal(2)
            @inbounds for i ∈ 2:order_a-1
                c[i] = (a[i-1] - a[i+1]) / ExactReal(2i)
            end
            @inbounds c[order_a] = a[order_a-1] / ExactReal(2order_a)
            @inbounds c[order_a+1] = a[order_a] / ExactReal(2(order_a+1))
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
            C₁ .= A₀ ./ ExactReal(2)
        elseif ord == 1
            @inbounds A₁ = selectdim(A, 1, 2)
            C₀ .= A₀ .- A₁ ./ ExactReal(2)
            C₁ .= A₀ ./ ExactReal(2)
            @inbounds selectdim(C, 1, 3) .= A₁ ./ ExactReal(4)
        else
            C₀ .= zero(T)
            @inbounds for i ∈ 2:2:ord-1
                C₀ .+= selectdim(A, 1, i+2) ./ ExactReal((i+1)^2-1) .- selectdim(A, 1, i+1) ./ ExactReal(i^2-1)
            end
            if iseven(ord)
                @inbounds C₀ .-= selectdim(A, 1, ord+1) ./ ExactReal(ord^2-1)
            end
            @inbounds C₀ .= ExactReal(2) .* C₀ .+ A₀ .- selectdim(A, 1, 2) ./ ExactReal(2)
            @inbounds C₁ .= (A₀ .- selectdim(A, 1, 3)) ./ ExactReal(2)
            @inbounds for i ∈ 2:ord-1
                selectdim(C, 1, i+1) .= (selectdim(A, 1, i) .- selectdim(A, 1, i+2)) ./ ExactReal(2i)
            end
            @inbounds selectdim(C, 1, ord+1) .= selectdim(A, 1, ord) ./ ExactReal(2ord)
            @inbounds selectdim(C, 1, ord+2) .= selectdim(A, 1, ord+1) ./ ExactReal(2(ord+1))
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
            C₁ .= A ./ ExactReal(2)
        elseif ord == 1
            @inbounds A₁ = selectdim(A, D, 2)
            C₀ .= A₀ .- A₁ ./ ExactReal(2)
            C₁ .= A₀ ./ ExactReal(2)
            @inbounds selectdim(C, D, 3) .= A₁ ./ ExactReal(4)
        else
            C₀ .= zero(CoefType)
            @inbounds for i ∈ 2:2:ord-1
                C₀ .+= selectdim(A, D, i+2) ./ ExactReal((i+1)^2-1) .- selectdim(A, D, i+1) ./ ExactReal(i^2-1)
            end
            if iseven(ord)
                @inbounds C₀ .-= selectdim(A, D, ord+1) ./ ExactReal(ord^2-1)
            end
            @inbounds C₀ .= ExactReal(2) .* C₀ .+ A₀ .- selectdim(A, D, 2) ./ ExactReal(2)
            @inbounds C₁ .= (A₀ .- selectdim(A, D, 3)) ./ ExactReal(2)
            @inbounds for i ∈ 2:ord-1
                selectdim(C, D, i+1) .= (selectdim(A, D, i) .- selectdim(A, D, i+2)) ./ ExactReal(2i)
            end
            @inbounds selectdim(C, D, ord+1) .= selectdim(A, D, ord) ./ ExactReal(2ord)
            @inbounds selectdim(C, D, ord+2) .= selectdim(A, D, ord+1) ./ ExactReal(2(ord+1))
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
                return convert(T, -one(T) / ExactReal(2))
            elseif iseven(j)
                return convert(T, ExactReal(2) * one(T) / ExactReal(1-j^2))
            else
                return convert(T, ExactReal(2) * one(T) / ExactReal(j^2-1))
            end
        elseif i == 1 && j == 0
            return convert(T, one(T) / ExactReal(2))
        elseif i == 2 && j == 1
            return convert(T, one(T) / ExactReal(4))
        else
            if i+1 == j
                return convert(T, -one(T) / ExactReal(2i))
            else # i == j+1
                return convert(T, one(T) / ExactReal(2i))
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end



#

"""
    Laplacian <: SpecialOperator

Laplacian operator.
"""
struct Laplacian <: SpecialOperator end

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
    new_space = image(Δ, space_a)
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
    new_space = image(Δ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, Laplacian(a) has space $new_space"))
    _apply!(c, Δ, a)
    return c
end

"""
    project(Δ::Laplacian, domain::VectorSpace, codomain::VectorSpace, ::Type{T}=_coeftype(Δ, domain, Float64))

Represent `Δ` as a [`LinearOperator`](@ref) from `domain` to `codomain`.

See also: [`project!(::LinearOperator, ::Laplacian)`](@ref) and [`Laplacian`](@ref).
"""
function project(Δ::Laplacian, domain::VectorSpace, codomain::VectorSpace, ::Type{T}=_coeftype(Δ, domain, Float64)) where {T}
    image_domain = image(Δ, domain)
    _iscompatible(image_domain, codomain) || return throw(ArgumentError("spaces must be compatible: image of domain under $Δ is $image_domain, codomain is $codomain"))
    ind_domain = _findposition_nzind_domain(Δ, domain, codomain)
    ind_codomain = _findposition_nzind_codomain(Δ, domain, codomain)
    C = LinearOperator(domain, codomain, SparseArrays.sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), dimension(codomain), dimension(domain)))
    _project!(C, Δ)
    return C
end

"""
    project!(C::LinearOperator, Δ::Laplacian)

Represent `Δ` as a [`LinearOperator`](@ref) from `domain(C)` to `codomain(C)`.
The result is stored in `C` by overwriting it.

See also: [`project(::Laplacian, ::VectorSpace, ::VectorSpace)`](@ref) and
[`Laplacian`](@ref)
"""
function project!(C::LinearOperator, Δ::Laplacian)
    image_domain = image(Δ, domain(C))
    codomain_C = codomain(C)
    _iscompatible(image_domain, codomain_C) || return throw(ArgumentError("spaces must be compatible: image of domain(C) under $Δ is $image_domain, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, Δ)
    return C
end

#

image(::Laplacian, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    image(Derivative(ntuple(i -> 2, Val(N))), s)

image(::Laplacian, s::BaseSpace) = image(Derivative(2), s)


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


_findposition_nzind_domain(Δ::Laplacian, domain, codomain) =
    _findposition(_nzind_domain(Δ, domain, codomain), domain)

_findposition_nzind_codomain(Δ::Laplacian, domain, codomain) =
    _findposition(_nzind_codomain(Δ, domain, codomain), codomain)


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
        image(ℱ::$F, s::CartesianPower) =
            CartesianPower(image(ℱ, space(s)), nspaces(s))

        image(ℱ::$F, s::CartesianProduct) =
            CartesianProduct(map(sᵢ -> image(ℱ, sᵢ), spaces(s)))

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

        function _findposition_nzind_domain(ℱ::$F, domain::CartesianSpace, codomain::CartesianSpace)
            u = map((dom, codom) -> _findposition_nzind_domain(ℱ, dom, codom), spaces(domain), spaces(codomain))
            len = sum(length, u)
            v = Vector{Int}(undef, len)
            δ = δδ = 0
            @inbounds for (i, uᵢ) in enumerate(u)
                δ_ = δ
                δ += length(uᵢ)
                view(v, 1+δ_:δ) .= δδ .+ uᵢ
                δδ += dimension(domain[i])
            end
            return v
        end

        function _findposition_nzind_codomain(ℱ::$F, domain::CartesianSpace, codomain::CartesianSpace)
            u = map((dom, codom) -> _findposition_nzind_codomain(ℱ, dom, codom), spaces(domain), spaces(codomain))
            len = sum(length, u)
            v = Vector{Int}(undef, len)
            δ = δδ = 0
            @inbounds for (i, uᵢ) in enumerate(u)
                δ_ = δ
                δ += length(uᵢ)
                view(v, 1+δ_:δ) .= δδ .+ uᵢ
                δδ += dimension(codomain[i])
            end
            return v
        end

        function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ℱ::$F)
            @inbounds for i ∈ 1:nspaces(domain(C))
                _project!(component(C, i, i), ℱ)
            end
            return C
        end
    end
end
