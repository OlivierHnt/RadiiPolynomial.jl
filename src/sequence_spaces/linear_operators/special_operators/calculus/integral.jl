"""
    Integral{T<:Union{Int,Tuple{Vararg{Int}}}} <: AbstractLinearOperator

Generic integral operator.

Field:
- `order :: T`

Constructors:
- `Integral(::Int)`
- `Integral(::Tuple{Vararg{Int}})`
- `Integral(order::Int...)`: equivalent to `Integral(order)`

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

Base.:*(ℐ₁::Integral{Int}, ℐ₂::Integral{Int}) = Integral(order(ℐ₁) + order(ℐ₂))
Base.:*(ℐ₁::Integral{NTuple{N,Int}}, ℐ₂::Integral{NTuple{N,Int}}) where {N} = Integral(map(+, order(ℐ₁), order(ℐ₂)))

Base.:^(ℐ::Integral{Int}, n::Integer) = Integral(order(ℐ) * n)
Base.:^(ℐ::Integral{<:Tuple{Vararg{Int}}}, n::Integer) = Integral(map(αᵢ -> *(αᵢ, n), order(ℐ)))
Base.:^(ℐ::Integral{NTuple{N,Int}}, n::NTuple{N,Integer}) where {N} = Integral(map(*, order(ℐ), n))

# Tensor space

function domain(I::Integral{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    s_out = map((αᵢ, sᵢ) -> domain(Integral(αᵢ), sᵢ), order(I), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return TensorSpace(s_out)
end

codomain(ℱ::Integral{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((αᵢ, sᵢ) -> codomain(Integral(αᵢ), sᵢ), order(ℱ), spaces(s)))

_coeftype(ℱ::Integral{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Integral(order(ℱ)[1]), s[1], T), _coeftype(Integral(Base.tail(order(ℱ))), Base.tail(s), T))
_coeftype(ℱ::Integral{Tuple{Int}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Integral(order(ℱ)[1]), s[1], T)

getcoefficient(ℱ::Integral{NTuple{N,Int}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Integral(order(ℱ)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Integral(Base.tail(order(ℱ))), (Base.tail(codom), T, Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(ℱ::Integral{Tuple{Int}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Integral(order(ℱ)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

# Taylor

domain(I::Integral, s::Taylor) = codomain(Derivative(order(I)), s)

codomain(ℐ::Integral, s::Taylor) = Taylor(order(s)+order(ℐ))

_coeftype(::Integral, ::Taylor, ::Type{T}) where {T} = typeof(inv(one(T))*zero(T))

function getcoefficient(ℐ::Integral, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    n = order(ℐ)
    i != j+n && return zero(T)
    p = one(real(T))
    for k ∈ 1:n
        p = exact(j+k) * p
    end
    return convert(T, inv(p))
end

# Fourier

domain(I::Integral, s::Fourier) = iszero(order(I)) ? s : EmptySpace() # flags an error

codomain(::Integral, s::Fourier) = s

_coeftype(::Integral, ::Fourier{T}, ::Type{S}) where {T,S} = complex(typeof(inv(one(real(S))*one(T))*zero(S)))

function getcoefficient(ℐ::Integral, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T}
    n = order(ℐ)
    if n == 0
        return ifelse(i == j, one(T), zero(T))
    elseif n == 1
        if i == 0
            j == 0 && return zero(T)
            return convert(T, -inv(im * one(real(T)) * frequency(dom) * exact(j)))
        else
            i != j && return zero(T)
            return convert(T, inv(im * one(real(T)) * frequency(dom) * exact(j)))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Chebyshev

domain(I::Integral, s::Chebyshev) = iszero(order(I)) ? s : EmptySpace() # flags an error

codomain(ℐ::Integral, s::Chebyshev) = Chebyshev(order(s)+order(ℐ))

_coeftype(::Integral, ::Chebyshev, ::Type{T}) where {T} = T

function getcoefficient(ℐ::Integral, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    n = order(ℐ)
    if n == 0
        return ifelse(i == j, one(T), zero(T))
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
            elseif i == j+1
                return convert(T, one(T) / exact(2i))
            else
                return zero(T)
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end



# action

Base.:*(ℐ::Integral, a::AbstractSequence) = integrate(a, order(ℐ))

function integrate(a::Sequence, α::Union{Int,Tuple{Vararg{Int}}}=1)
    ℐ = Integral(α)
    space_a = space(a)
    new_space = codomain(ℐ, space_a)
    CoefType = _coeftype(ℐ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, ℐ, a)
    return c
end

function integrate!(c::Sequence, a::Sequence, α::Union{Int,Tuple{Vararg{Int}}}=1)
    ℐ = Integral(α)
    space_c = space(c)
    new_space = codomain(ℐ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℐ(a) has space $new_space"))
    _apply!(c, ℐ, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, ℱ::Integral, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, ℱ, space_a, A)
    return c
end

_apply!(C, ℱ::Integral, space::TensorSpace, A) =
    @inbounds _apply!(C, Integral(order(ℱ)[1]), space[1], _apply(Integral(Base.tail(order(ℱ))), Base.tail(space), A))
_apply!(C, ℱ::Integral, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply!(C, Integral(order(ℱ)[1]), space[1], A)

_apply(ℱ::Integral, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply(Integral(order(ℱ)[1]), space[1], Val(N₂-N₁+1), _apply(Integral(Base.tail(order(ℱ))), Base.tail(space), A))
_apply(ℱ::Integral, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _apply(Integral(order(ℱ)[1]), space[1], Val(N), A)

# Taylor

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
            c[i+n] = getcoefficient(ℐ, (space_a, i+n), (space_a, i), CoefType) * a[i]
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
            selectdim(C, 1, i+n+1) .= getcoefficient(ℐ, (space, i+n), (space, i), T) * selectdim(A, 1, i+1)
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
            selectdim(C, D, i+n+1) .= getcoefficient(ℐ, (space, i+n), (space, i), CoefType) * selectdim(A, D, i+1)
        end
        return C
    end
end

# Fourier

function _apply!(c::Sequence{<:Fourier}, ℐ::Integral, a)
    n = order(ℐ)
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        @inbounds iszero(a[0]) || return throw(DomainError("Fourier coefficient of order zero must be zero"))
        ω = one(real(eltype(a)))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        @inbounds for j ∈ 1:order(c)
            ω⁻¹j⁻¹ = inv(ω * exact(j))
            aⱼ = a[j]
            a₋ⱼ = a[-j]
            c[0] += ω⁻¹j⁻¹ * im * (aⱼ - a₋ⱼ)
            c[j] = complex(ω⁻¹j⁻¹ * imag(aⱼ), -ω⁻¹j⁻¹ * real(aⱼ))
            c[-j] = complex(-ω⁻¹j⁻¹ * imag(a₋ⱼ), ω⁻¹j⁻¹ * real(a₋ⱼ))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Fourier, A) where {T}
    n = order(ℐ)
    if n == 0
        C .= A
    elseif n == 1
        ord = order(space)
        @inbounds all(iszero, selectdim(A, 1, ord+1)) || return throw(DomainError("Fourier coefficients of order zero along dimension 1 must be zero"))
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, ord+1) .= zero(T)
        @inbounds for j ∈ 1:ord
            ω⁻¹j⁻¹ = inv(ω * exact(j))
            Aⱼ = selectdim(A, 1, ord+1+j)
            A₋ⱼ = selectdim(A, 1, ord+1-j)
            selectdim(C, 1, ord+1) .+= (ω⁻¹j⁻¹ * im) .* (Aⱼ .- A₋ⱼ)
            selectdim(C, 1, ord+1+j) .= complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
            selectdim(C, 1, ord+1-j) .= complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

function _apply(ℐ::Integral, space::Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(ℐ)
    CoefType = _coeftype(ℐ, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif n == 1
        ord = order(space)
        @inbounds all(iszero, selectdim(A, D, ord+1)) || return throw(DomainError("Fourier coefficient of order zero along dimension $D must be zero"))
        ω = one(real(T))*frequency(space)
        C = Array{CoefType,N}(undef, size(A))
        @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
        @inbounds for j ∈ 1:ord
            ω⁻¹j⁻¹ = inv(ω * exact(j))
            Aⱼ = selectdim(A, D, ord+1+j)
            A₋ⱼ = selectdim(A, D, ord+1-j)
            selectdim(C, D, ord+1) .+= (ω⁻¹j⁻¹ * im) .* (Aⱼ .- A₋ⱼ)
            selectdim(C, D, ord+1+j) .= complex.(ω⁻¹j⁻¹ .* imag.(Aⱼ), (-ω⁻¹j⁻¹) .* real.(Aⱼ))
            selectdim(C, D, ord+1-j) .= complex.((-ω⁻¹j⁻¹) .* imag.(A₋ⱼ), ω⁻¹j⁻¹ .* real.(A₋ⱼ))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

# Chebyshev

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





#

function integrate(a::InfiniteSequence, α::Union{Int,Tuple{Vararg{Int}}}=1)
    c = integrate(sequence(a), α)
    X = banachspace(a)
    factor = _integral_error(X, space(a), α)
    new_err = factor * sequence_error(a)
    return InfiniteSequence(c, new_err, X)
end

_integral_error(X::Ell1{<:NTuple{N,Weight}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    mapreduce((wᵢ, sᵢ, αᵢ) -> _integral_error(Ell1(wᵢ), sᵢ, αᵢ), *, weight(X), spaces(s), α)

function _integral_error(X::Ell1{<:GeometricWeight}, s::Taylor, α::Int)
    α == 0 && return one(rate(X))
    ν = rate(X)
    # Tail-only bound in ℓ¹(ν) with N = order(s):
    #   ‖ℐ^α ε‖_{ℓ¹(ν)} ≤ ν^α · (N+1)!/(N+α+1)! · sequence_error
    factor = ν ^ exact(α)
    N = order(s)
    for j ∈ 1:α
        factor = factor / exact(N + j + 1)
    end
    return factor
end

function _integral_error(X::Ell1{<:GeometricWeight}, s::Fourier, α::Int)
    α == 0 && return one(rate(X))
    return throw(DomainError(Fourier, "integral error on Fourier InfiniteSequence is not implemented"))
    # α == 1 || return throw(DomainError(α, "integral error on Fourier is only implemented for α ≤ 1"))
    # ν = rate(X)
    # ω = abs(frequency(s)) * one(ν)
    # N = order(s)
    # # Tail-only bound in ℓ¹(ν):
    # #   ‖ℐ ε‖_{ℓ¹(ν)} ≤ (1 + ν^{-(N+1)}) / (|ω|(N+1)) · sequence_error
    # return (one(ν) + one(ν) / ν ^ exact(N+1)) / (ω * exact(N+1))
end

_integral_error(::Ell1{<:GeometricWeight}, ::Chebyshev, ::Int) =
    throw(DomainError(Chebyshev, "integral error on Chebyshev InfiniteSequence is not implemented"))
