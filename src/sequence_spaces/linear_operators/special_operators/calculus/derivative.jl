"""
    Derivative{T<:Union{Int,Tuple{Vararg{Int}}}} <: AbstractLinearOperator

Generic derivative operator.

Field:
- `order :: T`

Constructors:
- `Derivative(::Int)`
- `Derivative(::Tuple{Vararg{Int}})`
- `Derivative(order::Int...)`: equivalent to `Derivative(order)`

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

Base.:*(𝒟₁::Derivative{Int}, 𝒟₂::Derivative{Int}) = Derivative(order(𝒟₁) + order(𝒟₂))
Base.:*(𝒟₁::Derivative{NTuple{N,Int}}, 𝒟₂::Derivative{NTuple{N,Int}}) where {N} = Derivative(map(+, order(𝒟₁), order(𝒟₂)))

Base.:^(𝒟::Derivative{Int}, n::Integer) = Derivative(order(𝒟) * n)
Base.:^(𝒟::Derivative{<:Tuple{Vararg{Int}}}, n::Integer) = Derivative(map(αᵢ -> *(αᵢ, n), order(𝒟)))
Base.:^(𝒟::Derivative{NTuple{N,Int}}, n::NTuple{N,Integer}) where {N} = Derivative(map(*, order(𝒟), n))

# Tensor space

function domain(D::Derivative{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N}
    s_out = map((αᵢ, sᵢ) -> domain(Derivative(αᵢ), sᵢ), order(D), spaces(s))
    any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
    return TensorSpace(s_out)
end

codomain(ℱ::Derivative{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((αᵢ, sᵢ) -> codomain(Derivative(αᵢ), sᵢ), order(ℱ), spaces(s)))

_coeftype(ℱ::Derivative{NTuple{N,Int}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Derivative(order(ℱ)[1]), s[1], T), _coeftype(Derivative(Base.tail(order(ℱ))), Base.tail(s), T))
_coeftype(ℱ::Derivative{Tuple{Int}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Derivative(order(ℱ)[1]), s[1], T)

getcoefficient(ℱ::Derivative{NTuple{N,Int}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Derivative(order(ℱ)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Derivative(Base.tail(order(ℱ))), (Base.tail(codom), T, Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(ℱ::Derivative{Tuple{Int}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Derivative(order(ℱ)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

# Taylor

domain(D::Derivative, s::Taylor) = codomain(Integral(order(D)), s)

codomain(𝒟::Derivative, s::Taylor) = Taylor(max(0, order(s)-order(𝒟)))

_coeftype(::Derivative, ::Taylor, ::Type{T}) where {T} = T

function getcoefficient(𝒟::Derivative, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    n = order(𝒟)
    i != j-n && return zero(T)
    p = one(real(T))
    for k ∈ 1:n
        p = exact(i+k) * p
    end
    return convert(T, p)
end

# Fourier

domain(::Derivative, s::Fourier) = s

codomain(::Derivative, s::Fourier) = s

_coeftype(::Derivative, ::Fourier{T}, ::Type{S}) where {T,S} = complex(typeof(zero(T)*zero(S)))

function getcoefficient(𝒟::Derivative, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T}
    i != j && return zero(T)
    n = order(𝒟)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(real(T)) * frequency(dom) * exact(j)) ^ exact(n)
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

# Chebyshev

domain(D::Derivative, s::Chebyshev) = iszero(order(D)) ? s : EmptySpace() # flags an error

codomain(𝒟::Derivative, s::Chebyshev) = Chebyshev(max(0, order(s)-order(𝒟)))

_coeftype(::Derivative, ::Chebyshev, ::Type{T}) where {T} = T

function getcoefficient(𝒟::Derivative, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    n = order(𝒟)
    if n == 0
        return ifelse(i == j, one(T), zero(T))
    elseif n == 1
        return ifelse(i ∈ (0+iseven(j)):2:(j-1), convert(T, exact(2j)), zero(T))
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Symmetric space

function domain(D::Derivative, s::SymmetricSpace)
    V = domain(D, desymmetrize(s))
    G = unsafe_group!(Set(_groupelem_antiderivative(D, g, desymmetrize(s))
                  for g ∈ elements(symmetry(s))))
    return SymmetricSpace(V, G)
end

function _groupelem_antiderivative(D::Derivative, g::GroupElement, ::Fourier)
    c = g.index_action.matrix[1]^order(D)
    new_va = CoefAction(g.coef_action.amplitude / exact(c),
                      g.coef_action.phase)
    return GroupElement(g.index_action, new_va)
end

function codomain(D::Derivative, s::SymmetricSpace)
    V = codomain(D, desymmetrize(s))
    G = unsafe_group!(Set(_groupelem_derivative(D, g, desymmetrize(s)) for g ∈ elements(symmetry(s))))
    return SymmetricSpace(V, G)
end

function _groupelem_derivative(D::Derivative, g::GroupElement, ::Fourier)
    c = g.index_action.matrix[1]^order(D)
    new_va = CoefAction(g.coef_action.amplitude * exact(c),
                      g.coef_action.phase)
    return GroupElement(g.index_action, new_va)
end

_coeftype(D::Derivative, s::SymmetricSpace, ::Type{T}) where {T} =
    _coeftype(D, desymmetrize(s), T)



# action

Base.:*(𝒟::Derivative, a::AbstractSequence) = differentiate(a, order(𝒟))

function differentiate(a::Sequence, α::Union{Int,Tuple{Vararg{Int}}}=1)
    𝒟 = Derivative(α)
    space_a = space(a)
    new_space = codomain(𝒟, space_a)
    CoefType = _coeftype(𝒟, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒟, a)
    return c
end

function differentiate!(c::Sequence, a::Sequence, α::Union{Int,Tuple{Vararg{Int}}}=1)
    𝒟 = Derivative(α)
    space_c = space(c)
    new_space = codomain(𝒟, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒟(a) has space $new_space"))
    _apply!(c, 𝒟, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, ℱ::Derivative, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, ℱ, space_a, A)
    return c
end

_apply!(C, ℱ::Derivative, space::TensorSpace, A) =
    @inbounds _apply!(C, Derivative(order(ℱ)[1]), space[1], _apply(Derivative(Base.tail(order(ℱ))), Base.tail(space), A))
_apply!(C, ℱ::Derivative, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply!(C, Derivative(order(ℱ)[1]), space[1], A)

_apply(ℱ::Derivative{NTuple{N₁,Int}}, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply(Derivative(order(ℱ)[1]), space[1], Val(N₂-N₁+1), _apply(Derivative(Base.tail(order(ℱ))), Base.tail(space), A))
_apply(ℱ::Derivative{Tuple{Int}}, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _apply(Derivative(order(ℱ)[1]), space[1], Val(N), A)

# Taylor

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
                c[i-n] = getcoefficient(𝒟, space_a, space_a, CoefType, i-n, i) * a[i]
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
                selectdim(C, 1, i-n+1) .= getcoefficient(𝒟, space, space, T, i-n, i) .* selectdim(A, 1, i+1)
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
                selectdim(C, D, i-n+1) .= getcoefficient(𝒟, space, space, CoefType, i-n, i) .* selectdim(A, D, i+1)
            end
            return C
        end
    end
end

# Fourier

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

# Chebyshev

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





# Cauchy estimates

function differentiate(a::InfiniteSequence, α::Union{Int,Tuple{Vararg{Int}}}=1)
    c = differentiate(sequence(a), α)
    X = banachspace(a)
    factor = _derivative_error(X, space(a), α)
    new_err = factor * sequence_error(a)
    return InfiniteSequence(c, new_err, Ell1())
end

_derivative_error(X::Ell1{<:NTuple{N,Weight}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    mapreduce((wᵢ, sᵢ, αᵢ) -> _derivative_error(Ell1(wᵢ), sᵢ, αᵢ), *, weight(X), spaces(s), α)

function _derivative_error(X::Ell1{<:GeometricWeight}, s::Taylor, α::Int)
    α == 0 && return one(rate(X))
    ν = rate(X)
    factα = one(ν)
    for k ∈ 1:α
        factα = exact(k) * factα
    end
    # Tail-only bound at ν' = 1 leveraging N = order(s):
    #   ‖D^α ε‖_{ℓ¹(1)} ≤ (Σ_{k>N} k!/(k-α)! ν^{-k}) · sequence_error
    # Computed iteratively as α!·ν/(ν-1)^(α+1) − Σ_{k=α}^{N} k!/(k-α)! · ν^{-k}.
    cur = factα * ν / (ν - exact(1)) ^ exact(α+1)
    term_coef = factα
    k = α
    while k ≤ order(s)
        cur = cur - term_coef / ν ^ exact(k)
        k += 1
        term_coef = term_coef * exact(k) / exact(k - α)
    end
    return cur
end

function _derivative_error(X::Ell1{<:GeometricWeight}, s::Fourier, α::Int)
    α == 0 && return one(rate(X))
    α == 1 || return throw(DomainError(α, "derivative error on Fourier is only implemented for α ≤ 1"))
    ν = rate(X)
    ω = abs(frequency(s)) * one(ν)
    # Tail-only bound at ν' = 1 with N = order(s):
    #   ‖D ε‖_{ℓ¹(1)} ≤ 2|ω| (Σ_{m>N} m ν^{-m}) · sequence_error
    cur = exact(2) * ω * ν / (ν - exact(1)) ^ exact(2)
    m = 1
    while m ≤ order(s)
        cur = cur - exact(2) * ω * exact(m) / ν ^ exact(m)
        m += 1
    end
    return cur
end

_derivative_error(::Ell1{<:GeometricWeight}, s::Chebyshev, ::Int) =
    throw(DomainError(s, "derivative error on Chebyshev InfiniteSequence is not implemented"))
