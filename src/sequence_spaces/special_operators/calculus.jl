"""
    Derivative{T<:Union{Int,Tuple{Vararg{Int}}}}

Generic derivative operator.

See also: [`differentiate`](@ref) and [`differentiate!`](@ref).

# Examples
```jldoctest
julia> Derivative(1)
Derivative{Int64}(1)

julia> Derivative(1, 2)
Derivative{Tuple{Int64, Int64}}((1, 2))
```
"""
struct Derivative{T<:Union{Int,Tuple{Vararg{Int}}}}
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

"""
    Integral{T<:Union{Int,Tuple{Vararg{Int}}}}

Generic integral operator.

See also: [`integrate`](@ref) and [`integrate!`](@ref).

# Examples
```jldoctest
julia> Integral(1)
Integral{Int64}(1)

julia> Integral(1, 2)
Integral{Tuple{Int64, Int64}}((1, 2))
```
"""
struct Integral{T<:Union{Int,Tuple{Vararg{Int}}}}
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

# fallback arithmetic methods

for F ∈ (:Derivative, :Integral)
    @eval begin
        function Base.:+(A::LinearOperator, ℱ::$F)
            domain_A = domain(A)
            return A + project(ℱ, domain_A, codomain(A), _coeftype(ℱ, domain_A, eltype(A)))
        end
        function Base.:+(ℱ::$F, A::LinearOperator)
            domain_A = domain(A)
            return project(ℱ, domain_A, codomain(A), _coeftype(ℱ, domain_A, eltype(A))) + A
        end
        function Base.:-(A::LinearOperator, ℱ::$F)
            domain_A = domain(A)
            return A - project(ℱ, domain_A, codomain(A), _coeftype(ℱ, domain_A, eltype(A)))
        end
        function Base.:-(ℱ::$F, A::LinearOperator)
            domain_A = domain(A)
            return project(ℱ, domain_A, codomain(A), _coeftype(ℱ, domain_A, eltype(A))) - A
        end

        add!(C::LinearOperator, A::LinearOperator, ℱ::$F) = add!(C, A, project(ℱ, domain(A), codomain(A), eltype(C)))
        add!(C::LinearOperator, ℱ::$F, A::LinearOperator) = add!(C, project(ℱ, domain(A), codomain(A), eltype(C)), A)
        sub!(C::LinearOperator, A::LinearOperator, ℱ::$F) = sub!(C, A, project(ℱ, domain(A), codomain(A), eltype(C)))
        sub!(C::LinearOperator, ℱ::$F, A::LinearOperator) = sub!(C, project(ℱ, domain(A), codomain(A), eltype(C)), A)

        radd!(A::LinearOperator, ℱ::$F) = radd!(A, project(ℱ, domain(A), codomain(A), eltype(A)))
        rsub!(A::LinearOperator, ℱ::$F) = rsub!(A, project(ℱ, domain(A), codomain(A), eltype(A)))

        ladd!(ℱ::$F, A::LinearOperator) = ladd!(project(ℱ, domain(A), codomain(A), eltype(A)), A)
        lsub!(ℱ::$F, A::LinearOperator) = lsub!(project(ℱ, domain(A), codomain(A), eltype(A)), A)

        function Base.:*(ℱ::$F, A::LinearOperator)
            codomain_A = codomain(A)
            return project(ℱ, codomain_A, image(ℱ, codomain_A), _coeftype(ℱ, codomain_A, eltype(A))) * A
        end

        mul!(c::Sequence, ℱ::$F, a::Sequence, α::Number, β::Number) = mul!(c, project(ℱ, space(a), space(c), eltype(c)), a, α, β)
        mul!(C::LinearOperator, ℱ::$F, A::LinearOperator, α::Number, β::Number) = mul!(C, project(ℱ, codomain(A), codomain(C), eltype(C)), A, α, β)
        mul!(C::LinearOperator, A::LinearOperator, ℱ::$F, α::Number, β::Number) = mul!(C, A, project(ℱ, domain(C), domain(A), eltype(C)), α, β)
    end
end

#

"""
    differentiate(a::Sequence, α=1)

Computes the `α`-th derivative of `a`.

See also: [`differentiate!`](@ref) and [`Derivative`](@ref).
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

Computes the `α`-th derivative of `a`. The result is stored in `c` by overwritting it.

See also: [`differentiate`](@ref) and [`Derivative`](@ref).
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
    integrate(a::Sequence, α=1)

Computes the `α`-th integral of `a`.

See also: [`integrate!`](@ref) and [`Integral`](@ref).
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

Computes the `α`-th integral of `a`. The result is stored in `c` by overwritting it.

See also: [`integrate`](@ref) and [`Integral`](@ref).
"""
function integrate!(c::Sequence, a::Sequence, α=1)
    ℐ = Integral(α)
    space_c = space(c)
    new_space = image(ℐ, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $ℐ(a) has space $new_space"))
    _apply!(c, ℐ, a)
    return c
end

for (F, f) ∈ ((:Derivative, :differentiate), (:Integral, :integrate))
    @eval begin
        Base.:*(ℱ₁::$F{Int}, ℱ₂::$F{Int}) = $F(ℱ₁.order + ℱ₂.order)
        Base.:*(ℱ₁::$F{NTuple{N,Int}}, ℱ₂::$F{NTuple{N,Int}}) where {N} = $F(map(+, ℱ₁.order, ℱ₂.order))

        Base.:^(ℱ::$F{Int}, n::Int) = $F(ℱ.order * n)
        Base.:^(ℱ::$F{NTuple{N,Int}}, n::Int) where {N} = $F(map(αᵢ -> *(αᵢ, n), ℱ.order))
        Base.:^(ℱ::$F{NTuple{N,Int}}, α::NTuple{N,Int}) where {N} = $F(map(*, ℱ.order, α))

        (ℱ::$F)(a::Sequence) = *(ℱ, a)
        Base.:*(ℱ::$F, a::Sequence) = $f(a, ℱ.order)

        function project(ℱ::$F, domain::VectorSpace, codomain::VectorSpace, ::Type{T}) where {T}
            _iscompatible(domain, codomain) || return throw(ArgumentError("spaces must be compatible: domain is $domain, codomain is $codomain"))
            ind_domain = _findposition_nzind_domain(ℱ, domain, codomain)
            ind_codomain = _findposition_nzind_codomain(ℱ, domain, codomain)
            C = LinearOperator(domain, codomain, SparseArrays.sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), dimension(codomain), dimension(domain)))
            _project!(C, ℱ)
            return C
        end

        function project!(C::LinearOperator, ℱ::$F)
            domain_C = domain(C)
            codomain_C = codomain(C)
            _iscompatible(domain_C, codomain_C) || return throw(ArgumentError("spaces must be compatible: C has domain $domain_C, C has codomain $codomain_C"))
            coefficients(C) .= zero(eltype(C))
            _project!(C, ℱ)
            return C
        end

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
            TensorSpace(map((αᵢ, sᵢ) -> image($F(αᵢ), sᵢ), ℱ.order, spaces(s)))

        _coeftype(ℱ::$F, s::TensorSpace, ::Type{T}) where {T} =
            @inbounds promote_type(_coeftype($F(ℱ.order[1]), s[1], T), _coeftype($F(Base.tail(ℱ.order)), Base.tail(s), T))
        _coeftype(ℱ::$F, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
            @inbounds _coeftype($F(ℱ.order[1]), s[1], T)

        function _apply!(c::Sequence{<:TensorSpace}, ℱ::$F, a)
            space_a = space(a)
            A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
            C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
            _apply!(C, ℱ, space_a, A)
            return c
        end

        _apply!(C, ℱ::$F, space::TensorSpace, A) =
            @inbounds _apply!(C, $F(ℱ.order[1]), space[1], _apply($F(Base.tail(ℱ.order)), Base.tail(space), A))

        _apply!(C, ℱ::$F, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
            @inbounds _apply!(C, $F(ℱ.order[1]), space[1], A)

        _apply(ℱ::$F, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
            @inbounds _apply($F(ℱ.order[1]), space[1], Val(N₂-N₁+1), _apply($F(Base.tail(ℱ.order)), Base.tail(space), A))

        _apply(ℱ::$F, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
            @inbounds _apply($F(ℱ.order[1]), space[1], Val(N), A)
    end
end

for F ∈ (:Derivative, :Integral)
    for (_f, __f) ∈ ((:_nzind_domain, :__nzind_domain), (:_nzind_codomain, :__nzind_codomain))
        @eval begin
            $_f(ℱ::$F{NTuple{N,Int}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
                TensorIndices($__f(ℱ, domain, codomain))
            $__f(ℱ::$F, domain::TensorSpace, codomain) =
                @inbounds ($_f($F(ℱ.order[1]), domain[1], codomain[1]), $__f($F(Base.tail(ℱ.order)), Base.tail(domain), Base.tail(codomain))...)
            $__f(ℱ::$F, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain) =
                @inbounds ($_f($F(ℱ.order[1]), domain[1], codomain[1]),)
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
            @inbounds _nzval($F(ℱ.order[1]), domain[1], codomain[1], T, α[1], β[1]) * _nzval($F(Base.tail(ℱ.order)), Base.tail(domain), Base.tail(codomain), T, Base.tail(α), Base.tail(β))
        _nzval(ℱ::$F{Tuple{Int}}, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}, α, β) where {T} =
            @inbounds _nzval($F(ℱ.order[1]), domain[1], codomain[1], T, α[1], β[1])
    end
end

# Taylor

image(𝒟::Derivative, s::Taylor) = Taylor(max(0, order(s)-𝒟.order))

_coeftype(::Derivative, ::Taylor, ::Type{T}) where {T} = typeof(zero(T)*0)

function _apply!(c::Sequence{Taylor}, 𝒟::Derivative, a)
    n = 𝒟.order
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        order_a = order(a)
        if order_a < n
            @inbounds c[0] = zero(eltype(c))
        elseif n == 1
            @inbounds for i ∈ 1:order_a
                c[i-1] = i * a[i]
            end
        else
            space_a = space(a)
            CoefType_a = eltype(a)
            @inbounds for i ∈ n:order_a
                c[i-n] = _nzval(𝒟, space_a, space_a, CoefType_a, i-n, i) * a[i]
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::Taylor, A) where {T}
    n = 𝒟.order
    if n == 0
        C .= A
    else
        ord = order(space)
        if ord < n
            C .= zero(T)
        elseif n == 1
            @inbounds for i ∈ 1:ord
                selectdim(C, 1, i) .= i .* selectdim(A, 1, i+1)
            end
        else
            CoefType_A = eltype(A)
            @inbounds for i ∈ n:ord
                selectdim(C, 1, i-n+1) .= _nzval(𝒟, space, space, CoefType_A, i-n, i) .* selectdim(A, 1, i+1)
            end
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = 𝒟.order
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
                selectdim(C, D, i) .= i .* selectdim(A, D, i+1)
            end
            return C
        else
            C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord-n+1, size(A, i)), Val(N)))
            @inbounds for i ∈ n:ord
                selectdim(C, D, i-n+1) .= _nzval(𝒟, space, space, T, i-n, i) .* selectdim(A, D, i+1)
            end
            return C
        end
    end
end

_nzind_domain(𝒟::Derivative, domain::Taylor, codomain::Taylor) =
    𝒟.order:min(order(domain), order(codomain)+𝒟.order)

_nzind_codomain(𝒟::Derivative, domain::Taylor, codomain::Taylor) =
    0:min(order(domain)-𝒟.order, order(codomain))

function _nzval(𝒟::Derivative, ::Taylor, ::Taylor, ::Type{T}, i, j) where {T}
    n = 𝒟.order
    p = one(T)*1
    for k ∈ 1:n
        p *= i+k
    end
    return convert(T, p)
end

image(ℐ::Integral, s::Taylor) = Taylor(order(s)+ℐ.order)

_coeftype(::Integral, ::Taylor, ::Type{T}) where {T} = typeof(inv(one(T)*1)*zero(T))

function _apply!(c::Sequence{Taylor}, ℐ::Integral, a)
    n = ℐ.order
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        @inbounds c[0] = zero(eltype(c))
        @inbounds for i ∈ 0:order(a)
            c[i+1] = a[i] / (i+1)
        end
    else
        space_a = space(a)
        CoefType_a = eltype(a)
        @inbounds view(c, 0:n-1) .= zero(eltype(c))
        @inbounds for i ∈ 0:order(a)
            c[i+n] = _nzval(ℐ, space_a, space_a, CoefType_a, i+n, i) * a[i]
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Taylor, A) where {T}
    n = ℐ.order
    if n == 0
        C .= A
    elseif n == 1
        ord = order(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        @inbounds for i ∈ 0:ord
            selectdim(C, 1, i+2) .= selectdim(A, 1, i+1) ./ (i+1)
        end
    else
        CoefType_A = eltype(A)
        ord = order(space)
        @inbounds selectdim(C, 1, 1:n) .= zero(T)
        @inbounds for i ∈ 0:ord
            selectdim(C, 1, i+n+1) .= _nzval(ℐ, space, space, CoefType_A, i+n, i) .* selectdim(A, 1, i+1)
        end
    end
    return C
end

function _apply(ℐ::Integral, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = ℐ.order
    CoefType = _coeftype(ℐ, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    elseif n == 1
        ord = order(space)
        C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord+2, size(A, i)), Val(N)))
        @inbounds selectdim(C, D, 1) .= zero(CoefType)
        @inbounds for i ∈ 0:ord
            selectdim(C, D, i+2) .= selectdim(A, D, i+1) ./ (i+1)
        end
        return C
    else
        ord = order(space)
        C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, ord+n+1, size(A, i)), Val(N)))
        @inbounds selectdim(C, D, 1:n) .= zero(CoefType)
        @inbounds for i ∈ 0:ord
            selectdim(C, D, i+n+1) .= _nzval(ℐ, space, space, T, i+n, i) .* selectdim(A, D, i+1)
        end
        return C
    end
end

_nzind_domain(ℐ::Integral, domain::Taylor, codomain::Taylor) =
    0:min(order(domain), order(codomain)-ℐ.order)

_nzind_codomain(ℐ::Integral, domain::Taylor, codomain::Taylor) =
    ℐ.order:min(order(domain)+ℐ.order, order(codomain))

_nzval(ℐ::Integral, s₁::Taylor, s₂::Taylor, ::Type{T}, i, j) where {T} =
    convert(T, inv(_nzval(Derivative(ℐ.order), s₁, s₂, T, j, i)))

# Fourier

image(::Derivative, s::Fourier) = s

_coeftype(::Derivative, ::Fourier{T}, ::Type{S}) where {T,S} = complex(typeof(zero(T)*0*zero(S)))

function _apply!(c::Sequence{<:Fourier}, 𝒟::Derivative, a)
    n = 𝒟.order
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(eltype(a))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        if n == 1
            iω = im*ω
            @inbounds for j ∈ 1:order(c)
                iωj = iω*j
                c[j] = iωj * a[j]
                c[-j] = -iωj * a[-j]
            end
        else
            if isodd(n)
                iⁿ = complex(0, ifelse(n%4 == 1, 1, -1))
                @inbounds for j ∈ 1:order(c)
                    iⁿωⁿjⁿ = iⁿ*(ω*j)^n
                    c[j] = iⁿωⁿjⁿ * a[j]
                    c[-j] = -iⁿωⁿjⁿ * a[-j]
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
                    c[j] = iⁿωⁿjⁿ_real * a[j]
                    c[-j] = iⁿωⁿjⁿ_real * a[-j]
                end
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::Fourier, A) where {T}
    n = 𝒟.order
    if n == 0
        C .= A
    else
        ord = order(space)
        ω = one(eltype(A))*frequency(space)
        @inbounds selectdim(C, 1, ord+1) .= zero(T)
        if n == 1
            iω = im*ω
            @inbounds for j ∈ 1:ord
                iωj = iω*j
                selectdim(C, 1, ord+1+j) .= iωj .* selectdim(A, 1, ord+1+j)
                selectdim(C, 1, ord+1-j) .= -iωj .* selectdim(A, 1, ord+1-j)
            end
        else
            if isodd(n)
                iⁿ = complex(0, ifelse(n%4 == 1, 1, -1))
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ = iⁿ*(ω*j)^n
                    selectdim(C, 1, ord+1+j) .= iⁿωⁿjⁿ .* selectdim(A, 1, ord+1+j)
                    selectdim(C, 1, ord+1-j) .= -iⁿωⁿjⁿ .* selectdim(A, 1, ord+1-j)
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
                    selectdim(C, 1, ord+1+j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, ord+1+j)
                    selectdim(C, 1, ord+1-j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, ord+1-j)
                end
            end
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = 𝒟.order
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        C = Array{CoefType,N}(undef, size(A))
        ord = order(space)
        ω = one(T)*frequency(space)
        @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
        if n == 1
            iω = im*ω
            @inbounds for j ∈ 1:ord
                iωj = iω*j
                selectdim(C, D, ord+1+j) .= iωj .* selectdim(A, D, ord+1+j)
                selectdim(C, D, ord+1-j) .= -iωj .* selectdim(A, D, ord+1-j)
            end
        else
            if isodd(n)
                iⁿ = complex(0, ifelse(n%4 == 1, 1, -1))
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ = iⁿ*(ω*j)^n
                    selectdim(C, D, ord+1+j) .= iⁿωⁿjⁿ .* selectdim(A, D, ord+1+j)
                    selectdim(C, D, ord+1-j) .= -iⁿωⁿjⁿ .* selectdim(A, D, ord+1-j)
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
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
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return -ord:ord
end

function _nzind_codomain(::Derivative, domain::Fourier, codomain::Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return -ord:ord
end

function _nzval(𝒟::Derivative, domain::Fourier, ::Fourier, ::Type{T}, i, j) where {T}
    n = 𝒟.order
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(T)*frequency(domain)*j)^n
        r = n % 4
        if r == 0
            return convert(T, ωⁿjⁿ)
        elseif r == 1
            return convert(T, im*ωⁿjⁿ)
        elseif r == 2
            return convert(T, -ωⁿjⁿ)
        else
            return convert(T, -im*ωⁿjⁿ)
        end
    end
end

image(::Integral, s::Fourier) = s

_coeftype(::Integral, ::Fourier{T}, ::Type{S}) where {T,S} = complex(typeof(inv(one(S)*one(T)*1)*zero(S)))

function _apply!(c::Sequence{<:Fourier}, ℐ::Integral, a)
    n = ℐ.order
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        @inbounds iszero(a[0]) || return throw(DomainError("Fourier coefficient of order zero must be zero"))
        ω = one(eltype(a))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        if n == 1
            @inbounds for j ∈ 1:order(c)
                iω⁻¹j⁻¹ = im*inv(ω*j)
                c[j] = -iω⁻¹j⁻¹ * a[j]
                c[-j] = iω⁻¹j⁻¹ * a[-j]
            end
        else
            if isodd(n)
                iⁿ = complex(0, ifelse(n%4 == 1, 1, -1))
                @inbounds for j ∈ 1:order(c)
                    iⁿω⁻ⁿj⁻ⁿ = iⁿ*inv(ω*j)^n
                    c[j] = -iⁿω⁻ⁿj⁻ⁿ * a[j]
                    c[-j] = iⁿω⁻ⁿj⁻ⁿ * a[-j]
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:order(c)
                    iⁿω⁻ⁿj⁻ⁿ_real = iⁿ_real*inv(ω*j)^n
                    c[j] = iⁿω⁻ⁿj⁻ⁿ_real * a[j]
                    c[-j] = iⁿω⁻ⁿj⁻ⁿ_real * a[-j]
                end
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Fourier, A) where {T}
    n = ℐ.order
    if n == 0
        C .= A
    else
        ord = order(space)
        @inbounds iszero(selectdim(A, 1, ord+1)) || return throw(DomainError("Fourier coefficients of order zero along dimension 1 must be zero"))
        ω = one(eltype(A))*frequency(space)
        @inbounds selectdim(C, 1, ord+1) .= zero(T)
        if n == 1
            @inbounds for j ∈ 1:ord
                iω⁻¹j⁻¹ = im*inv(ω*j)
                selectdim(C, 1, ord+1+j) .= -iω⁻¹j⁻¹ .* selectdim(A, 1, ord+1+j)
                selectdim(C, 1, ord+1-j) .= iω⁻¹j⁻¹ .* selectdim(A, 1, ord+1-j)
            end
        else
            if isodd(n)
                iⁿ = complex(0, ifelse(n%4 == 1, 1, -1))
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ = iⁿ*inv(ω*j)^n
                    selectdim(C, 1, ord+1+j) .= -iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, 1, ord+1+j)
                    selectdim(C, 1, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, 1, ord+1-j)
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ_real = iⁿ_real*inv(ω*j)^n
                    selectdim(C, 1, ord+1+j) .= iⁿω⁻ⁿj⁻ⁿ_real .* selectdim(A, 1, ord+1+j)
                    selectdim(C, 1, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ_real .* selectdim(A, 1, ord+1-j)
                end
            end
        end
    end
    return C
end

function _apply(ℐ::Integral, space::Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = ℐ.order
    CoefType = _coeftype(ℐ, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        ord = order(space)
        @inbounds iszero(selectdim(A, D, ord+1)) || return throw(DomainError("Fourier coefficient of order zero along dimension $D must be zero"))
        ω = one(T)*frequency(space)
        C = Array{CoefType,N}(undef, size(A))
        @inbounds selectdim(C, D, ord+1) .= zero(CoefType)
        if n == 1
            @inbounds for j ∈ 1:ord
                iω⁻¹j⁻¹ = im*inv(ω*j)
                selectdim(C, D, ord+1+j) .= -iω⁻¹j⁻¹ .* selectdim(A, D, ord+1+j)
                selectdim(C, D, ord+1-j) .= iω⁻¹j⁻¹ .* selectdim(A, D, ord+1-j)
            end
        else
            if isodd(n)
                iⁿ = complex(0, ifelse(n%4 == 1, 1, -1))
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ = iⁿ*inv(ω*j)^n
                    selectdim(C, D, ord+1+j) .= -iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, D, ord+1+j)
                    selectdim(C, D, ord+1-j) .= iⁿω⁻ⁿj⁻ⁿ .* selectdim(A, D, ord+1-j)
                end
            else
                iⁿ_real = ifelse(n%4 == 0, 1, -1)
                @inbounds for j ∈ 1:ord
                    iⁿω⁻ⁿj⁻ⁿ_real = iⁿ_real*inv(ω*j)^n
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
    n = ℐ.order
    if n == 0
        return one(T)
    else
        if j == 0
            return zero(T)
        else
            ω⁻ⁿj⁻ⁿ = inv(one(T)*frequency(domain)*j)^n
            r = n % 4
            if r == 0
                return convert(T, ω⁻ⁿj⁻ⁿ)
            elseif r == 1
                return convert(T, -im*ω⁻ⁿj⁻ⁿ)
            elseif r == 2
                return convert(T, -ω⁻ⁿj⁻ⁿ)
            else
                return convert(T, im*ω⁻ⁿj⁻ⁿ)
            end
        end
    end
end

# Chebyshev

image(𝒟::Derivative, s::Chebyshev) = Chebyshev(max(0, order(s)-𝒟.order))

_coeftype(::Derivative, ::Chebyshev, ::Type{T}) where {T} = typeof(zero(T)*0)

function _apply!(c::Sequence{Chebyshev}, 𝒟::Derivative, a)
    n = 𝒟.order
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
                    c[i] += j * a[j]
                end
                c[i] *= 2
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::Chebyshev, A) where {T}
    n = 𝒟.order
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
                    Cᵢ .+= (2j) .* selectdim(A, 1, j+1)
                end
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

function _apply(𝒟::Derivative, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = 𝒟.order
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
                    Cᵢ .+= (2j) .* selectdim(A, D, j+1)
                end
            end
            return C
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_domain(𝒟::Derivative, domain::Chebyshev, codomain::Chebyshev)
    if 𝒟.order == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif 𝒟.order == 1
        len = sum(j -> length((j-1)%2:2:min(j-1, order(codomain))), 1:order(domain))
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
    if 𝒟.order == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif 𝒟.order == 1
        len = sum(j -> length((j-1)%2:2:min(j-1, order(codomain))), 1:order(domain))
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
    n = 𝒟.order
    if n == 0
        return one(T)
    elseif n == 1
        return convert(T, 2j)
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

image(ℐ::Integral, s::Chebyshev) = Chebyshev(order(s)+ℐ.order)

_coeftype(::Integral, ::Chebyshev, ::Type{T}) where {T} = typeof(zero(T)/1)

function _apply!(c::Sequence{Chebyshev}, ℐ::Integral, a)
    n = ℐ.order
    if n == 0
        coefficients(c) .= coefficients(a)
    elseif n == 1
        order_a = order(a)
        if order_a == 0
            @inbounds c[0] = a[0]
            @inbounds c[1] = a[0] / 2
        elseif order_a == 1
            @inbounds c[0] = a[0] - a[1] / 2
            @inbounds c[1] = a[0] / 2
            @inbounds c[2] = a[1] / 4
        else
            @inbounds c[0] = zero(eltype(c))
            @inbounds for i ∈ 2:2:order_a-1
                c[0] += a[i+1] / ((i+1)^2-1) - a[i] / (i^2-1)
            end
            if iseven(order_a)
                @inbounds c[0] -= a[order_a] / (order_a^2-1)
            end
            @inbounds c[0] = 2 * c[0] + a[0] - a[1] / 2
            @inbounds c[1] = (a[0] - a[2]) / 2
            @inbounds for i ∈ 2:order_a-1
                c[i] = (a[i-1] - a[i+1]) / (2i)
            end
            @inbounds c[order_a] = a[order_a-1] / (2order_a)
            @inbounds c[order_a+1] = a[order_a] / (2(order_a+1))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C::AbstractArray{T}, ℐ::Integral, space::Chebyshev, A) where {T}
    n = ℐ.order
    if n == 0
        C .= A
    elseif n == 1
        ord = order(space)
        @inbounds C₀ = selectdim(C, 1, 1)
        @inbounds C₁ = selectdim(C, 1, 2)
        @inbounds A₀ = selectdim(A, 1, 1)
        if ord == 0
            C₀ .= A₀
            C₁ .= A₀ ./ 2
        elseif ord == 1
            @inbounds A₁ = selectdim(A, 1, 2)
            C₀ .= A₀ .- A₁ ./ 2
            C₁ .= A₀ ./ 2
            @inbounds selectdim(C, 1, 3) .= A₁ ./ 4
        else
            C₀ .= zero(T)
            @inbounds for i ∈ 2:2:ord-1
                C₀ .+= selectdim(A, 1, i+2) ./ ((i+1)^2-1) .- selectdim(A, 1, i+1) ./ (i^2-1)
            end
            if iseven(ord)
                @inbounds C₀ .-= selectdim(A, 1, ord+1) ./ (ord^2-1)
            end
            @inbounds C₀ .= 2 .* C₀ .+ A₀ .- selectdim(A, 1, 2) ./ 2
            @inbounds C₁ .= (A₀ .- selectdim(A, 1, 3)) ./ 2
            @inbounds for i ∈ 2:ord-1
                selectdim(C, 1, i+1) .= (selectdim(A, 1, i) .- selectdim(A, 1, i+2)) ./ (2i)
            end
            @inbounds selectdim(C, 1, ord+1) .= selectdim(A, 1, ord) ./ (2ord)
            @inbounds selectdim(C, 1, ord+2) .= selectdim(A, 1, ord+1) ./ (2(ord+1))
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

function _apply(ℐ::Integral, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = ℐ.order
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
            C₁ .= A₀ ./ 2
        elseif ord == 1
            @inbounds A₁ = selectdim(A, D, 2)
            C₀ .= A₀ .- A₁ ./ 2
            C₁ .= A₀ ./ 2
            @inbounds selectdim(C, D, 3) .= A₁ ./ 4
        else
            C₀ .= zero(CoefType)
            @inbounds for i ∈ 2:2:ord-1
                C₀ .+= selectdim(A, D, i+2) ./ ((i+1)^2-1) .- selectdim(A, D, i+1) ./ (i^2-1)
            end
            if iseven(ord)
                @inbounds C₀ .-= selectdim(A, D, ord+1) ./ (ord^2-1)
            end
            @inbounds C₀ .= 2 .* C₀ .+ A₀ .- selectdim(A, D, 2) ./ 2
            @inbounds C₁ .= (A₀ .- selectdim(A, D, 3)) ./ 2
            @inbounds for i ∈ 2:ord-1
                selectdim(C, D, i+1) .= (selectdim(A, D, i) .- selectdim(A, D, i+2)) ./ (2i)
            end
            @inbounds selectdim(C, D, ord+1) .= selectdim(A, D, ord) ./ (2ord)
            @inbounds selectdim(C, D, ord+2) .= selectdim(A, D, ord+1) ./ (2(ord+1))
        end
        return C
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_domain(ℐ::Integral, domain::Chebyshev, codomain::Chebyshev)
    if ℐ.order == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif ℐ.order == 1
        v = mapreduce(vcat, 0:order(domain)) do j
            if j < 2
                j+1 ≤ order(codomain) && return [j, j]
                return [j]
            else
                j+1 ≤ order(codomain) && return [j, j, j]
                j-1 ≤ order(codomain) && return [j, j]
                return [j]
            end
        end
        return v
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzind_codomain(ℐ::Integral, domain::Chebyshev, codomain::Chebyshev)
    if ℐ.order == 0
        return collect(0:min(order(domain), order(codomain)))
    elseif ℐ.order == 1
        v = mapreduce(vcat, 0:order(domain)) do j
            if j < 2
                j+1 ≤ order(codomain) && return [0, j+1]
                return [0]
            else
                j+1 ≤ order(codomain) && return [0, j-1, j+1]
                j-1 ≤ order(codomain) && return [0, j-1]
                return [0]
            end
        end
        return v
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

function _nzval(ℐ::Integral, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j) where {T}
    n = ℐ.order
    if n == 0
        return one(T)
    elseif n == 1
        if i == 0
            if j == 0
                return one(T)
            elseif j == 1
                return convert(T, -one(T)/2)
            elseif iseven(j)
                return convert(T, 2one(T)/(1-j^2))
            else
                return convert(T, 2one(T)/(j^2-1))
            end
        elseif i == 1 && j == 0
            return convert(T, one(T)/2)
        elseif i == 2 && j == 1
            return convert(T, one(T)/4)
        else
            if i+1 == j
                return convert(T, -one(T)/(2i))
            else # i == j+1
                return convert(T, one(T)/(2i))
            end
        end
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Cartesian spaces

for F ∈ (:Derivative, :Integral)
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
