abstract type SymBaseSpace <: BaseSpace end

desymmetrize(s::SymBaseSpace) = s.space
desymmetrize(s::TensorSpace) = TensorSpace(map(desymmetrize, spaces(s)))
desymmetrize(s::BaseSpace) = s
desymmetrize(s::ParameterSpace) = s
desymmetrize(s::CartesianPower) = CartesianPower(desymmetrize(space(s)), nspaces(s))
desymmetrize(s::CartesianProduct) = CartesianProduct(map(desymmetrize, spaces(s)))

order(s::SymBaseSpace) = order(desymmetrize(s))
frequency(s::SymBaseSpace) = frequency(desymmetrize(s))

Base.issubset(s₁::SymBaseSpace, s₂::SymBaseSpace) = false
Base.issubset(s₁::SymBaseSpace, s₂::BaseSpace) = issubset(desymmetrize(s₁), s₂)
Base.union(s₁::SymBaseSpace, s₂::SymBaseSpace) = union(desymmetrize(s₁), desymmetrize(s₂))
Base.union(s₁::SymBaseSpace, s₂::BaseSpace) = union(desymmetrize(s₁), s₂)
Base.union(s₁::BaseSpace, s₂::SymBaseSpace) = union(s₁, desymmetrize(s₂))





#





struct CosFourier{T<:Real} <: SymBaseSpace
    space :: Fourier{T}
    CosFourier{T}(space::Fourier{T}) where {T<:Real} = new{T}(space)
end
CosFourier(space::Fourier{T}) where {T<:Real} = CosFourier{T}(space)
CosFourier{T}(order::Int, frequency::T) where {T<:Real} = CosFourier(Fourier{T}(order, frequency))
CosFourier(order::Int, frequency::Real) = CosFourier(Fourier(order, frequency))

Base.:(==)(s₁::CosFourier, s₂::CosFourier) = desymmetrize(s₁) == desymmetrize(s₂)
Base.issubset(s₁::CosFourier, s₂::CosFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
Base.intersect(s₁::CosFourier, s₂::CosFourier) = CosFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
Base.union(s₁::CosFourier, s₂::CosFourier) = CosFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

indices(s::CosFourier) = 0:order(s)

_findindex_constant(s::CosFourier) = 0

_findposition(i::Int, ::CosFourier) = i + 1
_findposition(u::AbstractRange{Int}, ::CosFourier) = u .+ 1
_findposition(u::AbstractVector{Int}, s::CosFourier) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::CosFourier) = c

Base.convert(::Type{T}, s::T) where {T<:CosFourier} = s
Base.convert(::Type{CosFourier{T}}, s::CosFourier) where {T<:Real} =
    CosFourier{T}(order(s), convert(T, frequency(s)))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:CosFourier} = T
Base.promote_rule(::Type{CosFourier{T}}, ::Type{CosFourier{S}}) where {T<:Real,S<:Real} =
    CosFourier{promote_type(T, S)}

_iscompatible(s₁::CosFourier, s₂::CosFourier) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂))

_prettystring(s::CosFourier) = "CosFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"



struct SinFourier{T<:Real} <: SymBaseSpace
    space :: Fourier{T}
    SinFourier{T}(space::Fourier{T}) where {T<:Real} = new{T}(space)
end
SinFourier(space::Fourier{T}) where {T<:Real} = SinFourier{T}(space)
SinFourier{T}(order::Int, frequency::T) where {T<:Real} = SinFourier(Fourier{T}(order, frequency))
SinFourier(order::Int, frequency::Real) = SinFourier(Fourier(order, frequency))

Base.:(==)(s₁::SinFourier, s₂::SinFourier) = desymmetrize(s₁) == desymmetrize(s₂)
Base.issubset(s₁::SinFourier, s₂::SinFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
Base.intersect(s₁::SinFourier, s₂::SinFourier) = SinFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
Base.union(s₁::SinFourier, s₂::SinFourier) = SinFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

indices(s::SinFourier) = 1:order(s)

_findposition(i::Int, ::SinFourier) = i
_findposition(u::AbstractRange{Int}, ::SinFourier) = u
_findposition(u::AbstractVector{Int}, s::SinFourier) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::SinFourier) = c

Base.convert(::Type{T}, s::T) where {T<:SinFourier} = s
Base.convert(::Type{SinFourier{T}}, s::SinFourier) where {T<:Real} =
    SinFourier{T}(order(s), convert(T, frequency(s)))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:SinFourier} = T
Base.promote_rule(::Type{SinFourier{T}}, ::Type{SinFourier{S}}) where {T<:Real,S<:Real} =
    SinFourier{promote_type(T, S)}

_iscompatible(s₁::SinFourier, s₂::SinFourier) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂))

_prettystring(s::SinFourier) = "SinFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"





#





image(::typeof(+), s₁::CosFourier, s₂::CosFourier) = CosFourier(image(+, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(*), s₁::CosFourier, s₂::CosFourier) = CosFourier(image(*, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(add_bar), s₁::CosFourier, s₂::CosFourier) = CosFourier(image(add_bar, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(mul_bar), s₁::CosFourier, s₂::CosFourier) = CosFourier(image(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))



image(::typeof(+), s₁::SinFourier, s₂::SinFourier) = SinFourier(image(+, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(*), s₁::SinFourier, s₂::SinFourier) = CosFourier(image(*, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(add_bar), s₁::SinFourier, s₂::SinFourier) = SinFourier(image(add_bar, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(mul_bar), s₁::SinFourier, s₂::SinFourier) = CosFourier(image(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))



image(::typeof(+), s₁::CosFourier, s₂::SinFourier) = image(+, desymmetrize(s₁), desymmetrize(s₂))
image(::typeof(+), s₁::SinFourier, s₂::CosFourier) = image(+, desymmetrize(s₁), desymmetrize(s₂))
image(::typeof(*), s₁::CosFourier, s₂::SinFourier) = SinFourier(image(*, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(*), s₁::SinFourier, s₂::CosFourier) = SinFourier(image(*, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(add_bar), s₁::CosFourier, s₂::SinFourier) = image(add_bar, desymmetrize(s₁), desymmetrize(s₂))
image(::typeof(add_bar), s₁::SinFourier, s₂::CosFourier) = image(add_bar, desymmetrize(s₁), desymmetrize(s₂))
image(::typeof(mul_bar), s₁::CosFourier, s₂::SinFourier) = SinFourier(image(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))
image(::typeof(mul_bar), s₁::SinFourier, s₂::CosFourier) = SinFourier(image(mul_bar, desymmetrize(s₁), desymmetrize(s₂)))

# Convolution

function __convolution!(C, A, B, α, ::CosFourier, space_a::CosFourier, space_b::CosFourier, i)
    order_a = order(space_a)
    order_b = order(space_b)
    Cᵢ = zero(promote_type(eltype(A), eltype(B)))
    @inbounds @simd for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(space_a, space_b, i)
        Cᵢ += A[abs(i-j)+1] * B[abs(j)+1]
    end
    @inbounds C[i+1] += Cᵢ * α
    return C
end
function _convolution!(C::AbstractArray{T,N}, A, B, α, ::CosFourier, current_space_a::CosFourier, current_space_b::CosFourier, remaining_space_c, remaining_space_a, remaining_space_b, i) where {T,N}
    order_a = order(current_space_a)
    order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b) # _convolution_indices(current_space_a, current_space_b, i)
        _add_mul!(Cᵢ,
            selectdim(A, N, abs(i-j)+1),
            selectdim(B, N, abs(j)+1),
            α, remaining_space_c, remaining_space_a, remaining_space_b)
    end
    return C
end

_convolution_indices(s₁::CosFourier, s₂::CosFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

_symmetry_action(::CosFourier, ::Int, ::Int) = 1
_symmetry_action(::CosFourier, ::Int) = 1

_extract_valid_index(::CosFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::CosFourier, i::Int) = abs(i)



_convolution_indices(s₁::SinFourier, s₂::SinFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

function _symmetry_action(::SinFourier, i::Int, j::Int)
    x = i-j
    return ifelse(x == 0, 0, flipsign(1, x))
end
_symmetry_action(::SinFourier, i::Int) = ifelse(i == 0, 0, flipsign(1, i))

_extract_valid_index(::SinFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::SinFourier, i::Int) = abs(i)



_convolution_indices(s₁::CosFourier, s₂::SinFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))
_convolution_indices(s₁::SinFourier, s₂::CosFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

# Derivative

image(𝒟::Derivative, s::CosFourier) = iseven(order(𝒟)) ? s : SinFourier(desymmetrize(s))

_coeftype(::Derivative, ::CosFourier{T}, ::Type{S}) where {T,S} = typeof(zero(T)*0*zero(S))

function _apply!(c::Sequence{<:CosFourier}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(eltype(a))*frequency(a)
        @inbounds c[0] = zero(eltype(c))
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:order(c)
            iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
            c[j] = iⁿωⁿjⁿ_real * a[j]
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::CosFourier, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    else
        ord = order(space)
        ω = one(eltype(A))*frequency(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
            selectdim(C, 1, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j+1)
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::CosFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        C = Array{CoefType,N}(undef, size(A))
        ord = order(space)
        ω = one(T)*frequency(space)
        @inbounds selectdim(C, D, 1) .= zero(CoefType)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
            selectdim(C, D, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j+1)
        end
        return C
    end
end

function _nzind_domain(::Derivative, domain::CosFourier, codomain::CosFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 0:ord
end
function _nzind_domain(::Derivative, domain::CosFourier, codomain::SinFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzind_codomain(::Derivative, domain::CosFourier, codomain::CosFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 0:ord
end
function _nzind_codomain(::Derivative, domain::SinFourier, codomain::CosFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzval(𝒟::Derivative, domain::Union{CosFourier,SinFourier}, ::CosFourier, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(T)*frequency(domain)*j)^n
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ)) # (n%4 == 0) | (n%4 == 1)
    end
end



image(𝒟::Derivative, s::SinFourier) = iseven(order(𝒟)) ? s : CosFourier(desymmetrize(s))

_coeftype(::Derivative, ::SinFourier{T}, ::Type{S}) where {T,S} = typeof(zero(T)*0*zero(S))

function _apply!(c::Sequence{<:SinFourier}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(eltype(a))*frequency(a)
        iⁿ_real = ifelse(n%4 < 2, -1, 1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:order(c)
            iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
            c[j] = iⁿωⁿjⁿ_real * a[j]
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒟::Derivative, space::SinFourier, A) where {T}
    n = order(𝒟)
    if n == 0
        C .= A
    else
        ord = order(space)
        ω = one(eltype(A))*frequency(space)
        iⁿ_real = ifelse(n%4 < 2, -1, 1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
            selectdim(C, 1, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j+1)
        end
    end
    return C
end

function _apply(𝒟::Derivative, space::SinFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(𝒟)
    CoefType = _coeftype(𝒟, space, T)
    if n == 0
        return convert(Array{CoefType,N}, A)
    else
        C = Array{CoefType,N}(undef, size(A))
        ord = order(space)
        ω = one(T)*frequency(space)
        iⁿ_real = ifelse(n%4 < 2, -1, 1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = iⁿ_real*(ω*j)^n
            selectdim(C, D, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j+1)
        end
        return C
    end
end

function _nzind_domain(::Derivative, domain::SinFourier, codomain::Union{CosFourier,SinFourier})
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzind_codomain(::Derivative, domain::Union{CosFourier,SinFourier}, codomain::SinFourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = min(order(domain), order(codomain))
    return 1:ord
end

function _nzval(𝒟::Derivative, domain::Union{CosFourier,SinFourier}, ::SinFourier, ::Type{T}, i, j) where {T}
    n = order(𝒟)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (one(T)*frequency(domain)*j)^n
        return convert(T, ifelse(n%4 < 2, -ωⁿjⁿ, ωⁿjⁿ)) # (n%4 == 0) | (n%4 == 1)
    end
end

# Evaluation

_memo(::CosFourier, ::Type) = nothing

image(::Evaluation{Nothing}, s::CosFourier) = s
image(::Evaluation, s::CosFourier) = CosFourier(0, frequency(s))

_coeftype(::Evaluation{Nothing}, ::CosFourier, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, s::CosFourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cos(frequency(s)*zero(T))), S)

function _apply!(c::Sequence{<:CosFourier}, ::Evaluation{Nothing}, a)
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c::Sequence{<:CosFourier}, ℰ::Evaluation, a)
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
            @inbounds c[0] *= cos(ωx*ord)
            @inbounds for j ∈ ord-1:-1:1
                c[0] += a[j] * cos(ωx*j)
            end
        end
        @inbounds c[0] = 2c[0] + a[0]
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
            C .*= cos(ωx*ord)
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, 1, j+1) .* cos(ωx*j)
            end
        end
        @inbounds C .= 2 .* C .+ selectdim(A, 1, 1)
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
            C .*= cos(ωx*ord)
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, D, j+1) .* cos(ωx*j)
            end
        end
        @inbounds C .= 2 .* C .+ selectdim(A, D, 1)
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
            return convert(T, 2one(T))
        else
            return convert(T, 2cos(frequency(domain)*j*x))
        end
    else
        return zero(T)
    end
end



_memo(::SinFourier, ::Type) = nothing

image(::Evaluation{Nothing}, s::SinFourier) = s
# image(::Evaluation, s::SinFourier) = Fourier(0, frequency(s))

_coeftype(::Evaluation{Nothing}, ::SinFourier, ::Type{T}) where {T} = T
# _coeftype(::Evaluation{T}, s::SinFourier, ::Type{S}) where {T,S} =
#     promote_type(typeof(sin(frequency(s)*zero(T))), S)

function _apply!(c::Sequence{<:SinFourier}, ::Evaluation{Nothing}, a)
    coefficients(c) .= coefficients(a)
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::SinFourier, A)
    C .= A
    return C
end

_apply(::Evaluation{Nothing}, ::SinFourier, ::Val, A::AbstractArray) = A

_getindex(::Evaluation{Nothing}, ::SinFourier, ::SinFourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))

# Multiplication

function _project!(C::LinearOperator{<:CosFourier,<:CosFourier}, ℳ::Multiplication)
    C_ = LinearOperator(Chebyshev(order(domain(C))), Chebyshev(order(codomain(C))), coefficients(C))
    a = sequence(ℳ)
    ℳ_ = Multiplication(Sequence(Chebyshev(order(space(a))), coefficients(a)))
    _project!(C_, ℳ_)
    return C
end

_mult_domain_indices(s::CosFourier) = _mult_domain_indices(Chebyshev(order(s)))
_isvalid(s::CosFourier, i::Int, j::Int) = _isvalid(Chebyshev(order(s)), i, j)

# Norm

_getindex(weight::GeometricWeight, ::CosFourier, i::Int) = weight.rate ^ i
_getindex(weight::GeometricWeight{<:Interval}, ::CosFourier, i::Int) = pow(weight.rate, i)

_getindex(weight::AlgebraicWeight, ::CosFourier, i::Int) = (one(weight.rate) + i) ^ weight.rate
_getindex(weight::AlgebraicWeight{<:Interval}, ::CosFourier, i::Int) = pow(one(weight.rate) + i, weight.rate)





_apply(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + 2sum(abs, view(A, 2:length(A)))
function _apply(::Ell1{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= 2 .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]), maximum(abs, view(A, 2:length(A)))/2)
function _apply_dual(::Ell1{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(s ./ 2, abs.(selectdim(A, N, 1)))
    end
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(X.weight)
    ord = order(space)
    @inbounds s = 1abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = 2s * ν + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
    CoefType = typeof(2abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= 2 .* s .* ν .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector{T}) where {T}
    ν = rate(X.weight)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ½ = one(ν⁻¹)/2
    @inbounds s = abs(A[1]) * one(ν⁻ⁱ½)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ½ *= ν⁻¹
        s = max(s, abs(A[i+1]) * ν⁻ⁱ½)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
    ν⁻¹ = abs(one(T))/ν
    ν⁻ⁱ½ = one(ν⁻¹)/2
    CoefType = typeof(ν⁻ⁱ½)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        ν⁻ⁱ½ *= ν⁻¹
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* ν⁻ⁱ½)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = 1abs(A[ord+1]) * _getindex(X.weight, space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) * _getindex(X.weight, space, i)
        end
        @inbounds s = 2s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T))*_getindex(X.weight, space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(X.weight, space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) .* _getindex(X.weight, space, i)
        end
        @inbounds s .= 2 .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = (abs(A[ord+1]) / _getindex(X.weight, space, ord)) / 1
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) / _getindex(X.weight, space, i))
        end
        @inbounds s = max(s/2, abs(A[1]))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof((abs(zero(T))/_getindex(X.weight, space, 0))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(X.weight, space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(X.weight, space, i))
        end
        @inbounds s .= max.(s ./ 2, abs.(selectdim(A, N, 1)))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + 2sum(abs2, view(A, 2:length(A))))
function _apply(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(2abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(2 .* s .+ abs2.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + sum(abs2, view(A, 2:length(A)))/2)
function _apply_dual(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))/2))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(s ./ 2 .+ abs2.(selectdim(A, N, 1)))
    return s
end

_apply(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]), 2maximum(abs, view(A, 2:length(A))))
function _apply(::EllInf{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(2 .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + sum(abs, view(A, 2:length(A)))/2
function _apply_dual(::EllInf{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s ./ 2 .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(X.weight)
    νⁱ2 = 2one(ν)
    @inbounds s = abs(A[1]) * one(νⁱ)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s = max(s, abs(A[i+1]) * νⁱ2)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
    νⁱ2 = 2one(ν)
    CoefType = typeof(abs(zero(T))*νⁱ2)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ2)
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(X.weight)
    ord = order(space)
    @inbounds s = (abs(A[ord+1]) * one(ν)) / 1
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = (s * ν)/2 + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(X.weight)
    CoefType = typeof((abs(zero(T))*ν)/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= (s .* ν) ./ 2 .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = 1abs(A[ord+1]) * _getindex(X.weight, space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) * _getindex(X.weight, space, i))
        end
        @inbounds s = max(2s, abs(A[1]))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(2abs(zero(T))*_getindex(X.weight, space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(X.weight, space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(X.weight, space, i))
        end
        @inbounds s .= max.(2 .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = (abs(A[ord+1]) / _getindex(X.weight, space, ord)) / 1
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) / _getindex(X.weight, space, i)
        end
        @inbounds s = s/2 + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof((abs(zero(T))/_getindex(X.weight, space, 0))/2)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(X.weight, space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(X.weight, space, i)
        end
        @inbounds s .= s ./ 2 .+ abs.(selectdim(A, N, 1))
    end
    return s
end
