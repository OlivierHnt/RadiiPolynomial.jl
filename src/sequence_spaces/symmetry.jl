# conjugacy_symmetry!(a::Sequence{<:VectorSpace,<:AbstractVector{<:Real}}) = a

conjugacy_symmetry!(a::Sequence) = _conjugacy_symmetry!(a)

_conjugacy_symmetry!(::Sequence) = throw(DomainError) # TODO: lift restriction

function _conjugacy_symmetry!(a::Sequence{ParameterSpace})
    @inbounds a[1] = real(a[1])
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:Fourier})
    ord = order(a)
    @inbounds a[0] = real(a[0])
    @inbounds view(a, -ord:-1) .= conj.(view(a, ord:-1:1))
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:CartesianSpace})
    for aᵢ ∈ eachcomponent(a)
        _conjugacy_symmetry!(aᵢ)
    end
    return a
end

function _conjugacy_symmetry!(a::Sequence{CartesianProduct{T}}) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _conjugacy_symmetry!(component(a, 1))
    @inbounds _conjugacy_symmetry!(component(a, 2:N))
    return a
end
function _conjugacy_symmetry!(a::Sequence{CartesianProduct{T}}) where {T<:Tuple{VectorSpace}}
    @inbounds _conjugacy_symmetry!(component(a, 1))
    return a
end





#





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

_findindex_constant(::CosFourier) = 0

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
_inverse_symmetry_action(::CosFourier, ::Int) = 1

_extract_valid_index(::CosFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::CosFourier, i::Int) = abs(i)



_convolution_indices(s₁::SinFourier, s₂::SinFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

function _symmetry_action(::SinFourier, i::Int, j::Int)
    x = j-i
    y = ifelse(x == 0, 0, flipsign(1, x))
    return complex(0, y)
end
function _symmetry_action(::SinFourier, i::Int)
    y = ifelse(i == 0, 0, flipsign(1, -i))
    return complex(0, y)
end
_inverse_symmetry_action(::SinFourier, ::Int) = complex(0, 1)

_extract_valid_index(::SinFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::SinFourier, i::Int) = abs(i)



_convolution_indices(s₁::CosFourier, s₂::SinFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))
_convolution_indices(s₁::SinFourier, s₂::CosFourier, i::Int) =
    max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

# FFT

_dfs_dimension(s::CosFourier) = 2order(s)+1
function _preprocess!(C::AbstractVector, space::CosFourier)
    len = length(C)
    @inbounds for i ∈ 2:order(space)+1
        C[len+2-i] = C[i]
    end
    return C
end
function _preprocess!(C::AbstractArray, space::CosFourier, ::Val{D}) where {D}
    len = size(C, D)
    @inbounds for i ∈ 2:order(space)+1
        selectdim(C, D, len+2-i) .= selectdim(C, D, i)
    end
    return C
end
_postprocess!(C, ::CosFourier) = C
_postprocess!(C, ::CosFourier, ::Val) = C

_dfs_dimension(s::SinFourier) = 2order(s)+1
function _preprocess!(C::AbstractVector, space::SinFourier)
    CoefType = eltype(C)
    len = length(C)
    @inbounds for i ∈ order(space)+1:-1:2
        C[i] = complex(zero(CoefType), -C[i-1])
        C[len+2-i] = -C[i]
    end
    @inbounds C[1] = zero(CoefType)
    return C
end
function _preprocess!(C::AbstractArray, space::SinFourier, ::Val{D}) where {D}
    CoefType = eltype(C)
    len = size(C, D)
    @inbounds for i ∈ order(space)+1:-1:2
        selectdim(C, D, i) .= complex.(zero(CoefType), .- selectdim(C, D, i-1))
        selectdim(C, D, len+2-i) .= .- selectdim(C, D, i)
    end
    @inbounds selectdim(C, D, 1) .= zero(CoefType)
    return C
end
function _postprocess!(C, space::SinFourier)
    CoefType = eltype(C)
    @inbounds for i ∈ 1:order(space)
        C[i] = complex(zero(CoefType), C[i+1])
    end
    return C
end
function _postprocess!(C, space::SinFourier, ::Val{D}) where {D}
    CoefType = eltype(C)
    @inbounds for i ∈ 1:order(space)
        selectdim(C, D, i) .= complex.(zero(CoefType), selectdim(C, D, i+1))
    end
    return C
end

# Derivative

image(𝒟::Derivative, s::CosFourier) = iseven(order(𝒟)) ? s : SinFourier(desymmetrize(s))

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
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
            selectdim(C, 1, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j+1)
        end
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
            selectdim(C, D, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j+1)
        end
        return C
    else
        C = Array{CoefType,N}(undef, ntuple(i -> size(A, i) - ifelse(i == D, 1, 0), Val(N)))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
        ωⁿjⁿ = _safe_pow(_safe_mul(one(real(T))*frequency(domain), j), n)
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ)) # (n%4 == 0) | (n%4 == 1)
    end
end



image(𝒟::Derivative, s::SinFourier) = iseven(order(𝒟)) ? s : CosFourier(desymmetrize(s))

_coeftype(::Derivative, ::SinFourier{T}, ::Type{S}) where {T,S} = typeof(zero(T)*zero(S))

function _apply!(c::Sequence{<:SinFourier}, 𝒟::Derivative, a)
    n = order(𝒟)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        ω = one(real(eltype(a)))*frequency(a)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:order(c)
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
            selectdim(C, 1, j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j)
        end
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
            iⁿωⁿjⁿ_real = _safe_mul(iⁿ_real, _safe_pow(_safe_mul(ω, j), n))
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
        ωⁿjⁿ = _safe_pow(_safe_mul(one(real(T))*frequency(domain), j), n)
        return convert(T, ifelse((n+1)%4 < 2, ωⁿjⁿ, -ωⁿjⁿ)) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
    end
end

# Evaluation

_memo(::CosFourier, ::Type) = nothing

image(::Evaluation{Nothing}, s::CosFourier) = s
image(::Evaluation, s::CosFourier) = CosFourier(0, frequency(s))

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
            @inbounds c[0] *= cos(_safe_mul(ωx, ord))
            @inbounds for j ∈ ord-1:-1:1
                c[0] += a[j] * cos(_safe_mul(ωx, j))
            end
        end
        @inbounds c[0] = _safe_mul(2, c[0]) + a[0]
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
            C .*= cos(_safe_mul(ωx, ord))
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, 1, j+1) .* cos(_safe_mul(ωx, j))
            end
        end
        @inbounds C .= _safe_mul.(2, C) .+ selectdim(A, 1, 1)
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
            C .*= cos(_safe_mul(ωx, ord))
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, D, j+1) .* cos(_safe_mul(ωx, j))
            end
        end
        @inbounds C .= _safe_mul.(2, C) .+ selectdim(A, D, 1)
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
            return _safe_convert(T, 2)
        else
            return convert(T, _safe_mul(2, cos(_safe_mul(frequency(domain)*x, j))))
        end
    else
        return zero(T)
    end
end



_memo(::SinFourier, ::Type) = nothing

image(::Evaluation{Nothing}, s::SinFourier) = s
image(::Evaluation, s::SinFourier) = Fourier(0, frequency(s))

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
        @inbounds c[0] = a[ord] * sin(_safe_mul(ωx, ord))
        @inbounds for j ∈ ord-1:-1:1
            c[0] += a[j] * sin(_safe_mul(ωx, j))
        end
        @inbounds c[0] = _safe_mul(2, c[0])
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
        @inbounds C .= selectdim(A, 1, ord) .* sin(_safe_mul(ωx, ord))
        @inbounds for j ∈ ord-1:-1:1
            C .+= selectdim(A, 1, j) .* sin(_safe_mul(ωx, j))
        end
        C .= _safe_mul.(2, C)
    end
    return C
end

_apply(::Evaluation{Nothing}, ::SinFourier, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::SinFourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = value(ℰ)
    CoefType = _coeftype(ℰ, space, T)
    @inbounds Aᵢ = selectdim(A, D, ord)
    C = Array{CoefType,N-1}(undef, size(Aᵢ))
    if iszero(x)
        C .= zero(CoefType)
    else
        ωx = frequency(space)*x
        @inbounds C .= Aᵢ .* sin(_safe_mul(ωx, ord))
        @inbounds for j ∈ ord-1:-1:1
            C .+= selectdim(A, D, j) .* sin(_safe_mul(ωx, j))
        end
        C .= _safe_mul.(2, C)
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::SinFourier, ::SinFourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::SinFourier, ::Fourier, ::Type{T}, i, j, memo) where {T}
    if i == 0 && !iszero(x)
        x = value(ℰ)
        return convert(T, _safe_mul(2, sin(_safe_mul(frequency(domain)*x, j))))
    else
        return zero(T)
    end
end

# Multiplication

_mult_domain_indices(s::CosFourier) = _mult_domain_indices(Chebyshev(order(s)))
_isvalid(s::CosFourier, i::Int, j::Int) = _isvalid(Chebyshev(order(s)), i, j)

_mult_domain_indices(s::SinFourier) = -order(s):order(s)
_isvalid(s::SinFourier, i::Int, j::Int) = (0 < abs(j)) & (0 < abs(i-j) ≤ order(s))

# Norm

_getindex(weight::GeometricWeight, ::Union{CosFourier,SinFourier}, i::Int) = _safe_pow(rate(weight), i)

_getindex(weight::AlgebraicWeight, ::Union{CosFourier,SinFourier}, i::Int) = _safe_pow(1 + i, rate(weight))





_apply(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + _safe_mul(2, sum(abs, view(A, 2:length(A))))
function _apply(::Ell1{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= _safe_mul.(2, s) .+ abs.(selectdim(A, N, 1))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]), _safe_div(maximum(abs, view(A, 2:length(A))), 2))
function _apply_dual(::Ell1{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(_safe_div.(s, 2), abs.(selectdim(A, N, 1)))
    end
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = _safe_mul(2, ν) * s + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= _safe_mul(2, ν) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector{T}) where {T}
    ν = inv(rate(weight(X)))
    νⁱ½ = _safe_div(one(ν), 2)
    @inbounds s = abs(A[1]) * one(νⁱ½)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s = max(s, abs(A[i+1]) * νⁱ½)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ½ = _safe_div(one(ν), 2)
    CoefType = typeof(abs(zero(T))*νⁱ½)
    @inbounds A₀ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₀))
    s .= abs.(A₀)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s .= max.(s, abs.(selectdim(A, N, i+1)) .* νⁱ½)
    end
    return s
end

function _apply(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) * _getindex(weight(X), space, i)
        end
        @inbounds s = _safe_mul(2, s) + abs(A[1])
    end
    return s
end
function _apply(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i)
        end
        @inbounds s .= _safe_mul.(2, s) .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) / _getindex(weight(X), space, i))
        end
        @inbounds s = max(_safe_div(s, 2), abs(A[1]))
    end
    return s
end
function _apply_dual(X::Ell1{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(_safe_div.(s, 2), abs.(selectdim(A, N, 1)))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + _safe_mul(2, sum(abs2, view(A, 2:length(A)))))
function _apply(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(_safe_mul.(2, s) .+ abs2.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + _safe_div(sum(abs2, view(A, 2:length(A))), 2))
function _apply_dual(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(_safe_div.(s, 2) .+ abs2.(selectdim(A, N, 1)))
    return s
end

_apply(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]), _safe_mul(2, maximum(abs, view(A, 2:length(A)))))
function _apply(::EllInf{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)))
        end
        @inbounds s .= max.(_safe_mul.(2, s), abs.(selectdim(A, N, 1)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + _safe_div(sum(abs, view(A, 2:length(A))), 2)
function _apply_dual(::EllInf{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= _safe_div.(s, 2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ2 = _safe_mul(2, one(ν))
    @inbounds s = abs(A[1]) * one(νⁱ)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s = max(s, abs(A[i+1]) * νⁱ2)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ2 = _safe_mul(2, one(ν))
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
    ν = inv(rate(weight(X)))
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = s * ν + abs(A[i+1])
        end
        @inbounds s = s * _safe_div(ν, 2) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= s .* ν .+ abs.(selectdim(A, N, i+1))
        end
        @inbounds s .= s .* _safe_div(ν, 2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s = max(s, abs(A[i+1]) * _getindex(weight(X), space, i))
        end
        @inbounds s = max(_safe_mul(2, s), abs(A[1]))
    end
    return s
end
function _apply(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))*_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) .* _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .= max.(s, abs.(selectdim(A, N, i+1)) .* _getindex(weight(X), space, i))
        end
        @inbounds s .= max.(_safe_mul.(2, s), abs.(selectdim(A, N, 1)))
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractVector)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) / _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s += abs(A[i+1]) / _getindex(weight(X), space, i)
        end
        @inbounds s = _safe_div(s, 2) + abs(A[1])
    end
    return s
end
function _apply_dual(X::EllInf{<:AlgebraicWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T))/_getindex(weight(X), space, 0))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ) ./ _getindex(weight(X), space, ord)
    if ord > 0
        @inbounds for i ∈ ord-1:-1:1
            s .+= abs.(selectdim(A, N, i+1)) ./ _getindex(weight(X), space, i)
        end
        @inbounds s .= _safe_div.(s, 2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end
