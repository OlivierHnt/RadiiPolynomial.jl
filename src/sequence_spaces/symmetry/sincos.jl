# conjugacy_symmetry!(a::Sequence{<:VectorSpace,<:AbstractVector{<:Real}}) = a

conjugacy_symmetry!(a::Sequence) = _conjugacy_symmetry!(a)

_conjugacy_symmetry!(::Sequence) = throw(DomainError) # TODO: lift restriction

function _conjugacy_symmetry!(a::Sequence{ParameterSpace})
    @inbounds a[1] = real(a[1])
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:Fourier})
    a .= (a .+ conj.(reverse(a))) ./ 2
    return a
end

function _conjugacy_symmetry!(a::Sequence{<:TensorSpace{<:Tuple{Vararg{Fourier}}}})
    A = _no_alloc_reshape(coefficients(a), dimensions(space(a)))
    A .= (A .+ conj.(reverse(A))) ./ 2
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

_compatible_space_with_constant_index(s::CosFourier) = s
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

IntervalArithmetic.interval(::Type{T}, s::CosFourier) where {T} = CosFourier(interval(T, desymmetrize(s)))
IntervalArithmetic.interval(s::CosFourier) = CosFourier(interval(desymmetrize(s)))

_prettystring(s::CosFourier) = "CosFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

_zero_space(::Type{CosFourier{T}}) where {T<:Real} = CosFourier(0, one(T))

_float_space(s::CosFourier) = CosFourier(_float_space(desymmetrize(s)))
_big_space(s::CosFourier) = CosFourier(_big_space(desymmetrize(s)))
_node(::CosFourier, j, N) = 2π*j[1]/N



struct SinFourier{T<:Real} <: SymBaseSpace
    space :: Fourier{T}
    function SinFourier{T}(space::Fourier{T}) where {T<:Real}
        order(space) < 1 && return throw(DomainError(order, "SinFourier is only defined for orders greater or equal to 1"))
        return new{T}(space)
    end
end
SinFourier(space::Fourier{T}) where {T<:Real} = SinFourier{T}(space)
SinFourier{T}(order::Int, frequency::T) where {T<:Real} = SinFourier(Fourier{T}(order, frequency))
SinFourier(order::Int, frequency::Real) = SinFourier(Fourier(order, frequency)) # may fail since it can normalize to order 0

Base.:(==)(s₁::SinFourier, s₂::SinFourier) = desymmetrize(s₁) == desymmetrize(s₂)
Base.issubset(s₁::SinFourier, s₂::SinFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
Base.intersect(s₁::SinFourier, s₂::SinFourier) = SinFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
Base.union(s₁::SinFourier, s₂::SinFourier) = SinFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

indices(s::SinFourier) = 1:order(s)

_compatible_space_with_constant_index(s::SinFourier) = desymmetrize(s)

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

IntervalArithmetic.interval(::Type{T}, s::SinFourier) where {T} = SinFourier(interval(T, desymmetrize(s)))
IntervalArithmetic.interval(s::SinFourier) = SinFourier(interval(desymmetrize(s)))

_prettystring(s::SinFourier) = "SinFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

_zero_space(::Type{SinFourier{T}}) where {T<:Real} = SinFourier(1, one(T))

_float_space(s::SinFourier) = SinFourier(_float_space(desymmetrize(s)))
_big_space(s::SinFourier) = SinFourier(_big_space(desymmetrize(s)))
_node(::SinFourier, j, N) = 2π*j[1]/N





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

_convolution_indices(s₁::CosFourier, s₂::CosFourier, i::Int) =
    _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)

_symmetry_action(::CosFourier, ::Int, ::Int) = 1
_symmetry_action(::CosFourier, ::Int) = 1
_inverse_symmetry_action(::CosFourier, ::Int) = 1

_extract_valid_index(::CosFourier, i::Int, j::Int) = abs(i-j)
_extract_valid_index(::CosFourier, i::Int) = abs(i)



_convolution_indices(s₁::SinFourier, s₂::SinFourier, i::Int) =
    _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)

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
    _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)
_convolution_indices(s₁::SinFourier, s₂::CosFourier, i::Int) =
    _convolution_indices(desymmetrize(s₁), desymmetrize(s₂), i)

# FFT

_dft_dimension(s::CosFourier) = 2order(s)+1

_preprocess!(C::AbstractVector, space::CosFourier) = _preprocess!(C, Chebyshev(order(space)))
_preprocess!(C::AbstractArray, space::CosFourier, ::Val{D}) where {D} = _preprocess!(C, Chebyshev(order(space)), Val(D))

_ifft_get_index(n, space::CosFourier) = _ifft_get_index(n, Chebyshev(order(space)))
_postprocess!(C::AbstractVector, space::CosFourier) = _postprocess!(C, Chebyshev(order(space)))
_postprocess!(C::AbstractArray, space::CosFourier, ::Val{D}) where {D} = _postprocess!(C, Chebyshev(order(space)), Val(D))



_dft_dimension(s::SinFourier) = 2order(s)+1

function _preprocess!(C::AbstractVector, space::SinFourier)
    len = length(C)
    ord = order(space)
    view(C, 2:ord+1) .= view(C, 1:ord) .* complex(ExactReal(false), ExactReal(true))
    C[1] = zero(eltype(C))
    view(C, len:-1:len+1-ord) .= .- view(C, 2:ord+1)
    return C
end

function _preprocess!(C::AbstractArray, space::SinFourier, ::Val{D}) where {D}
    len = size(C, D)
    ord = order(space)
    selectdim(C, D, 2:ord+1) .= selectdim(C, D, 1:ord) .* complex(ExactReal(false), ExactReal(true))
    selectdim(C, D, 1) .= zero(eltype(C))
    selectdim(C, D, len:-1:len+1-ord) .= .- selectdim(C, D, 2:ord+1)
    return C
end

_ifft_get_index(n, space::SinFourier) = 1:min(n÷2, dimension(space)), 1:min(n÷2, dimension(space))

function _postprocess!(C::AbstractVector, ::SinFourier)
    ord = length(C) ÷ 2
    view(C, 1:ord) .= -complex(ExactReal(false), ExactReal(true)) .* view(C, 2:ord+1)
    return C
end

function _postprocess!(C::AbstractArray, ::SinFourier, ::Val{D}) where {D}
    ord = size(C, D) ÷ 2
    selectdim(C, D, 1:ord) .= -complex(ExactReal(false), ExactReal(true)) .* selectdim(C, D, 2:ord+1)
    return C
end

# used for Projection

_infer_domain(𝒟::Derivative, s::CosFourier) = image(Derivative(order(𝒟)), s)
_infer_domain(𝒟::Derivative, s::SinFourier) = image(Derivative(order(𝒟)), s)

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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
            selectdim(C, 1, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j+1)
        end
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
            selectdim(C, D, j+1) .= iⁿωⁿjⁿ_real .* selectdim(A, D, j+1)
        end
        return C
    else
        C = Array{CoefType,N}(undef, ntuple(i -> size(A, i) - ifelse(i == D, 1, 0), Val(N)))
        ord = order(space)
        ω = one(real(T))*frequency(space)
        iⁿ_real = ifelse((n+1)%4 < 2, 1, -1) # ((n+1)%4 == 0) | ((n+1)%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
        ωⁿjⁿ = (one(real(T)) * frequency(domain) * ExactReal(j)) ^ ExactReal(n)
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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
            selectdim(C, 1, j) .= iⁿωⁿjⁿ_real .* selectdim(A, 1, j)
        end
    else
        ord = order(space)
        ω = one(real(eltype(A)))*frequency(space)
        @inbounds selectdim(C, 1, 1) .= zero(T)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j ∈ 1:ord
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
            iⁿωⁿjⁿ_real = ExactReal(iⁿ_real) * (ω * ExactReal(j)) ^ ExactReal(n)
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
        ωⁿjⁿ = (one(real(T)) * frequency(domain) * ExactReal(j)) ^ ExactReal(n)
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
            @inbounds c[0] *= cos(ωx * ExactReal(ord))
            @inbounds for j ∈ ord-1:-1:1
                c[0] += a[j] * cos(ωx * ExactReal(j))
            end
        end
        @inbounds c[0] = ExactReal(2) * c[0] + a[0]
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
            C .*= cos(ωx * ExactReal(ord))
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, 1, j+1) .* cos(ωx * ExactReal(j))
            end
        end
        @inbounds C .= ExactReal(2) .* C .+ selectdim(A, 1, 1)
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
            C .*= cos(ωx * ExactReal(ord))
            @inbounds for j ∈ ord-1:-1:1
                C .+= selectdim(A, D, j+1) .* cos(ωx * ExactReal(j))
            end
        end
        @inbounds C .= ExactReal(2) .* C .+ selectdim(A, D, 1)
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
            return convert(T, ExactReal(2))
        else
            return convert(T, ExactReal(2) * cos(frequency(domain) * x * ExactReal(j)))
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
        @inbounds c[0] = a[ord] * sin(ωx * ExactReal(ord))
        @inbounds for j ∈ ord-1:-1:1
            c[0] += a[j] * sin(ωx * ExactReal(j))
        end
        @inbounds c[0] *= ExactReal(2)
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
        @inbounds C .= selectdim(A, 1, ord) .* sin(ωx * ExactReal(ord))
        @inbounds for j ∈ ord-1:-1:1
            C .+= selectdim(A, 1, j) .* sin(ωx * ExactReal(j))
        end
        C .*= ExactReal(2)
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
        @inbounds C .= Aᵢ .* sin(ωx * ExactReal(ord))
        @inbounds for j ∈ ord-1:-1:1
            C .+= selectdim(A, D, j) .* sin(ωx * ExactReal(j))
        end
        C .*= ExactReal(2)
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::SinFourier, ::SinFourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::SinFourier, ::Fourier, ::Type{T}, i, j, memo) where {T}
    if i == 0 && !iszero(x)
        x = value(ℰ)
        return convert(T, ExactReal(2) * sin(frequency(domain) * x * ExactReal(j)))
    else
        return zero(T)
    end
end

# Multiplication

_mult_domain_indices(s::CosFourier) = _mult_domain_indices(desymmetrize(s))
_isvalid(::CosFourier, s::CosFourier, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)
_isvalid(::SinFourier, s::CosFourier, i::Int, j::Int) = (0 < abs(j)) & _checkbounds_indices(abs(i-j), s)

_mult_domain_indices(s::SinFourier) = _mult_domain_indices(desymmetrize(s))
_isvalid(::SinFourier, s::SinFourier, i::Int, j::Int) = (0 < abs(j)) & _checkbounds_indices(abs(i-j), s)
_isvalid(::CosFourier, s::SinFourier, i::Int, j::Int) = _checkbounds_indices(abs(i-j), s)

# Norm

_getindex(weight::GeometricWeight, ::Union{CosFourier,SinFourier}, i::Int) = rate(weight) ^ ExactReal(i)

_getindex(weight::AlgebraicWeight, ::Union{CosFourier,SinFourier}, i::Int) = ExactReal(1 + i) ^ rate(weight)

function _linear_regression(::CosFourier, f, A)
    j = length(A)
    β, err = β_, err_ = __linear_regression(f, A, j)
    while err_ ≤ err && j ≥ 3
        j -= 1
        β, err = β_, err_
        β_, err_ = __linear_regression(f, A, j)
    end
    return β, err, j
end

function _linear_regression(::SinFourier, f, A)
    j = length(A)
    β, err = β_, err_ = __linear_regression(f, A, j)
    while err_ ≤ err && j ≥ 3
        j -= 1
        β, err = β_, err_
        β_, err_ = __linear_regression(f, A, j)
    end
    return β, err, j
end



_apply(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + ExactReal(2) * sum(abs, view(A, 2:length(A)))
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
        @inbounds s .= ExactReal(2) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]), maximum(abs, view(A, 2:length(A))) / ExactReal(2))
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
        @inbounds s .= max.(s ./ ExactReal(2), abs.(selectdim(A, N, 1)))
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
        @inbounds s = (ExactReal(2) * ν) * s + abs(A[1])
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
        @inbounds s .= (ExactReal(2) * ν) .* s .+ abs.(selectdim(A, N, 1))
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractVector{T}) where {T}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / ExactReal(2)
    @inbounds s = abs(A[1]) * one(νⁱ½)
    @inbounds for i ∈ 1:order(space)
        νⁱ½ *= ν
        s = max(s, abs(A[i+1]) * νⁱ½)
    end
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = inv(rate(weight(X)))
    νⁱ½ = one(ν) / ExactReal(2)
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
        @inbounds s = ExactReal(2) * s + abs(A[1])
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
        @inbounds s .=  ExactReal(2) .* s .+ abs.(selectdim(A, N, 1))
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
        @inbounds s = max(s / ExactReal(2), abs(A[1]))
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
        @inbounds s .= max.(s ./ ExactReal(2), abs.(selectdim(A, N, 1)))
    end
    return s
end

_apply(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) +  ExactReal(2) * sum(abs2, view(A, 2:length(A))))
function _apply(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(ExactReal(2) .* s .+ abs2.(selectdim(A, N, 1)))
    return s
end
_apply_dual(::Ell2{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds sqrt(abs2(A[1]) + sum(abs2, view(A, 2:length(A))) / ExactReal(2))
function _apply_dual(::Ell2{IdentityWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(sqrt(abs2(zero(T))))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord+1)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs2.(Aᵢ)
    for i ∈ ord-1:-1:1
        s .+= abs2.(selectdim(A, N, i+1))
    end
    @inbounds s .= sqrt.(s ./ ExactReal(2) .+ abs2.(selectdim(A, N, 1)))
    return s
end

_apply(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds max(abs(A[1]),  ExactReal(2) * maximum(abs, view(A, 2:length(A))))
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
        @inbounds s .= max.(ExactReal(2) .* s, abs.(selectdim(A, N, 1)))
    end
    return s
end
_apply_dual(::EllInf{IdentityWeight}, ::CosFourier, A::AbstractVector) =
    @inbounds abs(A[1]) + sum(abs, view(A, 2:length(A))) / ExactReal(2)
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
        @inbounds s .= s ./ ExactReal(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end

function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractVector)
    ν = rate(weight(X))
    νⁱ2 = ExactReal(2) * one(ν)
    @inbounds s = abs(A[1]) * one(νⁱ)
    @inbounds for i ∈ 1:order(space)
        νⁱ2 *= ν
        s = max(s, abs(A[i+1]) * νⁱ2)
    end
    return s
end
function _apply(X::EllInf{<:GeometricWeight}, space::CosFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    νⁱ2 = ExactReal(2) * one(ν)
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
        @inbounds s = s * (ν / ExactReal(2)) + abs(A[1])
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
        @inbounds s .= s .* (ν / ExactReal(2)) .+ abs.(selectdim(A, N, 1))
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
        @inbounds s = max(ExactReal(2) * s, abs(A[1]))
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
        @inbounds s .= max.(ExactReal(2) .* s, abs.(selectdim(A, N, 1)))
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
        @inbounds s = s / ExactReal(2) + abs(A[1])
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
        @inbounds s .= s ./ ExactReal(2) .+ abs.(selectdim(A, N, 1))
    end
    return s
end





_apply(::Ell1{IdentityWeight}, ::SinFourier, A::AbstractVector) = ExactReal(2) * sum(abs, A)
function _apply(::Ell1{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i))
    end
    s .*= ExactReal(2)
    return s
end
_apply_dual(::Ell1{IdentityWeight}, ::SinFourier, A::AbstractVector) = maximum(abs, A) / ExactReal(2)
function _apply_dual(::Ell1{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    s ./= ExactReal(2)
    return s
end

function _apply(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractVector)
    ν = rate(weight(X))
    ord = order(space)
    @inbounds s = abs(A[ord]) * one(ν)
    @inbounds for i ∈ ord-1:-1:1
        s = s * ν + abs(A[i])
    end
    s *= ExactReal(2) * ν
    return s
end
function _apply(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    ν = rate(weight(X))
    CoefType = typeof(abs(zero(T))*ν)
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= s .* ν .+ abs.(selectdim(A, N, i))
    end
    s .*= ExactReal(2) * ν
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractVector{T}) where {T}
    ν = νⁱ = inv(rate(weight(X)))
    @inbounds s = abs(A[1]) * νⁱ
    @inbounds for i ∈ 2:order(space)
        νⁱ *= ν
        s = max(s, abs(A[i]) * νⁱ)
    end
    s /= ExactReal(2)
    return s
end
function _apply_dual(X::Ell1{<:GeometricWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    ν = νⁱ = inv(rate(weight(X)))
    CoefType = typeof(abs(zero(T))*νⁱ)
    @inbounds A₁ = selectdim(A, N, 1)
    s = Array{CoefType,N-1}(undef, size(A₁))
    s .= abs.(A₁) .* νⁱ
    @inbounds for i ∈ 2:order(space)
        νⁱ *= ν
        s .= max.(s, abs.(selectdim(A, N, i)) .* νⁱ)
    end
    s ./= ExactReal(2)
    return s
end

_apply(::EllInf{IdentityWeight}, ::SinFourier, A::AbstractVector) = ExactReal(2) * maximum(abs, A)
function _apply(::EllInf{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .= max.(s, abs.(selectdim(A, N, i)))
    end
    s .*= ExactReal(2)
    return s
end
_apply_dual(::EllInf{IdentityWeight}, ::SinFourier, A::AbstractVector) = sum(abs, A) / ExactReal(2)
function _apply_dual(::EllInf{IdentityWeight}, space::SinFourier, A::AbstractArray{T,N}) where {T,N}
    CoefType = typeof(abs(zero(T)))
    ord = order(space)
    @inbounds Aᵢ = selectdim(A, N, ord)
    s = Array{CoefType,N-1}(undef, size(Aᵢ))
    s .= abs.(Aᵢ)
    @inbounds for i ∈ ord-1:-1:1
        s .+= abs.(selectdim(A, N, i))
    end
    s ./= ExactReal(2)
    return s
end
