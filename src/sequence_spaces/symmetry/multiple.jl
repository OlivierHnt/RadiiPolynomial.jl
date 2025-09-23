# struct NFourier{T<:Real} <: SymBaseSpace
#     order :: Int
#     frequency :: T
#     multiple :: Int
#     function NFourier{T}(order::Int, frequency::T, multiple::Int=1) where {T<:Real}
#         (multiple > 0) & (order ≥ 0) & (inf(frequency) ≥ 0) || return throw(DomainError((order, frequency), "NFourier is only defined for positive orders and frequencies"))
#         return new{T}(order-(order % multiple), frequency, multiple)
#     end
# end

# NFourier(order::Int, frequency::T, multiple::Int=1) where {T<:Real} = NFourier{T}(order, frequency, multiple)

# order(s::NFourier) = s.order

# frequency(s::NFourier) = s.frequency

# multiple(s::Fourier) = s.multiple

# Base.:(==)(s₁::NFourier, s₂::NFourier) = _safe_isequal(s₁.frequency, s₂.frequency) & (s₁.order == s₂.order) & (s₁.multiple == s₂.multiple)
# Base.issubset(s₁::NFourier, s₂::NFourier) = _safe_isequal(s₁.frequency, s₂.frequency) & (s₁.order ≤ s₂.order) & iszero(s₁.multiple % s₂.multiple)
# function Base.intersect(s₁::NFourier{T}, s₂::NFourier{S}) where {T<:Real,S<:Real}
#     _safe_isequal(s₁.frequency, s₂.frequency) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $(s₁.frequency), s₂ has frequency $(s₂.frequency)"))
#     R = promote_type(T, S)
#     t = iszero(s₁.multiple % s₂.multiple) | iszero(s₂.multiple % s₁.multiple)
#     ord = ifelse(t, min(s₁.order, s₂.order), 0)
#     mult = ifelse(t, max(s₁.multiple, s₂.multiple), 1)
#     return NFourier(ord, convert(R, s₁.frequency), mult)
# end
# function Base.union(s₁::NFourier{T}, s₂::NFourier{S}) where {T<:Real,S<:Real}
#     _safe_isequal(s₁.frequency, s₂.frequency) || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $(s₁.frequency), s₂ has frequency $(s₂.frequency)"))
#     R = promote_type(T, S)
#     return NFourier(max(s₁.order, s₂.order), convert(R, s₁.frequency), gcd(s₁.multiple, s₂.multiple))
# end

# dimension(s::NFourier) = 2(s.order ÷ s.multiple) + 1
# _firstindex(s::NFourier) = -s.order
# _lastindex(s::NFourier) = s.order
# indices(s::NFourier) = -s.order:s.multiple:s.order

# __checkbounds_indices(α::Int, s::NFourier) = iszero(α % s.multiple) & (-s.order ≤ α ≤ s.order)

# _compatible_space_with_constant_index(s::NFourier) = s
# _findindex_constant(::NFourier) = 0

# _findposition(i::Int, s::NFourier) = (i + s.order) ÷ s.multiple + 1
# _findposition(u::AbstractRange{Int}, s::NFourier) = (u .+ s.order) .÷ s.multiple .+ 1
# _findposition(u::AbstractVector{Int}, s::NFourier) = map(i -> _findposition(i, s), u)
# _findposition(c::Colon, ::NFourier) = c

# # promotion

# Base.convert(::Type{NFourier{T}}, s::NFourier) where {T<:Real} =
#     NFourier{T}(s.order, convert(T, s.frequency), s.multiple)

# Base.promote_rule(::Type{NFourier{T}}, ::Type{NFourier{S}}) where {T<:Real,S<:Real} =
#     NFourier{promote_type(T, S)}

# #

# _iscompatible(s₁::NFourier, s₂::NFourier) = _safe_isequal(frequency(s₁), frequency(s₂)) & (multiple(s₁) == multiple(s₂))

# #

# IntervalArithmetic.interval(::Type{T}, s::NFourier) where {T} = NFourier(order(s), interval(T, frequency(s)))
# IntervalArithmetic.interval(s::NFourier) = NFourier(order(s), interval(frequency(s)), multiple(s))

# # show

# _prettystring(s::NFourier) = "NFourier(" * string(order(s)) * ", " * string(frequency(s)) * ", " * string(multiple(s)) * ")"

# #

# for (f, g) ∈ ((:float, :_float_space), (:big, :_big_space))
#     @eval $g(s::NFourier) = NFourier(order(s), $f(frequency(s)), multiple(s))
# end








###

# TODO: CosNFourier, SinNFourier
# however, this means copy pasting a lot of redundant code. Maybe it's better to create some Cos(NFourier) / Sin(NFourier)
# or some Multiple(CosFourier), Multiple(SinFourier)




# struct CosFourier{T<:Real} <: SymBaseSpace
#     space :: Fourier{T}
#     CosFourier{T}(space::Fourier{T}) where {T<:Real} = new{T}(space)
# end
# CosFourier(space::Fourier{T}) where {T<:Real} = CosFourier{T}(space)
# CosFourier{T}(order::Int, frequency::T) where {T<:Real} = CosFourier(Fourier{T}(order, frequency))
# CosFourier(order::Int, frequency::Real, multiple::Int=1) = CosFourier(Fourier(order, frequency, multiple))

# multiple(s::CosFourier) = multiple(desymmetrize(s))

# Base.:(==)(s₁::CosFourier, s₂::CosFourier) = desymmetrize(s₁) == desymmetrize(s₂)
# Base.issubset(s₁::CosFourier, s₂::CosFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
# Base.intersect(s₁::CosFourier, s₂::CosFourier) = CosFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
# Base.union(s₁::CosFourier, s₂::CosFourier) = CosFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

# indices(s::CosFourier) = 0:multiple(s):order(s)

# _compatible_space_with_constant_index(s::CosFourier) = s
# _findindex_constant(::CosFourier) = 0

# _findposition(i::Int, s::CosFourier) = i ÷ multiple(s) + 1
# _findposition(u::AbstractRange{Int}, s::CosFourier) = u .÷ multiple(s) .+ 1
# _findposition(u::AbstractVector{Int}, s::CosFourier) = map(i -> _findposition(i, s), u)
# _findposition(c::Colon, ::CosFourier) = c

# Base.convert(::Type{T}, s::T) where {T<:CosFourier} = s
# Base.convert(::Type{CosFourier{T}}, s::CosFourier) where {T<:Real} =
#     CosFourier{T}(order(s), convert(T, frequency(s)))

# Base.promote_rule(::Type{T}, ::Type{T}) where {T<:CosFourier} = T
# Base.promote_rule(::Type{CosFourier{T}}, ::Type{CosFourier{S}}) where {T<:Real,S<:Real} =
#     CosFourier{promote_type(T, S)}

# _iscompatible(s₁::CosFourier, s₂::CosFourier) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂))

# IntervalArithmetic.interval(::Type{T}, s::CosFourier) where {T} = CosFourier(interval(T, desymmetrize(s)))
# IntervalArithmetic.interval(s::CosFourier) = CosFourier(interval(desymmetrize(s)))

# _prettystring(s::CosFourier) = "CosFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

# _zero_space(::Type{CosFourier{T}}) where {T<:Real} = CosFourier(0, one(T))

# _float_space(s::CosFourier) = CosFourier(_float_space(desymmetrize(s)))
# _big_space(s::CosFourier) = CosFourier(_big_space(desymmetrize(s)))
# _node(::CosFourier, j, N) = 2π*j[1]/N



# struct SinFourier{T<:Real} <: SymBaseSpace
#     space :: Fourier{T}
#     function SinFourier{T}(space::Fourier{T}) where {T<:Real}
#         order(space) < 1 && return throw(DomainError(order, "SinFourier is only defined for orders greater or equal to 1"))
#         return new{T}(space)
#     end
# end
# SinFourier(space::Fourier{T}) where {T<:Real} = SinFourier{T}(space)
# SinFourier{T}(order::Int, frequency::T) where {T<:Real} = SinFourier(Fourier{T}(order, frequency))
# SinFourier(order::Int, frequency::Real, multiple::Int=1) = SinFourier(Fourier(order, frequency, multiple)) # may fail since it can normalize to order 0

# multiple(s::SinFourier) = multiple(desymmetrize(s))

# Base.:(==)(s₁::SinFourier, s₂::SinFourier) = desymmetrize(s₁) == desymmetrize(s₂)
# Base.issubset(s₁::SinFourier, s₂::SinFourier) = issubset(desymmetrize(s₁), desymmetrize(s₂))
# Base.intersect(s₁::SinFourier, s₂::SinFourier) = SinFourier(intersect(desymmetrize(s₁), desymmetrize(s₂)))
# Base.union(s₁::SinFourier, s₂::SinFourier) = SinFourier(union(desymmetrize(s₁), desymmetrize(s₂)))

# indices(s::SinFourier) = multiple(s):multiple(s):order(s)

# _compatible_space_with_constant_index(s::SinFourier) = desymmetrize(s)

# _findposition(i::Int, s::SinFourier) = i ÷ multiple(s)
# _findposition(u::AbstractRange{Int}, s::SinFourier) = u .÷ multiple(s)
# _findposition(u::AbstractVector{Int}, s::SinFourier) = map(i -> _findposition(i, s), u)
# _findposition(c::Colon, ::SinFourier) = c

# Base.convert(::Type{T}, s::T) where {T<:SinFourier} = s
# Base.convert(::Type{SinFourier{T}}, s::SinFourier) where {T<:Real} =
#     SinFourier{T}(order(s), convert(T, frequency(s)))

# Base.promote_rule(::Type{T}, ::Type{T}) where {T<:SinFourier} = T
# Base.promote_rule(::Type{SinFourier{T}}, ::Type{SinFourier{S}}) where {T<:Real,S<:Real} =
#     SinFourier{promote_type(T, S)}

# _iscompatible(s₁::SinFourier, s₂::SinFourier) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂))

# IntervalArithmetic.interval(::Type{T}, s::SinFourier) where {T} = SinFourier(interval(T, desymmetrize(s)))
# IntervalArithmetic.interval(s::SinFourier) = SinFourier(interval(desymmetrize(s)))

# _prettystring(s::SinFourier) = "SinFourier(" * string(order(s)) * ", " * string(frequency(s)) * ")"

# _zero_space(::Type{SinFourier{T}}) where {T<:Real} = SinFourier(1, one(T))

# _float_space(s::SinFourier) = SinFourier(_float_space(desymmetrize(s)))
# _big_space(s::SinFourier) = SinFourier(_big_space(desymmetrize(s)))
# _node(::SinFourier, j, N) = 2π*j[1]/N
