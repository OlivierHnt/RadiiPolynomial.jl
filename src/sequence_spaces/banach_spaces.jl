"""
    Weight

Abstract type for all weights.
"""
abstract type Weight end

Base.:(==)(w₁::Weight, w₂::Weight) = false
Base.min(w₁::NTuple{N,Weight}, w₂::NTuple{N,Weight}) where {N} = map(min, w₁, w₂)

_getindex(weight::NTuple{N,Weight}, s::TensorSpace{<:NTuple{N,BaseSpace}}, α::NTuple{N,Int}) where {N} =
    @inbounds _getindex(weight[1], s[1], α[1]) * _getindex(Base.tail(weight), Base.tail(s), Base.tail(α))
_getindex(weight::Tuple{Weight}, s::TensorSpace{<:Tuple{BaseSpace}}, α::Tuple{Int}) =
    @inbounds _getindex(weight[1], s[1], α[1])

"""
    IdentityWeight <: Weight

Identity weight.
"""
struct IdentityWeight <: Weight end

Base.:(==)(::IdentityWeight, ::IdentityWeight) = true
Base.min(::IdentityWeight, ::IdentityWeight) = IdentityWeight()
Base.min(::IdentityWeight, ::Weight) = IdentityWeight()
Base.min(::Weight, ::IdentityWeight) = IdentityWeight()

"""
    GeometricWeight{T<:Real} <: Weight

Geometric weight associated with a given `rate` satisfying `isfinite(rate) & (rate ≥ 1)`.

Field:
- `rate :: T`

See also: [`geometricweight`](@ref), [`IdentityWeight`](@ref),
[`AlgebraicWeight`](@ref), [`algebraicweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> w = GeometricWeight(1.0)
GeometricWeight(1.0)

julia> rate(w)
1.0
```
"""
struct GeometricWeight{T<:Real} <: Weight
    rate :: T
    function GeometricWeight{T}(rate::T) where {T<:Real}
        isfinite(inf(rate)) & (inf(rate) ≥ 1) || return throw(DomainError(rate, "rate must be finite and greater or equal to one"))
        return new{T}(rate)
    end
end

GeometricWeight(rate::T) where {T<:Real} = GeometricWeight{T}(rate)

rate(weight::GeometricWeight) = weight.rate

Base.:(==)(w₁::GeometricWeight, w₂::GeometricWeight) = _safe_isequal(rate(w₁), rate(w₂))
Base.min(w₁::GeometricWeight, w₂::GeometricWeight) = GeometricWeight(min(rate(w₁), rate(w₂)))

_getindex(weight::GeometricWeight, ::Taylor, i::Int) = rate(weight) ^ exact(i)

_getindex(weight::GeometricWeight, ::Fourier, i::Int) = rate(weight) ^ exact(abs(i))

_getindex(weight::GeometricWeight, ::Chebyshev, i::Int) = rate(weight) ^ exact(i)

_getindex(weight::GeometricWeight, ::Union{CosFourier,SinFourier}, i::Int) = rate(weight) ^ exact(i)

IntervalArithmetic.interval(::Type{T}, weight::GeometricWeight) where {T} = GeometricWeight(interval(T, rate(weight)))
IntervalArithmetic.interval(weight::GeometricWeight) = GeometricWeight(interval(rate(weight)))

"""
    AlgebraicWeight{T<:Real} <: Weight

Algebraic weight associated with a given `rate` satisfying `isfinite(rate) & (rate ≥ 0)`.

Field:
- `rate :: T`

See also: [`algebraicweight`](@ref), [`IdentityWeight`](@ref),
[`GeometricWeight`](@ref), [`geometricweight`](@ref) and [`BesselWeight`](@ref).

# Examples

```jldoctest
julia> w = AlgebraicWeight(1.0)
AlgebraicWeight(1.0)

julia> rate(w)
1.0
```
"""
struct AlgebraicWeight{T<:Real} <: Weight
    rate :: T
    function AlgebraicWeight{T}(rate::T) where {T<:Real}
        isfinite(inf(rate)) & (inf(rate) ≥ 0) || return throw(DomainError(rate, "rate must be finite and positive"))
        return new{T}(rate)
    end
end

AlgebraicWeight(rate::T) where {T<:Real} = AlgebraicWeight{T}(rate)

rate(weight::AlgebraicWeight) = weight.rate

Base.:(==)(w₁::AlgebraicWeight, w₂::AlgebraicWeight) = _safe_isequal(rate(w₁), rate(w₂))
Base.min(w₁::AlgebraicWeight, w₂::AlgebraicWeight) = AlgebraicWeight(min(rate(w₁), rate(w₂)))

_getindex(weight::AlgebraicWeight, ::Taylor, i::Int) = exact(1 + i) ^ rate(weight)

_getindex(weight::AlgebraicWeight, ::Fourier, i::Int) = exact(1 + abs(i)) ^ rate(weight)

_getindex(weight::AlgebraicWeight, ::Chebyshev, i::Int) = exact(1 + i) ^ rate(weight)

_getindex(weight::AlgebraicWeight, ::Union{CosFourier,SinFourier}, i::Int) = exact(1 + i) ^ rate(weight)

IntervalArithmetic.interval(::Type{T}, weight::AlgebraicWeight) where {T} = AlgebraicWeight(interval(T, rate(weight)))
IntervalArithmetic.interval(weight::AlgebraicWeight) = AlgebraicWeight(interval(rate(weight)))

"""
    BesselWeight{T<:Real} <: Weight

Bessel weight associated with a given `rate` satisfying `isfinite(rate) & (rate ≥ 0)`.

Field:
- `rate :: T`

See also: [`IdentityWeight`](@ref), [`GeometricWeight`](@ref), [`geometricweight`](@ref),
[`AlgebraicWeight`](@ref) and [`algebraicweight`](@ref).

# Examples

```jldoctest
julia> w = BesselWeight(1.0)
BesselWeight(1.0)

julia> rate(w)
1.0
```
"""
struct BesselWeight{T<:Real} <: Weight
    rate :: T
    function BesselWeight{T}(rate::T) where {T<:Real}
        isfinite(rate) & (rate ≥ 0) || return throw(DomainError(rate, "rate must be finite and positive"))
        return new{T}(rate)
    end
end

BesselWeight(rate::T) where {T<:Real} = BesselWeight{T}(rate)

rate(weight::BesselWeight) = weight.rate

Base.:(==)(w₁::BesselWeight, w₂::BesselWeight) = _safe_isequal(rate(w₁), rate(w₂))
Base.min(w₁::BesselWeight, w₂::BesselWeight) = BesselWeight(min(rate(w₁), rate(w₂)))

_getindex(weight::BesselWeight, ::TensorSpace{<:NTuple{N,Fourier}}, α::NTuple{N,Int}) where {N} =
    (one(rate(weight)) + mapreduce(abs2, +, α)) ^ rate(weight)
_getindex(weight::BesselWeight{<:Interval}, ::TensorSpace{<:NTuple{N,Fourier}}, α::NTuple{N,Int}) where {N} =
    (one(rate(weight)) + interval(mapreduce(abs2, +, α))) ^ rate(weight)

_getindex(weight::BesselWeight, ::Fourier, i::Int) = (one(rate(weight)) + i*i) ^ rate(weight)
_getindex(weight::BesselWeight{<:Interval}, ::Fourier, i::Int) = (one(rate(weight)) + interval(i*i)) ^ rate(weight)

IntervalArithmetic.interval(::Type{T}, weight::BesselWeight) where {T} = BesselWeight(interval(T, rate(weight)))
IntervalArithmetic.interval(weight::BesselWeight) = BesselWeight(interval(rate(weight)))

#

Base.min(w₁::AlgebraicWeight, w₂::GeometricWeight) = AlgebraicWeight(ifelse(isone(rate(w₂)), zero(rate(w₁)), rate(w₁)))
Base.min(w₁::GeometricWeight, w₂::AlgebraicWeight) = AlgebraicWeight(ifelse(isone(rate(w₁)), zero(rate(w₂)), rate(w₂)))

# show

Base.show(io::IO, ::MIME"text/plain", weight::Weight) = print(io, _prettystring(weight))

_prettystring(weight::Weight) = string(weight)
_prettystring(weight::GeometricWeight) = "GeometricWeight(" * string(rate(weight)) * ")"
_prettystring(weight::AlgebraicWeight) = "AlgebraicWeight(" * string(rate(weight)) * ")"
_prettystring(weight::BesselWeight) = "BesselWeight(" * string(rate(weight)) * ")"

#

"""
    BanachSpace

Abstract type for all Banach spaces.
"""
abstract type BanachSpace end

Base.:(==)(::BanachSpace, ::BanachSpace) = false
Base.intersect(s₁::BanachSpace, s₂::BanachSpace) = throw(MethodError(intersect, (s₁, s₂)))

"""
    Ell1{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^1`` space.

Field:
- `weight :: T`

Constructors:
- `Ell1(::Weight)`
- `Ell1(::Tuple{Vararg{Weight}})`
- `Ell1()`: equivalent to `Ell1(IdentityWeight())`
- `Ell1(weight::Weight...)`: equivalent to `Ell1(weight)`

Unicode alias [`ℓ¹`](@ref) can be typed by `\\ell<tab>\\^1<tab>` in the Julia REPL
and in many editors.

See also: [`Ell2`](@ref) and [`EllInf`](@ref).

# Examples

```jldoctest
julia> Ell1()
ℓ¹()

julia> Ell1(GeometricWeight(1.0))
ℓ¹(GeometricWeight(1.0))

julia> Ell1(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
struct Ell1{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    Ell1{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    Ell1{Tuple{}}(::Tuple{}) = throw(ArgumentError("Ell1 is only defined for at least one Weight"))
end

Ell1(weight::T) where {T<:Weight} = Ell1{T}(weight)
Ell1(weight::T) where {T<:Tuple{Vararg{Weight}}} = Ell1{T}(weight)
Ell1() = Ell1{IdentityWeight}(IdentityWeight())
Ell1(weight::Weight...) = Ell1(weight)

weight(X::Ell1) = X.weight
rate(X::Ell1{<:Weight}) = rate(weight(X))
rate(X::Ell1{<:Tuple{Vararg{Weight}}}) = rate.(weight(X))

Base.:(==)(X₁::Ell1, X₂::Ell1) = weight(X₁) == weight(X₂)
Base.intersect(X₁::Ell1, X₂::Ell1) = Ell1(min(weight(X₁), weight(X₂)))

IntervalArithmetic.interval(::Type{T}, X::Ell1{<:Weight}) where {T} = Ell1(interval(T, weight(X)))
IntervalArithmetic.interval(X::Ell1{<:Weight}) = Ell1(interval(weight(X)))
IntervalArithmetic.interval(::Type{T}, X::Ell1{<:Tuple{Vararg{Weight}}}) where {T} = Ell1(map(w -> interval(T, w), weight(X)))
IntervalArithmetic.interval(X::Ell1{<:Tuple{Vararg{Weight}}}) = Ell1(map(w -> interval(w), weight(X)))

"""
    ℓ¹(::Weight)
    ℓ¹(::Tuple{Vararg{Weight}})
    ℓ¹()
    ℓ¹(::Weight...)

Unicode alias of [`Ell1`](@ref) representing the weighted ``\\ell^1`` space.

# Examples

```jldoctest
julia> ℓ¹()
ℓ¹()

julia> ℓ¹(GeometricWeight(1.0))
ℓ¹(GeometricWeight(1.0))

julia> ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
const ℓ¹ = Ell1

"""
    Ell2{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^2`` space.

Field:
- `weight :: T`

Constructors:
- `Ell2(::Weight)`
- `Ell2(::Tuple{Vararg{Weight}})`
- `Ell2()`: equivalent to `Ell2(IdentityWeight())`
- `Ell2(weight::Weight...)`: equivalent to `Ell2(weight)`

Unicode alias [`ℓ²`](@ref) can be typed by `\\ell<tab>\\^2<tab>` in the Julia REPL
and in many editors.

See also: [`Ell1`](@ref) and [`EllInf`](@ref).

# Examples

```jldoctest
julia> Ell2()
ℓ²()

julia> Ell2(BesselWeight(1.0))
ℓ²(BesselWeight(1.0))

julia> Ell2(BesselWeight(1.0), GeometricWeight(2.0))
ℓ²(BesselWeight(1.0), GeometricWeight(2.0))
```
"""
struct Ell2{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    Ell2{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    Ell2{Tuple{}}(::Tuple{}) = throw(ArgumentError("Ell2 is only defined for at least one Weight"))
end

Ell2(weight::T) where {T<:Weight} = Ell2{T}(weight)
Ell2(weight::T) where {T<:Tuple{Vararg{Weight}}} = Ell2{T}(weight)
Ell2() = Ell2{IdentityWeight}(IdentityWeight())
Ell2(weight::Weight...) = Ell2(weight)

weight(X::Ell2) = X.weight
rate(X::Ell2{<:Weight}) = rate(weight(X))
rate(X::Ell2{<:Tuple{Vararg{Weight}}}) = rate.(weight(X))

Base.:(==)(X₁::Ell2, X₂::Ell2) = weight(X₁) == weight(X₂)
Base.intersect(X₁::Ell2, X₂::Ell2) = Ell2(min(weight(X₁), weight(X₂)))

IntervalArithmetic.interval(::Type{T}, X::Ell2{<:Weight}) where {T} = Ell2(interval(T, weight(X)))
IntervalArithmetic.interval(X::Ell2{<:Weight}) = Ell2(interval(weight(X)))
IntervalArithmetic.interval(::Type{T}, X::Ell2{<:Tuple{Vararg{Weight}}}) where {T} = Ell2(map(w -> interval(T, w), weight(X)))
IntervalArithmetic.interval(X::Ell2{<:Tuple{Vararg{Weight}}}) = Ell2(map(w -> interval(w), weight(X)))

"""
    ℓ²(::Weight)
    ℓ²(::Tuple{Vararg{Weight}})
    ℓ²()
    ℓ²(::Weight...)

Unicode alias of [`Ell2`](@ref) representing the weighted ``\\ell^2`` space.

# Examples

```jldoctest
julia> ℓ²()
ℓ²()

julia> ℓ²(BesselWeight(1.0))
ℓ²(BesselWeight(1.0))

julia> ℓ²(BesselWeight(1.0), GeometricWeight(2.0))
ℓ²(BesselWeight(1.0), GeometricWeight(2.0))
```
"""
const ℓ² = Ell2

"""
    EllInf{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace

Weighted ``\\ell^\\infty`` space.

Field:
- `weight :: T`

Constructors:
- `EllInf(::Weight)`
- `EllInf(::Tuple{Vararg{Weight}})`
- `EllInf()`: equivalent to `EllInf(IdentityWeight())`
- `EllInf(weight::Weight...)`: equivalent to `EllInf(weight)`

Unicode alias [`ℓ∞`](@ref) can be typed by `\\ell<tab>\\infty<tab>` in the Julia REPL
and in many editors.

See also: [`Ell1`](@ref) and [`Ell2`](@ref).

# Examples

```jldoctest
julia> EllInf()
ℓ∞()

julia> EllInf(GeometricWeight(1.0))
ℓ∞(GeometricWeight(1.0))

julia> EllInf(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ∞(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
struct EllInf{T<:Union{Weight,Tuple{Vararg{Weight}}}} <: BanachSpace
    weight :: T
    EllInf{T}(weight::T) where {T<:Union{Weight,Tuple{Vararg{Weight}}}} = new{T}(weight)
    EllInf{Tuple{}}(::Tuple{}) = throw(ArgumentError("EllInf is only defined for at least one Weight"))
end

EllInf(weight::T) where {T<:Weight} = EllInf{T}(weight)
EllInf(weight::T) where {T<:Tuple{Vararg{Weight}}} = EllInf{T}(weight)
EllInf() = EllInf{IdentityWeight}(IdentityWeight())
EllInf(weight::Weight...) = EllInf(weight)

weight(X::EllInf) = X.weight
rate(X::EllInf{<:Weight}) = rate(weight(X))
rate(X::EllInf{<:Tuple{Vararg{Weight}}}) = rate.(weight(X))

Base.:(==)(X₁::EllInf, X₂::EllInf) = weight(X₁) == weight(X₂)
Base.intersect(X₁::EllInf, X₂::EllInf) = EllInf(min(weight(X₁), weight(X₂)))

IntervalArithmetic.interval(::Type{T}, X::EllInf{<:Weight}) where {T} = EllInf(interval(T, weight(X)))
IntervalArithmetic.interval(X::EllInf{<:Weight}) = EllInf(interval(weight(X)))
IntervalArithmetic.interval(::Type{T}, X::EllInf{<:Tuple{Vararg{Weight}}}) where {T} = EllInf(map(w -> interval(T, w), weight(X)))
IntervalArithmetic.interval(X::EllInf{<:Tuple{Vararg{Weight}}}) = EllInf(map(w -> interval(w), weight(X)))

"""
    ℓ∞(::Weight)
    ℓ∞(::Tuple{Vararg{Weight}})
    ℓ∞()
    ℓ∞(::Weight...)

Unicode alias of [`EllInf`](@ref) representing the weighted ``\\ell^\\infty`` space.

# Examples

```jldoctest
julia> ℓ∞()
ℓ∞()

julia> ℓ∞(GeometricWeight(1.0))
ℓ∞(GeometricWeight(1.0))

julia> ℓ∞(GeometricWeight(1.0), AlgebraicWeight(2.0))
ℓ∞(GeometricWeight(1.0), AlgebraicWeight(2.0))
```
"""
const ℓ∞ = EllInf

# normed cartesian space

"""
    NormedCartesianSpace{T<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},S<:BanachSpace} <: BanachSpace

Cartesian Banach space.

Fields:
- `inner :: T`
- `outer :: S`

See also: [`Ell1`](@ref), [`Ell2`](@ref) and [`EllInf`](@ref).

# Examples

```jldoctest
julia> NormedCartesianSpace(Ell1(), EllInf())
NormedCartesianSpace(ℓ¹(), ℓ∞())

julia> NormedCartesianSpace((Ell1(), Ell2()), EllInf())
NormedCartesianSpace((ℓ¹(), ℓ²()), ℓ∞())
```
"""
struct NormedCartesianSpace{T<:Union{BanachSpace,Tuple{Vararg{BanachSpace}}},S<:BanachSpace} <: BanachSpace
    inner :: T
    outer :: S
end

# show

Base.show(io::IO, ::MIME"text/plain", X::BanachSpace) = print(io, _prettystring(X))

_prettystring(X::BanachSpace) = string(X)

_prettystring(::Ell1{IdentityWeight}) = "ℓ¹()"
_prettystring(X::Ell1{<:Weight}) = "ℓ¹(" * _prettystring(weight(X)) * ")"
_prettystring(X::Ell1{<:Tuple{Vararg{Weight}}}) = "ℓ¹(" * mapreduce(_prettystring, (x, y) -> x * ", " * y, weight(X)) * ")"

_prettystring(::Ell2{IdentityWeight}) = "ℓ²()"
_prettystring(X::Ell2{<:Weight}) = "ℓ²(" * _prettystring(weight(X)) * ")"
_prettystring(X::Ell2{<:Tuple{Vararg{Weight}}}) = "ℓ²(" * mapreduce(_prettystring, (x, y) -> x * ", " * y, weight(X)) * ")"

_prettystring(::EllInf{IdentityWeight}) = "ℓ∞()"
_prettystring(X::EllInf{<:Weight}) = "ℓ∞(" * _prettystring(weight(X)) * ")"
_prettystring(X::EllInf{<:Tuple{Vararg{Weight}}}) = "ℓ∞(" * mapreduce(_prettystring, (x, y) -> x * ", " * y, weight(X)) * ")"

_prettystring(X::NormedCartesianSpace{<:BanachSpace}) =
    "NormedCartesianSpace(" * _prettystring(X.inner) * ", " *  _prettystring(X.outer) * ")"

_prettystring(X::NormedCartesianSpace{<:Tuple{Vararg{BanachSpace}}}) =
    "NormedCartesianSpace((" * mapreduce(_prettystring, (x, y) -> x * ", " * y, X.inner) * "), " *  _prettystring(X.outer) * ")"
