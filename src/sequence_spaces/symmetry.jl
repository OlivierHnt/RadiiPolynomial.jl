struct IndexAction{N}
    matrix :: StaticArrays.SMatrix{N,N,Int}
end

IndexAction(a::AbstractMatrix{Int}) = IndexAction(StaticArrays.SMatrix{size(a)...,Int}(a))

(A::IndexAction{1})(k::Integer) = A.matrix[1] * k
(A::IndexAction{2})(k::NTuple{2,Integer}) =
    (A.matrix[1,1]*k[1] + A.matrix[1,2]*k[2],
     A.matrix[2,1]*k[1] + A.matrix[2,2]*k[2])
(A::IndexAction{3})(k::NTuple{3,Integer}) =
    (A.matrix[1,1]*k[1] + A.matrix[1,2]*k[2] + A.matrix[1,3]*k[3],
     A.matrix[2,1]*k[1] + A.matrix[2,2]*k[2] + A.matrix[2,3]*k[3],
     A.matrix[3,1]*k[1] + A.matrix[3,2]*k[2] + A.matrix[3,3]*k[3],)
function (A::IndexAction{N})(k::NTuple{N,Integer}) where {N}
    l = A.matrix * StaticArrays.SVector{N}(k)
    return ntuple(i -> l[i], Val(N))
end

Base.:*(A::IndexAction, B::IndexAction) = IndexAction(A.matrix * B.matrix)

Base.:(==)(A::IndexAction, B::IndexAction) = A.matrix == B.matrix

Base.hash(A::IndexAction, h::UInt) = hash(A.matrix, h)

#

struct CoefAction{N,T<:Number}
    amplitude :: T
    phase     :: StaticArrays.SVector{N,Rational{Int}} # factor of π
    CoefAction{N,T}(amplitude::T, phase::StaticArrays.SVector{N,Rational{Int}}) where {T<:Number,N} = new{N,T}(amplitude, mod.(phase, 2))
end

CoefAction(amplitude::T, phase::StaticArrays.SVector{N,Rational{Int}}) where {T<:Number,N} = CoefAction{N,T}(amplitude, phase)

CoefAction(amplitude::Number, phase::AbstractVector{Rational{Int}}) = CoefAction(amplitude, StaticArrays.SVector{length(phase),Rational{Int}}(phase))

(v::CoefAction{1,<:Interval})(k::Integer) = v.amplitude * cispi(interval(exact(v.phase[1]) * exact(k)))
(v::CoefAction{N,<:Interval})(k::NTuple{N,Integer}) where {N} = v.amplitude * cispi(interval(mapreduce(*, +, exact.(v.phase), exact.(k))))
(v::CoefAction)(k) = v.amplitude * cispi(mapreduce(*, +, v.phase, k))

Base.:*(v::CoefAction, w::CoefAction) = CoefAction(
    v.amplitude * w.amplitude,
    mod.(v.phase + w.phase, 2))

Base.:(==)(v::CoefAction, w::CoefAction) = (v.amplitude == w.amplitude) & (v.phase == w.phase)

Base.hash(v::CoefAction, h::UInt) = hash(v.amplitude, hash(v.phase, h))

#

struct GroupElement{N,T<:Number}
    index_action :: IndexAction{N}
    coef_action  :: CoefAction{N,T}
end

Base.:∘(g::GroupElement, h::GroupElement) = GroupElement(g.index_action * h.index_action, g.coef_action * h.coef_action)

Base.:(==)(g::GroupElement, h::GroupElement) = (g.index_action == h.index_action) & (g.coef_action == h.coef_action)
Base.hash(g::GroupElement, h::UInt) = hash(g.index_action, hash(g.coef_action, h))

#

struct Group{N,T<:Number}
    elements :: Set{GroupElement{N,T}}
    global function unsafe_group!(elements::Set{GroupElement{N,T}}) where {N,T<:Number}
        # modify in-place the input set of group elements until it is closed under composition
        changed = true
        while changed
            changed = false
            for g ∈ elements, h ∈ elements
                gh = g ∘ h
                if gh ∉ elements
                    push!(elements, gh)
                    changed = true
                end
            end
        end
        return new{N,T}(elements)
    end
end

Group(g::GroupElement{N,T}, h::GroupElement{N,T}...) where {N,T<:Number} = unsafe_group!(Set{GroupElement{N,T}}((g, h...)))

elements(g::Group) = g.elements

Base.:(==)(G₁::Group, G₂::Group) = elements(G₁) == elements(G₂)
Base.issubset(G₁::Group, G₂::Group) = issubset(elements(G₁), elements(G₂))
Base.intersect(G₁::Group, G₂::Group) = unsafe_group!(intersect(elements(G₁), elements(G₂)))
Base.union(G₁::Group, G₂::Group) = Group(elements(G₁)..., elements(G₂)...)

Base.hash(g::Group, h::UInt) = hash(g.elements, h)

_orbit(sym::Group{1}, k::T) where {T<:Integer} = Set{T}(g.index_action(k) for g ∈ elements(sym))
_orbit(sym::Group{N}, k::NTuple{N,T}) where {N,T<:Integer} = Set{NTuple{N,T}}(g.index_action(k) for g ∈ elements(sym))

function _orbit_representatives(sym::Group, inds) # slow
    sym_elements = elements(sym)
    T_k = eltype(inds)
    k_reps = Vector{T_k}(undef, length(inds))
    for (i, k) ∈ enumerate(inds)
        k_rep = k
        for g ∈ sym_elements
            k_g = g.index_action(k)
            if k_g > k_rep
                k_rep = k_g
            end
        end
        @inbounds k_reps[i] = k_rep
    end
    return k_reps
end

function _filter_valid_representatives(sym::Group, k_reps)
    sym_elements = elements(sym)
    T_k = eltype(k_reps)
    reps = T_k[]
    seen_reps = Set{T_k}()
    for k_rep ∈ k_reps
        k_rep in seen_reps && continue
        push!(seen_reps, k_rep)

        is_valid_orbit = true # if `k` maps to itself and has a conflicting phase, then it is not a valid orbit
        for g ∈ sym_elements
            if g.index_action(k_rep) == k_rep && g.coef_action(k_rep) != 1
                is_valid_orbit = false
                break
            end
        end
        if is_valid_orbit
            push!(reps, k_rep)
        end
    end
    return sort!(reps)
end

function _compute_action_map(sym::Group, inds, k_reps)
    sym_elements = elements(sym)
    return map(enumerate(inds)) do (i, k)
        k_rep = k_reps[i]
        for g ∈ sym_elements
            if g.index_action(k_rep) == k
                return (k_rep, g.coef_action(k_rep))
            end
        end
        return throw(ArgumentError("Symmetry group consistency error"))
    end
end



#

const NoSymSpace = Union{BaseSpace,TensorSpace}

struct SymmetricSpace{S<:NoSymSpace,G<:Group,I,R} <: SequenceSpace
    space    :: S
    symmetry :: G
    indices  :: I
    rep_idx_action :: Vector{R}
    #= NOTE
    Constructing the orbit on the fly is slow. The field `rep_idx_action` stores the index representative of the orbit along with the action of the corresponding coefficient action. The issue is that this gets reconstructed every time. It might be possible to have a more global dictonary containing the `rep_idx_action` per symmetry group.
    =#
    function SymmetricSpace(space::S, sym::G) where {S<:NoSymSpace,G<:Group}
        inds = indices(space)
        k_reps = _orbit_representatives(sym, inds)
        reps = _filter_valid_representatives(sym, k_reps)
        inds2 = _maybe_range(reps)
        rep_idx_action = _compute_action_map(sym, inds, k_reps)
        return new{S,G,typeof(inds2),eltype(rep_idx_action)}(space, sym, inds2, rep_idx_action)
    end
end

_maybe_range(v) = v
function _maybe_range(v::Vector{<:Integer})
    n = length(v)
    n == 0 && return v
    n == 1 && return @inbounds v[1]:v[1]
    @inbounds d = v[2] - v[1]
    @inbounds for i ∈ 3:n
        if v[i] - v[i-1] != d
            return v
        end
    end
    d == 1 && return @inbounds v[1]:v[n]
    return @inbounds v[1]:d:v[n]
end

SymmetricSpace(space::NoSymSpace) = SymmetricSpace(space, symmetry(space))

SymmetricSpace(space::SymmetricSpace) = space

SymmetricSpace(space::SymmetricSpace, sym::Group) = SymmetricSpace(space, symmetry(space) ∪ sym)

desymmetrize(s::SymmetricSpace) = s.space
desymmetrize(s::NoSymSpace) = s
desymmetrize(s::ScalarSpace) = s
desymmetrize(s::CartesianPower) = CartesianPower(desymmetrize(space(s)), nspaces(s))
desymmetrize(s::CartesianProduct) = CartesianProduct(map(desymmetrize, spaces(s)))

symmetry(s::SymmetricSpace) = s.symmetry
symmetry(::BaseSpace) = # identity
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(1)),
                       CoefAction(exact(1), StaticArrays.SVector{1,Rational{Int}}(0//1))))
symmetry(::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = # identity
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{N,N,Int}(I)),
                       CoefAction(exact(1), StaticArrays.SVector{N,Rational{Int}}(ntuple(_ -> 0//1, Val(N))))))

indices(s::SymmetricSpace) = s.indices

order(s::SymmetricSpace) = order(desymmetrize(s))
frequency(s::SymmetricSpace) = frequency(desymmetrize(s))

Base.:(==)(s₁::SymmetricSpace, s₂::SymmetricSpace) = (desymmetrize(s₁) == desymmetrize(s₂)) & (symmetry(s₁) == symmetry(s₂))
Base.issubset(s₁::SymmetricSpace, s₂::SymmetricSpace) = issubset(desymmetrize(s₁), desymmetrize(s₂)) & issubset(symmetry(s₂), symmetry(s₁))
Base.intersect(s₁::SymmetricSpace, s₂::SymmetricSpace) = SymmetricSpace(intersect(desymmetrize(s₁), desymmetrize(s₂)), union(symmetry(s₁), symmetry(s₂)))
Base.union(s₁::SymmetricSpace, s₂::SymmetricSpace) = SymmetricSpace(union(desymmetrize(s₁), desymmetrize(s₂)), intersect(symmetry(s₁), symmetry(s₂)))

Base.hash(s::SymmetricSpace, h::UInt) = hash(s.space, hash(s.symmetry, h))

function _findindex_constant(s::SymmetricSpace)
    k0 = _findindex_constant(desymmetrize(s))
    k0 ∈ indices(s) && return k0
    return nothing
end

_findposition(k, s::SymmetricSpace) = findfirst(==(k), indices(s))
_findposition(u::AbstractRange, s::SymmetricSpace) = map(i -> _findposition(i, s), u)
_findposition(u::AbstractVector, s::SymmetricSpace) = map(i -> _findposition(i, s), u)
_findposition(c::Colon, ::SymmetricSpace) = c

_iscompatible(s₁::SymmetricSpace, s₂::SymmetricSpace) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂)) # & (symmetry(s₁) == symmetry(s₂))
_iscompatible(s₁::SymmetricSpace, s₂::NoSymSpace) = _iscompatible(desymmetrize(s₁), s₂)
_iscompatible(s₁::NoSymSpace, s₂::SymmetricSpace) = _iscompatible(s₁, desymmetrize(s₂))

IntervalArithmetic._infer_numtype(s::SymmetricSpace) = IntervalArithmetic._infer_numtype(desymmetrize(s))
IntervalArithmetic._interval_infsup(::Type{T}, s₁::SymmetricSpace, s₂::SymmetricSpace, d::IntervalArithmetic.Decoration) where {T<:IntervalArithmetic.NumTypes} =
    SymmetricSpace(IntervalArithmetic._interval_infsup(T, desymmetrize(s₁), desymmetrize(s₂), d), intersect(interval(symmetry(s₁)), interval(symmetry(s₂))))

IntervalArithmetic.interval(g::Group) = unsafe_group!(Set(interval(h) for h in elements(g)))
IntervalArithmetic.interval(g::GroupElement) = GroupElement(g.index_action, interval(g.coef_action))
IntervalArithmetic.interval(g::CoefAction) = CoefAction(interval(g.amplitude), g.phase)

#

_prettystring(s::SymmetricSpace{<:TensorSpace}, iscompact::Bool) = string("(", _prettystring(desymmetrize(s), iscompact), ")_sym")
_prettystring(s::SymmetricSpace, iscompact::Bool) = string(_prettystring(desymmetrize(s), iscompact), "_sym")



#

evensym(s::Taylor) = SymmetricSpace(s,
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(1)),
                       CoefAction(exact(1//1), StaticArrays.SVector{1,Rational{Int}}(1//1)))))
oddsym(s::Taylor)  = SymmetricSpace(s,
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(1)),
                       CoefAction(exact(-1//1), StaticArrays.SVector{1,Rational{Int}}(1//1)))))

evensym(s::Fourier) = SymmetricSpace(s,
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(-1)),
                       CoefAction(exact(1//1), StaticArrays.SVector{1,Rational{Int}}(0//1)))))
oddsym(s::Fourier)  = SymmetricSpace(s,
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(-1)),
                       CoefAction(exact(-1//1), StaticArrays.SVector{1,Rational{Int}}(0//1)))))

evensym(s::Chebyshev) = SymmetricSpace(s,
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(1)),
                       CoefAction(exact(1//1), StaticArrays.SVector{1,Rational{Int}}(1//1)))))
oddsym(s::Chebyshev)  = SymmetricSpace(s,
    Group(GroupElement(IndexAction(StaticArrays.SMatrix{1,1,Int}(1)),
                       CoefAction(exact(-1//1), StaticArrays.SVector{1,Rational{Int}}(1//1)))))

#

d4sym(s::TensorSpace{T}) where {T<:Tuple{<:Fourier,<:Fourier}} = SymmetricSpace(s,
    Group(
        GroupElement(IndexAction(StaticArrays.SMatrix{2,2,Int}([0 -1 ; 1 0])),
                       CoefAction(exact(1), StaticArrays.SVector{2,Rational{Int}}(0//1, 0//1))),
        GroupElement(IndexAction(StaticArrays.SMatrix{2,2,Int}([0  1 ; 1 0])),
                       CoefAction(exact(1), StaticArrays.SVector{2,Rational{Int}}(0//1, 0//1)))))
