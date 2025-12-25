abstract type IndexAction end

struct LinearIdx <: IndexAction
    matrix :: Matrix{Int}
end

(A::LinearIdx)(k::Integer) = A.matrix[1] * k
(A::LinearIdx)(k::Tuple) = ((A.matrix * [k...])...,)

Base.:*(A::LinearIdx, B::LinearIdx) = LinearIdx(A.matrix * B.matrix)

Base.:(==)(A::LinearIdx, B::LinearIdx) = A.matrix == B.matrix
Base.hash(A::LinearIdx, h::UInt) = hash(A.matrix, h)

#

abstract type ValueAction end

struct PhaseVal{T<:Number,S<:Number} <: ValueAction
    amplitude :: T
    phase     :: Vector{S} # factor of π
    conjugacy :: Bool
    PhaseVal{T,S}(amplitude::T, phase::Vector{S}, conjugacy::Bool) where {T,S} = new{T,S}(amplitude, mod.(phase, 2), conjugacy)
end

PhaseVal(amplitude::T, phase::Vector{S}, conjugacy::Bool) where {T,S} = PhaseVal{T,S}(amplitude, phase, conjugacy)

(v::PhaseVal)(k, x) = v.amplitude * cispi(mapreduce(*, +, v.phase, k)) * ifelse(v.conjugacy, conj(x), x)

Base.:*(v::PhaseVal, w::PhaseVal) = PhaseVal(
    v.amplitude * ifelse(v.conjugacy, conj(w.amplitude), w.amplitude),
    mod.(v.phase + ifelse(v.conjugacy, -w.phase, w.phase), 2),
    xor(v.conjugacy, w.conjugacy))

Base.:(==)(v::PhaseVal, w::PhaseVal) = (v.amplitude == w.amplitude) & (v.phase == w.phase) & (v.conjugacy == w.conjugacy)
Base.hash(v::PhaseVal, h::UInt) = hash(v.amplitude, hash(v.phase, hash(v.conjugacy, h)))

#

struct GroupElement{I<:IndexAction,V<:ValueAction}
    idx_action :: I
    val_action :: V
end

idx_action(g::GroupElement) = g.idx_action
val_action(g::GroupElement) = g.val_action

Base.:∘(g::GroupElement, h::GroupElement) = GroupElement(idx_action(g) * idx_action(h), val_action(g) * val_action(h))

Base.:(==)(g::GroupElement, h::GroupElement) = (g.idx_action == h.idx_action) & (g.val_action == h.val_action)
Base.hash(g::GroupElement, h::UInt) = hash(g.idx_action, hash(g.val_action, h))

#

struct Group{G<:GroupElement}
    elements :: Set{G}
    function Group{G}(elements::Set{G}) where {G<:GroupElement}
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
        return new{G}(elements)
    end
end

Group(elements::Set{G}) where {G<:GroupElement} = Group{G}(elements)
Group(g::GroupElement, h::GroupElement...) = Group(Set((g, h...)))

elements(g::Group) = g.elements

Base.:(==)(G₁::Group, G₂::Group) = elements(G₁) == elements(G₂)
Base.issubset(G₁::Group, G₂::Group) = issubset(elements(G₁), elements(G₂))
Base.intersect(G₁::Group, G₂::Group) = Group(intersect(elements(G₁), elements(G₂)))
Base.union(G₁::Group, G₂::Group) = Group(elements(G₁)..., elements(G₂)...)

_orbit(sym::Group, k) = Set(idx_action(g)(k) for g ∈ elements(sym))

function _orbit_representatives(sym::Group, inds)
    reps = eltype(inds)[]
    seen = Set{eltype(inds)}()
    for k ∈ inds
        k ∈ seen && continue
        O = _orbit(sym, k)
        push!(reps, minimum(O))
        union!(seen, O)
    end
    return reps
end

function _get_representative_and_action(sym::Group, k)
    orbit = _orbit(sym, k)
    k_rep = minimum(orbit)
    for g ∈ elements(sym)
        if idx_action(g)(k_rep) == k
            return k_rep, val_action(g)(k_rep, one(ComplexF64))
        end
    end
    return throw(ArgumentError("Symmetry group consistency error"))
end

function group_product(G₁::Group, G₂::Group)
    D₁ = Dict(idx_action(g) => val_action(g) for g ∈ elements(G₁))
    D₂ = Dict(idx_action(g) => val_action(g) for g ∈ elements(G₂))
    elements = Set{GroupElement}()
    for (idx, val₁) ∈ D₁
        haskey(D₂, idx) || continue # symmetry lost
        val₂ = D₂[idx]
        push!(elements, GroupElement(idx, val₁ * val₂))
    end
    return Group(elements)
end

function _fixed_dim(sym::Group, k)
    for g ∈ elements(sym)
        idx_action(g)(k) == k || continue
        val_action(g)(k, one(ComplexF64)) == 1 || return false
    end
    return true
end

#

const NoSymSpace = Union{BaseSpace,TensorSpace}

struct SymmetricSpace{S<:NoSymSpace,G<:Group} <: SequenceSpace
    space    :: S
    symmetry :: G
end

desymmetrize(s::SymmetricSpace) = s.space
desymmetrize(s::NoSymSpace) = s
desymmetrize(s::ParameterSpace) = s
desymmetrize(s::CartesianPower) = CartesianPower(desymmetrize(space(s)), nspaces(s))
desymmetrize(s::CartesianProduct) = CartesianProduct(map(desymmetrize, spaces(s)))

symmetry(s::SymmetricSpace) = s.symmetry
symmetry(s::NoSymSpace) = Group(GroupElement(LinearIdx(I(nspaces(s))), PhaseVal(true, fill(false, nspaces(s)), false))) # identity

SymmetricSpace(space::SequenceSpace) = SymmetricSpace(space, symmetry(space))

SymmetricSpace(space::SymmetricSpace, sym::Group) = SymmetricSpace(space, symmetry(space) ∪ sym)

order(s::SymmetricSpace) = order(desymmetrize(s))
frequency(s::SymmetricSpace) = frequency(desymmetrize(s))

Base.:(==)(s₁::SymmetricSpace, s₂::SymmetricSpace) = (desymmetrize(s₁) == desymmetrize(s₂)) & (symmetry(s₁) == symmetry(s₂))
Base.issubset(s₁::SymmetricSpace, s₂::SymmetricSpace) = issubset(desymmetrize(s₁), desymmetrize(s₂)) & issubset(symmetry(s₂), symmetry(s₁))
Base.intersect(s₁::SymmetricSpace, s₂::SymmetricSpace) = SymmetricSpace(intersect(desymmetrize(s₁), desymmetrize(s₂)), union(symmetry(s₁), symmetry(s₂)))
Base.union(s₁::SymmetricSpace, s₂::SymmetricSpace) = SymmetricSpace(union(desymmetrize(s₁), desymmetrize(s₂)), intersect(symmetry(s₁), symmetry(s₂)))

indices(s::SymmetricSpace) = [k for k ∈ _orbit_representatives(symmetry(s), indices(desymmetrize(s))) if _fixed_dim(symmetry(s), k)]

# function nz_indices(s::SymmetricSpace)
#     inds = indices(desymmetrize(s))
#     out_inds = Set{eltype(indices(desymmetrize(s)))}()
#     for k ∈ inds
#         union!(out_inds, _orbit(symmetry(s), k))
#     end
#     return collect(out_inds)
# end

function _findindex_constant(s::SymmetricSpace)
    k0 = _findindex_constant(space(s))
    k0 ∈ indices(s) && return k0
    return nothing
end

_iscompatible(s₁::SymmetricSpace, s₂::SymmetricSpace) = _iscompatible(desymmetrize(s₁), desymmetrize(s₂)) & (symmetry(s₁) == symmetry(s₂))

IntervalArithmetic.interval(::Type{T}, s::SymmetricSpace) where {T} = SymmetricSpace(interval(T, desymmetrize(s)), symmetry(s))
IntervalArithmetic.interval(s::SymmetricSpace) = SymmetricSpace(interval(desymmetrize(s)), symmetry(s))

#

evensym(s::Taylor) = SymmetricSpace(s, Group(GroupElement(LinearIdx([1;;]), PhaseVal(true, [1], false))))
oddsym(s::Taylor) = SymmetricSpace(s, Group(GroupElement(LinearIdx([1;;]), PhaseVal(-1, [1], false))))

evensym(s::Fourier) = SymmetricSpace(s, Group(GroupElement(LinearIdx([-1;;]), PhaseVal(true, [false], false))))
oddsym(s::Fourier) = SymmetricSpace(s, Group(GroupElement(LinearIdx([-1;;]), PhaseVal(true, [1], false))))

evensym(s::Chebyshev) = SymmetricSpace(s, Group(GroupElement(LinearIdx([1;;]), PhaseVal(true, [1], false))))
oddsym(s::Chebyshev) = SymmetricSpace(s, Group(GroupElement(LinearIdx([1;;]), PhaseVal(-1, [1], false))))
