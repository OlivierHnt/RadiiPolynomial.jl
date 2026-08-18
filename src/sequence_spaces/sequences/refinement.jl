# enclosure [-μ, μ] for interval coefficients, midpoint 0 for floating point ones; `μ` is
# passed as is — the constructor encloses an interval radius by itself, and reading `sup(μ)`
# first would launder an NG envelope into a guaranteed-looking box
_envelope_box(::Type{T}, μ) where {T<:RealOrComplexI} =
    interval(zero(T), μ; format = :midpoint)
_envelope_box(::Type{T}, μ) where {T} = zero(T)

# envelope μ on |aₖ| given ‖a‖_X ≤ err and the weight wₖ at the index k
_coefficient_bound(::Union{Ell1,EllInf}, err, w) = err / w
_coefficient_bound(::Ell2, err, w) = sqrt(err^exact(2) / w) # ‖a‖² = √(∑ |aₖ|² wₖ)





# support enforcement

function _zero_outside!(C, sc::BaseSpace, cmin, cmax)
    z = zero(eltype(C))
    n = length(C)
    offset = first(indices(sc)) - 1
    head_end = clamp(cmin - offset - 1, 0, n)
    tail_start = clamp(cmax - offset + 1, 1, n + 1)
    @inbounds view(C, 1:head_end) .= z
    @inbounds view(C, tail_start:n) .= z
    return C
end
function _zero_outside!(C, sc::BaseSpace, cmin, cmax, ::Val{D}) where {D}
    z = zero(eltype(C))
    n = size(C, D)
    offset = first(indices(sc)) - 1
    head_end = clamp(cmin - offset - 1, 0, n)
    tail_start = clamp(cmax - offset + 1, 1, n + 1)
    @inbounds selectdim(C, D, 1:head_end) .= z
    @inbounds selectdim(C, D, tail_start:n) .= z
    return C
end





# polishing against the fitted decay

"""
    polish!(a::Sequence)

Zero every coefficient violating the decay envelope fitted on `a` itself
(cf. [`geometricweight`](@ref) and [`algebraicweight`](@ref)). The envelope is a
least-squares fit.

This is a numerical hygiene tool and nothing rigorous is known about the
discarded modes.
"""
polish!(a::Sequence{ScalarSpace}) = a

polish!(a::Sequence{<:TensorSpace}) = a

function polish!(a::Sequence{<:BaseSpace})
    w, ord = _weight(a)
    s = space(a)
    norm_a = norm(a, 1)
    for i ∈ indices(s)
        if abs(i) > ord
            val = norm_a / _getindex(w, s, i)
            if abs(a[i]) > val
                a[i] = 0
            end
        end
    end
    return a
end

function polish!(a::Sequence{<:CartesianSpace})
    for i ∈ 1:nspaces(space(a))
        polish!(component(a, i))
    end
    return a
end
