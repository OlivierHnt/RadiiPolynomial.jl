include("multiplication.jl")
    export Multiplication, sequence
include("calculus/derivative.jl")
    export Derivative, differentiate, differentiate!
include("calculus/integral.jl")
    export Integral, integrate, integrate!
include("calculus/laplacian.jl")
    export Laplacian, laplacian, laplacian!
include("evaluation.jl")
    export Evaluation, evaluate, evaluate!, value
include("scale.jl")
    export Scale, scale, scale!
include("shift.jl")
    export Shift, shift, shift!

#

for S ∈ (:Evaluation, :Multiplication, :Derivative, :Integral, :Laplacian, :Scale, :Shift)
    @eval begin
        getcoefficient(A::$S, (codom, α)::Tuple{SymmetricSpace,Any}, (dom, β)::Tuple{SymmetricSpace,Any}, ::Type{T}) where {T} =
            _sym_getcoefficient(A, dom, codom, α, β, T)

        function _sym_getcoefficient(A::$S, dom::SymmetricSpace, codom::SymmetricSpace, α, β, ::Type{T}) where {T}
            v = zero(T)
            orbit_α = _orbit(symmetry(codom), α)
            for l ∈ _orbit(symmetry(dom), β)
                _checkbounds_indices(l, desymmetrize(dom)) || continue
                _, factor_l = _unsafe_get_representative_and_action(dom, l)
                for k ∈ orbit_α
                    _checkbounds_indices(k, desymmetrize(codom)) || continue
                    _, factor_k = _unsafe_get_representative_and_action(codom, k)
                    v += factor_l * getcoefficient(A, (desymmetrize(codom), k), (desymmetrize(dom), l), T) / factor_k
                end
            end
            return convert(T, v / exact(length(orbit_α)))
        end

        function _apply!(c, A::$S, a::Sequence{<:SymmetricSpace})
            @inbounds for k ∈ indices(space(c))
                c[k] = zero(eltype(c))
                for l ∈ indices(space(a))
                    c[k] += _sym_getcoefficient(A, space(a), space(c), k, l, eltype(c)) * a[l]
                end
            end
            return c
        end
    end

    # Cartesian spaces

    if S != :Multiplication
        @eval begin
            function domain(A::$S, s::CartesianPower)
                s_out = domain(A, space(s))
                s_out isa EmptySpace && return EmptySpace()
                return CartesianPower(s_out, nspaces(s))
            end

            function domain(A::$S, s::CartesianProduct)
                s_out = map(sᵢ -> domain(A, sᵢ), spaces(s))
                any(sᵢ -> sᵢ isa EmptySpace, s_out) && return EmptySpace()
                return CartesianProduct(s_out)
            end

            codomain(A::$S, s::CartesianPower) =
                CartesianPower(codomain(A, space(s)), nspaces(s))

            codomain(A::$S, s::CartesianProduct) =
                CartesianProduct(map(sᵢ -> codomain(A, sᵢ), spaces(s)))

            _coeftype(A::$S, s::CartesianPower, ::Type{T}) where {T} =
                _coeftype(A, space(s), T)

            _coeftype(A::$S, s::CartesianProduct, ::Type{T}) where {T} =
                @inbounds promote_type(_coeftype(A, s[1], T), _coeftype(A, Base.tail(s), T))
            _coeftype(A::$S, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
                @inbounds _coeftype(A, s[1], T)

            function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, A::$S)
                @inbounds for i ∈ 1:nspaces(domain(C))
                    _project!(block(C, i, i), A)
                end
                return C
            end

            #

            function _apply!(c::Sequence{<:CartesianPower}, A::$S, a)
                @inbounds for i ∈ 1:nspaces(space(c))
                    _apply!(block(c, i), A, block(a, i))
                end
                return c
            end

            function _apply!(c::Sequence{CartesianProduct{T}}, A::$S, a) where {N,T<:NTuple{N,VectorSpace}}
                @inbounds _apply!(block(c, 1), A, block(a, 1))
                @inbounds _apply!(block(c, 2:N), A, block(a, 2:N))
                return c
            end
            function _apply!(c::Sequence{CartesianProduct{T}}, A::$S, a) where {T<:Tuple{VectorSpace}}
                @inbounds _apply!(block(c, 1), A, block(a, 1))
                return c
            end
        end
    end
end
