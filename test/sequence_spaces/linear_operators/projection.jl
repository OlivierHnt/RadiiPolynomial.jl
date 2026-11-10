@testset "Projection" begin

    𝒯 = Taylor(2)
    ℱ = Fourier(2, 1.0)
    # reflection in x on a two-dimensional lattice, and a reflection with a cocycle factor which is not ±1
    reflection_x = Group(GroupElement(LatticeAut([-1 0 ; 0 1]), Cocycle(0//1, Rational{Int}[0//1, 0//1])))
    third = SymmetricSpace(ℱ, Group(GroupElement(LatticeAut([-1;;]), Cocycle(0//1, Rational{Int}[1//3]))))

    @testset "construction, domain, codomain" begin
        Π = Projection(𝒯)
        @test Π isa Projection{Taylor,ExactReal{Bool}}
        @test domain(Π) == codomain(Π) == 𝒯
        @test eltype(Π) == ExactReal{Bool}
        @test eltype(Projection(𝒯, ComplexF64)) == ComplexF64
        @test eltype(Projection{Taylor,Float64}) == Float64
        @test coefficients(Π) == LinearOperator(𝒯, 𝒯, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0])

        # the projected space is returned whenever the argument is compatible with it
        @test domain(Π, Taylor(5)) == 𝒯
        @test codomain(Π, Taylor(0)) == 𝒯
        @test_throws ArgumentError domain(Projection(ℱ), Fourier(2, 2.0))
        @test_throws ArgumentError codomain(Projection(ℱ), Fourier(2, 2.0))
        @test_throws ArgumentError domain(Π, UndefSpace())
        @test_throws ArgumentError codomain(Π, UndefSpace())
    end

    @testset "interval(Projection)" begin
        for Π_I ∈ (interval(Float64, Projection(𝒯)), interval(Projection(𝒯)))
            @test Π_I.space == 𝒯
            @test eltype(Π_I) == Interval{Float64}
        end
        Πf = interval(Float64, Projection(ℱ))
        @test isequal_interval(frequency(Πf.space), interval(1.0))
        @test eltype(Πf) == Interval{Float64}
    end

    @testset "project/project! (Sequence)" begin

        @testset "base, tensor and cartesian spaces: enlarging zero-pads, shrinking truncates" begin
            s1 = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
            s2 = ScalarSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
            s3 = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0))^1
            s4 = ScalarSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0))^1
            a = Sequence(Taylor(1), [1.0, 2.0])
            b = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
            c = Sequence(Chebyshev(1), [1.0, 2.0])
            t = Sequence(Taylor(1) ⊗ Chebyshev(1), [1.0, 2.0, 3.0, 4.0])
            p = Sequence(Taylor(1)^2, [1.0, 2.0, 3.0, 4.0])
            v = Sequence(s1, [1.0 ; 1.0:12.0])
            scalar = Sequence(ScalarSpace(), [7.0])
            cases = (
                (a, Taylor(1), [1.0, 2.0]),
                (a, Taylor(3), [1.0, 2.0, 0.0, 0.0]),
                (a, Taylor(0), [1.0]),
                (b, Fourier(2, 1.0), [0.0, 1.0, 2.0, 3.0, 0.0]),
                (b, Fourier(0, 1.0), [2.0]),
                (c, Chebyshev(3), [1.0, 2.0, 0.0, 0.0]),
                (c, Chebyshev(0), [1.0]),
                (t, Taylor(2) ⊗ Chebyshev(0), [1.0, 2.0, 0.0]),
                (p, Taylor(2)^2, [1.0, 2.0, 0.0, 3.0, 4.0, 0.0]),
                (v, s1, coefficients(v)),
                (v, s2, [1.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 7.0, 8.0, 0.0, 9.0, 10.0, 0.0, 11.0, 12.0, 0.0]),
                (v, s3, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                (v, s4, [1.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0]),
                # a scalar is the coefficient of order 0
                (scalar, Taylor(2), [7.0, 0.0, 0.0]),
                (scalar, Fourier(1, 1.0), [0.0, 7.0, 0.0]),
                (scalar, Chebyshev(2), [7.0, 0.0, 0.0]),
                (Sequence(Taylor(2), [1.0, 2.0, 3.0]), ScalarSpace(), [1.0]),
                (Sequence(Taylor(1), ComplexF64[1.0 + 1.0im, 2.0 - 1.0im]), Taylor(3), ComplexF64[1.0 + 1.0im, 2.0 - 1.0im, 0.0, 0.0]),
            )
            for (x, s, expected) ∈ cases
                @test project(x, s) == Sequence(s, expected)
            end
            @test project!(fill(Inf, Taylor(3)), a) == Sequence(Taylor(3), [1.0, 2.0, 0.0, 0.0])
            @test_throws ArgumentError project(b, Fourier(1, 2.0))
            @test_throws ArgumentError project(p, Taylor(1)^3)
        end

        @testset "SymmetricSpace" begin
            𝒯e2, 𝒯e4 = evensym(Taylor(2)), evensym(Taylor(4))
            @test indices(𝒯e2) == 0:2:2
            @test indices(𝒯e4) == 0:2:4
            ae = Sequence(𝒯e2, [1.0, 2.0])
            be = project(ae, 𝒯e4)
            @test be == Sequence(𝒯e4, [1.0, 2.0, 0.0])
            @test project(be, 𝒯e2) == ae

            ℱe1, ℱe3 = evensym(Fourier(1, 1.0)), evensym(Fourier(3, 1.0))
            af = Sequence(ℱe1, [1.0, 2.0])
            @test project(af, ℱe3) == Sequence(ℱe3, [1.0, 2.0, 0.0, 0.0])
            @test project(project(af, ℱe3), ℱe1) == af

            # across symmetry groups this is the orthogonal projection: the even part of an odd sequence vanishes
            @test project(Sequence(oddsym(Taylor(2)), [10.0]), 𝒯e4) == Sequence(𝒯e4, zeros(3))

            # and it is the identity when the target symmetry is a subgroup
            𝒯² = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
            ex = SymmetricSpace(𝒯², reflection_x)
            ad4 = Sequence(d4sym(𝒯²), [1.0, 2.0, 3.0])
            @test project(project(ad4, ex), 𝒯²) == project(ad4, 𝒯²)

            # a plain sequence is averaged over the orbits
            sine = Sequence(ℱ, [0.0, -1.0, 0.0, 1.0, 0.0])
            @test project(sine, evensym(ℱ)) == Sequence(evensym(ℱ), zeros(3))
            @test project(sine, oddsym(ℱ)) == Sequence(oddsym(ℱ), [1.0, 0.0])
            @test project(Sequence(ℱ, [1.0, 2.0, 3.0, 4.0, 5.0]), evensym(ℱ)) == Sequence(evensym(ℱ), [3.0, 3.0, 3.0])
            # a sequence carrying the symmetry round trips exactly, even over orbits of size 8
            ae2 = Sequence(evensym(ℱ), [1.0, 0.5, 0.2])
            @test project(project(ae2, ℱ), evensym(ℱ)) == ae2
            @test project(project(ad4, 𝒯²), d4sym(𝒯²)) == ad4
        end

        @testset "SymmetricSpace: operators" begin
            es, os = evensym(ℱ), oddsym(ℱ)
            xe, xo = Sequence(es, [1.0, 0.5, 0.2]), Sequence(os, [0.5, 0.2])
            A = project(Multiplication(Sequence(ℱ, [0.0, 0.3, 1.0, 0.3, 0.0])), ℱ, ℱ) # equivariant
            N = LinearOperator(ℱ, ℱ, reshape(1.0:25.0, 5, 5)) # not equivariant
            # the compression onto a symmetric space, in one or two steps, is desymmetrize-apply-project
            for (B, x) ∈ ((A, xe), (A, xo), (N, xe), (N, xo))
                s = space(x)
                y = project(B * project(x, ℱ), s)
                @test project(B, s, s) * x ≈ y
                @test (Projection(s) * (B * Projection(s))) * x ≈ y
                @test (Projection(s) * B) * Projection(s) * x ≈ y
                @test B * x ≈ B * project(x, ℱ)
            end
            # a column collects its orbit (x₋₁ = x₁ for a cosine series) while a row averages its orbit
            @test coefficients(A * Projection(es))[:, 2] == coefficients(A)[:, 2] + coefficients(A)[:, 4]
            @test coefficients(project(A, es, es)) == [1.0 0.6 0.0 ; 0.3 1.0 0.3 ; 0.0 0.3 1.0]
            @test coefficients(Projection(es) * N)[2, :] == (coefficients(N)[2, :] + coefficients(N)[4, :]) / 2
            # desymmetrizing one side of a symmetric operator agrees with the equivariant original
            Aes = project(A, es, es)
            @test Aes * Projection(ℱ) ≈ Projection(es) * A
            @test Projection(ℱ) * Aes ≈ A * Projection(es)
            @test project(Aes, ℱ, ℱ) ≈ project(Projection(ℱ) * Aes, ℱ, ℱ)
            # enlarging zero-pads
            es3 = evensym(Fourier(3, 1.0))
            @test coefficients(project(A, es3, es3)) == [coefficients(Aes) zeros(3) ; zeros(1, 4)]
            # composing operators with different symmetries projects the inner one onto the domain of the outer one
            @test (A * Aes) * xe ≈ A * project(Aes * xe, ℱ)
            @test (Aes * A) * project(xe, ℱ) ≈ Aes * project(A * project(xe, ℱ), es)
            # a weaker symmetry (reflections in x ⊂ d4) is read from the stored coefficients, as the full compression would
            𝒯² = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
            ex = SymmetricSpace(𝒯², reflection_x)
            Ad4 = LinearOperator(d4sym(𝒯²), d4sym(𝒯²), [1.0 2.0 3.0 ; 4.0 5.0 6.0 ; 7.0 8.0 9.0])
            Afull = project(Ad4, 𝒯², 𝒯²)
            for (dom, codom) ∈ ((ex, ex), (𝒯², ex), (ex, 𝒯²))
                @test project(Ad4, dom, codom) ≈ project(Afull, dom, codom)
            end
            # a scalar side is read as the zero space of the other side: a functional sums the orbits of its domain
            # and a column averages the orbits of its codomain
            a = Sequence(ℱ, [0.1, -0.2, 1.0, 0.4, 0.3])
            @test (project(adjoint(a), es, ScalarSpace()) * xe)[1] ≈ (adjoint(a) * project(xe, ℱ))[1]
            @test project(LinearOperator(a), ScalarSpace(), es) == LinearOperator(project(a, es))
            @test (project(adjoint(xe) * Multiplication(a), es, ScalarSpace()) * xe)[1] ≈ (adjoint(xe) * (a * project(xe, ℱ)))[1]
        end

        @testset "SymmetricSpace: identity and diagonal operators across symmetries" begin
            ℱ₁ = Fourier(1, 1.0)
            es, os = evensym(ℱ₁), oddsym(ℱ₁)
            ts = SymmetricSpace(ℱ₁, symmetry(ℱ₁)) # trivial group
            U = UniformScalingOperator(1.0)
            Iℱ = project(U, ℱ₁, ℱ₁)
            # the identity between different symmetries is the orthogonal projection, however it is materialized
            @test coefficients(project(U, ℱ₁, es)) == coefficients(project(Iℱ, ℱ₁, es)) == [0.0 1.0 0.0 ; 0.5 0.0 0.5]
            @test coefficients(project(U, ts, os)) == coefficients(project(Iℱ, ℱ₁, os)) == [-0.5 0.0 0.5]
            A = LinearOperator(ℱ₁, es, zeros(2, 3))
            @test coefficients(one(A)) == coefficients(project(A + I, ℱ₁, es)) == [0.0 1.0 0.0 ; 0.5 0.0 0.5]
            # a diagonal operator maps a space to itself: across symmetries it is materialized first, then compressed
            struct _LatticeDiag <: RadiiPolynomial.AbstractDiagonalOperator end
            RadiiPolynomial.getcoefficient(::_LatticeDiag, (codom, i)::Tuple{VectorSpace,Any}, (dom, j)::Tuple{VectorSpace,Any}, ::Type{T}) where {T} =
                ifelse(i == j, convert(T, 2), zero(T))
            D = _LatticeDiag()
            @test_throws ArgumentError project(D, ℱ₁, es, Float64)
            @test coefficients(project(project(D, ℱ₁, ℱ₁, Float64), ℱ₁, es)) == [0.0 2.0 0.0 ; 1.0 0.0 1.0]
            # the composition of projections is the projection onto the intersection
            @test (Projection(es) * Projection(ℱ₁)).space == (Projection(ℱ₁) ∘ Projection(es)).space == es
        end

        @testset "Interval{Float64} coefficients" begin
            a = Sequence(Taylor(1), interval.([1.0, 2.0]))
            b = project(a, Taylor(3), Interval{Float64})
            @test b == Sequence(Taylor(3), interval.([1.0, 2.0, 0.0, 0.0]))
            @test all(isguaranteed, coefficients(b))
            # a projection carries its own coefficient type
            @test eltype(Projection(𝒯, Complex{Interval{Float64}}) * a) == Complex{Interval{Float64}}
        end
    end

    @testset "project/project! (LinearOperator / AbstractLinearOperator)" begin

        @testset "AbstractLinearOperator: materialized through getcoefficient" begin
            # (∂a)ₖ = (k+1) aₖ₊₁
            expected = LinearOperator(Taylor(2), Taylor(1), [0.0 1.0 0.0 ; 0.0 0.0 2.0])
            @test project(Derivative(1), Taylor(2), Taylor(1), Float64) == expected
            @test project!(fill(Inf, Taylor(2), Taylor(1)), Derivative(1)) == expected
        end

        @testset "LinearOperator: enlarging and shrinking the domain and codomain" begin
            A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            enlarged = LinearOperator(Taylor(2), Taylor(2), [1.0 2.0 0.0 ; 3.0 4.0 0.0 ; 0.0 0.0 0.0])
            @test project(A, Taylor(2), Taylor(2)) == enlarged
            @test project!(fill(Inf, Taylor(2), Taylor(2)), A) == enlarged
            @test project(A, Taylor(0), Taylor(0)) == LinearOperator(Taylor(0), Taylor(0), [1.0;;])

            # each block of a cartesian operator is resized on its own
            B = LinearOperator(Taylor(1)^2, Taylor(1)^2, Float64[1 2 0 0 ; 3 4 0 0 ; 0 0 5 6 ; 0 0 7 8])
            @test project(B, Taylor(2)^2, Taylor(2)^2) == LinearOperator(Taylor(2)^2, Taylor(2)^2,
                [1.0 2.0 0.0 0.0 0.0 0.0
                 3.0 4.0 0.0 0.0 0.0 0.0
                 0.0 0.0 0.0 0.0 0.0 0.0
                 0.0 0.0 0.0 5.0 6.0 0.0
                 0.0 0.0 0.0 7.0 8.0 0.0
                 0.0 0.0 0.0 0.0 0.0 0.0])
            @test_throws ArgumentError project(B, Taylor(1)^3, Taylor(1)^2)
        end

        @testset "custom AbstractLinearOperator: the three documented methods suffice" begin
            # `domain`, `codomain` and the 3-argument `getcoefficient` are enough to materialize
            struct _MyDiagOp <: AbstractLinearOperator end
            RadiiPolynomial.domain(A::_MyDiagOp, codom::Taylor) = codom
            RadiiPolynomial.codomain(A::_MyDiagOp, dom::Taylor) = dom
            RadiiPolynomial.getcoefficient(A::_MyDiagOp, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}) =
                i == j ? inv(1.0+i) : 0.0

            A = _MyDiagOp()
            expected = LinearOperator(𝒯, 𝒯, [1.0 0.0 0.0 ; 0.0 0.5 0.0 ; 0.0 0.0 1/3])
            @test A * Projection(𝒯) == Projection(𝒯) * A == project(A, 𝒯, 𝒯) == expected
            @test RadiiPolynomial.getcoefficient(A, (𝒯, 1), (𝒯, 1), Float64) == 0.5
            # the entries are assumed to be representable in the requested type, `Float64` by default
            @test eltype(project(A, 𝒯, 𝒯)) == Float64
            @test eltype(Projection(𝒯, ComplexF64) * A) == ComplexF64
            @test eltype(project(A, 𝒯, 𝒯, ComplexF64)) == ComplexF64
        end

        @testset "custom AbstractLinearOperator with an undefined codomain" begin
            #= `A = (I - S/2)⁻¹ = Σₖ Sᵏ/2ᵏ` with `S` the shift `xʲ ↦ xʲ⁺¹`: every column is
               infinite, so `codomain` returns `UndefSpace()` and the operator can only be
               materialized once a codomain truncation is supplied. =#
            struct _MyResolvent <: AbstractLinearOperator end
            RadiiPolynomial.domain(::_MyResolvent, codom::SequenceSpace) = codom
            RadiiPolynomial.codomain(::_MyResolvent, ::SequenceSpace) = UndefSpace()
            RadiiPolynomial.getcoefficient(::_MyResolvent, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}) =
                i ≥ j ? 2.0^(j-i) : 0.0

            A = _MyResolvent()
            𝒯₁, 𝒯₂ = Taylor(1), Taylor(2)
            Π₁, Π₂ = Projection(𝒯₁), Projection(𝒯₂)
            expected = LinearOperator(𝒯₁, 𝒯₂, [1.0 0.0 ; 0.5 1.0 ; 0.25 0.5])

            # the supplied codomain is the truncation, and compatibility is checked through the domain
            @test project(A, 𝒯₁, 𝒯₂) == Π₂ * A * Π₁ == expected
            @test_throws ArgumentError project(A, Fourier(1, 1.0), 𝒯₂)
            # a left projection infers the domain from the codomain
            @test Π₂ * A == LinearOperator(𝒯₂, 𝒯₂, [1.0 0.0 0.0 ; 0.5 1.0 0.0 ; 0.25 0.5 1.0])
            # a right projection alone has nothing to materialize: the pair is wrapped lazily
            r = A * Π₁
            @test r isa ComposedOperator
            @test r.outer === A
            @test r.inner === Π₁
            @test Π₂ * r == expected
            # hence the action on a sequence needs a codomain
            a = Sequence(𝒯₁, [1.0, 2.0])
            @test_throws ArgumentError A * a
            @test Π₂ * A * a == expected * a == Sequence(𝒯₂, [1.0, 2.5, 1.25])

            # lazy arithmetic propagates the undefined codomain
            M = Multiplication(Sequence(Taylor(0), [2.0]))
            S = I - A * M
            @test codomain(S, 𝒯₁) == UndefSpace()
            @test domain(S, 𝒯₂) == 𝒯₂
            @test project(S, 𝒯₁, 𝒯₂) == Π₂ * S * Π₁ == LinearOperator(𝒯₁, 𝒯₂, [1.0 0.0 ; 0.0 1.0 ; 0.0 0.0] - 2coefficients(expected))
            # composing on either side: the intermediate truncation is the codomain of the inner
            # operator, or the domain of the outer operator when the former is undefined
            @test project(A ∘ M, 𝒯₁, 𝒯₂) == project(M ∘ A, 𝒯₁, 𝒯₂) == 2expected
            # the type of the outer operator is still inferred when the intermediate space is undefined
            Mc = Multiplication(Sequence(Taylor(0), [2.0im]))
            @test eltype(project(Mc ∘ A, 𝒯₁, 𝒯₂)) == ComplexF64
            @test M * A isa ComposedOperator
            @test project(M * project(A, 𝒯₁, 𝒯₂), 𝒯₁, 𝒯₂) == 2expected
            # both truncations undefined: nothing can be materialized
            @test_throws ArgumentError project(Evaluation(0.5) ∘ A, 𝒯₁, Taylor(0))

            # a block with an undefined codomain makes the matrix product elementwise
            v = [A 2.0] * Projection(𝒯₁ × 𝒯₁)
            @test v[1,1] isa ComposedOperator
            @test v[1,2] == LinearOperator(𝒯₁, 𝒯₁, [2.0 0.0 ; 0.0 2.0])
        end

        @testset "AbstractDiagonalOperator: only domain ∩ codomain indices are written" begin
            struct _ScalarDiag{T<:Number} <: RadiiPolynomial.AbstractDiagonalOperator
                λ :: T
            end
            RadiiPolynomial.getcoefficient(A::_ScalarDiag, (codom, i)::Tuple{VectorSpace,Any}, (dom, j)::Tuple{VectorSpace,Any}, ::Type{T}) where {T} =
                ifelse(i == j, convert(T, A.λ), zero(T))

            D = _ScalarDiag(2.0)
            @test project!(fill(Inf, 𝒯, 𝒯), D) == LinearOperator(𝒯, 𝒯, [2.0 0.0 0.0 ; 0.0 2.0 0.0 ; 0.0 0.0 2.0])
            # the rows 3 and 4 of a larger codomain are zero
            @test project!(fill(Inf, 𝒯, Taylor(4)), D) == LinearOperator(𝒯, Taylor(4),
                [2.0 0.0 0.0 ; 0.0 2.0 0.0 ; 0.0 0.0 2.0 ; 0.0 0.0 0.0 ; 0.0 0.0 0.0])
        end

        @testset "coefficient type inferred from the entries" begin
            # custom operators defined through `getcoefficient` alone, without `_coeftype`
            struct _ComplexDiag <: RadiiPolynomial.AbstractDiagonalOperator
                λ₁ :: Complex{Interval{Float64}}
                λ₂ :: Complex{Interval{Float64}}
            end
            function RadiiPolynomial.getcoefficient(L::_ComplexDiag, (codom, k)::T, (dom, l)::T) where {T<:Tuple{TensorSpace{NTuple{2,Taylor}},NTuple{2,Int}}}
                x = inv(exact(k[1]) * L.λ₁ + exact(k[2]) * L.λ₂)
                k == l && sum(k) ≥ 2 && return x
                return zero(x)
            end
            L = _ComplexDiag(interval(0.1) + im * interval(0.4), interval(0.1) - im * interval(0.4))
            s = Taylor(3) ⊗ Taylor(3)
            A = project(L, s, s)
            @test eltype(A) == Complex{Interval{Float64}}
            @test all(isequal_interval.(coefficients(A), coefficients(project(L, s, s, Complex{Interval{Float64}}))))
            @test eltype(L * Sequence(s, ones(16))) == Complex{Interval{Float64}}

            # an operator which is not diagonal
            struct _ComplexShiftUp <: AbstractLinearOperator end
            RadiiPolynomial.codomain(::_ComplexShiftUp, s::Taylor) = Taylor(order(s) + 1)
            RadiiPolynomial.getcoefficient(::_ComplexShiftUp, (codom, i)::Tuple{Taylor,Int}, (dom, j)::Tuple{Taylor,Int}) = complex(0.0, float(i == j + 1))
            B = project(_ComplexShiftUp(), Taylor(1), Taylor(2))
            @test eltype(B) == ComplexF64
            @test coefficients(B) == [0 0 ; im 0 ; 0 im]
        end

        @testset "Sequence <-> LinearOperator(ScalarSpace, ·) round trip" begin
            a = Sequence(𝒯, [1.0, 2.0, 3.0])
            A = project(a, ScalarSpace(), 𝒯)
            @test A isa LinearOperator{ScalarSpace,Taylor}
            @test coefficients(A) == reshape([1.0, 2.0, 3.0], 3, 1)
            @test project(A, 𝒯) == a
            @test project!(fill(Inf, ScalarSpace(), 𝒯), a) == A
        end
    end

    @testset "action, composition and materialization" begin
        Π₂, Π₃, Π₄ = Projection(Taylor(2)), Projection(Taylor(3)), Projection(Taylor(4))
        ∂ = Derivative(1)
        A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])

        @testset "Projection * Sequence, Projection * Projection" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            @test Π₄ * a == project(a, Taylor(4))
            @test eltype(Projection(Taylor(1), ComplexF64) * a) == ComplexF64
            @test (Π₂ * Π₄).space == (Π₂ ∘ Π₄).space == Taylor(2)
        end

        @testset "a projection on the left sets the codomain, on the right the domain" begin
            @test Π₂ * A == LinearOperator(Taylor(1), Taylor(2), [1.0 2.0 ; 3.0 4.0 ; 0.0 0.0])
            @test A * Π₂ == LinearOperator(Taylor(2), Taylor(1), [1.0 2.0 0.0 ; 3.0 4.0 0.0])
            # the other side of an `AbstractLinearOperator` is inferred from the projected one
            @test ∂ * Π₃ == LinearOperator(Taylor(3), Taylor(2), [0.0 1.0 0.0 0.0 ; 0.0 0.0 2.0 0.0 ; 0.0 0.0 0.0 3.0])
            @test Π₃ * ∂ == LinearOperator(Taylor(4), Taylor(3),
                [0.0 1.0 0.0 0.0 0.0
                 0.0 0.0 2.0 0.0 0.0
                 0.0 0.0 0.0 3.0 0.0
                 0.0 0.0 0.0 0.0 4.0])
        end

        @testset "Projection ± LinearOperator" begin
            Π₁ = Projection(Taylor(1))
            I2 = LinearOperator(Taylor(1), Taylor(1), [1.0 0.0 ; 0.0 1.0])
            @test Π₁ + A == I2 + A == LinearOperator(Taylor(1), Taylor(1), [2.0 2.0 ; 3.0 5.0])
            @test A + Π₁ == A + I2
            @test Π₁ - A == I2 - A == LinearOperator(Taylor(1), Taylor(1), [0.0 -2.0 ; -3.0 -3.0])
            @test A - Π₁ == A - I2
        end

        @testset "sums and negations of projections materialize" begin
            @test (Π₂ + Π₄) * ∂ == Π₂ * ∂ + Π₄ * ∂
            @test ∂ * (Π₂ + Π₄) == ∂ * Π₂ + ∂ * Π₄
            @test (Π₂ - Π₄) * ∂ == Π₂ * ∂ - Π₄ * ∂
            @test ∂ * (Π₂ - Π₄) == ∂ * Π₂ - ∂ * Π₄
            @test (-Π₂) * ∂ == -(Π₂ * ∂)
            @test ∂ * (-Π₂) == -(∂ * Π₂)
        end

        @testset "Projection(UndefSpace()): lazy ComposedOperator" begin
            # over an undefined space there is nothing to materialize
            Π_undef = Projection(UndefSpace())
            for (outer, inner) ∈ ((Π_undef, ∂), (∂, Π_undef))
                r = outer * inner
                @test r isa ComposedOperator
                @test r.outer === outer
                @test r.inner === inner
            end
        end
    end

    @testset "Projection{<:CartesianSpace} * Vector/Matrix/Diagonal" begin

        @testset "Projection * Vector{<:Sequence}" begin
            v = [Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(3), [4.0, 5.0, 6.0, 7.0])]
            @test Projection(Taylor(2)^2) * v == Sequence(Taylor(2)^2, [1.0, 2.0, 0.0, 4.0, 5.0, 6.0])
            @test_throws DimensionMismatch Projection(Taylor(2)^3) * v
        end

        @testset "Projection * Matrix{<:LinearOperator}" begin
            # the codomain of each block is resized, the domains are joined column-wise
            v11 = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            v12 = LinearOperator(Taylor(0), Taylor(0), [5.0;;])
            v21 = LinearOperator(Taylor(1), Taylor(1), [6.0 7.0 ; 8.0 9.0])
            v22 = LinearOperator(Taylor(0), Taylor(0), [10.0;;])
            r = Projection(Taylor(2) × Taylor(1)) * [v11 v12 ; v21 v22]
            @test domain(r) == Taylor(1) × Taylor(0)
            @test codomain(r) == Taylor(2) × Taylor(1)
            @test coefficients(r) == [1.0 2.0 5.0 ; 3.0 4.0 0.0 ; 0.0 0.0 0.0 ; 6.0 7.0 10.0 ; 8.0 9.0 0.0]
        end

        @testset "Matrix{<:LinearOperator} * Projection" begin
            # the domain of each block is resized, the codomains are joined row-wise
            v11 = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            v12 = LinearOperator(Taylor(1), Taylor(0), [5.0 6.0])
            v21 = LinearOperator(Taylor(0), Taylor(1), reshape([7.0, 8.0], 2, 1))
            v22 = LinearOperator(Taylor(0), Taylor(0), [9.0;;])
            r = [v11 v12 ; v21 v22] * Projection(Taylor(1) × Taylor(0))
            @test domain(r) == Taylor(1) × Taylor(0)
            @test codomain(r) == Taylor(1) × Taylor(1)
            @test coefficients(r) == [1.0 2.0 5.0 ; 3.0 4.0 0.0 ; 7.0 0.0 9.0 ; 8.0 0.0 0.0]
        end

        @testset "Projection * Diagonal / Diagonal * Projection" begin
            Π = Projection(Taylor(1)^2)
            D = RadiiPolynomial.LinearAlgebra.Diagonal([2.0, 3.0])
            @test Π * D == D * Π == LinearOperator(Taylor(1)^2, Taylor(1)^2, [2.0 0.0 0.0 0.0 ; 0.0 2.0 0.0 0.0 ; 0.0 0.0 3.0 0.0 ; 0.0 0.0 0.0 3.0])
        end
    end

    @testset "exact ones, storage promoted with the operands" begin
        # a default projection and `UniformScalingOperator()` carry exact ones, and the matrix
        # holding them takes the promoted type of the operands
        s = Fourier(3, exact(1.0))
        A = LinearOperator(s, s, interval.(reshape(1.0:49.0, 7, 7)))
        guaranteed(B) = all(isguaranteed, coefficients(B))
        @test guaranteed(Projection(s) - A)
        @test guaranteed(A + Projection(s))
        @test guaranteed(radd!(copy(A), UniformScalingOperator()))
        @test eltype(project(Projection(𝒯), 𝒯, 𝒯)) == ExactReal{Bool}
        @test eltype(Projection(Taylor(3)) * Derivative(1)) == Int
        @test eltype(Projection(𝒯) * Sequence(𝒯, [1, 2, 3])) == Int
        @test eltype(Projection(s) - LinearOperator(s, s, ones(7, 7))) == Float64
        # across symmetries, the orbit averages (a₋ₖ + aₖ)/2 are stored as well
        @test project(Projection(ℱ), ℱ, evensym(ℱ)) == LinearOperator(ℱ, evensym(ℱ), [0.0 0.0 1.0 0.0 0.0 ; 0.0 0.5 0.0 0.5 0.0 ; 0.5 0.0 0.0 0.0 0.5])
        @test eltype(project(Projection(ℱ), ℱ, third)) == ComplexF64
    end

    @testset "reads across symmetries" begin
        # a read across symmetry groups averages over the orbits: the result is floating, and
        # complex when a cocycle factor is not ±1, whatever the types of the operands
        e = evensym(ℱ)
        a = Sequence(ℱ, [1, 2, 3, 5, 5])
        L = LinearOperator(ℱ, ℱ, Matrix{Int}(I, 5, 5))
        O = LinearOperator(ℱ, ℱ, ones(5, 5))
        @test Projection(e) * a == project(a, e) == Sequence(e, [3.0, 3.5, 3.0])
        @test eltype(Projection(e) * L) == Float64
        @test eltype(Projection(e) * Multiplication(a)) == Float64
        @test eltype(project(Projection(ℱ), e, ℱ)) == Float64
        @test eltype(Projection(third) * Sequence(ℱ, ones(5))) == ComplexF64
        @test eltype(project(O, third, ℱ)) == ComplexF64
        @test eltype(O * Projection(third)) == ComplexF64
        @test eltype(Sequence(third, [1.0, 2.0, 3.0]) + Sequence(ℱ, ones(5))) == ComplexF64
        @test eltype(Projection(third) + O) == ComplexF64
        # between plain spaces the type is inferred
        b = Sequence(ℱ, [1.0, 2.0, 3.0, 4.0, 5.0])
        @test @inferred(project(b, Fourier(3, 1.0))) == project(b, Fourier(3, 1.0))
        @test @inferred(b + b) == 2b
    end
end
