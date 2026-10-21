@testset "Projection" begin

    𝒯 = Taylor(2)
    ℱ = Fourier(2, 1.0)

    @testset "construction, domain, codomain" begin
        Π = Projection(𝒯)
        @test Π isa Projection{Taylor,Float64}
        @test domain(Π) == codomain(Π) == 𝒯
        @test eltype(Π) == Float64

        ΠC = Projection(𝒯, ComplexF64)
        @test eltype(ΠC) == ComplexF64

        # `domain`/`codomain` return the projected space itself whenever the space
        # they are queried with is compatible with it, and throw otherwise
        @test domain(Π, Taylor(5)) == 𝒯
        @test codomain(Π, Taylor(0)) == 𝒯

        Πf = Projection(Fourier(2, 1.0))
        @test_throws ArgumentError domain(Πf, Fourier(2, 2.0))
        @test_throws ArgumentError codomain(Πf, Fourier(2, 2.0))
        @test_throws ArgumentError codomain(Π, UndefSpace())
    end

    @testset "coefficients(Projection) materializes the identity" begin
        Π = Projection(𝒯)
        @test coefficients(Π) == LinearOperator(𝒯, 𝒯, [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0])
    end

    @testset "interval(Projection)" begin
        Π = Projection(𝒯)
        Π_T = interval(Float64, Π)
        @test Π_T.space == 𝒯
        @test eltype(Π_T) == Interval{Float64}

        Πf = Projection(ℱ)
        Πf_T = interval(Float64, Πf)
        @test isequal_interval(frequency(Πf_T.space), interval(1.0))
        @test eltype(Πf_T) == Interval{Float64}
    end

    @testset "project/project! (Sequence)" begin

        @testset "Taylor" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            @test project(a, Taylor(1)) == a
            @test project(a, Taylor(3)) == Sequence(Taylor(3), [1.0, 2.0, 0.0, 0.0]) # enlarge: zero-pad
            @test project(a, Taylor(0)) == Sequence(Taylor(0), [1.0]) # shrink: truncate

            c = Sequence(Taylor(3), fill(Inf, 4))
            project!(c, a)
            @test c == Sequence(Taylor(3), [1.0, 2.0, 0.0, 0.0])
        end

        @testset "Fourier" begin
            b = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]) # b₋₁ = 1, b₀ = 2, b₁ = 3
            @test project(b, Fourier(2, 1.0)) == Sequence(Fourier(2, 1.0), [0.0, 1.0, 2.0, 3.0, 0.0])
            @test project(b, Fourier(0, 1.0)) == Sequence(Fourier(0, 1.0), [2.0])
            @test_throws ArgumentError project(b, Fourier(1, 2.0))
        end

        @testset "Chebyshev" begin
            c = Sequence(Chebyshev(1), [1.0, 2.0])
            @test project(c, Chebyshev(3)) == Sequence(Chebyshev(3), [1.0, 2.0, 0.0, 0.0])
            @test project(c, Chebyshev(0)) == Sequence(Chebyshev(0), [1.0])
        end

        @testset "TensorSpace" begin
            # Taylor(1) ⊗ Chebyshev(1) is ordered with Taylor varying fastest:
            # (0,0), (1,0), (0,1), (1,1)
            t = Sequence(Taylor(1) ⊗ Chebyshev(1), [1.0, 2.0, 3.0, 4.0])
            @test project(t, Taylor(2) ⊗ Chebyshev(0)) == Sequence(Taylor(2) ⊗ Chebyshev(0), [1.0, 2.0, 0.0])
        end

        @testset "CartesianPower / CartesianProduct" begin
            a = Sequence(Taylor(1)^2, [1.0, 2.0, 3.0, 4.0]) # component 1 = [1,2], component 2 = [3,4]
            @test project(a, Taylor(2)^2) == Sequence(Taylor(2)^2, [1.0, 2.0, 0.0, 3.0, 4.0, 0.0])
            @test_throws ArgumentError project(a, Taylor(1)^3)

            # a scalar block next to a one-component power of a three-factor tensor space
            s1 = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
            s2 = ScalarSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
            s3 = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0))^1
            s4 = ScalarSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(0))^1
            v = Sequence(s1, [1.0 ; 1.0:12.0])
            @test project(v, s1) == v
            @test project(v, s2) == Sequence(s2, [1.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0, 7.0, 8.0, 0.0, 9.0, 10.0, 0.0, 11.0, 12.0, 0.0])
            @test project(v, s3) == Sequence(s3, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            @test project(v, s4) == Sequence(s4, [1.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 5.0, 6.0, 0.0])
        end

        @testset "SymmetricSpace" begin
            𝒯e2, 𝒯e4 = evensym(Taylor(2)), evensym(Taylor(4))
            @test indices(𝒯e2) == 0:2:2
            @test indices(𝒯e4) == 0:2:4
            ae = Sequence(𝒯e2, [1.0, 2.0]) # a₀ = 1, a₂ = 2
            be = project(ae, 𝒯e4)
            @test be == Sequence(𝒯e4, [1.0, 2.0, 0.0]) # enlarge: zero-pad a₄
            @test project(be, 𝒯e2) == ae

            # an even Fourier space keeps only the |k| representatives: a cosine series
            ℱe1, ℱe3 = evensym(Fourier(1, 1.0)), evensym(Fourier(3, 1.0))
            af = Sequence(ℱe1, [1.0, 2.0])
            @test project(af, ℱe3) == Sequence(ℱe3, [1.0, 2.0, 0.0, 0.0])
            @test project(project(af, ℱe3), ℱe1) == af

            # projecting onto a different symmetry group is the orthogonal projection: the even part of an odd sequence vanishes
            𝒯o2 = oddsym(Taylor(2))
            ao = Sequence(𝒯o2, [10.0])
            @test project(ao, 𝒯e4) == Sequence(𝒯e4, zeros(3))

            # and it is the identity when the target symmetry is weaker (a subgroup): d4 ⊃ {id, reflection in x}
            𝒯² = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
            ex = SymmetricSpace(𝒯², Group(GroupElement(LatticeAut([-1 0 ; 0 1]), Cocycle(0//1, Rational{Int}[0//1, 0//1]))))
            ad4 = Sequence(d4sym(𝒯²), [1.0, 2.0, 3.0])
            @test project(project(ad4, ex), 𝒯²) == project(ad4, 𝒯²)

            # projecting a plain sequence onto a symmetric space is the orthogonal projection, i.e. the
            # average over the orbits: the even part of a sine vanishes
            ℱ2 = Fourier(2, 1.0)
            sine = Sequence(ℱ2, [0.0, -1.0, 0.0, 1.0, 0.0])
            @test project(sine, evensym(ℱ2)) == Sequence(evensym(ℱ2), zeros(3))
            @test project(sine, oddsym(ℱ2)) == Sequence(oddsym(ℱ2), [1.0, 0.0])
            @test project(Sequence(ℱ2, [1.0, 2.0, 3.0, 4.0, 5.0]), evensym(ℱ2)) == Sequence(evensym(ℱ2), [3.0, 3.0, 3.0])
            # and a sequence already carrying the symmetry round trips exactly, even over orbits of size 8
            ae2 = Sequence(evensym(ℱ2), [1.0, 0.5, 0.2])
            @test project(project(ae2, ℱ2), evensym(ℱ2)) == ae2
            @test project(project(ad4, 𝒯²), d4sym(𝒯²)) == ad4
        end

        @testset "SymmetricSpace: operators" begin
            ℱ = Fourier(2, 1.0)
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
                @test B * x ≈ B * project(x, ℱ) # acting on a symmetric sequence desymmetrizes it
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
            ex = SymmetricSpace(𝒯², Group(GroupElement(LatticeAut([-1 0 ; 0 1]), Cocycle(0//1, Rational{Int}[0//1, 0//1]))))
            Ad4 = LinearOperator(d4sym(𝒯²), d4sym(𝒯²), [1.0 2.0 3.0 ; 4.0 5.0 6.0 ; 7.0 8.0 9.0])
            Afull = project(Ad4, 𝒯², 𝒯²)
            @test project(Ad4, ex, ex) ≈ project(Afull, ex, ex)
            @test project(Ad4, 𝒯², ex) ≈ project(Afull, 𝒯², ex) && project(Ad4, ex, 𝒯²) ≈ project(Afull, ex, 𝒯²)
            # a scalar side is read as the zero space of the other side: a functional sums the orbits of its domain
            # and a column averages the orbits of its codomain, like any operator
            a = Sequence(ℱ, [0.1, -0.2, 1.0, 0.4, 0.3])
            @test (project(adjoint(a), es, ScalarSpace()) * xe)[1] ≈ (adjoint(a) * project(xe, ℱ))[1]
            @test project(LinearOperator(a), ScalarSpace(), es) == LinearOperator(project(a, es))
            @test (project(adjoint(xe) * Multiplication(a), es, ScalarSpace()) * xe)[1] ≈ (adjoint(xe) * (a * project(xe, ℱ)))[1]
        end

        @testset "SymmetricSpace: identity and diagonal operators across symmetries" begin
            ℱ = Fourier(1, 1.0)
            es, os = evensym(ℱ), oddsym(ℱ)
            ts = SymmetricSpace(ℱ, symmetry(ℱ)) # trivial group: the same sequences as ℱ, read through the symmetric machinery
            U = UniformScalingOperator(1.0)
            Iℱ = project(U, ℱ, ℱ)
            # the identity between different symmetries is the orthogonal projection, however it is materialized
            @test coefficients(project(U, ℱ, es)) == coefficients(project(Iℱ, ℱ, es)) == [0.0 1.0 0.0 ; 0.5 0.0 0.5]
            @test coefficients(project(U, ts, os)) == coefficients(project(Iℱ, ℱ, os)) == [-0.5 0.0 0.5]
            A = LinearOperator(ℱ, es, zeros(2, 3))
            @test coefficients(one(A)) == coefficients(project(A + I, ℱ, es)) == [0.0 1.0 0.0 ; 0.5 0.0 0.5]
            # a diagonal operator maps a space to itself: across symmetries it is materialized first, then compressed
            struct _LatticeDiag <: RadiiPolynomial.AbstractDiagonalOperator end
            RadiiPolynomial.getcoefficient(::_LatticeDiag, (codom, i)::Tuple{VectorSpace,Any}, (dom, j)::Tuple{VectorSpace,Any}, ::Type{T}) where {T} =
                ifelse(i == j, convert(T, 2), zero(T))
            D = _LatticeDiag()
            @test_throws ArgumentError project(D, ℱ, es, Float64)
            @test coefficients(project(project(D, ℱ, ℱ, Float64), ℱ, es)) == [0.0 2.0 0.0 ; 1.0 0.0 1.0]
            # the composition of projections is the projection onto the intersection
            @test (Projection(es) * Projection(ℱ)).space == (Projection(ℱ) ∘ Projection(es)).space == es
        end

        @testset "ScalarSpace <-> SequenceSpace isomorphism" begin
            # a scalar embeds as the order/mode-0 coefficient; conversely, projecting
            # a sequence onto ScalarSpace extracts its order/mode-0 coefficient
            s = Sequence(ScalarSpace(), [7.0])
            @test project(s, Taylor(2)) == Sequence(Taylor(2), [7.0, 0.0, 0.0])
            @test project(s, Fourier(1, 1.0)) == Sequence(Fourier(1, 1.0), [0.0, 7.0, 0.0])
            @test project(s, Chebyshev(2)) == Sequence(Chebyshev(2), [7.0, 0.0, 0.0])

            d = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            @test project(d, ScalarSpace()) == Sequence(ScalarSpace(), [1.0])
        end

        @testset "Interval{Float64} and ComplexF64 coefficients" begin
            a = Sequence(Taylor(1), interval.([1.0, 2.0]))
            b = project(a, Taylor(3), Interval{Float64})
            # `Sequence == Sequence` compares interval bounds, so it is safe here
            @test b == Sequence(Taylor(3), interval.([1.0, 2.0, 0.0, 0.0]))
            @test all(isguaranteed, coefficients(b))

            ac = Sequence(Taylor(1), ComplexF64[1.0 + 1.0im, 2.0 - 1.0im])
            @test project(ac, Taylor(3)) == Sequence(Taylor(3), ComplexF64[1.0 + 1.0im, 2.0 - 1.0im, 0.0, 0.0])

            # a projection carries its own coefficient type
            Π = Projection(Taylor(2), Complex{Interval{Float64}})
            r = Π * a
            @test eltype(r) == Complex{Interval{Float64}}
        end
    end

    @testset "project/project! (LinearOperator / AbstractLinearOperator)" begin

        @testset "AbstractLinearOperator (generic getcoefficient loop)" begin
            # ∂/∂x on Taylor(2): [x⁰,x¹,x²] ↦ [x⁰,x¹], (∂a)_k = (k+1) a_{k+1}
            ∂ = Derivative(1)
            expected = LinearOperator(Taylor(2), Taylor(1), [0.0 1.0 0.0 ; 0.0 0.0 2.0])
            @test project(∂, Taylor(2), Taylor(1), Float64) == expected

            C = LinearOperator(Taylor(2), Taylor(1), fill(Inf, 2, 3))
            project!(C, ∂)
            @test C == expected
        end

        @testset "LinearOperator (fast _radd! path): enlarging/shrinking domain and codomain" begin
            A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            enlarged = LinearOperator(Taylor(2), Taylor(2), [1.0 2.0 0.0 ; 3.0 4.0 0.0 ; 0.0 0.0 0.0])
            @test project(A, Taylor(2), Taylor(2)) == enlarged

            C = LinearOperator(Taylor(2), Taylor(2), fill(Inf, 3, 3))
            project!(C, A)
            @test C == enlarged

            shrunk = LinearOperator(Taylor(0), Taylor(0), reshape([1.0], 1, 1))
            @test project(A, Taylor(0), Taylor(0)) == shrunk
        end

        @testset "LinearOperator on CartesianPower domain/codomain" begin
            A = LinearOperator(Taylor(1)^2, Taylor(1)^2, Float64[1 2 0 0 ; 3 4 0 0 ; 0 0 5 6 ; 0 0 7 8])
            B = project(A, Taylor(2)^2, Taylor(2)^2)
            expected = LinearOperator(Taylor(2)^2, Taylor(2)^2,
                [1.0 2.0 0.0 0.0 0.0 0.0
                 3.0 4.0 0.0 0.0 0.0 0.0
                 0.0 0.0 0.0 0.0 0.0 0.0
                 0.0 0.0 0.0 5.0 6.0 0.0
                 0.0 0.0 0.0 7.0 8.0 0.0
                 0.0 0.0 0.0 0.0 0.0 0.0])
            @test B == expected
            @test_throws ArgumentError project(A, Taylor(1)^3, Taylor(1)^2)
        end

        @testset "custom AbstractLinearOperator: the three documented methods suffice" begin
            #= A custom operator needs only `domain`, `codomain` and `getcoefficient`:
               materialization goes through `getcoefficient` and never consults
               `Base.getindex`, so defining the latter instead is a MethodError. =#
            struct _MyDiagOp <: AbstractLinearOperator end
            RadiiPolynomial.domain(A::_MyDiagOp, codom::Taylor) = codom
            RadiiPolynomial.codomain(A::_MyDiagOp, dom::Taylor) = dom
            RadiiPolynomial.getcoefficient(A::_MyDiagOp, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}) =
                i == j ? inv(1.0+i) : 0.0

            A = _MyDiagOp()
            expected = LinearOperator(Taylor(2), Taylor(2), [1.0 0.0 0.0 ; 0.0 0.5 0.0 ; 0.0 0.0 1/3])
            @test A * Projection(Taylor(2)) == expected
            @test Projection(Taylor(2)) * A == expected
            @test project(A, Taylor(2), Taylor(2)) == expected
            # the 3-argument form is enough: the generic 4-argument method forwards to it
            @test RadiiPolynomial.getcoefficient(A, (Taylor(2), 1), (Taylor(2), 1), Float64) == 0.5
            # the entries are assumed to be representable in the requested type, `Float64` by default
            @test eltype(project(A, Taylor(2), Taylor(2))) == Float64
            @test eltype(Projection(Taylor(2), ComplexF64) * A) == ComplexF64
            @test eltype(project(A, Taylor(2), Taylor(2), ComplexF64)) == ComplexF64
        end

        @testset "custom AbstractLinearOperator with an undefined codomain" begin
            #= `A = (I - S/2)⁻¹ = Σₖ Sᵏ/2ᵏ` with `S` the shift `xʲ ↦ xʲ⁺¹`: every column is
               infinite, so `codomain` returns `UndefSpace()` and the operator can only be
               materialized once a codomain truncation is supplied (mirror of an undefined
               `domain`, which is materialized once a domain truncation is supplied). =#
            struct _MyResolvent <: AbstractLinearOperator end
            RadiiPolynomial.domain(::_MyResolvent, codom::SequenceSpace) = codom
            RadiiPolynomial.codomain(::_MyResolvent, ::SequenceSpace) = UndefSpace()
            RadiiPolynomial.getcoefficient(::_MyResolvent, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}) =
                i ≥ j ? 2.0^(j-i) : 0.0

            A = _MyResolvent()
            𝒯₁ = Taylor(1)
            𝒯₂ = Taylor(2)
            Π₁ = Projection(𝒯₁)
            Π₂ = Projection(𝒯₂)
            expected = LinearOperator(𝒯₁, 𝒯₂, [1.0 0.0 ; 0.5 1.0 ; 0.25 0.5])

            # the generic `project!` takes the supplied codomain as the truncation and checks
            # the compatibility of the truncations through the domain instead
            @test project(A, 𝒯₁, 𝒯₂) == expected
            @test_throws ArgumentError project(A, Fourier(1, 1.0), 𝒯₂)
            @test Π₂ * A * Π₁ == expected
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
            expected_S = LinearOperator(𝒯₁, 𝒯₂, [1.0 0.0 ; 0.0 1.0 ; 0.0 0.0] - 2coefficients(expected))
            @test project(S, 𝒯₁, 𝒯₂) == expected_S
            @test Π₂ * S * Π₁ == expected_S
            # composing on either side: the intermediate truncation is the codomain of the inner
            # operator, or the domain of the outer operator when the former is undefined
            @test project(A ∘ M, 𝒯₁, 𝒯₂) == project(M ∘ A, 𝒯₁, 𝒯₂) == 2expected
            # the type of the outer operator is still inferred when the intermediate space is undefined
            Mc = Multiplication(Sequence(Taylor(0), [2.0im]))
            @test eltype(project(Mc ∘ A, 𝒯₁, 𝒯₂)) == ComplexF64
            @test M * A isa ComposedOperator
            @test project(M * project(A, 𝒯₁, 𝒯₂), 𝒯₁, 𝒯₂) == 2expected
            # both truncations undefined: nothing can be materialized
            ℰ = Evaluation(0.5)
            @test_throws ArgumentError project(ℰ ∘ A, 𝒯₁, Taylor(0))

            # a block with an undefined codomain makes the matrix product elementwise
            v = [A 2.0] * Projection(𝒯₁ × 𝒯₁)
            @test v[1,1] isa ComposedOperator
            @test v[1,2] == LinearOperator(𝒯₁, 𝒯₁, [2.0 0.0 ; 0.0 2.0])
        end

        @testset "AbstractDiagonalOperator: only domain ∩ codomain indices are written" begin
            # a minimal custom diagonal operator, defined through `getcoefficient` alone
            struct _ScalarDiag{T<:Number} <: RadiiPolynomial.AbstractDiagonalOperator
                λ :: T
            end
            RadiiPolynomial.getcoefficient(A::_ScalarDiag, (codom, i)::Tuple{VectorSpace,Any}, (dom, j)::Tuple{VectorSpace,Any}, ::Type{T}) where {T} =
                ifelse(i == j, convert(T, A.λ), zero(T))

            D = _ScalarDiag(2.0)

            C = LinearOperator(Taylor(2), Taylor(2), fill(Inf, 3, 3))
            project!(C, D)
            @test C == LinearOperator(Taylor(2), Taylor(2), [2.0 0.0 0.0 ; 0.0 2.0 0.0 ; 0.0 0.0 2.0])

            # codomain bigger than domain: only the domain ∩ codomain = Taylor(2) indices
            # {0,1,2} get the diagonal value, the extra rows {3,4} stay zero
            C2 = LinearOperator(Taylor(2), Taylor(4), fill(Inf, 5, 3))
            project!(C2, D)
            @test C2 == LinearOperator(Taylor(2), Taylor(4),
                [2.0 0.0 0.0 ; 0.0 2.0 0.0 ; 0.0 0.0 2.0 ; 0.0 0.0 0.0 ; 0.0 0.0 0.0])
        end

        @testset "Sequence <-> LinearOperator(ScalarSpace, ·) round trip" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            A = project(a, ScalarSpace(), Taylor(2))
            @test A isa LinearOperator{ScalarSpace,Taylor}
            @test coefficients(A) == reshape([1.0, 2.0, 3.0], 3, 1)
            @test project(A, Taylor(2)) == a

            C = LinearOperator(ScalarSpace(), Taylor(2), fill(Inf, 3, 1))
            project!(C, a)
            @test C == A
        end
    end

    @testset "action, composition and materialization" begin
        Π₂ = Projection(Taylor(2))
        Π₄ = Projection(Taylor(4))

        @testset "Projection * Sequence (action = project)" begin
            a = Sequence(Taylor(1), [1.0, 2.0])
            @test Π₄ * a == project(a, Taylor(4))
            @test eltype(Projection(Taylor(1), ComplexF64) * a) == ComplexF64
        end

        @testset "Projection * Projection / ∘" begin
            @test (Π₂ * Π₄).space == Taylor(2) # intersect(Taylor(2), Taylor(4))
            @test (Π₂ ∘ Π₄).space == Taylor(2)
        end

        @testset "Projection * LinearOperator / LinearOperator * Projection" begin
            A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            # `Π * A`: resize the CODOMAIN of A to Π.space, keep A's domain
            @test Π₂ * A == LinearOperator(Taylor(1), Taylor(2), [1.0 2.0 ; 3.0 4.0 ; 0.0 0.0])
            # `A * Π`: resize the DOMAIN of A to Π.space, keep A's codomain
            @test A * Π₂ == LinearOperator(Taylor(2), Taylor(1), [1.0 2.0 0.0 ; 3.0 4.0 0.0])
        end

        @testset "Projection * AbstractLinearOperator / AbstractLinearOperator * Projection (materialize)" begin
            ∂ = Derivative(1)
            Π₃ = Projection(Taylor(3))
            # `∂ * Π₃`: materializes with domain = Π₃.space, codomain = codomain(∂, Π₃.space)
            r1 = ∂ * Π₃
            @test r1 == LinearOperator(Taylor(3), Taylor(2), [0.0 1.0 0.0 0.0 ; 0.0 0.0 2.0 0.0 ; 0.0 0.0 0.0 3.0])
            # `Π₃ * ∂`: materializes with codomain = Π₃.space, domain = domain(∂, Π₃.space)
            r2 = Π₃ * ∂
            @test r2 == LinearOperator(Taylor(4), Taylor(3),
                [0.0 1.0 0.0 0.0 0.0
                 0.0 0.0 2.0 0.0 0.0
                 0.0 0.0 0.0 3.0 0.0
                 0.0 0.0 0.0 0.0 4.0])
        end

        @testset "Projection ± LinearOperator (materialize Π first)" begin
            A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            Π₁ = Projection(Taylor(1))
            I2 = LinearOperator(Taylor(1), Taylor(1), [1.0 0.0 ; 0.0 1.0])
            @test Π₁ + A == I2 + A == LinearOperator(Taylor(1), Taylor(1), [2.0 2.0 ; 3.0 5.0])
            @test A + Π₁ == A + I2
            @test Π₁ - A == I2 - A == LinearOperator(Taylor(1), Taylor(1), [0.0 -2.0 ; -3.0 -3.0])
            @test A - Π₁ == A - I2
        end

        @testset "Add{Projection,±Projection} / Negate{Projection} trigger materialization" begin
            ∂ = Derivative(1)
            @test (Π₂ + Π₄) * ∂ == Π₂ * ∂ + Π₄ * ∂
            @test ∂ * (Π₂ + Π₄) == ∂ * Π₂ + ∂ * Π₄
            @test (Π₂ - Π₄) * ∂ == Π₂ * ∂ - Π₄ * ∂
            @test ∂ * (Π₂ - Π₄) == ∂ * Π₂ - ∂ * Π₄
            @test (-Π₂) * ∂ == -(Π₂ * ∂)
            @test ∂ * (-Π₂) == -(∂ * Π₂)
        end

        @testset "Projection(UndefSpace()) * AbstractLinearOperator: lazy ComposedOperator" begin
            # the domain over an undefined space is itself undefined, so there is nothing
            # to materialize and the pair is wrapped lazily instead
            Π_undef = Projection(UndefSpace())
            ∂ = Derivative(1)
            r = Π_undef * ∂
            @test r isa ComposedOperator
            @test r.outer === Π_undef
            @test r.inner === ∂
        end

        @testset "AbstractLinearOperator * Projection(UndefSpace()): lazy ComposedOperator" begin
            # mirror: the codomain over an undefined space is itself undefined
            Π_undef = Projection(UndefSpace())
            ∂ = Derivative(1)
            r = ∂ * Π_undef
            @test r isa ComposedOperator
            @test r.outer === ∂
            @test r.inner === Π_undef
        end
    end

    @testset "Projection{<:CartesianSpace} * Vector/Matrix/Diagonal" begin

        @testset "Projection * Vector{<:Sequence}" begin
            Π = Projection(Taylor(2)^2)
            v = [Sequence(Taylor(1), [1.0, 2.0]), Sequence(Taylor(3), [4.0, 5.0, 6.0, 7.0])]
            r = Π * v
            @test r == Sequence(Taylor(2)^2, [1.0, 2.0, 0.0, 4.0, 5.0, 6.0]) # enlarge/shrink each component
            @test_throws DimensionMismatch Projection(Taylor(2)^3) * v # one component too few
        end

        @testset "Projection * Matrix{<:LinearOperator}" begin
            Π = Projection(Taylor(2) × Taylor(1))
            v11 = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            v12 = LinearOperator(Taylor(0), Taylor(0), reshape([5.0], 1, 1))
            v21 = LinearOperator(Taylor(1), Taylor(1), [6.0 7.0 ; 8.0 9.0])
            v22 = LinearOperator(Taylor(0), Taylor(0), reshape([10.0], 1, 1))
            r = Π * [v11 v12 ; v21 v22]
            # each block's CODOMAIN is resized to Π.space[i]; the domain of the
            # assembled operator is the union, column-wise, of the (unresized) domains
            @test domain(r) == Taylor(1) × Taylor(0)
            @test codomain(r) == Taylor(2) × Taylor(1)
            @test coefficients(r) == [1.0 2.0 5.0 ; 3.0 4.0 0.0 ; 0.0 0.0 0.0 ; 6.0 7.0 10.0 ; 8.0 9.0 0.0]
        end

        @testset "Matrix{<:LinearOperator} * Projection" begin
            Π = Projection(Taylor(1) × Taylor(0))
            v11 = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
            v12 = LinearOperator(Taylor(1), Taylor(0), reshape([5.0, 6.0], 1, 2))
            v21 = LinearOperator(Taylor(0), Taylor(1), reshape([7.0, 8.0], 2, 1))
            v22 = LinearOperator(Taylor(0), Taylor(0), reshape([9.0], 1, 1))
            r = [v11 v12 ; v21 v22] * Π
            # each block's DOMAIN is resized to Π.space[j]; the codomain of the
            # assembled operator is the union, row-wise, of the (unresized) codomains
            @test domain(r) == Taylor(1) × Taylor(0)
            @test codomain(r) == Taylor(1) × Taylor(1)
            @test coefficients(r) == [1.0 2.0 5.0 ; 3.0 4.0 0.0 ; 7.0 0.0 9.0 ; 8.0 0.0 0.0]
        end

        @testset "Projection * Diagonal / Diagonal * Projection" begin
            Π = Projection(Taylor(1)^2)
            D = RadiiPolynomial.LinearAlgebra.Diagonal([2.0, 3.0])
            expected = LinearOperator(Taylor(1)^2, Taylor(1)^2, [2.0 0.0 0.0 0.0 ; 0.0 2.0 0.0 0.0 ; 0.0 0.0 3.0 0.0 ; 0.0 0.0 0.0 3.0])
            @test Π * D == expected
            @test D * Π == expected
        end
    end

    @testset "eltype covariance, interval wrapping, domain(UndefSpace)" begin
        # the type-level `eltype` matches a concrete instantiation, not just the UnionAll
        @test eltype(Projection{Taylor,Float64}) == Float64

        # the one-argument `interval` wraps the coefficient type of the projection
        Π = Projection(𝒯)
        Π_I = interval(Π)
        @test Π_I.space == 𝒯
        @test eltype(Π_I) == Interval{Float64}

        # both sides check compatibility, and `Taylor(2)` is incompatible with an
        # undefined space
        @test_throws ArgumentError domain(Π, UndefSpace())
        @test_throws ArgumentError codomain(Π, UndefSpace())
    end
end
