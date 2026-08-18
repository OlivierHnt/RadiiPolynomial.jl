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

            # projecting between different symmetry groups is an error
            𝒯o2 = oddsym(Taylor(2))
            ao = Sequence(𝒯o2, [10.0])
            @test_throws DomainError project(ao, 𝒯e4)
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
