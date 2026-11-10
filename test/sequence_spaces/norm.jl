# a weight on `Fourier` which is not even in the index, equal to itself
struct _OddWeight <: Weight end
RadiiPolynomial._getindex(::_OddWeight, ::Fourier, i::Int) = 2.0^i
Base.:(==)(::_OddWeight, ::_OddWeight) = true

_norm_desymmetrized(a, X) = norm(Projection(desymmetrize(space(a))) * a, X)

@testset "Norm" begin

    @testset "ScalarSpace" begin
        a = Sequence(ScalarSpace(), [-3.0])
        @test norm(a, Ell1()) == norm(a, Ell2()) == norm(a, EllInf()) ==
            norm(a, 1) == norm(a, 2) == norm(a, Inf) == norm(a) == 3.0

        ac = Sequence(ScalarSpace(), [3.0 + 4.0im])
        @test norm(ac, Ell1()) == norm(ac, Ell2()) == norm(ac, EllInf()) == 5.0

        A = LinearOperator(ScalarSpace(), ScalarSpace(), fill(-7.0, 1, 1))
        @test opnorm(A, Ell1()) == opnorm(A, Ell2()) == opnorm(A, EllInf()) ==
            opnorm(A, 1) == opnorm(A, 2) == opnorm(A, Inf) == opnorm(A) == 7.0
    end

    @testset "p-norm dispatch and guarded errors" begin
        a = Sequence(Taylor(2), [1.0, -2.0, 3.0])
        @test norm(a, 1) == norm(a, Ell1(IdentityWeight())) == 6.0
        @test norm(a, 2) == norm(a, Ell2(IdentityWeight())) == sqrt(14.0)
        @test norm(a, Inf) == norm(a, EllInf(IdentityWeight())) == norm(a) == 3.0
        @test_throws ArgumentError norm(a, 3)

        A = LinearOperator(Taylor(1), Taylor(1), [1.0 -2.0 ; 3.0 4.0])
        @test opnorm(A, 1) == opnorm(A, Ell1()) == 6.0                    # max column sum
        @test opnorm(A, 2) == opnorm(A, Ell2()) ≈ sqrt(30.0)
        @test opnorm(A, Inf) == opnorm(A, EllInf()) == opnorm(A) == 7.0   # max row sum
        @test_throws ArgumentError opnorm(A, 3)

        # `norm` is only defined for a `Sequence`
        @test_throws ArgumentError norm(A)
        @test_throws ArgumentError norm(A, Ell1())
        @test_throws ArgumentError norm(A, Ell1(), EllInf())
    end

    # Weight of the index i: νⁱ (geometric), (1+|i|)ᵖ (algebraic), 1+i² (Bessel). The weight of a
    # nonzero Chebyshev order is doubled.
    @testset "$(nameof(typeof(space(a))))" for (a, values) ∈ (
            Sequence(Taylor(3), [1.0, -2.0, 3.0, -4.0]) => (
                Ell1() => 10.0, Ell2() => sqrt(30.0), EllInf() => 4.0,
                Ell1(GeometricWeight(2.0)) => 49.0,           # 1·1 + 2·2 + 3·4 + 4·8
                Ell2(GeometricWeight(2.0)) => sqrt(173.0),    # 1·1 + 4·2 + 9·4 + 16·8
                EllInf(GeometricWeight(2.0)) => 32.0,
                Ell1(AlgebraicWeight(2.0)) => 100.0,          # 1·1 + 2·4 + 3·9 + 4·16
                Ell2(AlgebraicWeight(2.0)) => sqrt(354.0),    # 1·1 + 4·4 + 9·9 + 16·16
                EllInf(AlgebraicWeight(2.0)) => 64.0),
            Sequence(Fourier(2, 1.0), [1.0, -2.0, 3.0, -2.0, 1.0]) => (
                Ell1() => 9.0, Ell2() => sqrt(19.0), EllInf() => 3.0,
                Ell1(GeometricWeight(1.5)) => 13.5,           # 2·1·1.5² + 2·2·1.5 + 3
                Ell2(GeometricWeight(1.5)) => sqrt(25.5),     # 2·1·1.5² + 2·4·1.5 + 9
                EllInf(GeometricWeight(1.5)) => 3.0,          # max(2.25, 3, 3)
                Ell1(AlgebraicWeight(2.0)) => 37.0,           # 2·1·9 + 2·2·4 + 3
                EllInf(AlgebraicWeight(2.0)) => 9.0,
                Ell1(BesselWeight(1.0)) => 21.0,              # 2·1·5 + 2·2·2 + 3
                Ell2(BesselWeight(1.0)) => sqrt(35.0)),       # 2·1·5 + 2·4·2 + 9
            Sequence(Chebyshev(3), [1.0, -2.0, 3.0, -4.0]) => (
                Ell1() => 19.0,                               # 1 + 2·(2 + 3 + 4)
                Ell2() => sqrt(59.0),                         # 1 + 2·(4 + 9 + 16)
                EllInf() => 8.0,
                Ell1(GeometricWeight(2.0)) => 97.0,           # 1·1 + 2·4 + 3·8 + 4·16
                EllInf(GeometricWeight(2.0)) => 64.0,
                Ell1(AlgebraicWeight(2.0)) => 199.0))         # 1·1 + 2·8 + 3·18 + 4·32
        for (X, value) ∈ values
            @test norm(a, X) == value
        end
    end

    @testset "complex and interval coefficients" begin
        @test norm(Sequence(Taylor(1), [1.0 + 1.0im, 2.0 - 1.0im]), Ell1()) == sqrt(2.0) + sqrt(5.0)
        @test norm(Sequence(Fourier(1, 1.0), ComplexF64[1.0 + 1.0im, 2.0, 1.0 - 1.0im]), Ell1()) == 2sqrt(2.0) + 2.0

        # thin coefficients and a thin rate give a thin norm
        b = Sequence(Taylor(1), interval.([1.0, -2.0]))
        @test isequal_interval(norm(b, Ell1(GeometricWeight(interval(2.0)))), interval(5.0))

        c = Sequence(Taylor(1), [interval(1//3), interval(2//3)])
        @test in_interval(1.0, norm(c, Ell1()))
    end

    @testset "TensorSpace" begin
        # with unit coefficients the norm factors over the dimensions:
        # (1 + 2) for Taylor(1), (1 + 1 + 1) for Fourier(1), (1 + 2·2³) for Chebyshev(1)
        s = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1)
        a = Sequence(s, ones(dimension(s)))
        w = (GeometricWeight(2.0), IdentityWeight(), AlgebraicWeight(3.0))
        @test norm(a, Ell1(w)) == 153.0
        @test norm(a, Ell2(w)) == sqrt(153.0)
        @test norm(a, EllInf(w)) == 32.0   # 2 · 1 · 16

        # a lone weight applies to every dimension, so the Chebyshev doubling is kept
        b = ones(Chebyshev(2) ⊗ Chebyshev(2))
        @test norm(b, Ell1()) == norm(b, Ell1((IdentityWeight(), IdentityWeight()))) == 25.0   # (1+2+2)²
        @test norm(b, Ell2()) == 5.0
        @test norm(b, EllInf()) == 4.0
        @test norm(b, Ell1(GeometricWeight(2.0))) == norm(b, Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))) == 13.0^2
    end

    @testset "SymmetricSpace" begin
        # The weight of a representative is summed over its orbit. A Fourier reflection folds ±k
        # together, a Taylor or Chebyshev parity only restricts the indices.
        @testset "evensym and oddsym: $s" for (s, idx, value) ∈ (
                (evensym(Fourier(3, 1.0)), [0, 1, 2, 3], 19.0),   # 1 + 2·(2 + 3 + 4)
                (oddsym(Fourier(3, 1.0)), [1, 2, 3], 12.0),       # 2·(1 + 2 + 3)
                (evensym(Taylor(5)), [0, 2, 4], 6.0),
                (oddsym(Taylor(5)), [1, 3, 5], 6.0),
                (evensym(Chebyshev(5)), [0, 2, 4], 11.0),         # 1 + 2·(2 + 3)
                (oddsym(Chebyshev(5)), [1, 3, 5], 12.0))
            @test collect(indices(s)) == idx
            @test norm(Sequence(s, collect(1.0:length(idx))), Ell1()) == value
        end
        @test norm(Sequence(evensym(Fourier(3, 1.0)), [1.0, 2.0, 3.0, 4.0]), Ell1(GeometricWeight(2.0))) == 97.0

        @testset "d4sym" begin
            # the orbit of (0,0) has 1 element, the orbits of (1,0) and (1,1) have 4
            s = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
            @test collect(indices(s)) == [(0, 0), (1, 0), (1, 1)]
            a = Sequence(s, [1.0, 2.0, 3.0])
            @test norm(a, Ell1()) == 21.0   # 1·1 + 2·4 + 3·4
            @test norm(a, Ell1((GeometricWeight(1.5), GeometricWeight(1.5)))) ≈ 1.0 + 2.0 * 4 * 1.5 + 3.0 * 4 * 1.5^2
            @test norm(a, EllInf()) == 3.0  # the orbit length does not enter
        end

        @testset "the norm is that of the desymmetrization" begin
            odd = (_OddWeight(), _OddWeight())
            id = (IdentityWeight(), IdentityWeight())
            g23 = (GeometricWeight(2.0), GeometricWeight(3.0))
            swap = Group(GroupElement(LatticeAut([0 1 ; 1 0]), Cocycle(0//1, Rational{Int}[0//1, 0//1])))
            cases = (
                d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)) => (
                    Ell1((GeometricWeight(1.5), GeometricWeight(1.5))), Ell1(g23), Ell2(g23), EllInf(g23), EllInf(),
                    EllInf(BesselWeight(1.0)), Ell1((IdentityWeight(), GeometricWeight(1.0))),
                    Ell1((AlgebraicWeight(1.0), AlgebraicWeight(1.0))), Ell2((BesselWeight(1.0), BesselWeight(1.0))),
                    Ell1((AlgebraicWeight(0.0), IdentityWeight()))),
                # weights that the symmetry does not preserve
                evensym(Fourier(2, 1.0)) => (Ell1(_OddWeight()),),
                d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0)) => (Ell1(odd), EllInf(odd)),
                # the swap exchanges a Taylor index with a Chebyshev index, whose weight is doubled
                SymmetricSpace(Taylor(2) ⊗ Chebyshev(2), swap) => (Ell1(id), EllInf(id)),
                Chebyshev(2) ⊗ Chebyshev(2) ⊗ evensym(Fourier(2, 1.0)) => (
                    Ell1(), Ell1((GeometricWeight(1.1), GeometricWeight(1.2), GeometricWeight(1.3)))),
            )
            for (s, Xs) ∈ cases, X ∈ Xs
                a = Sequence(s, [sin(k) for k ∈ 1:dimension(s)])
                @test norm(a, X) ≈ _norm_desymmetrized(a, X)
            end
            @test !RadiiPolynomial._isinvariant(_OddWeight(), evensym(Fourier(2, 1.0)))
            @test !RadiiPolynomial._isinvariant(odd, d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0)))
            @test !RadiiPolynomial._isinvariant(id, SymmetricSpace(Taylor(2) ⊗ Chebyshev(2), swap))
        end

        @testset "dual norm and operator norm" begin
            # both are attained at the basis vectors, whose norms are those of their desymmetrization
            s = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
            n = dimension(s)
            e = [Sequence(s, Float64.(1:n .== i)) for i ∈ 1:n]
            c = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0]
            φ = LinearOperator(s, ScalarSpace(), transpose(c))
            A = LinearOperator(s, s, reshape(sin.(1:n^2), n, n))
            X₁ = Ell1((GeometricWeight(2.0), GeometricWeight(3.0)))
            X∞ = EllInf((GeometricWeight(2.0), GeometricWeight(3.0)))
            @test opnorm(φ, X₁) ≈ maximum(i -> abs(c[i]) / _norm_desymmetrized(e[i], X₁), 1:n)
            @test opnorm(φ, X∞) ≈ sum(i -> abs(c[i]) / _norm_desymmetrized(e[i], X∞), 1:n)
            @test opnorm(A, X₁) ≈ maximum(i -> _norm_desymmetrized(A * e[i], X₁) / _norm_desymmetrized(e[i], X₁), 1:n)
        end
    end

    @testset "NormedCartesianSpace" begin
        @testset "CartesianPower" begin
            a = Sequence(Taylor(1)^3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])   # block ℓ¹ norms 3, 7, 11
            @test norm(a, NormedCartesianSpace(Ell1(), Ell1())) == norm(a, Ell1()) == 21.0
            @test norm(a, NormedCartesianSpace(Ell1(), EllInf())) == 11.0
            @test norm(a, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(3.0^2 + 7.0^2 + 11.0^2)
        end

        @testset "CartesianProduct" begin
            dom = ScalarSpace() × Taylor(1)
            b = Sequence(dom, [2.0, -5.0, 3.0])
            A = LinearOperator(dom, ScalarSpace(), reshape([2.0, -5.0, 3.0], 1, 3))

            @test norm(b, NormedCartesianSpace((Ell1(), EllInf()), Ell1())) == 2.0 + 5.0
            @test norm(b, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 8.0^2)
            @test opnorm(A, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 5.0^2)

            # the dual of an outer ℓ¹ is a maximum over the blocks, the dual of an outer ℓ∞ is a sum,
            # and likewise within a block
            @test opnorm(A, NormedCartesianSpace((Ell1(), Ell1()), Ell1())) == 5.0     # max(2, max(5, 3))
            @test opnorm(A, NormedCartesianSpace((Ell1(), EllInf()), Ell1())) == 8.0   # max(2, 5 + 3)
            @test opnorm(A, NormedCartesianSpace((Ell1(), Ell1()), EllInf())) == 7.0   # 2 + max(5, 3)

            # three factors: the recursion unwinds through every level
            dom3 = ScalarSpace() × Taylor(1) × Taylor(1)
            b3 = Sequence(dom3, [2.0, -5.0, 3.0, 1.0, -1.0])
            A3 = LinearOperator(dom3, ScalarSpace(), reshape([2.0, -5.0, 3.0, 1.0, -1.0], 1, 5))
            @test norm(b3, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 8.0^2 + 2.0^2)
            @test opnorm(A3, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 5.0^2 + 1.0^2)

            # one inner space per component
            Xbad = NormedCartesianSpace((Ell1(), Ell1(), Ell1()), Ell1())
            @test_throws ArgumentError norm(b, Xbad)
            @test_throws ArgumentError opnorm(A, Xbad)
            @test_throws ArgumentError norm(Sequence(Taylor(1)^0, Float64[]), Ell1())
        end

        @testset "nested NormedCartesianSpace" begin
            s1 = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
            s2 = ScalarSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1
            a = Sequence(s1, collect(1.0:13.0))
            A = LinearOperator(s1, s2, ones(dimension(s2), dimension(s1)))
            w = (GeometricWeight(1.0), GeometricWeight(1.0), GeometricWeight(1.0))
            X = NormedCartesianSpace((EllInf(), NormedCartesianSpace(Ell1(w), EllInf())), EllInf())

            # the second block dominates: its coefficients of Chebyshev order 0 have weight 1,
            # those of Chebyshev order 1 have weight 2
            @test norm(a, X) == 2+4+6+3+5+7 + 2*(8+10+12+9+11+13) == 153.0
            @test opnorm(A, X, X) == 18.0
        end
    end

    @testset "opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)" begin
        A = LinearOperator(Taylor(1), Taylor(1), [1.0 -2.0 ; 3.0 4.0])

        # the X-dual norm of the vector of the Y-norms of the columns
        @test opnorm(A, Ell1(), EllInf()) == 4.0    # max(max(1, 3), max(2, 4))
        @test opnorm(A, EllInf(), Ell1()) == 10.0   # (1 + 3) + (2 + 4)
        @test opnorm(A, Ell1(), Ell1()) == opnorm(A, Ell1()) == 6.0
        @test opnorm(A, Ell2(), Ell2()) == opnorm(A, Ell2()) ≈ sqrt(30.0)

        s = Sequence(Taylor(2), [1.0, -2.0, 3.0])
        ℳ = Multiplication(s)
        @test opnorm(ℳ, Ell1()) == norm(s, Ell1()) == 6.0
        @test opnorm(ℳ, Ell2(GeometricWeight(2.0))) == norm(s, Ell2(GeometricWeight(2.0)))
    end

    @testset "opnorm of a functional" begin
        A = LinearOperator(Taylor(2), ScalarSpace(), reshape([1.0, -2.0, 3.0], 1, 3))
        w = GeometricWeight(2.0)   # weights 1, 2, 4
        @test opnorm(A, Ell1(w)) == 1.0          # max(1, 1, 0.75)
        @test opnorm(A, EllInf(w)) == 2.75       # 1 + 1 + 0.75
        @test opnorm(A, Ell2(w)) ≈ sqrt(5.25)    # sqrt(1 + 2 + 2.25)
    end

    @testset "InfiniteSequence" begin
        seq = Sequence(Taylor(2), [1.0, 0.5, 0.25])
        X0 = Ell1(GeometricWeight(2.0))
        a = InfiniteSequence(seq, X0; finite_error = 0.01, tail_error = 0.02)

        @test norm(seq, X0) == 3.0
        @test a.full_norm == norm(a) == norm(a, X0) == 3.03   # the norm of the sequence plus the total error

        # in a space where `banachspace(a)` embeds, the sharper of the two available bounds
        @test norm(a, Ell1(GeometricWeight(1.5))) == 1 + 0.5 * 1.5 + 0.25 * 1.5^2 + 0.03 == 2.3425
        # ℓ¹_ν shrinks as ν grows, so a bound says nothing about the norm for a larger rate
        @test_throws DomainError norm(a, Ell1(GeometricWeight(3.0)))

        b = InfiniteSequence(seq, Ell1(); tail_error = 0.1)
        @test_throws DomainError norm(b, Ell1(GeometricWeight(2.0)))
        @test norm(b, Ell1(GeometricWeight(1.0))) == norm(b, Ell1()) == norm(b)   # rate one is the identity weight

        d = differentiate(a)
        @test banachspace(d) == Ell1()
        @test_throws DomainError norm(d, Ell1(GeometricWeight(2.0)))
    end
end
