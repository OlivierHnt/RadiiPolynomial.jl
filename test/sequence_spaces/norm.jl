@testset "Norm" begin

    @testset "ScalarSpace" begin
        a = Sequence(ScalarSpace(), [-3.0])
        @test norm(a, Ell1()) == norm(a, Ell2()) == norm(a, EllInf()) ==
            norm(a, 1) == norm(a, 2) == norm(a, Inf) == norm(a) == 3.0

        ac = Sequence(ScalarSpace(), [3.0 + 4.0im])
        @test norm(ac, Ell1()) == norm(ac, Ell2()) == norm(ac, EllInf()) == abs(3.0 + 4.0im) == 5.0

        # opnorm of a functional ScalarSpace → ScalarSpace uses the same `abs(a[1])`
        A = LinearOperator(ScalarSpace(), ScalarSpace(), fill(-7.0, 1, 1))
        @test opnorm(A, Ell1()) == opnorm(A, Ell2()) == opnorm(A, EllInf()) ==
            opnorm(A, 1) == opnorm(A, 2) == opnorm(A, Inf) == opnorm(A) == 7.0
    end

    @testset "p-norm dispatch and guarded errors" begin
        a = Sequence(Taylor(2), [1.0, -2.0, 3.0])
        @test norm(a, 1) == norm(a, Ell1(IdentityWeight())) == 6.0          # |1|+|2|+|3|
        @test norm(a, 2) == norm(a, Ell2(IdentityWeight())) == sqrt(14.0)   # sqrt(1+4+9)
        @test norm(a, Inf) == norm(a, EllInf(IdentityWeight())) == norm(a) == 3.0
        @test_throws ArgumentError norm(a, 3)

        A = LinearOperator(Taylor(1), Taylor(1), [1.0 -2.0 ; 3.0 4.0])
        @test opnorm(A, 1) == opnorm(A, Ell1()) == 6.0            # max column ℓ¹-sum: max(1+3, 2+4)
        @test opnorm(A, 2) == opnorm(A, Ell2()) ≈ sqrt(30.0)
        @test opnorm(A, Inf) == opnorm(A, EllInf()) == opnorm(A) == 7.0   # max row ℓ¹-sum: max(1+2, 3+4)
        @test_throws ArgumentError opnorm(A, 3)

        # `norm` is only defined for `Sequence`; `opnorm` must be used for `LinearOperator`
        @test_throws ArgumentError norm(A)
        @test_throws ArgumentError norm(A, Ell1())
        @test_throws ArgumentError norm(A, Ell1(), EllInf())
    end

    @testset "Taylor" begin
        a = Sequence(Taylor(3), [1.0, -2.0, 3.0, -4.0])
        @test norm(a, Ell1()) == 10.0          # 1+2+3+4
        @test norm(a, Ell2()) == sqrt(30.0)    # sqrt(1+4+9+16)
        @test norm(a, EllInf()) == 4.0

        ν = 2.0
        w = GeometricWeight(ν)  # weight_i = ν^i : 1, 2, 4, 8
        @test norm(a, Ell1(w)) == 49.0          # 1·1+2·2+3·4+4·8
        @test norm(a, Ell2(w)) == sqrt(173.0)   # 1·1+4·2+9·4+16·8
        @test norm(a, EllInf(w)) == 32.0        # max(1,4,12,32)

        p = 2.0
        wa = AlgebraicWeight(p)  # weight_i = (1+i)^p : 1, 4, 9, 16
        @test norm(a, Ell1(wa)) == 100.0         # 1·1+2·4+3·9+4·16
        @test norm(a, Ell2(wa)) == sqrt(354.0)   # 1·1+4·4+9·9+16·16
        @test norm(a, EllInf(wa)) == 64.0        # max(1,8,27,64)

        ac = Sequence(Taylor(1), [1.0 + 1.0im, 2.0 - 1.0im])
        @test norm(ac, Ell1()) == sqrt(2.0) + sqrt(5.0)

        # interval coefficients and interval geometric rate: thin inputs ⇒ exact enclosure
        b = Sequence(Taylor(1), interval.([1.0, -2.0]))
        nb = norm(b, Ell1(GeometricWeight(interval(2.0))))
        @test in_interval(5.0, nb)   # 1·1 + 2·2 = 5
        @test isequal_interval(nb, interval(5.0))

        # non-exact interval coefficients: the true value 1 is still enclosed
        c = Sequence(Taylor(1), [interval(1//3), interval(2//3)])
        @test in_interval(1.0, norm(c, Ell1()))
    end

    @testset "Fourier" begin
        a = Sequence(Fourier(2, 1.0), [1.0, -2.0, 3.0, -2.0, 1.0])  # modes k = -2,…,2
        @test norm(a, Ell1()) == 9.0           # 1+2+3+2+1
        @test norm(a, Ell2()) == sqrt(19.0)    # sqrt(1+4+9+4+1)
        @test norm(a, EllInf()) == 3.0

        ρ = 1.5
        w = GeometricWeight(ρ)  # weight_|k| : |k|=2→ρ², |k|=1→ρ, k=0→1
        @test norm(a, Ell1(w)) == 13.5                                    # 1ρ²+2ρ+3+2ρ+1ρ²
        @test norm(a, Ell2(w)) == sqrt(1ρ^2 + 4ρ + 9 + 4ρ + 1ρ^2)
        @test norm(a, EllInf(w)) == 3.0                                   # max(1ρ², 2ρ, 3) = max(2.25,3,3)=3

        p = 2.0
        wa = AlgebraicWeight(p)  # weight_|k| = (1+|k|)^p : |k|=2→9, |k|=1→4, k=0→1
        @test norm(a, Ell1(wa)) == 37.0                                    # 1·9+2·4+3·1+2·4+1·9
        @test norm(a, EllInf(wa)) == 9.0                                   # max(9,8,3)

        bw = BesselWeight(1.0)  # weight_k = 1+k² : |k|=2→5, |k|=1→2, k=0→1
        @test norm(a, Ell1(bw)) == 21.0            # 1·5+2·2+3·1+2·2+1·5
        @test norm(a, Ell2(bw)) == sqrt(1*5.0 + 4*2.0 + 9*1.0 + 4*2.0 + 1*5.0)

        ac = Sequence(Fourier(1, 1.0), ComplexF64[1.0 + 1.0im, 2.0, 1.0 - 1.0im])
        @test norm(ac, Ell1()) == 2sqrt(2.0) + 2.0
    end

    @testset "Chebyshev" begin
        # `IdentityWeight`/`GeometricWeight`/`AlgebraicWeight` all double the weight of every
        # nonzero order (Chebyshev(0) is the constant `T₀`, Chebyshev(i>0) is `2Tᵢ`-normalized)
        a = Sequence(Chebyshev(3), [1.0, -2.0, 3.0, -4.0])
        @test norm(a, Ell1()) == 19.0          # 1·1 + 2·2 + 3·2 + 4·2
        @test norm(a, Ell2()) == sqrt(59.0)    # 1·1 + 4·2 + 9·2 + 16·2
        @test norm(a, EllInf()) == 8.0         # max(1,4,6,8)

        ν = 2.0
        w = GeometricWeight(ν)  # weight_i = doubling(i)·ν^i : 1, 4, 8, 16
        @test norm(a, Ell1(w)) == 97.0           # 1·1+2·4+3·8+4·16
        @test norm(a, EllInf(w)) == 64.0         # max(1,8,24,64)

        p = 2.0
        wa = AlgebraicWeight(p)  # weight_i = doubling(i)·(1+i)^p : 1, 8, 18, 32
        @test norm(a, Ell1(wa)) == 199.0         # 1·1+2·8+3·18+4·32
    end

    @testset "TensorSpace" begin
        # coefficients all equal to 1 ⇒ the ℓ¹/ℓ² weighted sum factors over the tensor
        # dimensions since the weight itself is a product of per-dimension weights
        s = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1)
        a = Sequence(s, ones(dimension(s)))
        ρ, p = 2.0, 3.0

        taylor_sum = 1 + ρ                # Taylor(1) geometric weight sum: ν^0+ν^1
        fourier_sum = 3.0                  # Fourier(1) identity weight sum: 1+1+1
        cheb_sum = 1 + 2*2^p               # Chebyshev(1) algebraic weight sum: 1 + doubling(1)·2^p
        total = taylor_sum * fourier_sum * cheb_sum

        X1 = Ell1((GeometricWeight(ρ), IdentityWeight(), AlgebraicWeight(p)))
        @test norm(a, X1) == total == 153.0

        X2 = Ell2((GeometricWeight(ρ), IdentityWeight(), AlgebraicWeight(p)))
        @test norm(a, X2) == sqrt(total)   # |a_α|² = 1 ⇒ same weighted sum, then √

        X3 = EllInf((GeometricWeight(ρ), IdentityWeight(), AlgebraicWeight(p)))
        @test norm(a, X3) == max(1, ρ) * max(1.0, 1.0, 1.0) * max(1, 2*2^p) == 32.0
    end

    @testset "SymmetricSpace" begin
        @testset "evensym/oddsym Fourier" begin
            # `evensym(Fourier)` folds ±k together: the orbit of k≠0 has 2 elements, so the
            # weight of every nonzero mode is doubled (analogous to the old `CosFourier`)
            s = evensym(Fourier(3, 1.0))
            @test collect(indices(s)) == [0, 1, 2, 3]
            a = Sequence(s, [1.0, 2.0, 3.0, 4.0])
            @test norm(a, Ell1()) == 19.0    # 1·1 + 2·2 + 3·2 + 4·2
            ρ = 2.0
            @test norm(a, Ell1(GeometricWeight(ρ))) == 97.0   # 1·1+2·(2ρ)+3·(2ρ²)+4·(2ρ³)

            s2 = oddsym(Fourier(3, 1.0))
            @test collect(indices(s2)) == [1, 2, 3]   # k=0 is not a valid orbit representative
            b = Sequence(s2, [1.0, 2.0, 3.0])
            @test norm(b, Ell1()) == 12.0    # every remaining mode has orbit length 2: 2(1+2+3)
        end

        @testset "evensym/oddsym Taylor" begin
            # index action is trivial for Taylor's sym ⇒ orbit length 1; only the parity of
            # the order is restricted (even/odd), the weight itself is untouched
            s = evensym(Taylor(5))
            @test collect(indices(s)) == [0, 2, 4]
            a = Sequence(s, [1.0, 2.0, 3.0])
            @test norm(a, Ell1()) == 6.0    # 1+2+3, no doubling

            s2 = oddsym(Taylor(5))
            @test collect(indices(s2)) == [1, 3, 5]
        end

        @testset "evensym/oddsym Chebyshev" begin
            # orbit length is again 1 (trivial index action), but Chebyshev's own doubling
            # of nonzero order still applies on top of the parity restriction
            s = evensym(Chebyshev(5))
            @test collect(indices(s)) == [0, 2, 4]
            a = Sequence(s, [1.0, 2.0, 3.0])
            @test norm(a, Ell1()) == 11.0   # 1·1 + 2·2 + 3·2

            s2 = oddsym(Chebyshev(5))
            @test collect(indices(s2)) == [1, 3, 5]
            b = Sequence(s2, [1.0, 2.0, 3.0])
            @test norm(b, Ell1()) == 12.0   # every order is nonzero: 2(1+2+3)
        end

        @testset "d4sym" begin
            # D4 orbit of (0,0) has 1 element; orbit of an axis point (k,0), k≠0, is
            # {(±k,0),(0,±k)} (4 elements); orbit of a diagonal point (k,k), k≠0, is
            # {(±k,±k)} with matching signs (4 elements)
            s = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
            @test collect(indices(s)) == [(0, 0), (1, 0), (1, 1)]
            a = Sequence(s, [1.0, 2.0, 3.0])
            @test norm(a, Ell1()) == 21.0   # 1·1 + 2·4 + 3·4
        end
    end

    @testset "NormedCartesianSpace" begin
        @testset "CartesianPower" begin
            dom = Taylor(1)^3
            a = Sequence(dom, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            block_norms = (3.0, 7.0, 11.0)  # ℓ¹ norm of each Taylor(1) block: (1+2),(3+4),(5+6)

            @test norm(a, NormedCartesianSpace(Ell1(), Ell1())) == norm(a, Ell1()) == sum(block_norms)
            @test norm(a, NormedCartesianSpace(Ell1(), EllInf())) == maximum(block_norms)
            @test norm(a, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(sum(abs2, block_norms))
        end

        @testset "CartesianProduct with heterogeneous inner spaces" begin
            dom = ScalarSpace() × Taylor(1)
            b = Sequence(dom, [2.0, -5.0, 3.0])

            X = NormedCartesianSpace((Ell1(), EllInf()), Ell1())
            @test norm(b, X) == abs(2.0) + max(5.0, 3.0)   # scalar block + EllInf-norm of Taylor block
        end

        @testset "nested NormedCartesianSpace" begin
            s1 = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
            s2 = ScalarSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1
            a = Sequence(s1, collect(1.0:13.0))
            A = LinearOperator(s1, s2, ones(dimension(s2), dimension(s1)))

            X = NormedCartesianSpace((EllInf(), NormedCartesianSpace(Ell1((GeometricWeight(1.0), GeometricWeight(1.0), GeometricWeight(1.0))), EllInf())), EllInf())

            # block1 = scalar 1.0 (EllInf-norm = 1); block2 = the 12 remaining coefficients
            # (2:13) with unit-rate geometric weight ⇒ Chebyshev's inherent doubling of
            # nonzero order still applies: half the entries (cheb order 0) keep weight 1,
            # the other half (cheb order 1) get weight 2; outer EllInf on the 1-component
            # block2 is trivial. Total = max(1, (2+4+6+3+5+7) + 2·(8+10+12+9+11+13)) = 153
            @test norm(a, X) == 2+4+6+3+5+7 + 2*(8+10+12+9+11+13) == 153.0
            @test opnorm(A, X, X) == 18.0
        end

        @testset "errors" begin
            s0 = Taylor(1)^0
            @test_throws ArgumentError norm(Sequence(s0, Float64[]), Ell1())

            dom = ScalarSpace() × Taylor(1)
            b = Sequence(dom, [2.0, -5.0, 3.0])
            Xbad = NormedCartesianSpace((Ell1(), Ell1(), Ell1()), Ell1())   # 3 inner spaces, 2 components
            @test_throws ArgumentError norm(b, Xbad)

            A = LinearOperator(dom, ScalarSpace(), reshape([2.0, -5.0, 3.0], 1, 3))
            @test_throws ArgumentError opnorm(A, Xbad)
        end

        @testset "Ell2 outer on CartesianProduct" begin
            # 2-factor CartesianProduct
            dom = ScalarSpace() × Taylor(1)
            b = Sequence(dom, [2.0, -5.0, 3.0])
            expected = sqrt(norm(component(b, 1), Ell1())^2 + norm(component(b, 2), Ell1())^2)
            @test norm(b, NormedCartesianSpace(Ell1(), Ell2())) == expected
            @test norm(b, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 8.0^2)   # ‖2‖₁=2, ‖(-5,3)‖₁=8

            A = LinearOperator(dom, ScalarSpace(), reshape([2.0, -5.0, 3.0], 1, 3))
            expectedA = sqrt(opnorm(component(A, 1), Ell1())^2 + opnorm(component(A, 2), Ell1())^2)
            @test opnorm(A, NormedCartesianSpace(Ell1(), Ell2())) == expectedA
            @test opnorm(A, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 5.0^2)   # opnorm(component2)=max(5,3)=5

            # 3-factor CartesianProduct: recursion must unwind through every level
            dom3 = ScalarSpace() × Taylor(1) × Taylor(1)
            b3 = Sequence(dom3, [2.0, -5.0, 3.0, 1.0, -1.0])
            @test norm(b3, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 8.0^2 + 2.0^2)   # ‖2‖₁=2, ‖(-5,3)‖₁=8, ‖(1,-1)‖₁=2

            A3 = LinearOperator(dom3, ScalarSpace(), reshape([2.0, -5.0, 3.0, 1.0, -1.0], 1, 5))
            @test opnorm(A3, NormedCartesianSpace(Ell1(), Ell2())) == sqrt(2.0^2 + 5.0^2 + 1.0^2)   # opnorm(component3)=max(1,1)=1
        end
    end

    @testset "opnorm(::LinearOperator, ::BanachSpace, ::BanachSpace)" begin
        A = LinearOperator(Taylor(1), Taylor(1), [1.0 -2.0 ; 3.0 4.0])

        # opnorm(A,X,Y) computes, for each column j, its Y-norm v_j = ‖A eⱼ‖_Y, then returns
        # the X-dual-norm of the vector v (see `_norm_dual`)
        @test opnorm(A, Ell1(), EllInf()) == 4.0    # v=(max(1,3),max(2,4))=(3,4); dual-Ell1(v)=max(3,4)=4
        @test opnorm(A, EllInf(), Ell1()) == 10.0   # v=(|1|+|3|,|-2|+|4|)=(4,6); dual-EllInf(v)=4+6=10
        @test opnorm(A, Ell1(), Ell1()) == opnorm(A, Ell1()) == 6.0     # max column ℓ¹-sum
        @test opnorm(A, Ell2(), Ell2()) == opnorm(A, Ell2()) ≈ sqrt(30.0)

        # opnorm(::Multiplication, X) == norm(sequence, X)
        s = Sequence(Taylor(2), [1.0, -2.0, 3.0])
        ℳ = Multiplication(s)
        @test opnorm(ℳ, Ell1()) == norm(s, Ell1()) == 6.0
        @test opnorm(ℳ, Ell2(GeometricWeight(2.0))) == norm(s, Ell2(GeometricWeight(2.0)))
    end

    @testset "opnorm functional (ScalarSpace codomain)" begin
        v = [1.0, -2.0, 3.0]
        A = LinearOperator(Taylor(2), ScalarSpace(), reshape(v, 1, 3))
        ρ = 2.0
        w = GeometricWeight(ρ)  # weight_i = ρ^i : 1, 2, 4

        @test opnorm(A, Ell1(w)) == 1.0            # dual-ℓ¹: max(|v_i|/w_i) = max(1, 1, 0.75)
        @test opnorm(A, EllInf(w)) == 2.75         # dual-ℓ∞: Σ|v_i|/w_i = 1+1+0.75
        @test opnorm(A, Ell2(w)) ≈ sqrt(5.25)      # dual-ℓ²: sqrt(Σ v_i²/w_i)

        @testset "cartesian domain" begin
            dom = ScalarSpace() × Taylor(1)
            B = LinearOperator(dom, ScalarSpace(), reshape([2.0, -5.0, 3.0], 1, 3))

            @test opnorm(B, NormedCartesianSpace((Ell1(), Ell1()), Ell1())) == 5.0
            # dual-Ell1 outer ⇒ max over blocks: max(|2|, dual-Ell1((-5,3))) = max(2, max(5,3)) = 5

            @test opnorm(B, NormedCartesianSpace((Ell1(), EllInf()), Ell1())) == 8.0
            # second block uses EllInf's dual (Σ|·|): max(2, 5+3) = 8

            @test opnorm(B, NormedCartesianSpace((Ell1(), Ell1()), EllInf())) == 7.0
            # dual-EllInf outer ⇒ sum over blocks: 2 + max(5,3) = 7
        end
    end

    @testset "InfiniteSequence" begin
        seq = Sequence(Taylor(2), [1.0, 0.5, 0.25])
        X0 = Ell1(GeometricWeight(2.0))
        a = InfiniteSequence(seq, X0; finite_error = 0.01, tail_error = 0.02)

        @test norm(seq, X0) == 3.0                        # 1·1 + 0.5·2 + 0.25·4
        @test a.full_norm == 3.03                          # sequence_norm + total_error(=0.03)
        @test norm(a) == norm(a, X0) == 3.03

        @testset "norm in a different BanachSpace" begin
            # for `X ≠ banachspace(a)`, `norm(a, X)` checks that `banachspace(a)` embeds in `X`
            # and returns the sharper of the two available bounds
            Xsmall = Ell1(GeometricWeight(1.5))   # a valid weaker enclosing space (rate ≤ 2.0)
            expected_tight = norm(seq, Xsmall) + 0.03   # 1+0.5·1.5+0.25·1.5² + 0.03 = 2.3425
            @test norm(a, Xsmall) == min(expected_tight, a.full_norm) == 2.3425

            # a stronger weight is not an enclosing space: ℓ¹_ν decreases as ν grows
            Xbig = Ell1(GeometricWeight(3.0))
            @test_throws DomainError norm(a, Xbig)
        end

        @testset "identity weight does not embed in a geometric weight" begin
            # ℓ¹ ⊄ ℓ¹_ν for ν > 1, so a bound in ℓ¹ says nothing about the ℓ¹_ν norm
            b = InfiniteSequence(seq, Ell1(); tail_error = 0.1)
            @test_throws DomainError norm(b, Ell1(GeometricWeight(2.0)))
            # rate one is the identity weight in disguise, so it is admissible
            @test norm(b, Ell1(GeometricWeight(1.0))) == norm(b, Ell1()) == norm(b)

            # reachable without ever constructing an `Ell1()` sequence by hand, since
            # `differentiate` collapses the weight of its argument to `IdentityWeight`
            d = differentiate(a)
            @test banachspace(d) == Ell1()
            @test_throws DomainError norm(d, Ell1(GeometricWeight(2.0)))
        end
    end
end
