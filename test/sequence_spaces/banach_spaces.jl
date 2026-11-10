@testset "Banach spaces" begin
    𝒯 = Taylor(3)
    ℱ = Fourier(3, 1.0)
    𝒞 = Chebyshev(3)

    @testset "weights: construction, comparison and interval promotion" begin
        for W ∈ (GeometricWeight, AlgebraicWeight, BesselWeight)
            w = W(2.0)
            @test w isa W{Float64}
            @test rate(w) == 2.0
            @test w == W(2.0)
            @test !(w == W(3.0))
            @test min(w, W(3.0)) == w
            @test W(interval(1.0, 2.0)) isa W{Interval{Float64}}
            @test interval(w) isa W{Interval{Float64}}
            @test isguaranteed(rate(interval(w)))
            @test interval(Float32, w) isa W{Interval{Float32}}
        end
        # the rate is finite, at least 1 for a geometric weight and nonnegative otherwise
        for (W, r) ∈ ((GeometricWeight, 0.5), (AlgebraicWeight, -1.0), (BesselWeight, -1.0))
            @test_throws DomainError W(r)
            @test_throws DomainError W(interval(r, 2.0))
            # the null logger silences the ill-formed interval warning raised for an infinite rate
            Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                @test_throws DomainError W(Inf)
            end
        end
        @test GeometricWeight(2) isa GeometricWeight{Int}
        @test GeometricWeight(2.0) == GeometricWeight(2)
        @test !(GeometricWeight(interval(1.0, 2.0)) == GeometricWeight(interval(1.0, 3.0)))

        w = IdentityWeight()
        @test w == IdentityWeight()
        @test !(w == GeometricWeight(1.0)) && !(GeometricWeight(1.0) == w)
        @test interval(w) === interval(Float64, w) === IdentityWeight()
        @test min(w, w) === min(w, GeometricWeight(2.0)) === min(AlgebraicWeight(2.0), w) === IdentityWeight()
    end

    @testset "weights: values" begin
        # a Chebyshev mode k > 0 counts twice (T₋ₖ = Tₖ)
        w = IdentityWeight()
        @test RadiiPolynomial._getindex(w, 𝒯, 0) == w[(𝒯, 2)] == 1
        @test w[(ℱ, -3)] == w[(ℱ, 0)] == w[(ℱ, 3)] == 1
        @test w[(𝒞, 0)] == 1
        @test w[(𝒞, 1)] == w[(𝒞, 3)] == 2

        # νⁱ on Taylor, ν^|i| on Fourier, 2νⁱ on Chebyshev for i > 0
        ν = GeometricWeight(2.0)
        @test ν[(𝒯, 0)] == ν[(𝒞, 0)] == 1.0
        @test ν[(𝒯, 3)] == RadiiPolynomial._getindex(ν, 𝒯, 3) == 8.0
        @test ν[(ℱ, -3)] == ν[(ℱ, 3)] == 8.0
        @test ν[(𝒞, 3)] == 16.0

        # (1+i)ˢ on Taylor, (1+|i|)ˢ on Fourier, 2(1+i)ˢ on Chebyshev for i > 0
        w = AlgebraicWeight(2.0)
        @test w[(𝒯, 0)] == w[(𝒞, 0)] == 1.0
        @test w[(𝒯, 3)] == RadiiPolynomial._getindex(w, 𝒯, 3) == 16.0
        @test w[(ℱ, -3)] == w[(ℱ, 3)] == 16.0
        @test AlgebraicWeight(1.0)[(𝒞, 3)] == 8.0

        # (1+|α|²)ˢ with |α|² = Σ αᵢ² on a tensor product of Fourier spaces
        w = BesselWeight(1.0)
        @test w[(ℱ, 0)] == 1.0
        @test w[(ℱ, -2)] == w[(ℱ, 2)] == 5.0
        @test BesselWeight(2.0)[(ℱ, 2)] == 25.0
        @test w[(ℱ ⊗ ℱ, (1, -2))] == 6.0
        @test BesselWeight(1.5)[(ℱ ⊗ ℱ, (1, -2))] == 6.0 ^ 1.5

        # an interval rate gives a guaranteed enclosure
        for (w, s, k, v) ∈ ((GeometricWeight(2.0), 𝒯, 3, 8), (AlgebraicWeight(2.0), 𝒯, 3, 16),
                            (BesselWeight(1.0), ℱ ⊗ ℱ, (1, -2), 6), (BesselWeight(1.0), ℱ, 2, 5))
            x = RadiiPolynomial._getindex(interval(w), s, k)
            @test isguaranteed(x) & in_interval(v, x)
        end
    end

    @testset "tuple of weights on tensor spaces" begin
        s = 𝒯 ⊗ ℱ
        # per-factor weights multiply: 2² ⋅ (1+|-1|)¹ = 8
        @test RadiiPolynomial._getindex((GeometricWeight(2.0), AlgebraicWeight(1.0)), s, (2, -1)) == 8.0
        # IdentityWeight factor contributes 1 on Taylor/Fourier, the Chebyshev doubling on Chebyshev
        @test RadiiPolynomial._getindex((IdentityWeight(), GeometricWeight(2.0)), 𝒞 ⊗ 𝒯, (1, 2)) == 8.0
        @test RadiiPolynomial._getindex((GeometricWeight(2.0), IdentityWeight()), s, (3, -1)) == 8.0
        @test RadiiPolynomial._getindex((GeometricWeight(2.0),), TensorSpace((𝒯,)), (3,)) == 8.0
        @test min((GeometricWeight(2.0), AlgebraicWeight(1.0)), (GeometricWeight(3.0), AlgebraicWeight(0.5))) ==
            (GeometricWeight(2.0), AlgebraicWeight(0.5))
        @test min((IdentityWeight(), GeometricWeight(2.0)), (GeometricWeight(3.0), GeometricWeight(1.5))) ==
            (IdentityWeight(), GeometricWeight(1.5))
    end

    @testset "weights on symmetric spaces" begin
        even = evensym(ℱ)
        odd = oddsym(ℱ)
        # weight is summed over the orbit: {k, -k} for k ≠ 0, {0} for k = 0
        @test IdentityWeight()[(even, 0)] == 1
        @test IdentityWeight()[(even, 2)] == 2
        @test GeometricWeight(2.0)[(even, 0)] == 1.0
        @test GeometricWeight(2.0)[(even, 2)] == 8.0 # 2 ⋅ 2²
        @test GeometricWeight(3.0)[(odd, 1)] == 6.0 # 2 ⋅ 3¹
        @test BesselWeight(1.0)[(even, 2)] == 10.0 # 2 ⋅ (1 + 2²)
        # d4sym orbits: |(0,0)| = 1, |(1,1)| = 4, |(0,2)| = 4, |(1,2)| = 8
        d4 = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
        @test IdentityWeight()[(d4, (0, 0))] == 1
        @test IdentityWeight()[(d4, (1, 1))] == 4
        @test IdentityWeight()[(d4, (0, 2))] == 4
        @test IdentityWeight()[(d4, (1, 2))] == 8
        @test BesselWeight(1.0)[(d4, (1, 2))] == 48.0 # 8 ⋅ (1 + 1 + 4)
        # per-dimension weights on a symmetric tensor space: sum of the tensor weight over the orbit
        @test RadiiPolynomial._getindex((GeometricWeight(2.0), GeometricWeight(2.0)), d4, (1, 2)) == 64.0 # 8 ⋅ 2¹ ⋅ 2²
        @test RadiiPolynomial._getindex((IdentityWeight(), GeometricWeight(3.0)), d4, (0, 2)) == 20.0 # 2 ⋅ 3⁰ + 2 ⋅ 3² over {(±2, 0), (0, ±2)}
        s3 = Chebyshev(2) ⊗ Chebyshev(2) ⊗ evensym(Fourier(2, 1.0))
        w3 = (GeometricWeight(2.0), IdentityWeight(), GeometricWeight(3.0))
        @test RadiiPolynomial._getindex(w3, s3, (1, 2, 2)) == 2 * RadiiPolynomial._getindex(w3, desymmetrize(s3), (1, 2, 2)) # orbit {(1, 2, ±2)}
        @test RadiiPolynomial._getindex(w3, s3, (1, 2, 0)) == RadiiPolynomial._getindex(w3, desymmetrize(s3), (1, 2, 0)) # orbit {(1, 2, 0)}
        # non-Fourier bases: evensym orbits have length 1 (each index maps to itself)
        @test IdentityWeight()[(evensym(𝒞), 2)] == 2 # 1 ⋅ 2 (Chebyshev doubling)
        @test GeometricWeight(3.0)[(evensym(Taylor(4)), 2)] == 9.0 # 1 ⋅ 3²
        # interval rate through a symmetric space: exact enclosure of 2 ⋅ 2² = 8
        v = RadiiPolynomial._getindex(interval(GeometricWeight(2.0)), even, 2)
        @test isguaranteed(v) & in_interval(8, v)
        # a weight constant on the orbits is carried by the orbit length, which agrees with the sum over the orbit
        @test RadiiPolynomial._isinvariant(GeometricWeight(2.0), even)
        @test RadiiPolynomial._isinvariant(BesselWeight(1.0), d4)
        @test RadiiPolynomial._isinvariant((GeometricWeight(2.0), GeometricWeight(2.0)), d4)
        @test RadiiPolynomial._isinvariant((IdentityWeight(), GeometricWeight(1.0)), d4) # both equal to the identity
        @test RadiiPolynomial._isinvariant((AlgebraicWeight(0.0), BesselWeight(0.0)), d4) # both equal to the identity
        @test RadiiPolynomial._isinvariant((AlgebraicWeight(1.0), AlgebraicWeight(1.0)), d4)
        @test RadiiPolynomial._isinvariant(w3, s3) # the symmetry does not mix the dimensions
        @test !RadiiPolynomial._isinvariant((GeometricWeight(2.0), GeometricWeight(3.0)), d4)
        for (X, s) ∈ ((Ell1(GeometricWeight(2.0)), even), (Ell1((GeometricWeight(2.0), GeometricWeight(2.0))), d4),
                      (Ell2(BesselWeight(1.0)), d4), (Ell1((IdentityWeight(), GeometricWeight(1.0))), d4), (Ell1(w3), s3))
            @test RadiiPolynomial._weights(X, s) ≈ [RadiiPolynomial._getindex(weight(X), s, k) for k ∈ indices(s)]
        end
    end

    @testset "min of mixed weight types" begin
        # a geometric rate 1 has no decay, a rate > 1 decays faster than any algebraic weight
        for (ν, w) ∈ ((1.0, AlgebraicWeight(0.0)), (2.0, AlgebraicWeight(3.0)))
            @test min(AlgebraicWeight(3.0), GeometricWeight(ν)) == min(GeometricWeight(ν), AlgebraicWeight(3.0)) == w
        end
    end

    @testset "$name" for (name, E, alias, w, w₂) ∈ (("Ell1", Ell1, ℓ¹, GeometricWeight(2.0), AlgebraicWeight(1.0)),
            ("Ell2", Ell2, ℓ², BesselWeight(1.5), GeometricWeight(2.0)), ("EllInf", EllInf, ℓ∞, AlgebraicWeight(1.0), GeometricWeight(2.0)))
        @test E === alias
        @test E() isa E{IdentityWeight}
        @test weight(E()) === IdentityWeight()
        X = E(w)
        @test X isa E{typeof(w)}
        @test weight(X) == w
        @test rate(X) == rate(w)
        Xt = E(w, w₂)
        @test Xt isa E{Tuple{typeof(w),typeof(w₂)}}
        @test Xt == E((w, w₂))
        @test weight(Xt) == (w, w₂)
        @test rate(Xt) == (rate(w), rate(w₂))
        @test E() == E()
        @test X == E(w)
        @test !(X == E(typeof(w)(3.0)))
        @test_throws ArgumentError E(())
    end

    @testset "Banach spaces of different types" begin
        @test !(Ell1() == Ell2())
        @test !(Ell1() == EllInf())
        @test !(Ell2() == EllInf())
    end

    @testset "intersect" begin
        # intersect keeps the weaker (smaller) weight, the space where both elements live
        @test intersect(Ell1(GeometricWeight(2.0)), Ell1(GeometricWeight(3.0))) == Ell1(GeometricWeight(2.0))
        @test intersect(Ell1(GeometricWeight(2.0)), Ell1()) == Ell1()
        @test intersect(Ell2(BesselWeight(1.0)), Ell2(BesselWeight(2.0))) == Ell2(BesselWeight(1.0))
        @test intersect(EllInf(AlgebraicWeight(1.0)), EllInf(AlgebraicWeight(2.5))) == EllInf(AlgebraicWeight(1.0))
        @test intersect(Ell1(GeometricWeight(2.0), AlgebraicWeight(1.0)), Ell1(GeometricWeight(3.0), AlgebraicWeight(0.5))) ==
            Ell1(GeometricWeight(2.0), AlgebraicWeight(0.5))
        @test_throws MethodError intersect(Ell1(), Ell2())
        @test_throws MethodError intersect(Ell2(), EllInf())
        @test_throws MethodError intersect(Ell1(), NormedCartesianSpace(Ell1(), EllInf()))
    end

    @testset "interval promotion of Banach spaces" begin
        @test interval(Ell1(GeometricWeight(2.0))) isa Ell1{GeometricWeight{Interval{Float64}}}
        @test interval(Float32, Ell1(GeometricWeight(2.0))) isa Ell1{GeometricWeight{Interval{Float32}}}
        @test interval(Ell2(BesselWeight(1.0))) isa Ell2{BesselWeight{Interval{Float64}}}
        @test interval(Float32, EllInf(AlgebraicWeight(1.0))) isa EllInf{AlgebraicWeight{Interval{Float32}}}
        @test interval(Ell1()) == Ell1()
        Xt = interval(Ell2(BesselWeight(1.0), GeometricWeight(2.0)))
        @test Xt isa Ell2{Tuple{BesselWeight{Interval{Float64}},GeometricWeight{Interval{Float64}}}}
        @test all(isguaranteed, rate(Xt))
        Yt = interval(Float32, EllInf(GeometricWeight(2.0), AlgebraicWeight(1.0)))
        @test Yt isa EllInf{Tuple{GeometricWeight{Interval{Float32}},AlgebraicWeight{Interval{Float32}}}}
    end

    @testset "NormedCartesianSpace" begin
        X = NormedCartesianSpace(Ell1(), EllInf())
        @test X isa NormedCartesianSpace{Ell1{IdentityWeight},EllInf{IdentityWeight}}
        @test X.inner == Ell1()
        @test X.outer == EllInf()
        # one inner norm per block
        Xt = NormedCartesianSpace((Ell1(GeometricWeight(2.0)), Ell2()), EllInf())
        @test Xt isa NormedCartesianSpace{Tuple{Ell1{GeometricWeight{Float64}},Ell2{IdentityWeight}},EllInf{IdentityWeight}}
        @test Xt.inner == (Ell1(GeometricWeight(2.0)), Ell2())
        @test Xt.outer == EllInf()
        Xn = NormedCartesianSpace(X, Ell1())
        @test Xn.inner === X
        @test Xn.outer == Ell1()
    end

    @testset "show" begin
        for (x, str) ∈ ((IdentityWeight(), "IdentityWeight()"), (GeometricWeight(2.0), "GeometricWeight(2.0)"),
                        (AlgebraicWeight(1.0), "AlgebraicWeight(1.0)"), (BesselWeight(1.5), "BesselWeight(1.5)"),
                        (Ell1(), "ℓ¹()"), (Ell2(), "ℓ²()"), (EllInf(), "ℓ∞()"),
                        (Ell1(GeometricWeight(2.0)), "ℓ¹(GeometricWeight(2.0))"), (Ell2(BesselWeight(1.0)), "ℓ²(BesselWeight(1.0))"),
                        (EllInf(AlgebraicWeight(1.0)), "ℓ∞(AlgebraicWeight(1.0))"),
                        (Ell1(GeometricWeight(1.0), AlgebraicWeight(2.0)), "ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))"),
                        (NormedCartesianSpace(Ell1(), EllInf()), "NormedCartesianSpace(ℓ¹(), ℓ∞())"),
                        (NormedCartesianSpace((Ell1(), Ell2()), EllInf()), "NormedCartesianSpace((ℓ¹(), ℓ²()), ℓ∞())"))
            @test repr(MIME("text/plain"), x) == str
        end
    end
end
