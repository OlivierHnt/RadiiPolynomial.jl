@testset "Banach spaces" begin
    𝒯 = Taylor(3)
    ℱ = Fourier(3, 1.0)
    𝒞 = Chebyshev(3)

    @testset "IdentityWeight" begin
        w = IdentityWeight()
        @test w == IdentityWeight()
        # different weight types are never equal (Weight == Weight fallback)
        @test !(w == GeometricWeight(1.0))
        @test !(GeometricWeight(1.0) == w)
        # identity weight is 1 at every index of Taylor and Fourier
        @test RadiiPolynomial._getindex(w, 𝒯, 0) == w[(𝒯, 2)] == 1
        @test w[(ℱ, -3)] == w[(ℱ, 0)] == w[(ℱ, 3)] == 1
        # Chebyshev counts each mode k > 0 twice (T₋ₖ = Tₖ), the zero mode once
        @test w[(𝒞, 0)] == 1
        @test w[(𝒞, 1)] == w[(𝒞, 3)] == 2
        # interval promotion is the identity
        @test interval(w) === IdentityWeight()
        @test interval(Float64, w) === IdentityWeight()
        # min against any weight gives the identity weight
        @test min(w, w) === IdentityWeight()
        @test min(w, GeometricWeight(2.0)) === min(AlgebraicWeight(2.0), w) === IdentityWeight()
    end

    @testset "GeometricWeight" begin
        ν = GeometricWeight(2.0)
        @test ν isa GeometricWeight{Float64}
        @test rate(ν) == 2.0
        # integer rate is preserved
        @test GeometricWeight(2) isa GeometricWeight{Int}
        @test rate(GeometricWeight(2)) == 2
        # equality compares rates (across number types)
        @test ν == GeometricWeight(2.0) == GeometricWeight(2)
        @test !(ν == GeometricWeight(3.0))
        @test min(ν, GeometricWeight(3.0)) == ν
        # Taylor: νⁱ ; Fourier: ν^|i| ; Chebyshev: 2νⁱ for i > 0, 1 at i = 0
        @test ν[(𝒯, 0)] == 1.0
        @test ν[(𝒯, 3)] == RadiiPolynomial._getindex(ν, 𝒯, 3) == 8.0
        @test ν[(ℱ, -3)] == ν[(ℱ, 3)] == 8.0
        @test ν[(𝒞, 0)] == 1.0
        @test ν[(𝒞, 3)] == 16.0
        # rate must be finite and ≥ 1
        @test_throws DomainError GeometricWeight(0.5)
        @test_throws DomainError GeometricWeight(interval(0.5, 2.0)) # inf(rate) < 1
        # rate = Inf is rejected (silence the IntervalArithmetic ill-formed interval warning from inf(Inf))
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            @test_throws DomainError GeometricWeight(Inf)
        end
        # interval-rate variants
        @test GeometricWeight(interval(1.0, 2.0)) isa GeometricWeight{Interval{Float64}}
        wint = interval(ν)
        @test wint isa GeometricWeight{Interval{Float64}}
        @test isguaranteed(rate(wint))
        @test interval(Float32, ν) isa GeometricWeight{Interval{Float32}}
        @test wint == interval(ν) # interval equality via isequal_interval
        @test !(GeometricWeight(interval(1.0, 2.0)) == GeometricWeight(interval(1.0, 3.0)))
        # ν = 2 at Taylor index 3 gives the exact enclosure of 8
        v = RadiiPolynomial._getindex(wint, 𝒯, 3)
        @test isguaranteed(v) & in_interval(8, v)
    end

    @testset "AlgebraicWeight" begin
        w = AlgebraicWeight(2.0)
        @test w isa AlgebraicWeight{Float64}
        @test rate(w) == 2.0
        @test w == AlgebraicWeight(2.0)
        @test !(w == AlgebraicWeight(1.0))
        @test min(w, AlgebraicWeight(1.0)) == AlgebraicWeight(1.0)
        # Taylor: (1+i)ˢ ; Fourier: (1+|i|)ˢ ; Chebyshev: 2(1+i)ˢ for i > 0, 1 at i = 0
        @test w[(𝒯, 0)] == 1.0
        @test w[(𝒯, 3)] == RadiiPolynomial._getindex(w, 𝒯, 3) == 16.0
        @test w[(ℱ, -3)] == w[(ℱ, 3)] == 16.0
        @test w[(𝒞, 0)] == 1.0
        @test AlgebraicWeight(1.0)[(𝒞, 3)] == 8.0
        # rate must be finite and ≥ 0
        @test_throws DomainError AlgebraicWeight(-1.0)
        @test_throws DomainError AlgebraicWeight(interval(-1.0, 1.0)) # inf(rate) < 0
        Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
            @test_throws DomainError AlgebraicWeight(Inf)
        end
        # interval-rate variants
        wint = interval(w)
        @test wint isa AlgebraicWeight{Interval{Float64}}
        @test isguaranteed(rate(wint))
        @test interval(Float32, w) isa AlgebraicWeight{Interval{Float32}}
        # s = 2 at Taylor index 3 gives the exact enclosure of (1+3)² = 16
        v = RadiiPolynomial._getindex(wint, 𝒯, 3)
        @test isguaranteed(v) & in_interval(16, v)
    end

    @testset "BesselWeight" begin
        w = BesselWeight(1.0)
        @test w isa BesselWeight{Float64}
        @test rate(w) == 1.0
        @test w == BesselWeight(1.0)
        @test !(w == BesselWeight(2.0))
        @test min(w, BesselWeight(2.0)) == w
        # Fourier: (1+i²)ˢ
        @test w[(ℱ, 0)] == 1.0
        @test w[(ℱ, -2)] == w[(ℱ, 2)] == 5.0
        @test BesselWeight(2.0)[(ℱ, 2)] == 25.0
        # tensor product of Fourier spaces: (1+|α|²)ˢ with |α|² = Σ αᵢ²
        @test w[(ℱ ⊗ ℱ, (1, -2))] == 6.0
        @test BesselWeight(1.5)[(ℱ ⊗ ℱ, (1, -2))] == 6.0 ^ 1.5
        # rate must be finite and ≥ 0
        @test_throws DomainError BesselWeight(-1.0)
        @test_throws DomainError BesselWeight(Inf)
        @test_throws DomainError BesselWeight(interval(-1.0, 1.0))
        # interval-rate variants (dedicated tensor method for BesselWeight{<:Interval})
        wint = interval(w)
        @test wint isa BesselWeight{Interval{Float64}}
        @test isguaranteed(rate(wint))
        @test interval(Float32, w) isa BesselWeight{Interval{Float32}}
        @test BesselWeight(interval(0.5, 1.0)) isa BesselWeight{Interval{Float64}}
        # s = 1 at α = (1,-2) gives the exact enclosure of 1 + 1 + 4 = 6
        v = RadiiPolynomial._getindex(wint, ℱ ⊗ ℱ, (1, -2))
        @test isguaranteed(v) & in_interval(6, v)
        v2 = RadiiPolynomial._getindex(wint, ℱ, 2)
        @test isguaranteed(v2) & in_interval(5, v2)
    end

    @testset "tuple of weights on tensor spaces" begin
        s = 𝒯 ⊗ ℱ
        # per-factor weights multiply: 2² ⋅ (1+|-1|)¹ = 8
        @test RadiiPolynomial._getindex((GeometricWeight(2.0), AlgebraicWeight(1.0)), s, (2, -1)) == 8.0
        # IdentityWeight factor contributes 1 on Taylor/Fourier, the Chebyshev doubling on Chebyshev
        @test RadiiPolynomial._getindex((IdentityWeight(), GeometricWeight(2.0)), 𝒞 ⊗ 𝒯, (1, 2)) == 8.0
        @test RadiiPolynomial._getindex((GeometricWeight(2.0), IdentityWeight()), s, (3, -1)) == 8.0
        # single-factor tensor space
        @test RadiiPolynomial._getindex((GeometricWeight(2.0),), TensorSpace((𝒯,)), (3,)) == 8.0
        # min acts componentwise on tuples of weights
        @test min((GeometricWeight(2.0), AlgebraicWeight(1.0)), (GeometricWeight(3.0), AlgebraicWeight(0.5))) ==
            (GeometricWeight(2.0), AlgebraicWeight(0.5))
        @test min((IdentityWeight(), GeometricWeight(2.0)), (GeometricWeight(3.0), GeometricWeight(1.5))) ==
            (IdentityWeight(), GeometricWeight(1.5))
    end

    @testset "weights on symmetric spaces" begin
        even = evensym(ℱ)
        odd = oddsym(ℱ)
        # weight is multiplied by the orbit length: {k, -k} for k ≠ 0, {0} for k = 0
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
        # non-Fourier bases: evensym orbits have length 1 (each index maps to itself)
        @test IdentityWeight()[(evensym(𝒞), 2)] == 2 # 1 ⋅ 2 (Chebyshev doubling)
        @test GeometricWeight(3.0)[(evensym(Taylor(4)), 2)] == 9.0 # 1 ⋅ 3²
        # interval rate through a symmetric space: exact enclosure of 2 ⋅ 2² = 8
        v = RadiiPolynomial._getindex(interval(GeometricWeight(2.0)), even, 2)
        @test isguaranteed(v) & in_interval(8, v)
    end

    @testset "min of mixed weight types" begin
        # a geometric rate 1 has no decay: min with any algebraic weight is AlgebraicWeight(0)
        @test min(AlgebraicWeight(3.0), GeometricWeight(1.0)) == AlgebraicWeight(0.0)
        @test min(GeometricWeight(1.0), AlgebraicWeight(3.0)) == AlgebraicWeight(0.0)
        # a geometric rate > 1 decays faster than any algebraic weight
        @test min(AlgebraicWeight(3.0), GeometricWeight(2.0)) == AlgebraicWeight(3.0)
        @test min(GeometricWeight(2.0), AlgebraicWeight(3.0)) == AlgebraicWeight(3.0)
    end

    @testset "Ell1 / ℓ¹" begin
        @test Ell1 === ℓ¹
        @test Ell1() isa Ell1{IdentityWeight}
        @test weight(Ell1()) === IdentityWeight()
        X = Ell1(GeometricWeight(2.0))
        @test X isa Ell1{GeometricWeight{Float64}}
        @test weight(X) == GeometricWeight(2.0)
        @test rate(X) == 2.0
        # vararg constructor packs into a tuple
        Xt = Ell1(GeometricWeight(2.0), AlgebraicWeight(1.0))
        @test Xt isa Ell1{Tuple{GeometricWeight{Float64},AlgebraicWeight{Float64}}}
        @test Xt == Ell1((GeometricWeight(2.0), AlgebraicWeight(1.0)))
        @test weight(Xt) == (GeometricWeight(2.0), AlgebraicWeight(1.0))
        @test rate(Xt) == (2.0, 1.0)
        # equality compares weights; different Banach space types are never equal
        @test Ell1() == Ell1()
        @test X == Ell1(GeometricWeight(2.0))
        @test !(X == Ell1(GeometricWeight(3.0)))
        @test !(Ell1() == Ell2())
        @test !(Ell1() == EllInf())
        # at least one weight is required
        @test_throws ArgumentError Ell1(())
    end

    @testset "Ell2 / ℓ²" begin
        @test Ell2 === ℓ²
        @test Ell2() isa Ell2{IdentityWeight}
        @test weight(Ell2()) === IdentityWeight()
        X = Ell2(BesselWeight(1.5))
        @test X isa Ell2{BesselWeight{Float64}}
        @test weight(X) == BesselWeight(1.5)
        @test rate(X) == 1.5
        Xt = Ell2(BesselWeight(1.0), GeometricWeight(2.0))
        @test Xt isa Ell2{Tuple{BesselWeight{Float64},GeometricWeight{Float64}}}
        @test Xt == Ell2((BesselWeight(1.0), GeometricWeight(2.0)))
        @test weight(Xt) == (BesselWeight(1.0), GeometricWeight(2.0))
        @test rate(Xt) == (1.0, 2.0)
        @test Ell2() == Ell2()
        @test !(X == Ell2(BesselWeight(1.0)))
        @test !(Ell2() == EllInf())
        @test_throws ArgumentError Ell2(())
    end

    @testset "EllInf / ℓ∞" begin
        @test EllInf === ℓ∞
        @test EllInf() isa EllInf{IdentityWeight}
        @test weight(EllInf()) === IdentityWeight()
        X = EllInf(AlgebraicWeight(1.0))
        @test X isa EllInf{AlgebraicWeight{Float64}}
        @test weight(X) == AlgebraicWeight(1.0)
        @test rate(X) == 1.0
        Xt = EllInf(GeometricWeight(2.0), AlgebraicWeight(1.0))
        @test Xt isa EllInf{Tuple{GeometricWeight{Float64},AlgebraicWeight{Float64}}}
        @test Xt == EllInf((GeometricWeight(2.0), AlgebraicWeight(1.0)))
        @test weight(Xt) == (GeometricWeight(2.0), AlgebraicWeight(1.0))
        @test rate(Xt) == (2.0, 1.0)
        @test EllInf() == EllInf()
        @test !(X == EllInf(AlgebraicWeight(2.0)))
        @test_throws ArgumentError EllInf(())
    end

    @testset "intersect" begin
        # intersect keeps the weaker (smaller) weight, the space where both elements live
        @test intersect(Ell1(GeometricWeight(2.0)), Ell1(GeometricWeight(3.0))) == Ell1(GeometricWeight(2.0))
        @test intersect(Ell1(GeometricWeight(2.0)), Ell1()) == Ell1()
        @test intersect(Ell2(BesselWeight(1.0)), Ell2(BesselWeight(2.0))) == Ell2(BesselWeight(1.0))
        @test intersect(EllInf(AlgebraicWeight(1.0)), EllInf(AlgebraicWeight(2.5))) == EllInf(AlgebraicWeight(1.0))
        # componentwise on tuples of weights
        @test intersect(Ell1(GeometricWeight(2.0), AlgebraicWeight(1.0)), Ell1(GeometricWeight(3.0), AlgebraicWeight(0.5))) ==
            Ell1(GeometricWeight(2.0), AlgebraicWeight(0.5))
        # intersecting different Banach space types is not allowed
        @test_throws MethodError intersect(Ell1(), Ell2())
        @test_throws MethodError intersect(Ell2(), EllInf())
        @test_throws MethodError intersect(Ell1(), NormedCartesianSpace(Ell1(), EllInf()))
    end

    @testset "interval promotion of Banach spaces" begin
        @test interval(Ell1(GeometricWeight(2.0))) isa Ell1{GeometricWeight{Interval{Float64}}}
        @test interval(Float32, Ell1(GeometricWeight(2.0))) isa Ell1{GeometricWeight{Interval{Float32}}}
        @test interval(Ell2(BesselWeight(1.0))) isa Ell2{BesselWeight{Interval{Float64}}}
        @test interval(Float32, EllInf(AlgebraicWeight(1.0))) isa EllInf{AlgebraicWeight{Interval{Float32}}}
        # identity weight is untouched
        @test interval(Ell1()) == Ell1()
        # tuple weights are promoted componentwise
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
        # nesting is allowed
        Xn = NormedCartesianSpace(X, Ell1())
        @test Xn.inner === X
        @test Xn.outer == Ell1()
    end

    @testset "show" begin
        @test repr(MIME("text/plain"), IdentityWeight()) == "IdentityWeight()"
        @test repr(MIME("text/plain"), GeometricWeight(2.0)) == "GeometricWeight(2.0)"
        @test repr(MIME("text/plain"), AlgebraicWeight(1.0)) == "AlgebraicWeight(1.0)"
        @test repr(MIME("text/plain"), BesselWeight(1.5)) == "BesselWeight(1.5)"
        @test repr(MIME("text/plain"), Ell1()) == "ℓ¹()"
        @test repr(MIME("text/plain"), Ell2()) == "ℓ²()"
        @test repr(MIME("text/plain"), EllInf()) == "ℓ∞()"
        @test repr(MIME("text/plain"), Ell1(GeometricWeight(2.0))) == "ℓ¹(GeometricWeight(2.0))"
        @test repr(MIME("text/plain"), Ell2(BesselWeight(1.0))) == "ℓ²(BesselWeight(1.0))"
        @test repr(MIME("text/plain"), EllInf(AlgebraicWeight(1.0))) == "ℓ∞(AlgebraicWeight(1.0))"
        @test repr(MIME("text/plain"), Ell1(GeometricWeight(1.0), AlgebraicWeight(2.0))) ==
            "ℓ¹(GeometricWeight(1.0), AlgebraicWeight(2.0))"
        @test repr(MIME("text/plain"), NormedCartesianSpace(Ell1(), EllInf())) ==
            "NormedCartesianSpace(ℓ¹(), ℓ∞())"
        @test repr(MIME("text/plain"), NormedCartesianSpace((Ell1(), Ell2()), EllInf())) ==
            "NormedCartesianSpace((ℓ¹(), ℓ²()), ℓ∞())"
    end
end
