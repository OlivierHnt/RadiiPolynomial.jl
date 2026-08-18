@testset "Utilities" begin

    @testset "_safe_iszero" begin
        @test RadiiPolynomial._safe_iszero(0.0) == true
        @test RadiiPolynomial._safe_iszero(1.0) == false
        @test RadiiPolynomial._safe_iszero(0) == true

        # must be a *thin* zero, not merely an interval containing 0
        @test RadiiPolynomial._safe_iszero(interval(0.0)) == true
        @test RadiiPolynomial._safe_iszero(interval(-0.1, 0.1)) == false

        @test RadiiPolynomial._safe_iszero(complex(interval(0.0), interval(0.0))) == true
        @test RadiiPolynomial._safe_iszero(complex(interval(0.0), interval(0.1))) == false
    end

    @testset "_safe_isone" begin
        @test RadiiPolynomial._safe_isone(1.0) == true
        @test RadiiPolynomial._safe_isone(0.0) == false
        @test RadiiPolynomial._safe_isone(1) == true

        # must be a *thin* one
        @test RadiiPolynomial._safe_isone(interval(1.0)) == true
        @test RadiiPolynomial._safe_isone(interval(0.9, 1.1)) == false

        @test RadiiPolynomial._safe_isone(complex(interval(1.0), interval(0.0))) == true
        @test RadiiPolynomial._safe_isone(complex(interval(1.0), interval(0.1))) == false
    end

    @testset "_safe_isequal" begin
        @test RadiiPolynomial._safe_isequal(1.0, 1.0) == true
        @test RadiiPolynomial._safe_isequal(1.0, 2.0) == false

        # bounds are compared, not membership: equal *bounds* compare equal even
        # though the interval is not thin
        @test RadiiPolynomial._safe_isequal(interval(0.0, 1.0), interval(0.0, 1.0)) == true
        @test RadiiPolynomial._safe_isequal(interval(0.0, 1.0), interval(0.0, 1.0000001)) == false

        cc1 = complex(interval(1.0), interval(2.0))
        cc2 = complex(interval(1.0), interval(2.0))
        cc3 = complex(interval(1.0), interval(2.1))
        @test RadiiPolynomial._safe_isequal(cc1, cc2) == true
        @test RadiiPolynomial._safe_isequal(cc1, cc3) == false
    end

    @testset "_setguarantee" begin
        Y = interval(0.25)
        @test isguaranteed(Y) == true

        Yfalse = RadiiPolynomial._setguarantee(Y, false)
        @test isguaranteed(Yfalse) == false
        @test isequal_interval(Yfalse, Y)
        @test decoration(Yfalse) == decoration(Y)

        Ytrue = RadiiPolynomial._setguarantee(Yfalse, true)
        @test isguaranteed(Ytrue) == true
        @test isequal_interval(Ytrue, Y)
        @test decoration(Ytrue) == decoration(Y)
    end

    @testset "_no_alloc_reshape: BaseSpace (identity, no reshape needed)" begin
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        r = RadiiPolynomial._no_alloc_reshape(a)
        @test r === coefficients(a)

        a2 = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
        r2 = RadiiPolynomial._no_alloc_reshape(a2)
        @test r2 === coefficients(a2)
    end

    @testset "_no_alloc_reshape: TensorSpace, aliased column-major reshape" begin
        s = Taylor(1) ⊗ Taylor(2)
        @test dimensions(s) == (2, 3)
        b = Sequence(s, collect(1.0:6.0))
        rb = RadiiPolynomial._no_alloc_reshape(b)
        @test size(rb) == (2, 3)
        @test rb == [1.0 3.0 5.0; 2.0 4.0 6.0]

        rb[2, 3] = 999.0
        @test coefficients(b) == [1.0, 2.0, 3.0, 4.0, 5.0, 999.0]

        coefficients(b)[1] = -1.0
        @test rb[1, 1] == -1.0
    end

    @testset "_no_alloc_reshape: TensorSpace, Interval{Float64} coefficients" begin
        s = Taylor(1) ⊗ Fourier(1, 1.0)
        @test dimensions(s) == (2, 3)
        b = Sequence(s, interval.(collect(1.0:6.0)))
        rb = RadiiPolynomial._no_alloc_reshape(b)
        @test size(rb) == (2, 3)
        @test isequal_interval(rb[2, 3], interval(6.0)) # column-major: linear index 6
        rb[1, 2] = interval(-42.0)
        @test isequal_interval(coefficients(b)[3], interval(-42.0)) # linear index 3 == (1,2)
    end
end
