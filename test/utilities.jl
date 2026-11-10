@testset "Utilities" begin

    # an interval must be thin to be zero or one, containing the value is not enough
    @testset "_safe_iszero and _safe_isone" begin
        for (f, v) ∈ ((RadiiPolynomial._safe_iszero, 0.0), (RadiiPolynomial._safe_isone, 1.0))
            @test f(v)
            @test f(Int(v))
            @test !f(v + 1)
            @test f(interval(v))
            @test !f(interval(v - 0.1, v + 0.1))
            @test f(complex(interval(v), interval(0.0)))
            @test !f(complex(interval(v), interval(0.1)))
        end
    end

    @testset "_safe_isequal" begin
        @test RadiiPolynomial._safe_isequal(1.0, 1.0)
        @test !RadiiPolynomial._safe_isequal(1.0, 2.0)

        # the bounds are compared, so an interval which is not thin is equal to itself
        @test RadiiPolynomial._safe_isequal(interval(0.0, 1.0), interval(0.0, 1.0))
        @test !RadiiPolynomial._safe_isequal(interval(0.0, 1.0), interval(0.0, 1.0000001))

        z = complex(interval(1.0), interval(2.0))
        @test RadiiPolynomial._safe_isequal(z, complex(interval(1.0), interval(2.0)))
        @test !RadiiPolynomial._safe_isequal(z, complex(interval(1.0), interval(2.1)))
    end

    @testset "_setguarantee" begin
        Y = interval(0.25)
        @test isguaranteed(Y)
        Yfalse = RadiiPolynomial._setguarantee(Y, false)
        Ytrue = RadiiPolynomial._setguarantee(Yfalse, true)
        @test !isguaranteed(Yfalse) && isguaranteed(Ytrue)
        for Z ∈ (Yfalse, Ytrue)
            @test isequal_interval(Z, Y)
            @test decoration(Z) == decoration(Y)
        end
    end

    @testset "_no_alloc_reshape" begin
        # a base space needs no reshape
        for a ∈ (Sequence(Taylor(2), [1.0, 2.0, 3.0]), Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]))
            @test RadiiPolynomial._no_alloc_reshape(a) === coefficients(a)
        end

        # a tensor space gives a column-major array sharing the memory of the coefficients
        s = Taylor(1) ⊗ Taylor(2)
        @test dimensions(s) == (2, 3)
        b = Sequence(s, collect(1.0:6.0))
        rb = RadiiPolynomial._no_alloc_reshape(b)
        @test rb == [1.0 3.0 5.0; 2.0 4.0 6.0]
        rb[2, 3] = 999.0
        @test coefficients(b) == [1.0, 2.0, 3.0, 4.0, 5.0, 999.0]
        coefficients(b)[1] = -1.0
        @test rb[1, 1] == -1.0

        c = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), interval.(collect(1.0:6.0)))
        rc = RadiiPolynomial._no_alloc_reshape(c)
        @test size(rc) == (2, 3)
        @test isequal_interval(rc[2, 3], interval(6.0))
        rc[1, 2] = interval(-42.0)
        @test isequal_interval(coefficients(c)[3], interval(-42.0))
    end
end
