@testset "Utilities" begin

    @testset "_safe_iszero" begin
        # generic fallback: iszero(x)
        @test RadiiPolynomial._safe_iszero(0.0) == true
        @test RadiiPolynomial._safe_iszero(1.0) == false
        @test RadiiPolynomial._safe_iszero(0) == true

        # Interval: must be a *thin* zero, not merely an interval containing 0
        @test RadiiPolynomial._safe_iszero(interval(0.0)) == true
        @test RadiiPolynomial._safe_iszero(interval(-0.1, 0.1)) == false

        # Complex{<:Interval}
        @test RadiiPolynomial._safe_iszero(complex(interval(0.0), interval(0.0))) == true
        @test RadiiPolynomial._safe_iszero(complex(interval(0.0), interval(0.1))) == false
    end

    @testset "_safe_isone" begin
        @test RadiiPolynomial._safe_isone(1.0) == true
        @test RadiiPolynomial._safe_isone(0.0) == false
        @test RadiiPolynomial._safe_isone(1) == true

        # Interval: must be a *thin* one
        @test RadiiPolynomial._safe_isone(interval(1.0)) == true
        @test RadiiPolynomial._safe_isone(interval(0.9, 1.1)) == false

        # Complex{<:Interval}
        @test RadiiPolynomial._safe_isone(complex(interval(1.0), interval(0.0))) == true
        @test RadiiPolynomial._safe_isone(complex(interval(1.0), interval(0.1))) == false
    end

    @testset "_safe_isequal" begin
        @test RadiiPolynomial._safe_isequal(1.0, 1.0) == true
        @test RadiiPolynomial._safe_isequal(1.0, 2.0) == false

        # Interval: isequal_interval compares bounds, not membership — equal *bounds*
        # compare equal even though the interval is not thin.
        @test RadiiPolynomial._safe_isequal(interval(0.0, 1.0), interval(0.0, 1.0)) == true
        @test RadiiPolynomial._safe_isequal(interval(0.0, 1.0), interval(0.0, 1.0000001)) == false

        # Complex{<:Interval}
        cc1 = complex(interval(1.0), interval(2.0))
        cc2 = complex(interval(1.0), interval(2.0))
        cc3 = complex(interval(1.0), interval(2.1))
        @test RadiiPolynomial._safe_isequal(cc1, cc2) == true
        @test RadiiPolynomial._safe_isequal(cc1, cc3) == false
    end

    #= _setguarantee(a, t) rebuilds the same bareinterval/decoration but forces the
       guarantee flag to `t`, regardless of the original flag. =#
    @testset "_setguarantee" begin
        Y = interval(0.25) # guaranteed, decoration com
        @test isguaranteed(Y) == true

        Yfalse = RadiiPolynomial._setguarantee(Y, false)
        @test isguaranteed(Yfalse) == false
        @test isequal_interval(Yfalse, Y) # bounds unchanged
        @test decoration(Yfalse) == decoration(Y) # decoration unchanged

        Ytrue = RadiiPolynomial._setguarantee(Yfalse, true)
        @test isguaranteed(Ytrue) == true
        @test isequal_interval(Ytrue, Y)
        @test decoration(Ytrue) == decoration(Y)
    end

    #= _no_alloc_reshape avoids the allocation that `reshape` incurs for certain array
       types (JuliaLang/julia#36313). For a BaseSpace sequence it is just the identity
       (there is nothing to reshape); for a TensorSpace sequence it reshapes the flat
       coefficient vector into an array of `dimensions(space(a))`, aliasing (not
       copying) the original coefficients. =#
    @testset "_no_alloc_reshape: BaseSpace (identity, no reshape needed)" begin
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        r = RadiiPolynomial._no_alloc_reshape(a)
        @test r === coefficients(a)

        a2 = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
        r2 = RadiiPolynomial._no_alloc_reshape(a2)
        @test r2 === coefficients(a2)
    end

    @testset "_no_alloc_reshape: TensorSpace, aliased column-major reshape" begin
        s = Taylor(1) ⊗ Taylor(2) # dimensions (2, 3), total dimension 6
        @test dimensions(s) == (2, 3)
        b = Sequence(s, collect(1.0:6.0))
        rb = RadiiPolynomial._no_alloc_reshape(b)
        @test size(rb) == (2, 3)
        # column-major fill: [1 3 5; 2 4 6]
        @test rb == [1.0 3.0 5.0; 2.0 4.0 6.0]

        # mutating the reshaped view mutates the original coefficient vector (no copy)
        rb[2, 3] = 999.0
        @test coefficients(b) == [1.0, 2.0, 3.0, 4.0, 5.0, 999.0]

        # and conversely, mutating the coefficients is reflected back in the reshape
        coefficients(b)[1] = -1.0
        @test rb[1, 1] == -1.0
    end

    @testset "_no_alloc_reshape: TensorSpace, Interval{Float64} coefficients" begin
        s = Taylor(1) ⊗ Fourier(1, 1.0) # dimensions (2, 3), total dimension 6
        @test dimensions(s) == (2, 3)
        b = Sequence(s, interval.(collect(1.0:6.0)))
        rb = RadiiPolynomial._no_alloc_reshape(b)
        @test size(rb) == (2, 3)
        @test isequal_interval(rb[2, 3], interval(6.0)) # column-major: linear index 6
        rb[1, 2] = interval(-42.0)
        @test isequal_interval(coefficients(b)[3], interval(-42.0)) # linear index 3 == (1,2)
    end
end
