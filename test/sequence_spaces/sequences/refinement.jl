@testset "Refinement" begin

    @testset "envelope primitives: the box in interval, its midpoint in float" begin
        # a coefficient below the envelope μ lies in the box [-μ, μ]
        box = RadiiPolynomial._envelope_box(Interval{Float64}, interval(0.5))
        @test inf(box) == -0.5
        @test sup(box) == 0.5

        cbox = RadiiPolynomial._envelope_box(Complex{Interval{Float64}}, interval(0.5))
        @test in_interval(0.3, real(cbox))
        @test in_interval(-0.4, imag(cbox))

        @test RadiiPolynomial._envelope_box(Float64, interval(0.5)) === 0.0
        @test RadiiPolynomial._envelope_box(ComplexF64, interval(0.5)) === zero(ComplexF64)

        # the box of an envelope which is not guaranteed is not guaranteed either
        μ_ng = 1.0 * interval(0.5)
        @test !isguaranteed(μ_ng)
        @test !isguaranteed(RadiiPolynomial._envelope_box(Interval{Float64}, μ_ng))
        @test isguaranteed(box)

        # the ℓ¹ envelope is err/w and the ℓ² one carries the square root
        @test RadiiPolynomial._coefficient_bound(Ell1(), 1.0, 4.0) == 0.25
        @test RadiiPolynomial._coefficient_bound(Ell2(), 1.0, 4.0) == 0.5
    end

    @testset "polish!" begin
        decay = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        outlier = [1.0, 0.5, 0.25, 0.125, 0.0625, 1000.0]
        polished = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0]

        # the rate 2 is fitted on the indices 0:4, and the last coefficient is zeroed when it exceeds the fit
        @test coefficients(polish!(Sequence(Taylor(5), copy(outlier)))) == polished
        @test coefficients(polish!(Sequence(Taylor(5), copy(decay)))) == decay
        @test coefficients(polish!(Sequence(Taylor(5)^2, [outlier ; decay]))) == [polished ; decay]

        # scalar and tensor spaces are left as they are
        s = Sequence(3.0)
        @test polish!(s) === s
        t = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
        @test polish!(t) === t
    end
end
