@testset "Sequence arithmetic" begin
    @testset "codomain" begin
        # the union of the spaces
        for (s₁, s₂, s) ∈ (
                (ScalarSpace(), ScalarSpace(), ScalarSpace()),
                (Taylor(1), Taylor(3), Taylor(3)),
                (Taylor(3), Taylor(1), Taylor(3)),
                (Fourier(3, 1.0), Fourier(1, 1.0), Fourier(3, 1.0)),
                (Chebyshev(0), Chebyshev(2), Chebyshev(2)),
                (Taylor(1) ⊗ Fourier(0, 1.0), Taylor(2) ⊗ Fourier(1, 1.0), Taylor(2) ⊗ Fourier(1, 1.0)),
                (evensym(Taylor(2)), evensym(Taylor(3)), evensym(Taylor(3))),
                (oddsym(Taylor(2)), oddsym(Taylor(3)), oddsym(Taylor(3))),
                (d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0)), d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0)), d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))),
                # no symmetry survives a space without one
                (evensym(Taylor(2)), Taylor(2), Taylor(2)),
                (Taylor(2), evensym(Taylor(2)), Taylor(2)),
                (Taylor(1)^2, Taylor(2)^2, Taylor(2)^2),
                (Taylor(1) × Fourier(1, 1.0), Taylor(2) × Fourier(0, 1.0), Taylor(2) × Fourier(1, 1.0)),
                (Taylor(1)^2, Taylor(1) × Taylor(2), Taylor(1) × Taylor(2)),
                (Taylor(1) × Taylor(2), Taylor(1)^2, Taylor(1) × Taylor(2)),
            )
            @test codomain(+, s₁, s₂) == codomain(-, s₁, s₂) == s
        end

        # different frequencies or numbers of components do not combine
        for (s₁, s₂) ∈ (
                (Fourier(1, 1.0), Fourier(1, 2.0)),
                (Taylor(1) ⊗ Fourier(1, 1.0), Taylor(1) ⊗ Fourier(1, 2.0)),
                (Taylor(1)^2, Taylor(1)^3),
                (Taylor(1) × Fourier(1, 1.0), (Taylor(1) × Fourier(1, 1.0)) × Chebyshev(1)),
            )
            @test_throws ArgumentError codomain(+, s₁, s₂)
        end
    end

    @testset "unary + and -" begin
        a = Sequence(Taylor(2), [1.0, -2.0, 3.0])
        @test +a == a
        @test -a == Sequence(Taylor(2), [-1.0, 2.0, -3.0])
    end

    @testset "scalar * and /" begin
        𝒯 = Taylor(2)
        a = Sequence(𝒯, [1.0, 2.0, 3.0])
        @test a * 3.0 == 3.0 * a == rmul!(copy(a), 3.0) == lmul!(3.0, copy(a)) == Sequence(𝒯, [3.0, 6.0, 9.0])
        @test a / 2.0 == 2.0 \ a == rdiv!(copy(a), 2.0) == ldiv!(2.0, copy(a)) == Sequence(𝒯, [0.5, 1.0, 1.5])

        af = Sequence(Fourier(1, 1.0), interval.([1.0, 2.0, 3.0]))
        @test all(isequal_interval.(coefficients(2.0 * af), interval.([2.0, 4.0, 6.0])))
        @test all(isequal_interval.(coefficients(rmul!(copy(af), 2.0)), interval.([2.0, 4.0, 6.0])))

        ac = Sequence(Chebyshev(1), ComplexF64[1.0+1.0im, 2.0-1.0im])
        @test 2.0 * ac == ac * 2.0 == rmul!(copy(ac), 2.0) == lmul!(2.0, copy(ac)) ==
            Sequence(Chebyshev(1), ComplexF64[2.0+2.0im, 4.0-2.0im])

        att = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(1.0:6.0))
        @test att * 2.0 == Sequence(Taylor(1) ⊗ Fourier(1, 1.0), collect(2.0:2.0:12.0))

        acart = Sequence(Taylor(1)^2, [1.0, 2.0, 3.0, 4.0])
        @test acart * 2.0 == rmul!(copy(acart), 2.0) == Sequence(Taylor(1)^2, [2.0, 4.0, 6.0, 8.0])
    end

    @testset "+ and - between sequences" begin
        @testset "Taylor" begin
            a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
            b = Sequence(Taylor(3), [10.0, 20.0, 30.0, 40.0])
            sum_ab = Sequence(Taylor(3), [11.0, 22.0, 33.0, 40.0])
            diff_ab = Sequence(Taylor(3), [-9.0, -18.0, -27.0, -40.0])
            @test a + b == a - (-b) == add!(fill(Inf, Taylor(3)), a, b) == sum_ab
            @test a - b == sub!(fill(Inf, Taylor(3)), a, b) == diff_ab

            # `radd!` and `rsub!` write into the space of `a`, `ladd!` and `lsub!` into the space of `b`
            @test radd!(copy(a), b) == Sequence(Taylor(2), [11.0, 22.0, 33.0])
            @test rsub!(copy(a), b) == Sequence(Taylor(2), [-9.0, -18.0, -27.0])
            @test ladd!(a, copy(b)) == sum_ab
            @test lsub!(a, copy(b)) == diff_ab

            # a target smaller than the union truncates the result
            @test add!(fill(Inf, Taylor(1)), a, b) == Sequence(Taylor(1), [11.0, 22.0])

            # the first operand is the bigger one
            b_small = Sequence(Taylor(1), [10.0, 20.0])
            @test b - b_small == sub!(fill(Inf, Taylor(3)), b, b_small) == Sequence(Taylor(3), [0.0, 0.0, 30.0, 40.0])

            # both operands on the same space
            c = Sequence(Taylor(2), [10.0, 20.0, 30.0])
            @test rsub!(copy(a), c) == lsub!(a, copy(c)) == Sequence(Taylor(2), [-9.0, -18.0, -27.0])
        end

        @testset "Fourier" begin
            a = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
            b = Sequence(Fourier(2, 1.0), [10.0, 20.0, 30.0, 40.0, 50.0])
            @test a + b == add!(fill(Inf, Fourier(2, 1.0)), a, b) == Sequence(Fourier(2, 1.0), [10.0, 21.0, 32.0, 43.0, 50.0])

            # the frequencies must agree, between the operands and with the target
            c = Sequence(Fourier(1, 2.0), [1.0, 2.0, 3.0])
            @test_throws ArgumentError a + c
            @test_throws ArgumentError radd!(copy(a), c)
            @test_throws ArgumentError ladd!(a, copy(c))
            @test_throws ArgumentError rsub!(copy(a), c)
            @test_throws ArgumentError lsub!(a, copy(c))
            @test_throws ArgumentError add!(zeros(Fourier(1, 2.0)), a, b)
            @test_throws ArgumentError sub!(zeros(Fourier(1, 2.0)), a, b)
        end

        @testset "Chebyshev" begin
            a = Sequence(Chebyshev(1), [1.0, 2.0])
            b = Sequence(Chebyshev(2), [10.0, 20.0, 30.0])
            @test a + b == add!(fill(Inf, Chebyshev(2)), a, b) == Sequence(Chebyshev(2), [11.0, 22.0, 30.0])
        end

        @testset "TensorSpace" begin
            # `a` only has the Fourier mode 0
            s = Taylor(2) ⊗ Fourier(1, 1.0)
            a = Sequence(Taylor(1) ⊗ Fourier(0, 1.0), [1.0, 2.0])
            b = Sequence(s, collect(1.0:9.0))
            @test a + b == add!(fill(Inf, s), a, b) == Sequence(s, [1.0, 2.0, 3.0, 5.0, 7.0, 6.0, 7.0, 8.0, 9.0])

            # each operand is the bigger one along one factor, so the codomain is neither of their spaces
            a = Sequence(Taylor(2) ⊗ Fourier(0, 1.0), [1.0, 2.0, 3.0])
            b = Sequence(Taylor(0) ⊗ Fourier(1, 1.0), [10.0, 20.0, 30.0])
            @test a - b == sub!(fill(Inf, s), a, b) == Sequence(s, [-10.0, 0.0, 0.0, -19.0, 2.0, 3.0, -30.0, 0.0, 0.0])
        end

        @testset "SymmetricSpace" begin
            # a symmetric term is read on the space without symmetry
            ae = Sequence(evensym(Taylor(2)), [1.0, 3.0]) # 1 + 3x²
            b = Sequence(Taylor(2), [1.0, 2.0, 4.0])
            @test ae + b == b + ae == add!(zeros(Taylor(2)), ae, b) == add!(zeros(Taylor(2)), b, ae) == Sequence(Taylor(2), [2.0, 2.0, 7.0])
            # an even and an odd term are read on their intersection, which has no symmetry
            ao = Sequence(oddsym(Taylor(2)), [5.0]) # 5x
            @test coefficients(ae + ao) == [1.0, 5.0, 3.0]

            # interval spaces keep the guarantee
            aei = Sequence(interval(evensym(Fourier(1, 1.0))), interval.([1.0, 3.0]))
            bi = Sequence(interval(Fourier(1, 1.0)), interval.([1.0, 2.0, 4.0]))
            @test all(isguaranteed, coefficients(aei + bi))
            @test all(in_interval.([4.0, 3.0, 7.0], coefficients(aei + bi)))
        end

        @testset "CartesianPower" begin
            s = Taylor(1)^2
            a = Sequence(s, [1.0, 2.0, 3.0, 4.0])
            b = Sequence(s, [10.0, 20.0, 30.0, 40.0])
            @test a + b == add!(fill(Inf, s), a, b) == radd!(copy(a), b) == ladd!(a, copy(b)) == Sequence(s, [11.0, 22.0, 33.0, 44.0])

            # the components are enlarged
            b = Sequence(Taylor(2)^2, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            sum_ab = Sequence(Taylor(2)^2, [11.0, 22.0, 30.0, 43.0, 54.0, 60.0])
            @test a + b == add!(fill(Inf, Taylor(2)^2), a, b) == ladd!(a, copy(b)) == sum_ab
            @test radd!(copy(a), b) == Sequence(s, [11.0, 22.0, 43.0, 54.0])
            @test lsub!(a, copy(b)) == Sequence(Taylor(2)^2, [-9.0, -18.0, -30.0, -37.0, -46.0, -60.0])

            @test_throws ArgumentError zeros(s) + zeros(Taylor(1)^3)
            @test_throws ArgumentError radd!(zeros(s), zeros(Taylor(1)^3))
        end

        @testset "CartesianProduct" begin
            s = Taylor(1) × Fourier(1, 1.0)
            a = Sequence(s, [1.0, 2.0, 3.0, 4.0, 5.0])
            b = Sequence(s, [10.0, 20.0, 30.0, 40.0, 50.0])
            @test a + b == add!(fill(Inf, s), a, b) == Sequence(s, [11.0, 22.0, 33.0, 44.0, 55.0])
            @test a - b == sub!(fill(Inf, s), a, b) == Sequence(s, [-9.0, -18.0, -27.0, -36.0, -45.0])

            # the components are enlarged
            s₁ = Taylor(1) × Fourier(0, 1.0) × Chebyshev(1)
            s₂ = Taylor(2) × Fourier(1, 1.0) × Chebyshev(1)
            a = Sequence(s₁, collect(1.0:5.0))
            b = Sequence(s₂, collect(1.0:8.0))
            @test a + b == add!(fill(Inf, s₂), a, b) == Sequence(s₂, [2.0, 4.0, 3.0, 4.0, 8.0, 6.0, 11.0, 13.0])
            @test radd!(copy(a), b) == Sequence(s₁, [2.0, 4.0, 8.0, 11.0, 13.0])

            a = Sequence(CartesianProduct(Taylor(1)), [1.0, 2.0])
            b = Sequence(CartesianProduct(Taylor(2)), [10.0, 20.0, 30.0])
            @test a + b == Sequence(CartesianProduct(Taylor(2)), [11.0, 22.0, 30.0])
        end
    end

    @testset "ScalarSpace" begin
        s1 = Sequence(ScalarSpace(), [2.0])
        s2 = Sequence(ScalarSpace(), [3.0])
        @test s1 * s2 == Sequence(ScalarSpace(), [6.0])
        @test s1 ^ 2 == Sequence(ScalarSpace(), [4.0])
        @test s1 ^ s2 == Sequence(ScalarSpace(), [8.0])
        @test s1 / s2 == Sequence(ScalarSpace(), [2.0/3.0])
        @test s1 \ s2 == Sequence(ScalarSpace(), [1.5])

        @test s1 + 1.0 == 1.0 + s1 == radd!(copy(s1), 1.0) == ladd!(1.0, copy(s1)) == Sequence(ScalarSpace(), [3.0])
        @test s1 - 1.0 == rsub!(copy(s1), 1.0) == Sequence(ScalarSpace(), [1.0])
        @test 1.0 - s1 == lsub!(1.0, copy(s1)) == Sequence(ScalarSpace(), [-1.0])

        # added to a sequence on a sequence space, the scalar is the constant coefficient
        ta = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        @test s1 + ta == ta + s1 == Sequence(Taylor(2), [3.0, 2.0, 3.0])
    end

    @testset "sequence ± number" begin
        # the number is added to the constant coefficient
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        @test a + 5.0 == 5.0 + a == radd!(copy(a), 5.0) == ladd!(5.0, copy(a)) == Sequence(Taylor(2), [6.0, 2.0, 3.0])
        @test a - 5.0 == rsub!(copy(a), 5.0) == Sequence(Taylor(2), [-4.0, 2.0, 3.0])
        @test 5.0 - a == lsub!(5.0, copy(a)) == Sequence(Taylor(2), [4.0, -2.0, -3.0])

        @test Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0]) + 5.0 == Sequence(Fourier(1, 1.0), [1.0, 7.0, 3.0])
        @test Sequence(Chebyshev(1), [1.0, 2.0]) + 5.0 == Sequence(Chebyshev(1), [6.0, 2.0])
        s = Taylor(1) ⊗ Fourier(1, 1.0)
        @test Sequence(s, collect(1.0:6.0)) + 5.0 == Sequence(s, [1.0, 2.0, 8.0, 4.0, 5.0, 6.0])

        # a constant is even, so an even space is kept
        r = Sequence(evensym(Taylor(2)), [1.0, 2.0]) + 5.0
        @test space(r) == evensym(Taylor(2))
        @test r == Sequence(evensym(Taylor(2)), [6.0, 2.0])

        # a constant is not odd: the sum lives on the space without symmetry, and cannot be taken in place
        odd = Sequence(oddsym(Taylor(2)), [3.0])
        r = odd + 5.0
        @test dimension(space(r)) == 3
        @test r == Sequence(space(r), [5.0, 3.0, 0.0])
        @test_throws ArgumentError radd!(copy(odd), 5.0)
    end

    @testset "cartesian sequence ± vector of numbers" begin
        s = Taylor(1)^3
        a = Sequence(s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v = [10.0, 20.0, 30.0]
        @test a + v == v + a == radd!(copy(a), v) == ladd!(v, copy(a)) == Sequence(s, [11.0, 2.0, 23.0, 4.0, 35.0, 6.0])
        @test a - v == rsub!(copy(a), v) == Sequence(s, [-9.0, 2.0, -17.0, 4.0, -25.0, 6.0])
        @test v - a == lsub!(v, copy(a)) == Sequence(s, [9.0, -2.0, 17.0, -4.0, 25.0, -6.0])

        # a nested cartesian space takes one number per innermost component
        s2 = (Taylor(1)^2) × Fourier(1, 1.0)
        b = Sequence(s2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        @test b + [100.0, 200.0, 300.0] == Sequence(s2, [101.0, 2.0, 203.0, 4.0, 5.0, 306.0, 7.0])
        @test b - [100.0, 200.0, 300.0] == Sequence(s2, [-99.0, 2.0, -197.0, 4.0, 5.0, -294.0, 7.0])

        @test_throws ArgumentError a + [10.0, 20.0]
        @test_throws ArgumentError radd!(copy(a), [10.0, 20.0])
    end

    @testset "Interval and Complex coefficient promotion" begin
        b = Sequence(Taylor(1), [3.0, 4.0])

        a = Sequence(Taylor(1), interval.([1.0, 2.0]))
        @test eltype(a + b) == Interval{Float64}
        @test all(isequal_interval.(coefficients(a + b), interval.([4.0, 6.0])))
        @test all(isequal_interval.(coefficients(add!(fill(interval(-9999.0), Taylor(1)), a, b)), interval.([4.0, 6.0])))

        ac = Sequence(Taylor(1), ComplexF64[1.0+2.0im, 3.0+4.0im])
        @test eltype(ac + b) == ComplexF64
        @test ac + b == Sequence(Taylor(1), ComplexF64[4.0+2.0im, 7.0+4.0im])

        aci = Sequence(Taylor(1), Complex{Interval{Float64}}[interval(1.0)+interval(1.0)*im, interval(2.0)])
        cci = aci + b
        @test eltype(cci) == Complex{Interval{Float64}}
        @test all(isequal_interval.(real.(coefficients(cci)), interval.([4.0, 6.0])))
        @test all(isequal_interval.(imag.(coefficients(cci)), interval.([1.0, 0.0])))
    end
end
