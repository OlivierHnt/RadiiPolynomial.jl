_infseq_errors(a) = (finite_error(a), tail_error(a), total_error(a))

@testset "InfiniteSequence" begin

    𝒯 = Taylor(2)
    a = Sequence(𝒯, [1.0, 2.0, 1.0])

    @testset "construction" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test sequence(ia) === a
        @test sequence_norm(ia) == 4.0
        @test _infseq_errors(ia) == (0.0, 0.1, 0.1)
        @test banachspace(ia) == Ell1()
        @test space(ia) == 𝒯
        @test coefficients(ia) == coefficients(a)
        @test eltype(ia) == eltype(typeof(ia)) == Float64

        # an error given alone sets the total error, and a total error given alone bounds both parts
        @test _infseq_errors(InfiniteSequence(a, Ell1())) == (0.0, 0.0, 0.0)
        @test _infseq_errors(InfiniteSequence(a, Ell1(); tail_error = 0.2)) == (0.0, 0.2, 0.2)
        @test _infseq_errors(InfiniteSequence(a, Ell1(); finite_error = 0.3)) == (0.3, 0.0, 0.3)
        @test _infseq_errors(InfiniteSequence(a, Ell1(); total_error = 0.5)) == (0.5, 0.5, 0.5)

        @test InfiniteSequence(𝒯, [1.0, 2.0, 1.0], Ell1()) == InfiniteSequence(a, Ell1())
        i_g = InfiniteSequence(𝒯, [1.0, 2.0, 1.0], 0.0, 0.1, 0.1, Ell1())
        @test sequence_norm(i_g) == 4.0 && total_error(i_g) == 0.1

        # the total error is min(finite + tail, total), and a zero total error resets the parts
        @test _infseq_errors(InfiniteSequence(a, 0.1, 0.2, 0.0, Ell1())) == (0.0, 0.0, 0.0)
        @test _infseq_errors(InfiniteSequence(a, 0.1, 0.2, 0.15, Ell1())) == (0.1, 0.2, 0.15)
        @test _infseq_errors(InfiniteSequence(a, 0.1, 0.2, 0.5, Ell1())) == (0.1, 0.2, 0.1 + 0.2)

        # a plain Sequence is its own finite part and carries no error
        @test sequence(a) === a
        @test _infseq_errors(a) == (0.0, 0.0, 0.0)

        for errors ∈ ((-0.1, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, -0.1))
            @test_throws ArgumentError InfiniteSequence(a, errors..., Ell1())
        end

        @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, 0.0, NormedCartesianSpace(Ell1(), Ell1()))

        # a tensor space needs one weight per factor
        d = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), zeros(6))
        w = (GeometricWeight(1.0), GeometricWeight(1.0))
        @test_throws ArgumentError InfiniteSequence(d, 0.0, 0.0, 0.0, Ell1((GeometricWeight(1.0),)))
        @test InfiniteSequence(d, 0.0, 0.0, 0.0, Ell1(w)) isa InfiniteSequence

        @testset "only ℓ¹ norms are accepted, ℓ² and ℓ^∞ not being Banach algebras" begin
            # the error of a product is bounded by ‖a‖ ‖b‖ in the norm the sequences carry
            b = Sequence(Taylor(40), ones(41))
            @test norm(b*b, Ell1())   ≤ norm(b, Ell1())^2
            @test norm(b*b, Ell2())   > norm(b, Ell2())^2
            @test norm(b*b, EllInf()) > norm(b, EllInf())^2

            for X ∈ (Ell2, EllInf)
                @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, 0.0, X())
                @test_throws ArgumentError InfiniteSequence(d, 0.0, 0.0, 0.0, X(w))
            end
        end
    end

    @testset "==" begin
        ia = InfiniteSequence(a, 0.0, 0.0, 0.0, Ell1())
        @test ia == InfiniteSequence(a, 0.0, 0.0, 0.0, Ell1())

        # two sequences are equal only when both carry no error
        ic = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test !(ic == ic)
        @test !(ia == ic)
    end

    @testset "zero, one, copy" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        z = zero(ia)
        @test sequence(z) == zeros(𝒯)
        @test _infseq_errors(z) == (0.0, 0.0, 0.0)
        @test banachspace(z) == banachspace(ia)

        o = one(ia)
        @test sequence(o) == Sequence(𝒯, [1.0, 0.0, 0.0])
        @test _infseq_errors(o) == (0.0, 0.0, 0.0)

        ib = copy(ia)
        @test sequence(ib) == a
        @test coefficients(ib) !== coefficients(ia)
        @test _infseq_errors(ib) == _infseq_errors(ia)
        @test norm(ib) == norm(ia)
        @test sequence(ia^1) == a
    end

    @testset "float, complex, real, imag, conj, conj!" begin
        fia = float(InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1()))
        @test sequence(fia) == a
        @test _infseq_errors(fia) == (0.0, 0.1, 0.1)

        z = Sequence(𝒯, [1 + 1im, 2 - 1im, 1 + 0im])
        iz = InfiniteSequence(z, 0.0, 0.1, 0.1, Ell1())
        @test sequence(complex(iz)) == z
        for f ∈ (real, imag, conj)
            @test sequence(f(iz)) == f(z)
            @test _infseq_errors(f(iz)) == (0.0, 0.1, 0.1)
        end
        @test sequence(conj!(iz)) == Sequence(𝒯, [1 - 1im, 2 + 1im, 1 - 0im])
    end

    @testset "interval" begin
        ia = InfiniteSequence(a, 0.1, 0.2, 0.25, Ell1(GeometricWeight(2.0)))
        iia = interval(ia)
        @test eltype(iia) == Interval{Float64}
        @test all(isequal_interval.(coefficients(iia), interval.([1.0, 2.0, 1.0])))
        @test all(isequal_interval.(_infseq_errors(iia), interval.((0.1, 0.2, 0.25))))
        @test banachspace(iia) == Ell1(GeometricWeight(interval(2.0)))
        # the norm is recomputed in interval arithmetic
        @test in_interval(sequence_norm(ia), sequence_norm(iia))
        @test isguaranteed(sequence_norm(iia)) && isguaranteed(finite_error(iia))

        ib = interval(BigFloat, ia)
        @test eltype(ib) == Interval{BigFloat}
        @test in_interval(0.1, finite_error(ib))
        @test in_interval(sequence_norm(ia), sequence_norm(ib))

        ic = interval(InfiniteSequence(Fourier(1, 1.0), [0.5, 0.0, 0.5], 0.0, 0.1, 0.1, Ell1()))
        @test space(ic) == Fourier(1, interval(1.0))

        @test isequal_interval(sequence_norm(interval(iia)), sequence_norm(iia))
    end

    @testset "permutedims" begin
        s = Taylor(1) ⊗ Fourier(1, 1.0)
        d = Sequence(s, collect(1.0:6.0))
        id_ = InfiniteSequence(d, 0.0, 0.05, 0.05, Ell1((GeometricWeight(1.0), GeometricWeight(1.0))))
        idp = permutedims(id_, [2, 1])
        @test space(idp) == Fourier(1, 1.0) ⊗ Taylor(1)
        @test sequence(idp) == permutedims(d, [2, 1])
        @test sequence_norm(idp) == sequence_norm(id_)
        @test _infseq_errors(idp) == (0.0, 0.05, 0.05)
    end

    @testset "norm" begin
        # the norm of the finite part plus the total error, in the Banach space the sequence carries
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test norm(a, Ell1()) == 4.0
        @test norm(ia) == norm(ia, banachspace(ia)) == 4.1
        @test norm(ia, Ell1(GeometricWeight(1.0))) == 4.1

        @test norm(a, Ell2()) == sqrt(6.0)
        @test norm(a, EllInf()) == 2.0

        # a Banach space with no known embedding of the one the sequence carries is refused
        for X ∈ (Ell1(GeometricWeight(2.0)), EllInf(), Ell2())
            @test_throws DomainError norm(ia, X)
        end
    end

    @testset "symmetric spaces" begin
        # a symmetric tensor space takes one weight per factor of the underlying space
        X = Ell1((GeometricWeight(1.1), IdentityWeight()))
        s = Chebyshev(2) ⊗ evensym(Fourier(3, 1.0))
        b = InfiniteSequence(Sequence(s, collect(1.0:dimension(s))), X; total_error = 0.1)
        @test space(b) == s
        @test banachspace(b) == X
        @test total_error(b) == 0.1
    end

    @testset "getindex, view, iterate" begin
        X = Ell1(GeometricWeight(interval(2.0)))
        c = Sequence(𝒯, interval.([1.0, 2.0, 3.0]))
        ia = InfiniteSequence(c, interval(0.5), interval(0.1), interval(0.6), X)

        # a coefficient carries the finite error over its weight: 0.5 / 2ᵏ
        enclosures = [interval(0.5, 1.5), interval(1.75, 2.25), interval(2.875, 3.125)]
        @test all(isequal_interval.([ia[k] for k ∈ 0:2], enclosures))
        @test all(isequal_interval.(ia[0:1], enclosures[1:2]))
        @test all(isequal_interval.(ia[:], enclosures))
        @test all(isequal_interval.(collect(ia), enclosures))
        @test all(isequal_interval.(coefficients(project(ia, 𝒯)), enclosures))
        @test isequal_interval(ia[end], enclosures[3])
        @test isequal_interval(first(ia), enclosures[1])
        @test isguaranteed(ia[1])
        @test isequal_interval(sequence(ia)[1], interval(2.0))
        @test_throws BoundsError ia[3]
        @test_throws ArgumentError view(ia, 1)

        # the tail error leaves the stored coefficients untouched
        ib = InfiniteSequence(c, X; tail_error = interval(0.1))
        @test ib[1] === c[1]
        @test all(ib[:] .=== coefficients(c))

        # an error which contains zero without being zero is charged in full
        ic = InfiniteSequence(c, X; finite_error = interval(0.0, 0.5))
        @test isequal_interval(project(ic, 𝒯)[1], interval(1.75, 2.25))
        @test isequal_interval(ic[1], interval(1.75, 2.25))

        # floating-point coefficients cannot enclose an error
        i_f = InfiniteSequence(a, 0.5, 0.1, 0.6, Ell1())
        @test i_f[1] == 2.0
        @test collect(i_f) == [1.0, 2.0, 1.0]

        s = 𝒯 ⊗ Fourier(1, 1.0)
        it = InfiniteSequence(Sequence(s, interval.(collect(1.0:dimension(s)))),
            Ell1((GeometricWeight(interval(2.0)), IdentityWeight())); finite_error = interval(0.5))
        @test isequal_interval(it[(1,-1)], interval(1.75, 2.25))
        @test all(isequal_interval.(it[(0:2,0)], project(it, s)[(0:2,0)]))
        s = evensym(Fourier(2, 1.0))
        ie = InfiniteSequence(Sequence(s, interval.([1.0, 2.0, 3.0])), X; finite_error = interval(0.5))
        @test isequal_interval(ie[1], interval(1.875, 2.125)) # the weight of the orbit {-1, 1} is 2 + 2
        @test all(isequal_interval.(ie[:], coefficients(project(ie, s))))
    end
end
