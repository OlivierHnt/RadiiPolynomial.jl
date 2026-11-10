@testset "InfiniteSequence" begin

    𝒯 = Taylor(2)
    a = Sequence(𝒯, [1.0, 2.0, 1.0]) # norm(a,1) = 1+2+1 = 4

    @testset "construction" begin
        # low-level constructor: (sequence, finite_error, tail_error, total_error, banachspace)
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test sequence(ia) == a
        @test sequence_norm(ia) == 4.0
        @test finite_error(ia) == 0.0
        @test tail_error(ia) == 0.1
        @test total_error(ia) == 0.1
        @test banachspace(ia) == Ell1()

        ib = InfiniteSequence(a, Ell1())
        @test finite_error(ib) == tail_error(ib) == total_error(ib) == 0.0

        # given only a tail error, the finite error defaults to 0 and the total to their sum
        ic = InfiniteSequence(a, Ell1(); tail_error = 0.2)
        @test finite_error(ic) == 0.0
        @test tail_error(ic) == 0.2
        @test total_error(ic) == 0.2

        id = InfiniteSequence(a, Ell1(); finite_error = 0.3)
        @test finite_error(id) == 0.3
        @test tail_error(id) == 0.0
        @test total_error(id) == 0.3

        # given only a total error, both the finite and the tail error default to that
        # same value, not to half of it
        ie = InfiniteSequence(a, Ell1(); total_error = 0.5)
        @test finite_error(ie) == 0.5
        @test tail_error(ie) == 0.5
        @test total_error(ie) == 0.5

        i_f = InfiniteSequence(𝒯, [1.0, 2.0, 1.0], Ell1())
        @test i_f == ib
        i_g = InfiniteSequence(𝒯, [1.0, 2.0, 1.0], 0.0, 0.1, 0.1, Ell1())
        @test sequence_norm(i_g) == 4.0 && total_error(i_g) == 0.1

        # invariant: if total_error is small enough to bind the min, it is stored
        # verbatim as `total_error`, but finite_error/tail_error are only reset to
        # zero when the bound is (safely) exactly zero.
        ih = InfiniteSequence(a, 0.1, 0.2, 0.0, Ell1())
        @test finite_error(ih) == 0.0 && tail_error(ih) == 0.0 && total_error(ih) == 0.0

        ii = InfiniteSequence(a, 0.1, 0.2, 0.15, Ell1()) # finite+tail = 0.3 > total_error = 0.15
        @test finite_error(ii) == 0.1 && tail_error(ii) == 0.2 && total_error(ii) == 0.15

        # errors must be non-negative
        @test_throws ArgumentError InfiniteSequence(a, -0.1, 0.0, 0.0, Ell1())
        @test_throws ArgumentError InfiniteSequence(a, 0.0, -0.1, 0.0, Ell1())
        @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, -0.1, Ell1())

        # the banach space must be compatible with the sequence space:
        # a plain SequenceSpace only accepts (tuples of) Ell1 weights
        @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, 0.0, NormedCartesianSpace(Ell1(), Ell1()))

        @testset "only ℓ¹-type norms: ℓ² and ℓ^∞ are not Banach algebras" begin
            #= A product of two infinite sequences bounds the result by ‖a‖·‖b‖ in whatever
               norm the sequences carry, which holds only for ℓ¹-type norms. For the all-ones
               sequence below ‖a‖₂ ≈ 6.40 and ‖a‖₂² = 41.0 while ‖a*a‖₂ ≈ 214.4, and ‖a‖∞ = 1
               while ‖a*a‖∞ = 41 — the bound would fall below the norm of the truncated part
               itself. Admitting those norms is therefore unsound. =#
            n = 40
            b = Sequence(Taylor(n), ones(n+1))
            @test norm(b*b, Ell1())   ≤ norm(b, Ell1())^2   # ℓ¹ is a Banach algebra
            @test norm(b*b, Ell2())   > norm(b, Ell2())^2   # ℓ² is not
            @test norm(b*b, EllInf()) > norm(b, EllInf())^2 # ℓ^∞ is not

            @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, 0.0, Ell2())
            @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, 0.0, EllInf())
            d2 = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), zeros(6))
            @test_throws ArgumentError InfiniteSequence(d2, 0.0, 0.0, 0.0, Ell2((GeometricWeight(1.0), GeometricWeight(1.0))))
            @test_throws ArgumentError InfiniteSequence(d2, 0.0, 0.0, 0.0, EllInf((GeometricWeight(1.0), GeometricWeight(1.0))))
        end

        # a tensor space needs one weight per factor
        d = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), zeros(6))
        @test_throws ArgumentError InfiniteSequence(d, 0.0, 0.0, 0.0, Ell1((GeometricWeight(1.0),)))
        @test InfiniteSequence(d, 0.0, 0.0, 0.0, Ell1((GeometricWeight(1.0), GeometricWeight(1.0)))) isa InfiniteSequence
    end

    @testset "accessors" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test space(ia) == space(a) == 𝒯
        @test coefficients(ia) == coefficients(a)
        @test eltype(ia) == eltype(typeof(ia)) == Float64
        @test sequence(ia) === a
        @test total_error(ia) == 0.1

        # the constructor normalizes the total error to min(finite + tail, total),
        # so total_error is always the sharpest bound on the whole sequence
        ii = InfiniteSequence(a, 0.1, 0.2, 0.15, Ell1())
        @test total_error(ii) == min(0.1 + 0.2, 0.15) == 0.15
        ij = InfiniteSequence(a, 0.1, 0.2, 0.5, Ell1())
        @test total_error(ij) == min(0.1 + 0.2, 0.5) == 0.1 + 0.2

        # a plain Sequence is its own finite part and carries no error
        @test sequence(a) === a
        @test finite_error(a) == tail_error(a) == total_error(a) == 0.0
    end

    @testset "==" begin
        ia = InfiniteSequence(a, 0.0, 0.0, 0.0, Ell1())
        ib = InfiniteSequence(a, 0.0, 0.0, 0.0, Ell1())
        @test ia == ib

        # equality requires *both* sides to carry zero error, even when
        # comparing an InfiniteSequence to itself.
        ic = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test !(ic == ic)
        @test !(ia == ic)
    end

    @testset "zero, one" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        z = zero(ia)
        @test sequence(z) == zeros(𝒯)
        @test finite_error(z) == tail_error(z) == total_error(z) == 0.0
        @test banachspace(z) == banachspace(ia)

        o = one(ia)
        @test sequence(o) == Sequence(𝒯, [1.0, 0.0, 0.0])
        @test finite_error(o) == tail_error(o) == total_error(o) == 0.0
    end

    @testset "float, complex, real, imag, conj, conj!" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        fia = float(ia)
        # `==` requires zero error on *both* sides, so compare the underlying
        # data and error bookkeeping directly instead
        @test sequence(fia) == sequence(ia)
        @test finite_error(fia) == 0.0 && tail_error(fia) == 0.1 && total_error(fia) == 0.1

        z = Sequence(𝒯, [1 + 1im, 2 - 1im, 1 + 0im])
        iz = InfiniteSequence(z, 0.0, 0.1, 0.1, Ell1())
        @test sequence(complex(iz)) == z
        @test sequence(real(iz)) == Sequence(𝒯, [1, 2, 1])
        @test sequence(imag(iz)) == Sequence(𝒯, [1, -1, 0])
        @test sequence(conj(iz)) == Sequence(𝒯, [1 - 1im, 2 + 1im, 1 - 0im])
        @test sequence(conj!(iz)) == Sequence(𝒯, [1 - 1im, 2 + 1im, 1 - 0im])
        # the error/norm bookkeeping is carried over unchanged
        for f ∈ (real, imag, conj)
            r = f(iz)
            @test finite_error(r) == 0.0 && tail_error(r) == 0.1 && total_error(r) == 0.1
        end
    end

    @testset "interval" begin
        ia = InfiniteSequence(a, 0.1, 0.2, 0.25, Ell1(GeometricWeight(2.0)))
        iia = interval(ia)
        @test eltype(iia) == Interval{Float64}
        @test all(isequal_interval.(coefficients(iia), interval.([1.0, 2.0, 1.0])))
        @test isequal_interval(finite_error(iia), interval(0.1))
        @test isequal_interval(tail_error(iia), interval(0.2))
        @test isequal_interval(total_error(iia), interval(0.25))
        # the weight of the Banach space is carried over as an interval
        @test banachspace(iia) == Ell1(GeometricWeight(interval(2.0)))
        # the derived norms are recomputed rather than thinly wrapped
        @test in_interval(sequence_norm(ia), sequence_norm(iia))
        @test isguaranteed(sequence_norm(iia)) && isguaranteed(finite_error(iia))

        # interval(T, ...) widens to the enclosure at the requested precision
        ib = interval(BigFloat, ia)
        @test eltype(ib) == Interval{BigFloat}
        @test in_interval(0.1, finite_error(ib))
        @test in_interval(sequence_norm(ia), sequence_norm(ib))

        # the frequency of a Fourier space becomes an interval too
        ℱ = Fourier(1, 1.0)
        ic = interval(InfiniteSequence(ℱ, [0.5, 0.0, 0.5], 0.0, 0.1, 0.1, Ell1()))
        @test space(ic) == Fourier(1, interval(1.0))

        # idempotent on data that is already interval-valued
        @test isequal_interval(sequence_norm(interval(iia)), sequence_norm(iia))
    end

    @testset "permutedims" begin
        𝒯′ = Taylor(1)
        ℱ = Fourier(1, 1.0)
        d = Sequence(𝒯′ ⊗ ℱ, collect(1.0:6.0))
        id_ = InfiniteSequence(d, 0.0, 0.05, 0.05, Ell1((GeometricWeight(1.0), GeometricWeight(1.0))))
        idp = permutedims(id_, [2, 1])
        @test space(sequence(idp)) == ℱ ⊗ 𝒯′
        @test sequence(idp) == permutedims(d, [2, 1])
        # norm/error bookkeeping is untouched by a mere relabelling of the axes
        @test sequence_norm(idp) == sequence_norm(id_)
        @test finite_error(idp) == 0.0 && tail_error(idp) == 0.05 && total_error(idp) == 0.05
    end

    @testset "norm" begin
        # `norm` of an InfiniteSequence is the norm of its finite part plus the total error,
        # measured in the Banach space the sequence carries
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test norm(a, Ell1()) == 4.0                    # 1+2+1
        @test norm(ia) == 4.1
        @test norm(ia, banachspace(ia)) == norm(ia)

        # ℓ² and ℓ^∞ remain available on a plain sequence; they are refused only as the
        # norm an infinite sequence carries, since they are not Banach algebras
        @test norm(a, Ell2()) == sqrt(6.0)              # √(1+4+1)
        @test norm(a, EllInf()) == 2.0                  # max(1,2,1)

        # `IdentityWeight` and `GeometricWeight(1.0)` describe the same ℓ¹ norm
        @test norm(ia, Ell1(GeometricWeight(1.0))) == 4.1

        # a Banach space with no known embedding from the sequence's own one is refused
        # rather than silently trusted
        @test_throws DomainError norm(ia, Ell1(GeometricWeight(2.0)))
        @test_throws DomainError norm(ia, EllInf())
        @test_throws DomainError norm(ia, Ell2())
    end

    @testset "symmetric spaces" begin
        # the Banach space is checked against the underlying space: a symmetric tensor space takes a
        # tuple of weights
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
        @test isequal_interval(ia[0], interval(0.5, 1.5))
        @test isequal_interval(ia[1], interval(1.75, 2.25))
        @test isequal_interval(ia[2], interval(2.875, 3.125))
        @test all(isequal_interval.(ia[0:1], [interval(0.5, 1.5), interval(1.75, 2.25)]))
        @test all(isequal_interval.(ia[:], coefficients(project(ia, 𝒯))))
        @test isequal_interval(ia[end], ia[2])
        @test all(isequal_interval.(collect(ia), ia[:]))
        @test isequal_interval(first(ia), ia[0])
        @test isguaranteed(ia[1])

        # the stored coefficients are those of `sequence`
        @test isequal_interval(sequence(ia)[1], interval(2.0))

        # the tail error bears on the indices which are not stored
        ib = InfiniteSequence(c, X; tail_error = interval(0.1))
        @test ib[1] === c[1]
        @test all(ib[:] .=== coefficients(c))
        @test_throws BoundsError ia[3]

        # floating-point coefficients cannot enclose an error
        i_f = InfiniteSequence(a, 0.5, 0.1, 0.6, Ell1())
        @test i_f[1] == 2.0
        @test collect(i_f) == [1.0, 2.0, 1.0]

        @test_throws ArgumentError view(ia, 1)

        # tensor and symmetric spaces
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

    @testset "copy" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        ib = copy(ia)
        @test sequence(ib) == a
        @test coefficients(ib) !== coefficients(ia)
        @test (finite_error(ib), tail_error(ib), total_error(ib)) == (finite_error(ia), tail_error(ia), total_error(ia))
        @test norm(ib) == norm(ia)
        @test sequence(ia^1) == a # `^` returns a copy for n = 1
    end
end
