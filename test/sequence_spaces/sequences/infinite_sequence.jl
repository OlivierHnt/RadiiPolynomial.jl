@testset "InfiniteSequence" begin

    𝒯 = Taylor(2)
    a = Sequence(𝒯, [1.0, 2.0, 1.0]) # norm(a,1) = 1+2+1 = 4

    @testset "construction" begin
        # low-level constructor: (sequence, finite_error, tail_error, total_error, banachspace)
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        @test sequence(ia) == a
        @test sequence_norm(ia) == 4.0 # ℓ¹ identity-weighted norm of [1,2,1]
        @test finite_error(ia) == 0.0
        @test tail_error(ia) == 0.1
        @test total_error(ia) == 0.1
        @test banachspace(ia) == Ell1()

        # kwarg constructor, no kwargs => all errors zero
        ib = InfiniteSequence(a, Ell1())
        @test finite_error(ib) == tail_error(ib) == total_error(ib) == 0.0

        # kwarg constructor, only tail_error given: finite_error defaults to 0,
        # total_error defaults to finite_error + tail_error
        ic = InfiniteSequence(a, Ell1(); tail_error = 0.2)
        @test finite_error(ic) == 0.0
        @test tail_error(ic) == 0.2
        @test total_error(ic) == 0.2

        # kwarg constructor, only finite_error given
        id = InfiniteSequence(a, Ell1(); finite_error = 0.3)
        @test finite_error(id) == 0.3
        @test tail_error(id) == 0.0
        @test total_error(id) == 0.3

        # kwarg constructor, only total_error given: both finite_error and
        # tail_error default to total_error itself (not to total_error/2)
        ie = InfiniteSequence(a, Ell1(); total_error = 0.5)
        @test finite_error(ie) == 0.5
        @test tail_error(ie) == 0.5
        @test total_error(ie) == 0.5

        # `space`/`coefficients`-based constructors, equivalent to wrapping a `Sequence` first
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
        # a plain SequenceSpace only accepts (tuples of) Ell1/Ell2/EllInf weights
        @test_throws ArgumentError InfiniteSequence(a, 0.0, 0.0, 0.0, NormedCartesianSpace(Ell1(), Ell1()))

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
        @test sequence(o) == Sequence(𝒯, [1.0, 0.0, 0.0]) # constant mode set to 1
        @test finite_error(o) == tail_error(o) == total_error(o) == 0.0
    end

    @testset "float, complex, real, imag, conj, conj!" begin
        ia = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell1())
        fia = float(ia)
        # `==` on InfiniteSequence requires zero error on *both* sides (see the "=="
        # testset), so compare the underlying data/bookkeeping directly instead.
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
        @test norm(ia, banachspace(ia)) == norm(ia)     # explicit X == banachspace(a)

        ia2 = InfiniteSequence(a, 0.0, 0.1, 0.1, Ell2())
        @test norm(a, Ell2()) == sqrt(6.0)              # √(1+4+1)
        @test norm(ia2) == sqrt(6.0) + 0.1

        ia3 = InfiniteSequence(a, 0.0, 0.1, 0.1, EllInf())
        @test norm(a, EllInf()) == 2.0                  # max(1,2,1)
        @test norm(ia3) == 2.1

        # a Banach space with no known embedding from the sequence's own one is refused rather
        # than silently trusted, even when the weights happen to coincide numerically (ν = 1)
        @test_throws DomainError norm(ia3, EllInf(GeometricWeight(1.0)))
    end

end
