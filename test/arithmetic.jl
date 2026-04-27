@testset "Arithmetic" begin
    @testset "Sequence" begin
        a = Sequence(ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:13.0))
        b = Sequence(ScalarSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:19.0))
        c = Sequence(ScalarSpace() × (Taylor(0) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:7.0))
        d = Sequence(ScalarSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:7.0))

        @test +(4.0\(8.0*a*3.0))/3.0 == a + a == a - (-a)

        @test a + b == a - (-b) == ladd!(a, copy(b)) == lsub!(a, -b)
        @test radd!(copy(a), b) == rsub!(copy(a), -b)

        @test a + c == a - (-c) == radd!(copy(a), c) == rsub!(copy(a), -c)
        @test ladd!(a, copy(c)) == lsub!(a, -c)

        @test a + d == a - (-d)
    end

    @testset "LinearOperator" begin
        𝒮₁ = ScalarSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        𝒮₂ = ScalarSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        𝒮₃ = ScalarSpace() × (Taylor(0) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        𝒮₄ = ScalarSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1

        A = LinearOperator(𝒮₁, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₁)])
        B = LinearOperator(𝒮₂, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₂)])
        C = LinearOperator(𝒮₃, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₃)])
        D = LinearOperator(𝒮₄, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₄)])

        @test +(4.0\(8.0*A*3.0))/3.0 == A + A == A - (-A)
        @test A == radd!(rsub!(copy(A), I), I) == -lsub!(I, ladd!(I, copy(A)))

        @test A + B == A - (-B) == ladd!(A, copy(B)) == lsub!(A, -B)
        @test radd!(copy(A), B) == rsub!(copy(A), -B)

        @test A + C == A - (-C) == radd!(copy(A), C) == rsub!(copy(A), -C)
        @test ladd!(A, copy(C)) == lsub!(A, -C)

        @test A + D == A - (-D)

        @test A*A == A^2
        @test A * B == mul!(similar(B), A, B, true, false)
        @test B * A == mul!(similar(A), B, A, true, false)
        @test A * C == mul!(similar(C), A, C, true, false)
        @test C * A == mul!(similar(A), C, A, true, false)
        @test A * D == mul!(similar(D), A, D, true, false)
        @test D * A == mul!(similar(A), D, A, true, false)
    end

    @testset "Convolution" begin
        function conv(a, b)
            space_c = codomain(*, space(a), space(b))
            c = Sequence(space_c, Vector{ComplexF64}(undef, dimension(space_c)))
            n = fft_size(space_c)
            to_seq!(c, to_grid(a, n) .* to_grid(b, n))
            return real(c)
        end

        a = Sequence(Taylor(1), [1.0, 2.0])
        b = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b

        a = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
        b = Sequence(Fourier(2, 1.0), [1.0, 2.0, 3.0, 4.0, 5.0])
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b

        a = Sequence(Chebyshev(1), [1.0, 2.0])
        b = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b

        a = Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1), collect(1.0:12.0))
        b = Sequence(Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1), collect(1.0:6.0))
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b
    end
end
