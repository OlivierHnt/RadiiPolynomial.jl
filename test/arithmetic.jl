@testset "Arithmetic" begin
    @testset "Sequence" begin
        a = Sequence(ParameterSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:13.0))
        b = Sequence(ParameterSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:19.0))
        c = Sequence(ParameterSpace() × (Taylor(0) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:7.0))
        d = Sequence(ParameterSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1, collect(1.0:7.0))

        @test +(4.0\(8.0*a*3.0))/3.0 == a + a == a - (-a) == a +̄ a == a -̄ (-a)

        @test a + b == a - (-b) == ladd!(a, copy(b)) == lsub!(a, -b)
        @test a +̄ b == a -̄ (-b) == radd!(copy(a), b) == rsub!(copy(a), -b)

        @test a + c == a - (-c) == radd!(copy(a), c) == rsub!(copy(a), -c)
        @test a +̄ c == a -̄ (-c) == ladd!(a, copy(c)) == lsub!(a, -c)

        @test a + d == a - (-d)
        @test a +̄ d == a -̄ (-d)
    end

    @testset "LinearOperator" begin
        𝒮₁ = ParameterSpace() × (Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        𝒮₂ = ParameterSpace() × (Taylor(2) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        𝒮₃ = ParameterSpace() × (Taylor(0) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1))^1
        𝒮₄ = ParameterSpace() × (Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1))^1

        A = LinearOperator(𝒮₁, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₁)])
        B = LinearOperator(𝒮₂, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₂)])
        C = LinearOperator(𝒮₃, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₃)])
        D = LinearOperator(𝒮₄, 𝒮₁, [i+j for i ∈ indices(𝒮₁), j ∈ indices(𝒮₄)])

        @test +(4.0\(8.0*A*3.0))/3.0 == A + A == A - (-A) == A +̄ A == A -̄ (-A)
        @test A == (A - I) + I == -(I - (I + A)) == radd!(rsub!(copy(A), I), I) == -lsub!(I, ladd!(I, copy(A)))

        @test A + B == A - (-B) == ladd!(A, copy(B)) == lsub!(A, -B)
        @test A +̄ B == A -̄ (-B) == radd!(copy(A), B) == rsub!(copy(A), -B)

        @test A + C == A - (-C) == radd!(copy(A), C) == rsub!(copy(A), -C)
        @test A +̄ C == A -̄ (-C) == ladd!(A, copy(C)) == lsub!(A, -C)

        @test A + D == A - (-D)
        @test A +̄ D == A -̄ (-D)

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
            n = fft_size(space(a), space(b))
            space_c = image(*, space(a), space(b))
            c = Sequence(space_c, Vector{Float64}(undef, dimension(space_c)))
            rifft!(c, fft(a, n) .* fft(b, n))
            return c
        end

        a = Sequence(Taylor(1), [1.0, 2.0])
        b = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        @test fft_size(space(a), space(a)) == fft_size(space(a), 2)
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b == mul!(a * b, a, b)
        @test a *̄ a == a ^̄ 2 == mul!(a *̄ a, a, a)
        @test 3a * b +̄ 4b == mul!(2b, a, b, 3, 2)

        a = Sequence(Fourier(1, 1.0), [1.0, 2.0, 3.0])
        b = Sequence(Fourier(2, 1.0), [1.0, 2.0, 3.0, 4.0, 5.0])
        @test fft_size(space(a), space(a)) == fft_size(space(a), 2)
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b == mul!(a * b, a, b)
        @test a *̄ a == a ^̄ 2 == mul!(a *̄ a, a, a)
        @test 3a * b +̄ 4b == mul!(2b, a, b, 3, 2)

        a = Sequence(Chebyshev(1), [1.0, 2.0])
        b = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        @test fft_size(space(a), space(a)) == fft_size(space(a), 2)
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b == mul!(a * b, a, b)
        @test a *̄ a == a ^̄ 2 == mul!(a *̄ a, a, a)
        @test 3a * b +̄ 4b == mul!(2b, a, b, 3, 2)

        a = Sequence(Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1), collect(1.0:12.0))
        b = Sequence(Taylor(2) ⊗ Fourier(0, 1.0) ⊗ Chebyshev(1), collect(1.0:6.0))
        @test fft_size(space(a), space(a)) == fft_size(space(a), 2)
        @test conv(a, a) ≈ a * a == a ^ 2
        @test conv(a, b) ≈ a * b == mul!(a * b, a, b)
        @test a *̄ a == a ^̄ 2 == mul!(a *̄ a, a, a)
        @test 3a * b +̄ 4b == mul!(2b, a, b, 3, 2)
    end
end
