@testset "Sequence" begin
    𝒯 = Taylor(1)
    ℱ = Fourier(1, 1.0)
    𝒞 = Chebyshev(2)
    𝑇 = 𝒯 ⊗ ℱ ⊗ 𝒞
    𝕊 = 𝑇^1
    coeffs = [inv(prod((2, 2, 2) .^ abs.(α))) for α ∈ indices(𝑇)]
    a = Sequence(𝕊, coeffs)
    #
    @test space(a) == 𝕊
    @test coefficients(a) == coeffs
    @test order(a) == order(𝕊)
    @test order(a, 1) == order(𝕊, 1)
    @test_throws MethodError frequency(a)
    @test frequency(block(a, 1), 2) == frequency(𝑇, 2)
    @test firstindex(a) == RadiiPolynomial._firstindex(𝕊)
    @test lastindex(a) == RadiiPolynomial._lastindex(𝕊)
    @test length(a) == length(coeffs)
    @test size(a) == size(coeffs)
    @test iterate(a) == iterate(coeffs)
    @test iterate(a, 2) == iterate(coeffs, 2)
    @test eltype(a) == eltype(typeof(a)) == eltype(coeffs)
    @test a[1:3] == coeffs[1:3]
    @test view(a, 1:3) == view(coeffs, 1:3)
    @test coefficients(setindex!(copy(a), 0.0, 1)) == setindex!(copy(coeffs), 0.0, 1)
    @test a == a ≈ a
    @test !iszero(a)
    @test copy(a) == a
    @test iszero(zero(a))
    @test float(a) == Sequence(𝕊, float(coeffs))
    @test complex(a) == Sequence(𝕊, complex(coeffs))
    @test real(a) == Sequence(𝕊, real(coeffs))
    @test imag(a) == Sequence(𝕊, imag(coeffs))
    @test conj(a) == Sequence(𝕊, conj(coeffs))
    @test conj!(copy(a)) == Sequence(𝕊, conj!(copy(coeffs)))
    #
    a_𝑇 = block(a, 1)
    @test coefficients(a_𝑇) == coeffs
    @test selectdim(a_𝑇, 2, 0) == selectdim(reshape(coeffs, 2, 3, 3), 2, 2)
end
