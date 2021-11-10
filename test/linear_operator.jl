@testset "LinearOperator" begin
    coeffs = rand(3*3, 2+3)
    A = LinearOperator(Taylor(1) × Chebyshev(2), Fourier(1, 1.0)^3, coeffs)
    @test domain(A) == Taylor(1) × Chebyshev(2)
    @test codomain(A) == Fourier(1, 1.0)^3
    @test coefficients(A) == coeffs
    @test firstindex(A, 1) == 1
    @test firstindex(A, 2) == 1
    @test lastindex(A, 1) == 9
    @test lastindex(A, 2) == 5
    @test size(A) == (9, 5)
    @test length(A) == 45
end
