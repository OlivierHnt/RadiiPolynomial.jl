```@contents
Pages = ["cubic_root_cont.md"]
Depth = 3
```

# Parameter continuation for the cubic root

In this example, we prove the existence of a one-parameter family of solutions to

```math
0 = f(u, \lambda) \bydef u^3 - \lambda - 2, \qquad \lambda \in [-1, 1].
```

### Step 1: Problem definition

The map ``f`` and its derivative ``D_u f`` are implemented as follows:

```@example cubic_root_cont
f(u, λ) = u^3 - λ - exact(2)

Duf(u, λ) = exact(3) * u^2
nothing # hide
```

Consider the fixed-point operator ``T : \mathbb{R} \times [-1,1] \to \mathbb{R}`` given by

```math
T(u, \lambda) \bydef u - A(\lambda) f(u, \lambda),
```

where ``A(\lambda) : \mathbb{R} \to \mathbb{R}`` is an injective operator corresponding to a numerical approximation of ``D_u f(\bar{u}(\lambda), \lambda)^{-1}``, for some approximate zero ``\bar{u}(\lambda) \in \mathbb{R}`` of ``f``, for all ``\lambda \in [-1, 1]``.

### Step 2: Approximate zero (floating-point arithmetic)

We use the [numerical continuation method](https://en.wikipedia.org/wiki/Numerical_continuation) to retrieve a numerical approximation of the curve.

We construct a grid of parameters and iterate Newton's method for each step, using the previous approximate zero as the **predictor** of the solution at the next step.

```@example cubic_root_cont
using RadiiPolynomial

K = 10
K_fft = fft_size(Chebyshev(K))
npts = K_fft ÷ 2 + 1

λ_grid = [-cospi(2j/K_fft) for j = 0:npts-1]
u_grid = Vector{Float64}(undef, npts)

# initialize

u_init = 1.0
u_init, success_newton = newton(u -> (f(u, λ_grid[1]), Duf(u, λ_grid[1])), u_init)

u_grid[1] = u_init

# run continuation scheme

for j = 2:npts
    w = u_grid[j-1] # predictor

    u_bar, success_newton_j = newton(u -> (f(u, λ_grid[j]), Duf(u, λ_grid[j])), w; verbose = true)
    success_newton_j || error()

    u_grid[j] = u_bar
end

# construct the approximation

u_fft = [reverse(u_grid) ; u_grid[begin+1:end-1]]
u_cheb = real(to_seq(u_fft, Chebyshev(K)))
```

### Step 3: Approximate inverse (floating-point arithmetic)

We construct the approximate inverse ``A(\lambda) \approx D_u f(\bar{u}(\lambda), \lambda)^{-1}`` across the continuation branch using standard floating-point arithmetic.

```@example cubic_root_cont
A_grid = inv.(Duf.(u_grid, λ_grid))
A_fft = [reverse(A_grid) ; A_grid[begin+1:end-1]]
A_cheb = real(to_seq(A_fft, Chebyshev(K)))
```

### Step 4: Bounds estimation (interval arithmetic)

To apply the Radii Polynomial Theorem, we need to theoretically derive and rigorously evaluate the bounds ``Y, Z_1, Z_2``. The computer-assisted proof is completed by evaluating these bounds with interval arithmetic:

```@example cubic_root_cont
λ_cheb_interval = interval(Sequence(Chebyshev(1), [0, 0.5]))
u_cheb_interval = interval(u_cheb)
A_cheb_interval = interval(A_cheb)

#- Y bound

Y = norm(A_cheb_interval * f(u_cheb_interval, λ_cheb_interval), 1)

#- Z₁ bound

Z₁ = norm(exact(1) - A_cheb_interval * Duf(u_cheb_interval, λ_cheb_interval), 1)

#- Z₂ bound

R = 10 * sup(Y)
Z₂ = exact(3) * norm(A_cheb_interval, 1) * (norm(u_cheb_interval, 1) + exact(2 * R))

# verify the contraction

ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R; verbose = true)
nothing # hide
```

```@example cubic_root_cont
inf(ie) # smallest error
```

The following figure[^1] shows the numerical approximation of the proven branch of the cubic root.

[^1]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@example cubic_root_cont
using GLMakie

fig = Figure()
ax = Axis(fig[1,1], xticks = -3:3)
lines!(ax, [Point2f(λ, cbrt(λ + 2)) for λ = LinRange(-3, 2, 501)];
    color = :black, label = L"(λ+2)^{1/3}")
lines!(ax, [Point2f(λ, u_cheb(λ)) for λ = LinRange(-1, 1, 501)];
    color = :blue, label = L"\bar{u}(λ)")
scatter!(ax, [Point2f(λ_grid[j], u_grid[j]) for j = 1:npts];
    color = :red)
axislegend(ax; position = :lt)
fig
```
