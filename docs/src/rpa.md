# Radii polynomial

The solution of many problems in dynamical systems can often be seen as the fixed point of an operator. The *radii polynomial approach* is a strategy in computer-assisted proofs to obtain a rigorous a posteriori error bound on a numerical approximation of the fixed point. More precisely, the Radii Polynomial theorem gives us sufficient conditions for the operator to be a contraction in a closed ball centred at the numerical approximation; this implies the existence and (local) uniqueness of the desired fixed point within this ball.

!!! info "Radii polynomial theorem"
    Let ``N \in \mathbb{N}``, ``X`` be a normed vector space, be ``U`` be an open subset of ``X`` and ``G \in C^m (U,X)`` where ``m \geq N``. Consider ``x_0`` and ``r_*>0`` such that ``\overline{B_{r_*}(x_0)} \subset U``.
    Suppose ``Y, Z_1, \dots, Z_{N-1} \geq 0`` and ``Z_N : [0,+\infty) \to [0,+\infty)`` satisfy

    ```math
    \begin{aligned}
    \| G(x_0) - x_0 \|_X &\leq Y,\\ \| D^k G(x_0) \|_{B(X^{\otimes k}, X)} &\leq Z_k \quad (k=1,\dots,N-1),\\ \sup_{y \in \overline{B_{r_*}(x_0)}} \| D^N G(y) \|_{B(X^{\otimes N}, X)} &\leq Z_N(r_*).
    \end{aligned}
    ```

    Define

    ```math
    p_N(r,r_*) \doteqdot Y - r + \sum_{k=1}^{N-1} \frac{r^k}{k!} Z_k + \frac{r^N}{N!} Z_N(r_*).
    ```

    If there exists ``r_0 \in [0, r_*]`` such that

    ```math
    \begin{aligned} p_N(r_0,r_*) &\leq 0,\\ \frac{r_0^{N-1}}{(N-1)!} Z_N(r_*) &< 1, \end{aligned}
    ```

    then ``G`` has a unique fixed point ``\tilde{x} \in \overline{B_{r_0}(x_0)}``.

!!! note
    The Radii Polynomial theorem is related to the [Newton-Kantorovich Theorem](https://en.wikipedia.org/wiki/Kantorovich_theorem).
