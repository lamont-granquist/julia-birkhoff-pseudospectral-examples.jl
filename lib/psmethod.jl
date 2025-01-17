using FastTransforms

function cgl_weights(N)
    w = (π / (N)) * ones(N+1)
    w[1] *= 0.5
    w[end] *= 0.5
    return w
end

function gen_sa(P, N, tau)
    Sa = zeros(N+1, N+1)
    rpi2 = 2.0/π
    rpi = 1.0/π
    for j in 1:N+1
        # k = 0
        Sa[j, 1] = rpi * (tau[j] - tau[1])
        # k = 1
        Sa[j, 2] = rpi * (tau[j]^2 - tau[1]^2)
        for n in 2:N
            sign = n % 2 == 1 ? -1 : 1
            Sa[j, n+1] = rpi2 * ( P[n+2,j] / (2*(n+1)) - P[n,j] / (2*(n-1)) - sign / (n^2-1) )
        end
    end
    return Sa
end

function gen_sb(P, N, tau)
    Sb = zeros(N+1, N+1)
    rpi2 = 2.0/π
    rpi = 1.0/π
    for j in 1:N+1
        # k = 0
        Sb[j, 1] = rpi * (tau[j] - tau[end])
        # k = 1
        Sb[j, 2] = rpi * (tau[j]^2 - tau[end]^2)
        for n in 2:N
            Sb[j, n+1] = rpi2 * ( P[n+2,j] / (2*(n+1)) - P[n,j] / (2*(n-1)) + 1 / (n^2-1) )
        end
    end
    return Sb
end

function chebeval(N::Int, tau::AbstractVector{<:Real})
    T = zeros(N, length(tau))
    T[1,:] .= 1.0
    T[2,:] .= tau
    for n in 3:N
        T[n,:] .= 2.0 .* tau .* T[n-1,:] .- T[n-2,:]
    end
    return T
end

# N is the number of segments, N+1 is the number of gridpoints
function psmethod(N)
    tau = reverse(clenshawcurtisnodes(Float64, N+1))
    w = cgl_weights(N)
    μ = FastTransforms.chebyshevmoments1(Float64, N+1)
    #wB = clenshawcurtisweights(μ) # this may be slighly more accurate than taking it from the last row of the Ba matrix?

    P = chebeval(N+2, tau)

    Q = P[1:end-1,:] .* w'
    Q[end, :] .= Q[end, :] ./ 2

    Sa = gen_sa(P, N, tau)
    Sb = gen_sb(P, N, tau)

    # Proposition 6
    Ba = Sa * Q
    Bb = Sb * Q

    wB = Ba[end, :]

    return tau, w, wB, Ba, Bb
end
