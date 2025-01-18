
using JuMP
import Ipopt
import Plots
using LinearAlgebra
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# number of grid points
const N = 100

#
# Earth constants
#

const µ🜨   = 3.986012e14   # m³/s²
const r🜨   = 6378145       # m
const rho0 = 1.225         # kg/m³
const H0   = 8500          # m
const g0   = 9.80665       # m/s²

#
# Vehicle constants
#

const Aref = 10
const Cd = 0.2
const Isp = 300
const mi = 5000
const mprop = 0.6*mi
const Tmax = mi*g0*2

#
# Initial conditions
#

const hi = 0
const vi = 0

#
# Scaling
#

const r_scale = norm(r🜨)
const v_scale = sqrt(µ🜨/r_scale)
const t_scale = r_scale / v_scale
const m_scale = mi
const a_scale = v_scale / t_scale
const f_scale = m_scale * a_scale
const area_scale = r_scale^2
const vol_scale = r_scale * area_scale
const d_scale = m_scale / vol_scale
const mdot_scale = m_scale / t_scale

#
# Applying scaling
#

const his = hi / r_scale
const vis = vi / v_scale
const mis = mi / m_scale
const r🜨s = r🜨 / r_scale
const mprops = mprop / m_scale
const Tmaxs = Tmax / f_scale
const H0s = H0 / r_scale
const rho0s = rho0 / d_scale
const Arefs = Aref / area_scale
const c = Isp*g0 / v_scale

#
# Computed values
#

const mfs = mis - mprops

function goddard()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "print_user_options", "yes")
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "tol", 1e-12)

    tau, w, wB, Ba, Bb = psmethod(N-1)

    #
    # Variables
    #

    @variable(model, 0 <= h[i=1:N] <= Inf, start=0)
    @variable(model, 0 <= v[i=1:N] <= Inf, start=0)
    @variable(model, mfs <= m[i=1:N] <= mis, start=mis)
    @variable(model, hdot[i=1:N], start = 0)
    @variable(model, vdot[i=1:N], start = 0)
    @variable(model, mdot[i=1:N], start = 0)
    @variable(model, 0 <= ha <= Inf, start = 0)
    @variable(model, 0 <= va <= Inf, start = 0)
    @variable(model, mfs <= ma <= mis, start=mis)
    @variable(model, 0 <= hb <= Inf, start = 0)
    @variable(model, 0 <= vb <= Inf, start = 0)
    @variable(model, mfs <= mb <= mis, start=mis)
    @variable(model, 0 <= T[i=1:N] <= Tmaxs, start=Tmaxs)
    @variable(model, 0 <= tf <= Inf, start=20/t_scale)
    @variable(model, 0 <= ti <= 0, start=0)

    #
    # Collected decision variables
    #

    X = hcat( h, v, m )
    V = hcat( hdot, vdot, mdot )
    Xa = hcat(ha, va, ma)
    Xb = hcat(hb, vb, mb)

    #
    # Fixed constraints
    #

    fix(ha, his, force=true)
    fix(va, vis, force=true)
    fix(ma, mis, force=true)
    fix(ti, 0, force=true)

    #
    # Dynamical constraints
    #

    @expression(model, rho[i=1:N], rho0s*exp(-h[i]/H0s))
    @expression(model, Drag[i=1:N], -0.5*Cd*Arefs*rho[i]*v[i]^2)
    @expression(model, rsqr[i=1:N], (r🜨s + h[i])^2)

    F = hcat(
              v,
              -1.0 ./ rsqr + (T + Drag) ./ m,
              -T ./ c,
             )

    @constraint(model, dyn1, 0 == X - Ba * V - Xa .* ones(N))
    @constraint(model, dyn2, 0 == (tf - ti) / 2 * F - V)
    @constraint(model, dyn3, 0 == Xa + wB' * V - Xb)

    #
    # Objective
    #

    @objective(model, Max, h[N])

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    #
    # Structure Detection
    #

    nu = 0.05

    T_val = value.(T)

    T_norm = ( T_val .- minimum(T_val) ) ./ ( 1 + maximum(T_val) - minimum(T_val) )

    midpoints = [(tau[i] + tau[i+1]) / 2 for i in 1:length(tau)-1]

    function cj(tau, Sidx, j)
        midx = length(Sidx)-1
        val = factorial(big(midx))
        for i in Sidx
            i == j && continue
            val /= tau[j] - tau[i]
        end
        return val
    end

    MMLm = []

    for t in midpoints
        sortidx = sortperm(abs.(tau .- t))
        arry = []
        for midx in 1:6 # mu = 6
            Sidx = sortidx[1:midx+1]
            temp = findall(x -> x > t, tau[Sidx])
            Splusidx = sortidx[temp]
            qm = 0
            for j in Splusidx
                qm += cj(tau, Sidx, j)
            end
            Lm = 0
            for j in Sidx
                Lm += cj(tau, Sidx, j) * T_norm[j]
            end
            Lm /= qm
            push!(arry, Lm)
        end
        if all(x -> x > 0, arry)
            push!(MMLm, minimum(arry))
            if minimum(arry) > nu
                val = (t * tf / 2.0 + tf / 2.0) * t_scale
                display(value(val))
                display(minimum(arry))
            end
        elseif all(x -> x < 0, arry)
            push!(MMLm,abs(maximum(arry)))
            if abs(maximum(arry)) > nu
                val = (t * tf / 2.0 + tf / 2.0) * t_scale
                display(value(val))
                display(maximum(arry))
            end
        else
            push!(MMLm, 0)
        end
    end

    #
    # Descale and interpolate the variables
    #

#    range = LinRange(-1,1,100)
#    L = lagrange_basis(ptau, range)

#    h = lagrange_interpolation(L, value.(hp)) * r_scale
#    v = lagrange_interpolation(L, value.(vp)) * v_scale
#    m = lagrange_interpolation(L, value.(mp)) * m_scale

#    L = lagrange_basis(tau, range)

#    T = lagrange_interpolation(L, value.(T)) * f_scale

    #
    # Construct ranges of real times at the interpolation points
    #

#    tf = value(tf)
#    t = (range * tf / 2.0 .+ tf / 2.0) * t_scale

    #
    # Do some plotting of interpolated results
    #

    p1 = Plots.plot(
                    tau,
                    value.(h),
                    xlabel = "Time (s)",
                    ylabel = "Height",
                    legend = false
                   )
    p2 = Plots.plot(
                    tau,
                    value.(v),
                    xlabel = "Time (s)",
                    ylabel = "Velocity",
                    legend = false
                   )
    p3 = Plots.plot(
                    tau,
                    value.(m),
                    xlabel = "Time (s)",
                    ylabel = "Mass",
                    legend = false
                   )
    p4 = Plots.plot(
                    tau,
                    value.(T),
                    xlabel = "Time (s)",
                    ylabel = "Thrust",
                    legend = false
                   )
    p5 = Plots.plot(
                    tau,
                    T_norm,
                    xlabel = "Tau",
                    ylabel = "T_norm",
                    legend = false
                   )
    p6 = Plots.plot(
                    midpoints,
                    MMLm,
                    xlabel = "Tau",
                    ylabel = "MMLm",
                    legend = false
                   )
    display(Plots.plot(p1, p2, p3, p4, p5, p6, layout=(2,3), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

goddard()
