using JuMP
import Ipopt
import Plots
using LinearAlgebra
using Printf
using Glob

foreach(include, glob("*.jl", "lib"))

# number of grid points
const N = 800

#
# Problem constants
#

const ti = 0
const tf = 1000
const xival = 1.5
const xfval = 1

const xmax = 50
const xmin = -xmax
const umax = 50
const umin = -umax

function hypersensitive()
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 5)
    set_optimizer_attribute(model, "print_user_options", "yes")
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "tol", 1e-12)

    tau, w, wB, Ba, Bb = psmethod(N-1)

    #
    # Variables
    #

    @variable(model, xmin <= x[i=1:N] <= xmax, start=0)
    @variable(model, v[i=1:N], start=0)
    @variable(model, umin <= u[i=1:N] <= umax, start=0)

    #
    # Endpoint variable slices
    #

    xi = vcat(x[1])
    xf = vcat(x[N])

    #
    # Endpoint constraints
    #

    fix(x[1], xival, force=true)
    fix(x[N], xfval, force=true)

    #
    # Dynamical constraints
    #

    @constraint(model, x == xi .* ones(N) + Ba * v)
    @constraint(model, v == (tf - ti) ./ 2 * ( -x.^3 .+ u ))
    @constraint(model, xf == xi + wB' * v)

    #
    # Objective
    #

    @objective(model, Min, dot(wB, 0.5*(x.^2 .+ u.^2)))

    #
    # Solve the problem
    #

    solve_and_print_solution(model)

    p1 = Plots.plot(
                    tau,
                    value.(x),
                    xlabel = "Tau",
                    ylabel = "State",
                    legend = false
                   )

    p2 = Plots.plot(
                    tau,
                    value.(u),
                    xlabel = "Tau",
                    ylabel = "Control",
                    legend = false
                   )

    display(Plots.plot(p1, p2, layout=(2,2), legend=false))
    readline()

    @assert is_solved_and_feasible(model)
end

hypersensitive()

