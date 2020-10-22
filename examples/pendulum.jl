# Pendulum Swing-Up Direct Transcription
using LinearAlgebra
const MOI = MathOptInterface

# pendulum dynamics
function pendulum_dynamics(x,u)
    ẋ = zero(x)
    m = 1.
    l = 0.5
    b = 0.1
    lc = 0.5
    I = 0.25
    g = 9.81
    ẋ[1] = x[2]
    ẋ[2] = (u[1] - m*g*lc*sin(x[1]) - b*x[2])/I

    return ẋ
end

n = 2; m = 1 # states, controls

# Problem
x0 = [0.;0.]
xf = [pi;0.]

N = 51 # knot points
NN = (n+m+1)*(N-1) + n # number of decision variables

p = n*N + n + (N-2)# number of equality constraints: dynamics, xf, h_k = h_{k+1}

Q = Diagonal(zeros(n))
R = Diagonal(zeros(m))
Qf = Diagonal(zeros(n))

tf0 = 3.
dt = tf0/(N-1)

u_max = 3.
u_min = -3.

h_max = dt
h_min = dt

function line_trajectory(x0::Vector,xf::Vector,N::Int)
    t = range(0,stop=N,length=N)
    slope = (xf .- x0)./N
    x_traj = [slope*t[k] for k = 1:N]
    x_traj[1] = x0
    x_traj[end] = xf
    x_traj
end

X = line_trajectory(x0,xf,N)
U = [0.01*rand(m) for k = 1:N-1]
H = [dt*ones(1) for k = 1:N-1]

# dynamics constraints
fc(z) = pendulum_dynamics(z[1:n],z[n .+ (1:m)])
fc(x,u) = pendulum_dynamics(x,u)
∇fc(z) = ForwardDiff.jacobian(fc,z)
∇fc(x,u) = ∇fc([x;u])
dfcdx(x,u) = ∇fc(x,u)[:,1:n]
dfcdu(x,u) = ∇fc(x,u)[:,n .+ (1:m)]

F(y,x,u,h) = y - x - h*fc(0.5*(y+x),u)
dFdy(y,x,u,h) = I - 0.5*h*dfcdx(0.5*(y+x),u)
dFdx(y,x,u,h) = -I - 0.5*h*dfcdx(0.5*(y+x),u)
dFdu(y,x,u,h) = -h*dfcdu(0.5*(y+x),u)
dFdh(y,x,u,h) = -fc(0.5*(y+x),u)

function packZ(X,U,H)
    Z = [X[1];U[1];H[1]]

    for k = 2:N-1
        append!(Z,[X[k];U[k];H[k]])
    end
    append!(Z,X[N])

    return Z
end

unpackZ(Z) = let n=n, m=m, N=N, NN=NN
    X = [k != N ? Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][1:n] : Z[((k-1)*(n+m+1) + 1):NN] for k = 1:N]
    U = [Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][n .+ (1:m)] for k = 1:N-1]
    H = [Z[((k-1)*(n+m+1) + 1):k*(n+m+1)][(n+m+1)] for k = 1:N-1]
    return X,U,H
end

unpackZ_timestep(Z) = let n=n, m=m, N=N, NN=NN
    Z_traj = [k != N ? Z[((k-1)*(n+m+1) + 1):k*(n+m+1)] : Z[((k-1)*(n+m+1) + 1):NN] for k = 1:N]
end

g(x,u,h) = let Q=Q, R=R
    (x'*Q*x + u'*R*u) + h
end

dgdx(x,u,h) = let Q=Q
    2*Q*x
end

dgdu(x,u,h) = let R=R
    2*R*u
end

dgdh(x,u,h) = 1.

gf(x) = let Qf=Qf
    x'*Qf*x
end

dgfdx(x) = let Qf=Qf
    2*Qf*x
end

cost(Z) = let n=n, m=m, N=N, NN=NN, Q=Q, R=R, Qf=Qf
    J = 0.
    Z_traj = unpackZ_timestep(Z)

    for k = 1:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        J += g(0.5*(y+x),u,h)
    end
    xN = Z_traj[N][1:n]
    J += gf(xN)

    return J
end

∇cost!(∇J,Z) = let n=n, m=m, N=N, NN=NN, Q=Q, R=R, Qf=Qf
    Z_traj = unpackZ_timestep(Z)
    k = 1
    y = Z_traj[k+1][1:n]
    x = Z_traj[k][1:n]
    u = Z_traj[k][n .+ (1:m)]
    h = Z_traj[k][n+m+1]
    xm = 0.5*(y+x)
    ∇J[1:(n+m+1+n)] = [0.5*dgdx(xm,u,h);dgdu(xm,u,h);dgdh(xm,u,h);0.5*dgdx(xm,u,h)]

    for k = 2:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        xm = 0.5*(y+x)
        ∇J[((k-1)*(n+m+1) + 1):(k*(n+m+1)+n)] = [0.5*dgdx(xm,u,h);dgdu(xm,u,h);dgdh(xm,u,h);0.5*dgdx(xm,u,h)]
    end
    xN = Z_traj[N][1:n]
    ∇J[(NN - (n-1)):NN] = dgfdx(xN)

    return nothing
end

constraints!(g,Z) = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf,p=p
    Z_traj = unpackZ_timestep(Z)
    k = 1
    y = Z_traj[k+1][1:n]
    x = Z_traj[k][1:n]
    u = Z_traj[k][n .+ (1:m)]
    h = Z_traj[k][n+m+1]
    con = x-x0

    for k = 1:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]
        append!(con,F(y,x,u,h))

        if k != N-1
            h₊ = Z_traj[k+1][n+m+1]
            append!(con,h - h₊)
        end
    end
    xN = Z_traj[N][1:n]
    append!(con,xN-xf)

    # g = con
    copyto!(view(g,1:p),con)
    return nothing
end

∇constraints_vec!(∇con,Z) = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf, p=p
    Z_traj = unpackZ_timestep(Z)

    shift = 0
    # con = x-x0
    copyto!(view(∇con,1:n^2),vec(Diagonal(ones(n))))
    shift += n^2

    for k = 1:N-1
        y = Z_traj[k+1][1:n]
        x = Z_traj[k][1:n]
        u = Z_traj[k][n .+ (1:m)]
        h = Z_traj[k][n+m+1]

        # dynamics constraint jacobians
        copyto!(view(∇con,shift .+ (1:(n+m+1+n)*(n))),vec([dFdx(y,x,u,h) dFdu(y,x,u,h) dFdh(y,x,u,h) dFdy(y,x,u,h)]))
        shift += (n+m+1+n)*n

        if k != N-1
            copyto!(view(∇con,shift .+ (1:2)),[1.;-1.])
            shift += 2
        end
    end
    xN = Z_traj[N][1:n]
    copyto!(view(∇con,shift .+ (1:n^2)),vec(Diagonal(ones(n))))

    return nothing
end

constraint_sparsity() = let n=n, m=m, N=N, NN=NN, x0=x0, xf=xf, p=p

    row = []
    col = []

    r = 1:n
    c = 1:n

    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end


    for k = 1:N-1
        # dynamics constraint jacobians
        c_idx = ((k-1)*(n+m+1)) .+ (1:(n+m+1+n))
        r_idx = (n + (k-1)*(n + 1)) .+ (1:n)

        for cc in c_idx
            for rr in r_idx
                push!(row,convert(Int,rr))
                push!(col,convert(Int,cc))
            end
        end

        if k != N-1
            c_idx = ((k-1)*(n+m+1) + n + m) .+ 1
            c2_idx = ((k)*(n+m+1) + n + m) .+ 1
            r_idx = (n + n + (k-1)*(n + 1)) .+ 1

            for cc in c_idx
                for rr in r_idx
                    push!(row,convert(Int,rr))
                    push!(col,convert(Int,cc))
                end
            end

            for cc in c2_idx
                for rr in r_idx
                    push!(row,convert(Int,rr))
                    push!(col,convert(Int,cc))
                end
            end
        end


    end
    k = N
    r = (p - (n-1)):p
    # println(r)
    r = ((n + (k-1)*(n+1)-1) .+ (1:n))
    # println(r)
    c = (NN - (n-1)):NN

    for cc in c
        for rr in r
            push!(row,convert(Int,rr))
            push!(col,convert(Int,cc))
        end
    end

    return collect(zip(row,col))
end

primal_bounds() = let n=n, m=m, N=N, NN=NN, u_max=u_max, u_min=u_min, h_max=h_max, h_min=h_min
    Z_low = [-Inf*ones(n);u_min*ones(m);h_min]
    Z_up = [Inf*ones(n);u_max*ones(m);h_max]

    for k = 2:N-1
        append!(Z_low,[-Inf*ones(n);u_min*ones(m);h_min])
        append!(Z_up,[Inf*ones(n);u_max*ones(m);h_max])
    end
    append!(Z_low,-Inf*ones(n))
    append!(Z_up,Inf*ones(n))

    return Z_low, Z_up
end

constraint_bounds() = let p=p
    c_low = zeros(p); c_up = zeros(p)

    return c_low, c_up
end

## set up optimization (MathOptInterface)
ZZ = packZ(X,U,H)

struct Problem <: MOI.AbstractNLPEvaluator
    enable_hessian::Bool
end

MOI.features_available(prob::Problem) = [:Grad, :Jac]
MOI.initialize(prob::Problem, features) = nothing
MOI.jacobian_structure(prob::Problem) = constraint_sparsity()
MOI.hessian_lagrangian_structure(prob::Problem) = []

function MOI.eval_objective(prob::Problem, Z)
    return cost(Z)
end

function MOI.eval_objective_gradient(prob::Problem, grad_f, Z)
    ∇cost!(grad_f, Z)
end

function MOI.eval_constraint(prob::Problem, g, Z)
    constraints!(g,Z)
end

function MOI.eval_constraint_jacobian(prob::Problem, jac, Z)
    ∇constraints_vec!(jac,Z)
end

MOI.eval_hessian_lagrangian(prob::Problem, H, x, σ, μ) = nothing

prob = Problem(false)

Z_low, Z_up = primal_bounds()
c_low, c_up = constraint_bounds()

nlp_bounds = MOI.NLPBoundsPair.(c_low,c_up)
block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

solver = SNOPT7.Optimizer()

Z = MOI.add_variables(solver,NN)

for i = 1:NN
    zi = MOI.SingleVariable(Z[i])
    MOI.add_constraint(solver, zi, MOI.LessThan(Z_up[i]))
    MOI.add_constraint(solver, zi, MOI.GreaterThan(Z_low[i]))
    MOI.set(solver, MOI.VariablePrimalStart(), Z[i], ZZ[i])
end

# Solve the problem
MOI.set(solver, MOI.NLPBlock(), block_data)
MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
MOI.optimize!(solver)

# Get the solution
res = MOI.get(solver, MOI.VariablePrimal(), Z)

X,U,H = unpackZ(res)
Xblk = zeros(n,N)
for k = 1:N
    Xblk[:,k] = X[k]
end

Ublk = zeros(m,N-1)
for k = 1:N-1
    Ublk[:,k] = U[k]
end

plot(Xblk')
plot(Ublk')
