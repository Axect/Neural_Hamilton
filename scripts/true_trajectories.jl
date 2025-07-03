using DifferentialEquations
using DataFrames
using Parquet2
using LinearAlgebra
using Parameters
using Printf
using PCHIPInterpolation
using ForwardDiff
using ProgressMeter
using Base.Threads

const NSENSORS = 100
const PI = π

# ┌─────────────────────────────────────────────────────────┐
#  Potential Definitions
# └─────────────────────────────────────────────────────────┘

# Simple Harmonic Oscillator
function V_SHO(q)
    return 8.0 * (q - 0.5)^2
end

function dV_SHO(q)
    return 16.0 * (q - 0.5)
end

# Double Well Potential
function V_DoubleWell(q)
    return 625.0 / 8.0 * (q - 0.2)^2 * (q - 0.8)^2
end

function dV_DoubleWell(q)
    return 625.0 / 2.0 * q^3 - 1875.0 / 4.0 * q^2 + 825.0 / 4.0 * q - 25.0
end

# Morse Potential
@with_kw struct MorseParams
    a::Float64 = 3.0 * log((1.0 + sqrt(5.0)) / 2.0)
    D_e::Float64 = 8.0 / (sqrt(5.0) - 1.0)^2
    r_e::Float64 = 1.0 / 3.0
end

function V_Morse(q, p::MorseParams)
    return p.D_e * (1.0 - exp(-p.a * (q - p.r_e)))^2
end

function dV_Morse(q, p::MorseParams)
    return 2.0 * p.D_e * (1.0 - exp(-p.a * (q - p.r_e))) * p.a * exp(-p.a * (q - p.r_e))
end

# Pendulum
@with_kw struct PendulumParams
    theta_L::Float64 = PI / 3.0
end

function V_Pendulum(q, p::PendulumParams)
    g = 2.0 / (1.0 - cos(p.theta_L))
    return g * (1.0 - cos(2.0 * p.theta_L * (q - 0.5)))
end

function dV_Pendulum(q, p::PendulumParams)
    g = 2.0 / (1.0 - cos(p.theta_L))
    return g * 2.0 * p.theta_L * sin(2.0 * p.theta_L * (q - 0.5))
end

# Mirrored Free Fall
function V_MirroredFreeFall(q)
    return 4.0 * abs(q - 0.5)
end

function dV_MirroredFreeFall(q)
    return 4.0 * sign(q - 0.5)
end

# Softened Mirrored Free Fall
function V_SoftenedMirroredFreeFall(q)
    alpha = 20.0
    if q == 0.5
        a = 4.0 * tanh(0.5 * alpha)
        return a / alpha
    end
    a = 4.0 * tanh(0.5 * alpha)
    return a * (q - 0.5) / tanh(alpha * (q - 0.5))
end

function dV_SoftenedMirroredFreeFall(q)
    alpha = 20.0
    a = 4.0 * tanh(0.5 * alpha)
    if q == 0.5
        return 0.0
    end
    tanh_val = tanh(alpha * (q - 0.5))
    sinh_val = sinh(alpha * (q - 0.5))
    return a * (1.0 / tanh_val - alpha * (q - 0.5) / (sinh_val^2))
end

# ATW
@with_kw struct ATWParams
    lambda::Float64 = 0.25
end

function V_ATW(q, p::ATWParams)
    lambda = p.lambda
    if q < lambda
        return 2.0 * (1.0 - q / lambda)
    else
        return 2.0 * (1.0 - (1.0 - q) / (1.0 - lambda))
    end
end

function dV_ATW(q, p::ATWParams)
    lambda = p.lambda
    if q == lambda
        return NaN
    end
    if q < lambda
        return -2.0 / lambda
    else
        return 2.0 / (1.0 - lambda)
    end
end

# ┌─────────────────────────────────────────────────────────┐
#  Hamiltonian System Definitions
# └─────────────────────────────────────────────────────────┘

# SHO
function hamiltonian_sho_p(p_momentum, q, params, t) # dp/dt
    return -dV_SHO(q)
end
function hamiltonian_sho_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Double Well
function hamiltonian_double_well_p(p_momentum, q, params, t) # dp/dt
    return -dV_DoubleWell(q)
end
function hamiltonian_double_well_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Morse
function hamiltonian_morse_p(p_momentum, q, params, t) # dp/dt
    return -dV_Morse(q, params)
end
function hamiltonian_morse_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Pendulum
function hamiltonian_pendulum_p(p_momentum, q, params, t) # dp/dt
    return -dV_Pendulum(q, params)
end
function hamiltonian_pendulum_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Mirrored Free Fall
function hamiltonian_stw_p(p_momentum, q, params, t) # dp/dt
    return -dV_MirroredFreeFall(q)
end
function hamiltonian_stw_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Softened Mirrored Free Fall
function hamiltonian_sstw_p(p_momentum, q, params, t) # dp/dt
    return -dV_SoftenedMirroredFreeFall(q)
end
function hamiltonian_sstw_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# ATW
function hamiltonian_atw_p(p_momentum, q, params, t) # dp/dt
    val_dV = dV_ATW(q, params)
    if isnan(val_dV)
    end
    return -val_dV
end
function hamiltonian_atw_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# ┌─────────────────────────────────────────────────────────┐
#  Loading Test Data and Generating Reference Data
# └─────────────────────────────────────────────────────────┘

# Read V data from Parquet2 file
function load_test_potentials(file_path)
    tbl = Parquet2.readfile(file_path)
    df = DataFrame(tbl)
    
    # Extract V field and divide into NSENSORS-sized chunks
    V_values = df.V
    t_values = df.t
    num_potentials = div(length(V_values), NSENSORS)
    
    potentials = []
    ts = []
    for i in 1:num_potentials
        start_idx = (i-1) * NSENSORS + 1
        end_idx = i * NSENSORS
        push!(potentials, V_values[start_idx:end_idx])
        push!(ts, t_values[start_idx:end_idx])
    end
    
    return potentials, ts
end

# Run simulation for given potential
function run_simulation_kl8(V_values, t_values)
    # Set interval and time
    tspan = (0.0, 2.0)
    dt = 1e-4  # Very small timestep to ensure accuracy
    
    # Uniform q coordinates (0.0 ~ 1.0)
    q_range = range(0.0, 1.0, length=NSENSORS)
    
    # Create spline for potential derivative calculation
    V_spline = Interpolator(q_range, V_values)
    dV_spline(q) = ForwardDiff.derivative(V_spline, q)
    
    # Define Hamiltonian system
    function hamiltonian_dp(p, q, params, t)
        q_val = clamp(q, 0.0, 1.0)  # Handle boundary cases
        return -dV_spline(q_val)  # Negative potential derivative
    end
    
    function hamiltonian_dq(p, q, params, t)
        return p  # Mass m=1
    end
    
    # Initial conditions
    q0 = 0.0
    p0 = 0.0
    
    # Define differential equation problem
    prob = DynamicalODEProblem(hamiltonian_dp, hamiltonian_dq, p0, q0, tspan, nothing)
    
    # Solve with Kahan-Li 8th order symplectic integrator
    sol = solve(prob, KahanLi8(), dt=dt, adaptive=false)
    
    # Sample at t_values
    solution = sol(t_values)
    
    # Extract results
    p_values = [solution[1, i] for i in 1:NSENSORS]
    q_values = [solution[2, i] for i in 1:NSENSORS]
    
    return q_values, p_values
end

# ┌─────────────────────────────────────────────────────────┐
#  Run Simulation and Save
# └─────────────────────────────────────────────────────────┘

function run_simulation(potential_name, p_fn_ham, q_fn_ham, initial_condition, params, tspan)
    # initial_condition is [q_initial, p_initial]
    q_initial = initial_condition[1]
    p_initial = initial_condition[2]

    # DynamicalODEProblem(dp/dt_func, dq/dt_func, p_initial, q_initial, tspan, params)
    prob = DynamicalODEProblem(p_fn_ham, q_fn_ham, p_initial, q_initial, tspan, params)

    dt = 1e-4 
    sol = solve(prob, KahanLi8(), dt=dt, adaptive=false)

    # Sample at uniform time intervals
    t_uniform = range(tspan[1], tspan[2], length=NSENSORS)
    solution = sol(t_uniform)

    # Extract results - DynamicalODEProblem(dp/dt, dq/dt, p0, q0, ...) format,
    # so each element of solution.u has [p, q] format.
    # solution[1, i] is p (momentum) at time step i
    # solution[2, i] is q (position) at time step i
    p_values = [solution[1, i] for i in 1:NSENSORS]
    q_values = [solution[2, i] for i in 1:NSENSORS]

    # Calculate potential values (based on q range)
    q_range_for_V = range(0.0, 1.0, length=NSENSORS) # Match original code (0.0 to 1.0)
    V_values = zeros(NSENSORS)

    # Calculate V values for each potential
    if potential_name == "sho"
        V_values = [V_SHO(q_v) for q_v in q_range_for_V]
    elseif potential_name == "double_well"
        V_values = [V_DoubleWell(q_v) for q_v in q_range_for_V]
    elseif potential_name == "morse"
        morse_params_local = MorseParams() # Ensure correct params are used for V plot
        V_values = [V_Morse(q_v, morse_params_local) for q_v in q_range_for_V]
    elseif potential_name == "pendulum"
        pendulum_params_local = PendulumParams()
        V_values = [V_Pendulum(q_v, pendulum_params_local) for q_v in q_range_for_V]
    elseif potential_name == "stw"
        V_values = [V_MirroredFreeFall(q_v) for q_v in q_range_for_V]
    elseif potential_name == "sstw"
        V_values = [V_SoftenedMirroredFreeFall(q_v) for q_v in q_range_for_V]
    elseif potential_name == "atw"
        atw_params_local = ATWParams()
        V_values = [V_ATW(q_v, atw_params_local) for q_v in q_range_for_V]
    end

    # Convert results to DataFrame
    df = DataFrame(
        V = V_values,      # Potential shape over a fixed q_range [0,1]
        t = collect(t_uniform), # Time points of the simulation
        q_true = q_values, # Simulated trajectory positions
        p_true = p_values  # Simulated trajectory momenta
    )

    return df
end

function run_all_simulations()
    # Create data storage directory
    mkpath("data_true")

    # Run simulations for all potentials
    # potentials = [ (name, dp/dt_func, dq/dt_func, [q0, p0], params_struct), ... ]
    potentials = [
        ("sho", hamiltonian_sho_p, hamiltonian_sho_q, [0.0, 0.0], nothing),
        ("double_well", hamiltonian_double_well_p, hamiltonian_double_well_q, [0.0, 0.0], nothing),
        ("morse", hamiltonian_morse_p, hamiltonian_morse_q, [0.0, 0.0], MorseParams()), # q0=0 for Morse can be far from r_e
        ("pendulum", hamiltonian_pendulum_p, hamiltonian_pendulum_q, [0.0, 0.0], PendulumParams()), # q0=0 means theta = -theta_L
        ("stw", hamiltonian_stw_p, hamiltonian_stw_q, [0.0, 0.0], nothing),
        ("sstw", hamiltonian_sstw_p, hamiltonian_sstw_q, [0.0, 0.0], nothing),
        ("atw", hamiltonian_atw_p, hamiltonian_atw_q, [0.0, 0.0], ATWParams()) # q0=0 is in the first slope segment
    ]

    for (name, p_fn, q_fn, init_cond, params_obj) in potentials
        @printf "Running simulation for %s...\n" name
        df = run_simulation(name, p_fn, q_fn, init_cond, params_obj, (0.0, 2.0))

        # Save as Parquet2 file
        output_file = "data_true/$(name).parquet"
        Parquet2.writefile(output_file, df)
        @printf "Saved to %s\n" output_file
    end

    println("All simulations completed!")
end

function run_simulation_reference(input_path::String = "data_test/test.parquet", output_path::String = "data_true/test_kl8.parquet")
    file_path = input_path

    @printf "Loading test potentials from %s...\n" file_path
    potentials, ts = load_test_potentials(file_path)
    @printf "Loaded %d potentials\n" length(potentials)

    n_total = length(potentials) * NSENSORS
    all_V = Vector{Float64}(undef, n_total)
    all_t = Vector{Float64}(undef, n_total)
    all_q_true = Vector{Float64}(undef, n_total)
    all_p_true = Vector{Float64}(undef, n_total)

    pb = Progress(length(potentials), desc="Obtain: ", showspeed=true)

    # Parallelize the simulation
    #Threads.@threads for i in 1:length(potentials)
    for i in 1:length(potentials)
        V_values = potentials[i]
        t_values = ts[i]
        q_values, p_values = run_simulation_kl8(V_values, t_values)
        
        # Store results in the preallocated arrays
        start_idx = (i-1) * NSENSORS + 1
        end_idx = i * NSENSORS
        
        all_V[start_idx:end_idx] = V_values
        all_t[start_idx:end_idx] = t_values
        all_q_true[start_idx:end_idx] = q_values
        all_p_true[start_idx:end_idx] = p_values
        
        # Update progress bar
        next!(pb)
    end
    finish!(pb)

    # Convert results to DataFrame
    df = DataFrame(
        V = all_V,
        t = all_t,
        q_true = all_q_true,
        p_true = all_p_true
    )

    # Save as Parquet2 file
    output_file = output_path
    @printf "Saving to %s...\n" output_file
    Parquet2.writefile(output_file, df)

    @printf "Saved to %s\n" output_file

    @printf "Reference simulation completed!\n"
end

# Run all simulations
run_all_simulations()
# Run reference simulation (test)
run_simulation_reference("data_test/test.parquet", "data_true/test_kl8.parquet")
# Run reference simulation (normal)
run_simulation_reference("data_normal/train.parquet", "data_true/train_normal_kl8.parquet")
run_simulation_reference("data_normal/val.parquet", "data_true/val_normal_kl8.parquet")
# Run reference simulation (more)
run_simulation_reference("data_more/train.parquet", "data_true/train_more_kl8.parquet")
run_simulation_reference("data_more/val.parquet", "data_true/val_more_kl8.parquet")
# Run reference simulation (much)
#run_simulation_reference("data_much/train.parquet", "data_true/train_much_kl8.parquet")
#run_simulation_reference("data_much/val.parquet", "data_true/val_much_kl8.parquet")
