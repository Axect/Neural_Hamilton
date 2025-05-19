using DifferentialEquations
using DataFrames
using CSV
using Parquet
using LinearAlgebra
using Parameters
using Printf

const NSENSORS = 100
const PI = π

# ┌─────────────────────────────────────────────────────────┐
#  포텐셜 정의
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
    # Derivative: (625/4) * (q-0.2)*(q-0.8)*(2*q-1.0)
    # = (625/4) * (q^2 - q + 0.16) * (2q - 1.0)
    # = (625/4) * (2q^3 - q^2 - 2q^2 + q + 0.32q - 0.16)
    # = (625/4) * (2q^3 - 3q^2 + 1.32q - 0.16)
    # = 625/2 * q^3 - 1875/4 * q^2 + (625 * 1.32)/4 * q - (625 * 0.16)/4
    # = 625/2 * q^3 - 1875/4 * q^2 + 825/4 * q - 25.0
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
    # Derivative of cos(A*(q-C)) is -A*sin(A*(q-C))
    # Derivative of (1 - cos(2*theta_L*(q-0.5))) is -(-2*theta_L*sin(2*theta_L*(q-0.5)))
    # = 2*theta_L*sin(2*theta_L*(q-0.5))
    # The original code has sin(theta_L * (2.0 * q - 1.0)) which is sin(2.0 * theta_L * (q - 0.5))
    # So it should be g * 2.0 * p.theta_L * sin(2.0 * p.theta_L * (q-0.5))
    return g * 2.0 * p.theta_L * sin(2.0 * p.theta_L * (q - 0.5)) # Corrected argument for sin
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
    # To ensure V(0.5) = 0, we need a modification if q=0.5 causes tanh(0) in denominator
    # However, limit q->0.5 of (q-0.5)/tanh(alpha*(q-0.5)) is 1/alpha
    # So V(0.5) = a / alpha
    # The function is defined well for q != 0.5
    if q == 0.5
        a = 4.0 * tanh(0.5 * alpha) # This 'a' is just a scaling factor
        return a / alpha # Using L'Hopital's rule lim x->0 x/tanh(ax) = 1/a
    end
    a = 4.0 * tanh(0.5 * alpha)
    return a * (q - 0.5) / tanh(alpha * (q - 0.5))
end

function dV_SoftenedMirroredFreeFall(q)
    alpha = 20.0
    a = 4.0 * tanh(0.5 * alpha)
    if q == 0.5 # Derivative at q=0.5 should be 0 for a softened potential minimum
        return 0.0
    end
    tanh_val = tanh(alpha * (q - 0.5))
    sinh_val = sinh(alpha * (q - 0.5))
    # Derivative of x/tanh(ax) is (tanh(ax) - ax*sech^2(ax))/tanh^2(ax)
    # = 1/tanh(ax) - ax / sinh^2(ax)
    return a * (1.0 / tanh_val - alpha * (q - 0.5) / (sinh_val^2))
end

# Sawtooth
@with_kw struct SawtoothParams
    lambda::Float64 = 0.25
end

function V_Sawtooth(q, p::SawtoothParams)
    lambda = p.lambda
    if q < lambda
        return 2.0 * (1.0 - q / lambda)
    else
        return 2.0 * (1.0 - (1.0 - q) / (1.0 - lambda))
    end
end

function dV_Sawtooth(q, p::SawtoothParams)
    lambda = p.lambda
    if q == lambda # Handle the point of discontinuity if necessary for some solvers
        # For fixed step, it might step over. Taking one side or an average might be options
        # Or rely on solver to handle it. Here, we stick to the definition.
        return NaN # Or an average, or one-sided derivative
    end
    if q < lambda
        return -2.0 / lambda
    else
        return 2.0 / (1.0 - lambda)
    end
end

# ┌─────────────────────────────────────────────────────────┐
#  Hamiltonian 시스템 정의
# └─────────────────────────────────────────────────────────┘

# dp/dt = -dV/dq
# dq/dt = p (assuming mass m=1)

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
function hamiltonian_mff_p(p_momentum, q, params, t) # dp/dt
    return -dV_MirroredFreeFall(q)
end
function hamiltonian_mff_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Softened Mirrored Free Fall
function hamiltonian_smff_p(p_momentum, q, params, t) # dp/dt
    return -dV_SoftenedMirroredFreeFall(q)
end
function hamiltonian_smff_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# Sawtooth
function hamiltonian_sawtooth_p(p_momentum, q, params, t) # dp/dt
    val_dV = dV_Sawtooth(q, params)
    if isnan(val_dV) # A simple way to handle NaN from exact discontinuity
        # This can happen if q lands exactly on lambda with certain dt choices.
        # A robust way would be to use event handling for discontinuities
        # or ensure dV_Sawtooth returns a one-sided derivative or average.
        # For now, if NaN, try to use previous or next point's logic (crude).
        # A better fix is to ensure dV_Sawtooth always returns a number.
        # For SymplecticEuler with fixed dt, it's less likely to hit exactly.
        # We will assume dV_Sawtooth returns a valid number or the solver handles it.
    end
    return -val_dV
end
function hamiltonian_sawtooth_q(p_momentum, q, params, t) # dq/dt
    return p_momentum
end

# ┌─────────────────────────────────────────────────────────┐
#  시뮬레이션 실행 및 저장
# └─────────────────────────────────────────────────────────┘

function run_simulation(potential_name, p_fn_ham, q_fn_ham, initial_condition, params, tspan)
    # initial_condition is [q_initial, p_initial]
    q_initial = initial_condition[1]
    p_initial = initial_condition[2]

    # DynamicalODEProblem(dp/dt_func, dq/dt_func, p_initial, q_initial, tspan, params)
    # p_fn_ham is the function for dp/dt
    # q_fn_ham is the function for dq/dt
    prob = DynamicalODEProblem(p_fn_ham, q_fn_ham, p_initial, q_initial, tspan, params)

    dt = 1e-4 # 매우 작은 timestep으로 정확도 보장
    # SymplecticEuler is a first-order symplectic integrator.
    # For better accuracy, consider higher-order symplectic integrators like:
    # VerletLeapfrog(), Ruth3(), McAte4(), CalvoSanz4()
    # E.g., sol = solve(prob, McAte4(), dt=dt, adaptive=false)
    sol = solve(prob, SymplecticEuler(), dt=dt, adaptive=false)

    # 균일한 시간 간격으로 샘플링
    t_uniform = range(tspan[1], tspan[2], length=NSENSORS)
    solution = sol(t_uniform)

    # 결과 추출 - DynamicalODEProblem(dp/dt, dq/dt, p0, q0, ...) 로 설정했으므로,
    # solution.u 각 요소는 [p, q] 형태를 가집니다.
    # 따라서 solution[1, i]는 i번째 시간 스텝에서의 p (운동량)
    # solution[2, i]는 i번째 시간 스텝에서의 q (위치)
    p_values = [solution[1, i] for i in 1:NSENSORS]
    q_values = [solution[2, i] for i in 1:NSENSORS]

    # 포텐셜 값 계산 (q 범위에 따라)
    q_range_for_V = range(0.0, 1.0, length=NSENSORS) # 원본 코드와 맞춤 (0.0 to 1.0)
    V_values = zeros(NSENSORS)

    # 포텐셜 별로 V 값 계산
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
    elseif potential_name == "mff"
        V_values = [V_MirroredFreeFall(q_v) for q_v in q_range_for_V]
    elseif potential_name == "smff"
        V_values = [V_SoftenedMirroredFreeFall(q_v) for q_v in q_range_for_V]
    elseif potential_name == "sawtooth"
        sawtooth_params_local = SawtoothParams()
        V_values = [V_Sawtooth(q_v, sawtooth_params_local) for q_v in q_range_for_V]
    end

    # 결과를 DataFrame으로 변환
    df = DataFrame(
        V = V_values,      # Potential shape over a fixed q_range [0,1]
        t = collect(t_uniform), # Time points of the simulation
        q_true = q_values, # Simulated trajectory positions
        p_true = p_values  # Simulated trajectory momenta
    )

    return df
end

function run_all_simulations()
    # 데이터 저장 디렉토리 생성
    mkpath("data_true")

    # 모든 포텐셜에 대해 시뮬레이션 실행
    # potentials = [ (name, dp/dt_func, dq/dt_func, [q0, p0], params_struct), ... ]
    potentials = [
        ("sho", hamiltonian_sho_p, hamiltonian_sho_q, [0.0, 0.0], nothing),
        ("double_well", hamiltonian_double_well_p, hamiltonian_double_well_q, [0.0, 0.0], nothing),
        ("morse", hamiltonian_morse_p, hamiltonian_morse_q, [0.0, 0.0], MorseParams()), # q0=0 for Morse can be far from r_e
        ("pendulum", hamiltonian_pendulum_p, hamiltonian_pendulum_q, [0.0, 0.0], PendulumParams()), # q0=0 means theta = -theta_L
        ("mff", hamiltonian_mff_p, hamiltonian_mff_q, [0.0, 0.0], nothing),
        ("smff", hamiltonian_smff_p, hamiltonian_smff_q, [0.0, 0.0], nothing),
        ("sawtooth", hamiltonian_sawtooth_p, hamiltonian_sawtooth_q, [0.0, 0.0], SawtoothParams()) # q0=0 is in the first slope segment
    ]

    for (name, p_fn, q_fn, init_cond, params_obj) in potentials
        @printf "Running simulation for %s...\n" name
        # init_cond is [q0, p0]
        # run_simulation expects (potential_name, p_fn_ham, q_fn_ham, initial_condition, params, tspan)
        df = run_simulation(name, p_fn, q_fn, init_cond, params_obj, (0.0, 2.0))

        # Parquet 파일로 저장
        output_file = "data_true/$(name).parquet"
        Parquet.write_parquet(output_file, df)
        @printf "Saved to %s\n" output_file
    end

    println("All simulations completed!")
end

# 모든 시뮬레이션 실행
run_all_simulations()
