import numpy as np

from qiskit_finance.circuit.library.probability_distributions.normal import NormalDistribution 
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import LinearAmplitudeFunction
# from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeManila 
from qiskit.primitives import Sampler

DEVICE = FakeManila()
NOISE_MODEL = NoiseModel.from_backend(DEVICE)
COUPLING_MAP = DEVICE.configuration().coupling_map


NOISE = True 
OBS_NOISE = 0.05**2

# TODO: Get real parameter ranges
# TODO: Pull strategy returns from an actual database (or calculate them)
# TODO: Wrap the reward function in an actual BO loop.

# I'm not sure what reasonable parameter ranges are here.
RANGES = {
    "ma_window": (5, 100),     # Moving average window size (days)
    "rsi_period": (5, 30),     # RSI period (days)
    "boll_width": (1.0, 3.0),  # Bollinger band width (multiplier)
    "vwap_window": (5, 100),   # VWAP window size (days)
}

# Placeholder for actual strategy returns
def random_returns():
    # This function returns a normal distribution of daily returns
    return np.random.normal(0, 0.01, 252) 

def rescale(param, name):
    low, high = RANGES[name]
    return param * (high - low) + low

def simulate_reward(param, eps):

    # rescale parameters
    ma = rescale(param[0], "ma_window")
    rsi = rescale(param[1], "rsi_period")
    boll = rescale(param[2], "boll_width")
    vwap = rescale(param[3], "vwap_window")

    # TODO: Pull strategy returns from an actual database (or calculate them)
    # I'm assuming the returns are daily
    # returns = get_returns(ma, rsi, boll, vwap)
    # Calculate sharpe ratio
    returns = random_returns()
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)
    sharpe = (mean_returns / (std_returns + 1e-10)) * np.sqrt(252)

    # creating an uncertainty model

    num_uncertainty_qubits = 6
    mu = sharpe
    sigma = np.sqrt(OBS_NOISE)

    low = mu - 3 * sigma
    high = mu + 3 * sigma

    uncertainty_model = NormalDistribution(
                            num_uncertainty_qubits,
                            mu=mu,
                            sigma=sigma,
                            bounds=(low, high)
                            )        

    c_approx = 1
    slopes = [1]
    offsets = [0]

    # The LinearAmplitudeFunction is a piecewise linear function
    linear_payoff = LinearAmplitudeFunction(
        num_uncertainty_qubits,
        slopes,
        offsets,
        domain=(low, high),
        image=(low, high),
        rescaling_factor=c_approx,
    )

    # construct an operator composing the uncertainty 
    # model and the objective function
    num_qubits = linear_payoff.num_qubits
    monte_carlo = QuantumCircuit(num_qubits)
    monte_carlo.append(uncertainty_model, range(num_uncertainty_qubits))
    monte_carlo.append(linear_payoff, range(num_qubits))

    basis_gates = DEVICE.configuration().basis_gates
    monte_carlo = transpile(monte_carlo, basis_gates=basis_gates)

    alpha = 0.05
    epsilon = np.clip(eps, 1e-6, 0.5)

    objective_qubits = [linear_payoff.num_qubits - 1]
    seed = 0

    max_shots = 32 * np.log(2/alpha*np.log2(np.pi/(4*epsilon))) 
    
    # construct amplitude estimation
    if NOISE:
        sampler = Sampler(options={"backend": AerSimulator(method='density_matrix',
                               shots=int(np.ceil(max_shots)),
                               coupling_map=COUPLING_MAP,
                               noise_model=NOISE_MODEL)})
    else:
        # Use a noiseless simulator
        options = {
            'method': 'statevector',
            'shots': int(np.ceil(max_shots))
        }
        sampler= Sampler(options={"backend":AerSimulator(backend_options=options)})


    # Create the estimation problem
    problem = EstimationProblem(state_preparation=monte_carlo,
                                objective_qubits=objective_qubits,
                                post_processing=linear_payoff.post_processing
                                )
    
    ae = IterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=alpha, sampler=sampler)

    # Running result
    result = ae.estimate(problem)
    est = result.estimation_processed
    
    # use the number of oracle calls given by the paper if we dont have a result
    num_oracle_queries = result.num_oracle_queries or (int(
        np.ceil((0.8 / epsilon) * np.log((2 / alpha) * np.log2(np.pi / (4 * epsilon))))
    ))

    return est, mu, num_oracle_queries
if __name__ == "__main__":
    print(simulate_reward([0.5, 0.5, 0.5, 0.5], 0.1))