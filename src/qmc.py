from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import NormalDistribution
from qiskit import QuantumCircuit
from qiskit.algorithms import EstimationProblem, IterativeAmplitudeEstimation
from qiskit_aer.primitives import Sampler
import numpy as np

class QMC:
    def __init__(self, num_uncertainty, means, obs_noise=0.0025):
        
        self.uncertainty_bits = num_uncertainty
        self.means = means
        self.variance = obs_noise
        self.std = np.sqrt(obs_noise)
        
        self.domain = (self.means - 3. * self.std, self.means + 3. * self.std)
        
        self.uncertainty_model = NormalDistribution(
            num_qubits=self.uncertainty_bits,
            mu=self.means,
            sigma=self.variance,
            bounds=self.domain
        )
        
        self.linear_ampf = LinearAmplitudeFunction(
            num_state_qubits=self.uncertainty_bits,
            slope=[1],
            offset=[0],
            domain=self.domain,
            image=self.domain
        )
        
        self.mc_circuit = QuantumCircuit(self.linear_ampf.num_qubits)
        self.mc_circuit.append(self.uncertainty_model, range(self.uncertainty_bits))
        self.mc_circuit.append(self.linear_ampf, range(self.linear_ampf.num_qubits))
        
        self.workflow = EstimationProblem(
            state_preparation=self.mc_circuit,
            objective_qubits=[0],
            post_processing=self.linear_ampf.post_processing
        )
    
    def estimate(self, eps, alpha=0.05):
        # TODO: add noise conditional ae
        epsilon = np.clip(eps / (3 * self.std), 1e-6, 0.5) # rescale
        shots = int(np.ceil(32 * np.log(2 / alpha * np.log2(np.pi / (4 * epsilon)))))
        estimator = IterativeAmplitudeEstimation(
            epsilon_target=epsilon,
            alpha=alpha,
            sampler=Sampler(run_options={"shots": shots, "seed_simulator": 0})
        )
        
        result = estimator.estimate(self.workflow)
        queries = result.num_oracle_queries
        if queries == 0:
            queries = int(np.ceil((0.8 / epsilon) * np.log((2. / alpha) * np.log2(np.pi / (4. * epsilon)))))
        
        return result, queries