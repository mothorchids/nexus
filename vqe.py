import os
from datetime import datetime
from dotenv import load_dotenv


from numpy import ndarray
from numpy.random import random_sample
from pytket import Circuit
from pytket.backends.backendresult import BackendResult
from pytket.circuit import Qubit
from pytket.circuit.display import render_circuit_jupyter
from pytket.partition import (
    MeasurementBitMap,
    MeasurementSetup,
    PauliPartitionStrat,
    measurement_reduction,
)
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils.operators import QubitPauliOperator
from scipy.optimize import minimize
from sympy import Symbol

import qnexus as qnx
from qnexus.client.auth import login_no_interaction

# ------------------------------------------------------------
# 1) Login to Nexus
# ------------------------------------------------------------
load_dotenv()
username = os.getenv("HQS_USERNAME")
password = os.getenv("HQS_PASSWORD")

if username is None or password is None:
    raise RuntimeError("HQS_USERNAME or HQS_PASSWORD missing in .env file.")

login_no_interaction(user=username, pwd=password)

# 1. Synthesise Symbolic State-Preparation Circuit (hardware efficient ansatz)

symbols = [Symbol(f"p{i}") for i in range(4)]
symbolic_circuit = Circuit(2)
symbolic_circuit.X(0)
symbolic_circuit.Ry(symbols[0], 0).Ry(symbols[1], 1)
symbolic_circuit.CX(0, 1)
symbolic_circuit.Ry(symbols[2], 0).Ry(symbols[3], 0)

print(symbolic_circuit)
# render_circuit_jupyter(symbolic_circuit)

# 2. Define Hamiltonian
# coefficients in the Hamiltonian are obtained from PhysRevX.6.031007

coeffs = [-0.4804, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910]
term0 = {
    QubitPauliString(
        {
            Qubit(0): Pauli.I,
            Qubit(1): Pauli.I,
        }
    ): coeffs[0]
}
term1 = {QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.I}): coeffs[1]}
term2 = {QubitPauliString({Qubit(0): Pauli.I, Qubit(1): Pauli.Z}): coeffs[2]}
term3 = {QubitPauliString({Qubit(0): Pauli.Z, Qubit(1): Pauli.Z}): coeffs[3]}
term4 = {QubitPauliString({Qubit(0): Pauli.X, Qubit(1): Pauli.X}): coeffs[4]}
term5 = {QubitPauliString({Qubit(0): Pauli.Y, Qubit(1): Pauli.Y}): coeffs[5]}
term_sum = {}
term_sum.update(term0)
term_sum.update(term1)
term_sum.update(term2)
term_sum.update(term3)
term_sum.update(term4)
term_sum.update(term5)
hamiltonian = QubitPauliOperator(term_sum)

# 3 Computing Expectation Values

# Computing Expectation Values for Pauli-Strings
def compute_expectation_paulistring(
    distribution: dict[tuple[int, ...], float], bitmap: MeasurementBitMap
) -> float:
    value = 0
    for bitstring, probability in distribution.items():
        value += probability * (sum(bitstring[i] for i in bitmap.bits) % 2)
    return ((-1) ** bitmap.invert) * (-2 * value + 1)

# 3.2 Computing Expectation Values for sums of Pauli-strings multiplied by coefficients
def compute_expectation_value(
    results: list[BackendResult],
    measurement_setup: MeasurementSetup,
    operator: QubitPauliOperator,
) -> float:
    energy = 0
    for pauli_string, bitmaps in measurement_setup.results.items():
        string_coeff = operator.get(pauli_string, 0.0)
        if string_coeff > 0:
            for bm in bitmaps:
                index = bm.circ_index
                distribution = results[index].get_distribution()
                value = compute_expectation_paulistring(distribution, bm)
                energy += complex(value * string_coeff).real
    return energy

# 4. Building our Objective function


class Objective:
    def __init__(
        self,
        symbolic_circuit: qnx.circuits.CircuitRef,
        problem_hamiltonian: QubitPauliOperator,
        n_shots_per_circuit: int,
        target: qnx.BackendConfig,
        iteration_number: int = 0,
        n_iterations: int = 10,
    ) -> None:
        """Returns the objective function needed for a variational
        procedure.
        """
        terms = [term for term in problem_hamiltonian._dict.keys()]
        self._symbolic_circuit: Circuit = symbolic_circuit.download_circuit()
        self._hamiltonian: QubitPauliOperator = problem_hamiltonian
        self._nshots: int = n_shots_per_circuit
        self._measurement_setup: MeasurementSetup = measurement_reduction(
            terms, strat=PauliPartitionStrat.CommutingSets
        )
        self._iteration_number: int = iteration_number
        self._niters: int = n_iterations
        self._target = target

    def __call__(self, parameter: ndarray) -> float:
        value = self._objective_function(parameter)
        self._iteration_number += 1
        if self._iteration_number >= self._niters:
            self._iteration_number = 0
        return value

    def _objective_function(
        self,
        parameters: ndarray,
    ) -> float:
        # Prepare the parameterised state preparation circuit
        assert len(parameters) == len(self._symbolic_circuit.free_symbols())
        symbol_dict = {
            s: p for s, p in zip(self._symbolic_circuit.free_symbols(), parameters)
        }
        state_prep_circuit = self._symbolic_circuit.copy()
        state_prep_circuit.symbol_substitution(symbol_dict)

        # Label each job with the properties associated with the circuit.
        properties = {str(sym): val for sym, val in symbol_dict.items()} | {
            "iteration": self._iteration_number
        }

        with qnx.context.using_properties(**properties):
            circuit_list = self._build_circuits(state_prep_circuit)

            # Execute circuits with Nexus
            results = qnx.execute(
                name=f"execute_job_VQE_{self._iteration_number}",
                programs=circuit_list,
                n_shots=[self._nshots] * len(circuit_list),
                backend_config=self._target,
                timeout=None,
            )

        expval = compute_expectation_value(
            results, self._measurement_setup, self._hamiltonian
        )
        return expval

    def _build_circuits(
        self, state_prep_circuit: Circuit
    ) -> list[qnx.circuits.CircuitRef]:
        # Requires properties to be set in the context

        # Upload the numerical state-prep circuit to Nexus
        qnx.circuits.upload(
            circuit=state_prep_circuit,
            name=f"state prep circuit {self._iteration_number}",
        )
        circuit_list = []
        for mc in self._measurement_setup.measurement_circs:
            c = state_prep_circuit.copy()
            c.append(mc)
            # Upload each measurement circuit to Nexus with correct params
            measurement_circuit_ref = qnx.circuits.upload(
                circuit=c,
                name=f"state prep circuit {self._iteration_number}",
            )
            circuit_list.append(measurement_circuit_ref)

        # Compile circuits with Nexus
        compiled_circuit_refs = qnx.compile(
            name=f"compile_job_VQE_{self._iteration_number}",
            programs=circuit_list,
            optimisation_level=2,
            backend_config=self._target,
            timeout=None,
        )

        return compiled_circuit_refs
    

# set up the project
project_ref = qnx.projects.create(
    name=f"VQE_example_{str(datetime.now())}",
    description="A VQE done with qnexus",
)

# set this in the context
qnx.context.set_active_project(project_ref)

# properties -> parameters
qnx.projects.add_property(
    name="iteration",
    property_type="int",
    description="The iteration number in my dihydrogen VQE experiment",
)

# Set up the properties for the symbolic circuit parameters
for sym in symbolic_circuit.free_symbols():
    qnx.projects.add_property(
        name=str(sym),
        property_type="float",
        description=f"Our VQE {str(sym)} parameter",
    )

# Upload our ansatz circuit

ansatz_ref = qnx.circuits.upload(
    circuit=symbolic_circuit,
    name="ansatz_circuit",
    description="The ansatz state-prep circuit for my dihydrogen VQE",
)

objective = Objective(
    symbolic_circuit=ansatz_ref,
    problem_hamiltonian=hamiltonian,
    n_shots_per_circuit=500,
    n_iterations=4,
    target=qnx.QuantinuumConfig(device_name="H1-1LE"),
)

# VQE Loop
initial_parameters = random_sample(len(symbolic_circuit.free_symbols()))

result = minimize(
    objective,
    initial_parameters,
    method="COBYLA",
    options={"disp": True, "maxiter": objective._niters},
    tol=1e-2,
)

print(result.fun)
print(result.x)

qnx.logout()