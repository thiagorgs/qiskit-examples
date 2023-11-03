from vqe_functions import*
import json
from scipy.interpolate import interp2d,CubicSpline,lagrange

from qiskit_aer.noise import NoiseModel
from qiskit.transpiler import CouplingMap
from qiskit.utils import algorithm_globals
from qiskit.providers.fake_provider import FakeManila,FakeQuito,FakeLima,FakeKolkata
from qiskit.providers.fake_provider import FakePerth,FakeLagos,FakeNairobi,FakeJakarta
from dataclasses import dataclass
from qiskit.compiler import transpile

path_cluster = '/home/alanduriez/qiskit/results/'
path_local = '/home/alan/Desktop/Projects/Python/qu_engine_std/Ising Simulations/results/'
path = path_local

# Backend setting
shots = 2**14
backend = service.backend("ibmq_qasm_simulator")

#####################################################
# EXACT OPTIONS
options_exct = Options()
options_exct.execution.shots = shots
options_exct.optimization_level = 3

# QUANTUM BACKEND OPTIONS
options_nairobi = Options()
options_nairobi.execution.shots = 2**14
options_nairobi.optimization_level = 3
options_nairobi.resilience_level = 1

# FAKE NAIROBI

fake_nairobi = FakeKolkata()
noise_nairobi = NoiseModel.from_backend(fake_nairobi)
options_noisy_nairobi = Options()
options_noisy_nairobi.execution.shots = shots
options_noisy_nairobi.simulator = {
    "noise_model": noise_nairobi,
    "basis_gates": fake_nairobi.configuration().basis_gates,
    "coupling_map": fake_nairobi.configuration().coupling_map,
    "seed_simulator": 42
}

options_noisy_nairobi.optimization_level = 0 # no optimization
options_noisy_nairobi.resilience_level = 0 # M3 for Sampler and T-REx for Estimator


# MITIGATED NAIROBI

options_mitigated_nairobi = Options()
options_mitigated_nairobi.execution.shots = shots
options_mitigated_nairobi.simulator = {
    "noise_model": noise_nairobi,
    "basis_gates": fake_nairobi.configuration().basis_gates,
    "coupling_map": fake_nairobi.configuration().coupling_map
}

options_mitigated_nairobi.optimization_level = 3
options_mitigated_nairobi.resilience_level = 1

#####################################################
# Setting lattice and model
n_qubits = 4
graph = get_line_graph(n_qubits)
model = 'ap'

#values of g
g_exct_values = np.linspace(0.,1.5,100) # exact
g_values = np.linspace(0.,1.6,30) # Simulation

#exact result
if n_qubits == 12:
    e_dat = np.loadtxt(''+path+'E0-12.dat')
    kk_dat = np.loadtxt(''+path+'kk-12.dat')
    g_exct_values = e_dat[:,0]
    exct_E = e_dat[:,1]
    exct_kk = kk_dat[:,1]

else:
    exct_E,exct_m,exct_kk = get_exact_values(graph=graph,g_values=g_exct_values,model=model)

# Optimization settings 
layers = 1
ansatz_str = 'hea'

if ansatz_str=='hea':
    num_param = 2*n_qubits*layers
if ansatz_str=='hva':
    num_param = 3*layers
initial_point = np.ones(num_param)

cobyla = COBYLA(maxiter=2,rhobeg=0.1)
spsa = SPSA(maxiter=20,trust_region=True,learning_rate=0.07,perturbation=0.1)
spsa_init = SPSA(maxiter=1,trust_region=True,learning_rate=0.07,perturbation=0.1)
slsqp = SLSQP(350)
slsqp_init = SLSQP(300)
bfgs = L_BFGS_B(2)

optimizer = spsa
initial_optimizer = spsa_init


# Initial point resampling
resampling = True
reps = 5

# Server
server = 'qasm'

# ZNE
zne = False
extrap = 'lin'

# Runtime options
options = options_exct

#loading angles
with open(path+f'angles_{len(graph.nodes)}qubits_{ansatz_str}{layers}_zne_{zne}_{extrap}_{server}.json') as json_file:
    angles_old = json.load(json_file)


E_new,angles_new = vqe_recycling(graph=graph,
                                            ansatz_str=ansatz_str,
                                            layers=layers,
                                            options=options_mitigated_nairobi,
                                            optimizer=optimizer,
                                            angles_opt=angles_old,
                                            service=service,
                                            backend=backend,
                                            server=server,
                                            device_aer=FakeKolkata(),
                                            zne=zne,
                                            shots=shots,
                                            extrap=extrap,
                                            resampling=resampling,
                                            reps=reps,
                                            path=path)

fontsize = 12
f,ax = plt.subplots()
plt.plot(g_exct_values,exct_E,label='exact')
plt.plot(g_values[::-1],E_new,'o',label='noisy')
plt.xlabel('g',fontsize=fontsize)
plt.ylabel('E',fontsize=fontsize)
plt.legend()
plt.savefig(path+f'graphErec_{n_qubits}q_{ansatz_str}{layers}_zne_{zne}_{extrap}_{server}.pdf',format='pdf')

# seed = 170
# device_aer = FakeKolkata()
# cost_operator = get_h_op(graph,ap=1.5)
# coupling_map = device_aer.configuration().coupling_map
# noise_model = NoiseModel.from_backend(device_aer)
# theta_list = ParameterVector('theta',2*n_qubits*layers)
# ansatz_k1 = get_ansatz_hea_ibm(graph,theta_list)
# ansatz_k2 = get_ansatz_hea_ibm_ZNE(graph,theta_list)

# estimator = get_estimator(server='qasm',device_aer=FakeKolkata())

# # def cost_function_vqe(theta):
# #     job = estimator.run(ansatz_k1,ansatz_k2,cost_operator,theta)
# #     value_k1 = job.result().values[0]
# #     #value_k2 = job.result().values[1]
# #     # job_k2 = estimator.run(ansatz_k2, cost_operator, theta)
# #     # value_k2 = job_k2.result().values[0]
# #     return value_k1

# def cost_function_vqe(theta):
#     job = estimator.run(ansatz_k1, cost_operator, theta)
#     job_2 = estimator.run(ansatz_k2, cost_operator, theta)
#     values = job.result().values[0]
#     return values

# spsa = SPSA(maxiter=3,trust_region=True,learning_rate=0.07,perturbation=0.1)

# result = spsa.minimize(cost_function_vqe,initial_point).fun

# print(result)