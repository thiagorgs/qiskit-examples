import numpy as np
import networkx as nx
from tqdm import tqdm
import rustworkx as rw
import matplotlib.pyplot as plt
import json
from qiskit.providers.fake_provider import FakeManila,FakeQuito,FakeLima,FakeKolkata,FakeNairobi
from qiskit.transpiler import CouplingMap

from qiskit.circuit import QuantumCircuit,ParameterVector,Parameter
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
#from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator,Sampler
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP,COBYLA,L_BFGS_B,QNSPSA,SPSA
from qiskit_aer.noise import NoiseModel

from qiskit_ibm_runtime import Session,Options,QiskitRuntimeService
from qiskit_ibm_runtime import Estimator as IBM_Estimator
from qiskit_ibm_runtime import Sampler as IBM_Sampler

from qiskit_aer.primitives import Estimator as AerEstimator

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='bbe37df5115273fe795d0cab351b2189aa8acd4889b464ccf2c2c8a22223b05a910d88d7294fd067c52e9997b898b88720022b14fdf7efe9ddc8d7c3afaf9ea7'
)


########################################################
# MISC
########################################################
def point_corrector(e_array):
    E_c = [e_array[2*i] for i in range(len(e_array)//2+1)]
    return E_c

def compute_deviation(aprox_E,exact_E):
    num_points = len(aprox_E)
    sum = 0.
    for i in range(num_points):
        sum += (aprox_E[i] - exact_E[i])**2
    dev = np.sqrt(sum)/num_points
    return dev

def get_deviations_layers(E_exct,E_list):
    """Returns the deviation with the 
    number of layers for a given exact list and
    a list of lists of energies"""
    devs = []
    for E in E_list:
        dev = compute_deviation(E,E_exct)
        devs.append(dev)
    return devs


########################################################
# GRAPHS
########################################################
def get_line_graph(n_qubits):
    graph_line = nx.Graph()
    graph_line.add_nodes_from(range(n_qubits))

    edge_list = []
    for i in graph_line.nodes:
        if i < n_qubits-1:
            edge_list.append((i,i+1))

    # Generate graph from the list of edges
    graph_line.add_edges_from(edge_list)
    return graph_line

def get_chain_graph(n_qubits):
    graph_chain = nx.Graph()
    graph_chain.add_nodes_from(range(n_qubits))

    edge_list = []
    for i in range(n_qubits):
        if i < n_qubits-1:
            edge_list.append((i,i+1))
        else:
            edge_list.append((i,0))

    # Generate graph from the list of edges
    graph_chain.add_edges_from(edge_list)
    return graph_chain

def get_btree_graph(floors):
    graph_bt_3 = nx.Graph()
    graph_bt_3.add_nodes_from([0, 1, 2])
    graph_bt_3.add_edges_from([(0, 1), (0, 2)])

    graph_bt_7 = nx.Graph()
    graph_bt_7.add_nodes_from([0, 2, 2, 3, 4, 5, 6])
    graph_bt_7.add_edges_from([(0, 1), (0, 2),(1,3),(1,4),(2,5),(2,6)])

    graph_bt_15 = nx.Graph()
    graph_bt_15.add_nodes_from(range(2**3-1))
    graph_bt_15.add_edges_from([(0, 1), (0, 2),(1,3),(1,4),(2,5),(2,6),(3,7),(3,8),(4,9),(4,10),(5,11),(5,12),(6,13),(6,14)])

    graph = nx.Graph()
    if floors == 2:
        graph = graph_bt_3
    elif floors == 3:
        graph = graph_bt_7
    elif floors == 4:
        graph = graph_bt_15
    return graph


########################################################
# ANSATZE
########################################################
def get_ansatz_transverse(graph, theta_list):
    """Creates a parametrized qaoa circuit to measure <ZiZj>
       Measures in the computational basis
    Args:
        graph: networkx graph
        theta: (list) [beta,gamma]
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//2 

    qc = QuantumCircuit(nqubits)

    # initial_state
    qc.h(range(nqubits))
    qc.barrier()

    for layer_index in range(n_layers):

        # problem unitary
        for pair in list(graph.edges()):
            qc.rzz(2 * theta_list[2*layer_index],pair[0],pair[1])
        qc.barrier()
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * theta_list[2*layer_index+1], qubit)
    return qc

def get_ansatz_uniform(graph, theta_list):
    """Creates a parametrized qaoa circuit to measure <ZiZj>
       Measures in the computational basis
    Args:
        graph: networkx graph
        theta: (list) [beta,gamma]
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//3 

    qc = QuantumCircuit(nqubits)

    # initial_state
    qc.h(range(nqubits))
    qc.barrier()

    for layer_index in range(n_layers):

        # problem unitary
        for pair in list(graph.edges()):
            qc.h(pair[1])
            qc.cz( pair[0], pair[1])
            qc.rx(2 * theta_list[3*layer_index],pair[1])
            qc.cz( pair[0], pair[1])
            qc.h(pair[1])
        qc.barrier()
        # layer of z-rotations
        for qubit in range(nqubits):
            qc.rz(2 * theta_list[3*layer_index+2], qubit) 
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * theta_list[3*layer_index+1], qubit)
    return qc

def get_ansatz_antiparallel(graph, theta_list):
    """Creates a parametrized qaoa circuit for the antiparallel
    model
    Args:
        graph: networkx graph
        theta: Parameter class list in the form
          [gamma_i,beta_i,alpha_i]
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//3 

    qc = QuantumCircuit(nqubits)
    
    even_edges = [edge for edge in graph.edges() if edge[0]%2==0]
    odd_edges = [edge for edge in graph.edges() if edge[0]%2!=0]

    

    # initial_state
    qc.h(range(nqubits))
    qc.barrier()

    for layer_index in range(n_layers):

        # problem unitary
        for pair in even_edges:
            qc.h(pair[1])
            qc.cz( pair[0], pair[1])
            qc.rx(2 * theta_list[3*layer_index],pair[1])
            qc.cz( pair[0], pair[1])
            qc.h(pair[1])
        for pair in odd_edges:
            qc.h(pair[1])
            qc.cz( pair[0], pair[1])
            qc.rx(2 * theta_list[3*layer_index],pair[1])
            qc.cz( pair[0], pair[1])
            qc.h(pair[1])
        # anti-paralel in the border of the line
        for qubit in graph.nodes:
            if qubit == 0:
                qc.rz(2 *theta_list[3*layer_index+2], qubit)
            if qubit == nqubits-1:
                qc.rz(-2 * theta_list[3*layer_index+2], qubit) 
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * theta_list[3*layer_index+1], qubit)
        if layer_index != n_layers-1:
            qc.barrier()
    return qc

def get_ansatz_antip_ibm(graph, theta_list):
    """Creates a parametrized qaoa circuit for the antiparallel
    model using the basis gates of IBM manilla
    Args:
        graph: networkx graph
        theta: Parameter class list in the form
          [gamma_i,beta_i,alpha_i]
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//3 

    qc = QuantumCircuit(nqubits)
    
    even_edges = [edge for edge in graph.edges() if edge[0]%2==0]
    odd_edges = [edge for edge in graph.edges() if edge[0]%2!=0]

    

    # initial_state
    qc.rz(np.pi/2,range(nqubits))
    qc.sx(range(nqubits))
    qc.rz(np.pi/2,range(nqubits))

    for layer_index in range(n_layers):

        # problem unitary
        for pair in even_edges:
            qc.cnot(pair[0],pair[1])
            qc.rz(2 * theta_list[3*layer_index],pair[1])
            qc.cnot(pair[0],pair[1])
        for pair in odd_edges:
            qc.cnot(pair[0],pair[1])
            qc.rz(2 * theta_list[3*layer_index],pair[1])
            qc.cnot(pair[0],pair[1])
        # anti-paralel in the border of the line
        for qubit in graph.nodes:
            if qubit == 0:
                qc.rz(2 *theta_list[3*layer_index+2], qubit)
            if qubit == nqubits-1:
                qc.rz(-2 * theta_list[3*layer_index+2], qubit) 
        # mixer unitary
        for qubit in range(nqubits):
            qc.rz(np.pi/2, qubit)
            qc.sx(qubit)
            qc.rz(2 * theta_list[3*layer_index+1]+np.pi, qubit)
            qc.sx(qubit)
            qc.rz(5*np.pi/2, qubit)

    return qc


def get_ansatz_antiparallel_alt(graph, theta_list):
    """Creates a parametrized qaoa circuit for the antiparallel
    model but using only two parameters
    Args:
        graph: networkx graph
        theta: Parameter class list in the form
          [gamma_i,beta_i]
    Returns:
        (QuantumCircuit) qiskit circuit
    """
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//2 

    qc = QuantumCircuit(nqubits)
    
    even_edges = [edge for edge in graph.edges() if edge[0]%2==0]
    odd_edges = [edge for edge in graph.edges() if edge[0]%2!=0]

    

    # initial_state
    qc.h(range(nqubits))
    qc.barrier()

    for layer_index in range(n_layers):

        # problem unitary
        for pair in even_edges:
            qc.h(pair[1])
            qc.cz( pair[0], pair[1])
            qc.rx(2 * theta_list[2*layer_index],pair[1])
            qc.cz( pair[0], pair[1])
            qc.h(pair[1])
        for pair in odd_edges:
            qc.h(pair[1])
            qc.cz( pair[0], pair[1])
            qc.rx(2 * theta_list[2*layer_index],pair[1])
            qc.cz( pair[0], pair[1])
            qc.h(pair[1])
        # anti-paralel in the border of the line
        for qubit in graph.nodes:
            if qubit == 0:
                qc.rz(2 *theta_list[2*layer_index], qubit)
            if qubit == nqubits-1:
                qc.rz(-2 * theta_list[2*layer_index], qubit) 
        # mixer unitary
        for qubit in range(nqubits):
            qc.rx(2 * theta_list[2*layer_index+1], qubit)
        if layer_index != n_layers-1:
            qc.barrier()
    return qc

def get_ansatz_hea_ibm(graph,theta_list):
    """Creates a parametrized qaoa circuit for the antiparallel
model using the basis gates of IBM manilla
Args:'
    graph: networkx graph
    theta: Parameter class list in the form
        [gamma_i,beta_i,alpha_i]
Returns:
    (QuantumCircuit) qiskit circuit
"""
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//(2*nqubits) 
    
    qc = QuantumCircuit(nqubits)
    
    even_edges = [edge for edge in graph.edges() if edge[0]%2==0]
    odd_edges = [edge for edge in graph.edges() if edge[0]%2!=0]
    reversed_edges = [edge for edge in graph.edges()][::-1]

    for layer_index in range(n_layers):
        for qubit in range(nqubits):
            qc.ry(theta_list[2*(nqubits)*layer_index+qubit], qubit)
        # for pair in reversed_edges:
        #     qc.cnot(pair[0],pair[1])
        for pair in even_edges:
            qc.cnot(pair[0],pair[1])
        for pair in odd_edges:
            qc.cnot(pair[0],pair[1])
        for qubit in range(nqubits):
            qc.ry(theta_list[nqubits+2*(nqubits)*layer_index+qubit], qubit)

    return qc

def get_ansatz_hea_ibm_ZNE(graph,theta_list):
    """Creates a parametrized qaoa circuit for the antiparallel
model using the basis gates of IBM. Here the CNOTS are folded 
for the ZNE squeme
Args:
    graph: networkx graph
    theta: Parameter class list in the form
        [gamma_i,beta_i,alpha_i]
Returns:
    (QuantumCircuit) qiskit circuit
"""
    nqubits = len(graph.nodes())
    n_layers = len(theta_list)//(2*nqubits) 
    
    qc = QuantumCircuit(nqubits)
    
    even_edges = [edge for edge in graph.edges() if edge[0]%2==0]
    odd_edges = [edge for edge in graph.edges() if edge[0]%2!=0]
    reversed_edges = [edge for edge in graph.edges()][::-1]

    for layer_index in range(n_layers):
        for qubit in range(nqubits):
            qc.ry(theta_list[2*(nqubits)*layer_index+qubit], qubit)
        # for pair in reversed_edges:
        #     qc.cnot(pair[0],pair[1])
        #folding even edges
        for pair in even_edges:
            qc.cnot(pair[0],pair[1])
        qc.barrier()
        for pair in even_edges:
            qc.cnot(pair[0],pair[1])
        qc.barrier()
        for pair in even_edges:
            qc.cnot(pair[0],pair[1])
        qc.barrier()
        #folding odd edges
        for pair in odd_edges:
            qc.cnot(pair[0],pair[1])
        qc.barrier()
        for pair in odd_edges:
            qc.cnot(pair[0],pair[1])
        qc.barrier()
        for pair in odd_edges:
            qc.cnot(pair[0],pair[1])
        qc.barrier()
        for qubit in range(nqubits):
            qc.ry(theta_list[nqubits+2*(nqubits)*layer_index+qubit], qubit)

        
        # for qubit in range(nqubits):
        #     qc.rz(np.pi/2, qubit)
        #     qc.sx(qubit)
        #     qc.rz(2 * theta_list[3*layer_index+1]+np.pi, qubit)
        #     qc.sx(qubit)
        #     qc.rz(5*np.pi/2, qubit)

    return qc


########################################################
# OPERATORS
########################################################
def get_h_op(graph,J=1.,hx=0.5,hz=0.,ap=0.):
    
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        # X field
        coeff = ('X',[qubit],-1*hx)
        sparse_list.append(coeff)
        # Z field
        coeff = ('Z',[qubit],-1*hz)
        sparse_list.append(coeff)

    # Anti-paralel field at the borders
    coeff = ('Z',[0],ap) #this is the positive field (order reversed)
    sparse_list.append(coeff)
    coeff = ('Z',[num_qubits-1],-1*ap)
    sparse_list.append(coeff)

    #Interaction field (ZZ)
    for i,j in graph.edges():
        coeff = ('ZZ',[i,j],-1*J)
        sparse_list.append(coeff)
    
    hamiltonian = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits)
    return hamiltonian

def get_mag_op(graph):
    
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        coeff = ('Z',[qubit],1)
        sparse_list.append(coeff)

    mag = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits)
    return mag

def get_kk_op(graph):
    sparse_list = []
    for i,j in graph.edges():
        coeff = ('II',[i,j],0.5)
        sparse_list.append(coeff)       
        coeff = ('ZZ',[i,j],-0.5)
        sparse_list.append(coeff)
    
    kk_op = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=len(graph.nodes))
    return kk_op

def get_mag_op(graph):
    
    num_qubits = len(graph.nodes())
    sparse_list = []
    # Uniform Z and X fields
    for qubit in graph.nodes():
        coeff = ('Z',[qubit],1)
        sparse_list.append(coeff)

    mag = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=num_qubits)
    return mag

def get_kk_op(graph):
    sparse_list = []
    for i,j in graph.edges():
        coeff = ('II',[i,j],0.5)
        sparse_list.append(coeff)       
        coeff = ('ZZ',[i,j],-0.5)
        sparse_list.append(coeff)
    
    kk_op = SparsePauliOp.from_sparse_list(sparse_list,num_qubits=len(graph.nodes))
    return kk_op

########################################################
# TRAINING
########################################################
def get_result(graph,p=1,method='COBYLA',hx=0.5,hz=0.,ap=0.,
               model='tr',reps=1):
    n_qubits = len(graph.nodes())
    
    estimator = Estimator(options={'shots':1024,'seed':170})
    
    if model == 'tr':
        theta_list = ParameterVector('theta',length=2*p)
        ansatz = get_ansatz_transverse(graph,theta_list)
        cost_operator = get_h_op(graph,hx=hx)
    
    elif model == 'un':
        theta_list = ParameterVector('theta',length=3*p)
        ansatz = get_ansatz_uniform(graph,theta_list)
        cost_operator = get_h_op(graph,hx=hx,hz=hz)
    
    elif model == 'ap':
        theta_list = ParameterVector('theta',length=3*p)
        ansatz = get_ansatz_antiparallel(graph,theta_list)
        cost_operator = get_h_op(graph,hx=hx,ap=ap)

    
    mag_op = get_mag_op(graph)
    kk_op = get_kk_op(graph)
    aux_operators = [mag_op,kk_op]
    
    if method == 'EXACT':
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(operator=cost_operator,aux_operators=aux_operators)
    
    else: 
        if method == 'COBYLA':
            optimizer = COBYLA()

        elif method == 'L_BFGS_B':
            optimizer = L_BFGS_B()
        

        resampled_energy = 0.
        for i in range(reps):
            initial_point = np.random.random(ansatz.num_parameters)
            vqe = VQE(estimator=estimator,ansatz=ansatz,
                        optimizer=optimizer,initial_point=initial_point)
            sample_result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                        aux_operators=aux_operators)
            eigenvalue = sample_result.eigenvalue
            
            if eigenvalue < resampled_energy:
                resampled_energy = eigenvalue
                result = sample_result
        
    
    return result

def get_aprox_values(graph,p,g_values,hx=0.5,hz=0.,ap=0.,optimizer='COBYLA',model='tr',reps=1):
    """Gets the aprox values of the energy for 
    given values of g using QAOA"""
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    mag_values = np.array([0. for i in range(len(g_values))])
    kk_values = np.array([0. for i in range(len(g_values))])
    angles_dict = {}
    
    for i,g in enumerate(tqdm(g_values)):
        
        if model == 'tr':
            result = get_result(graph=graph,p=p,method=optimizer,hx=g,model=model,reps=reps)
    
        elif model == 'un':
            result = get_result(graph=graph,p=p,method=optimizer,hz=g,model=model,reps=reps)
        
        elif model == 'ap':
            result = get_result(graph=graph,p=p,method=optimizer,ap=g,model=model,reps=reps)
        
        #quantities evaluated
        E_values[i] = result.eigenvalue
        mag_values[i] = result.aux_operators_evaluated[0][0]
        kk_values[i] = np.real_if_close(result.aux_operators_evaluated[1][0])
        #optimal angles storage
        angles = list(result.optimal_point)
        angles_dict[str(round(g,14))] = angles


    return E_values,mag_values,kk_values,angles_dict

def get_exact_values(graph,g_values,hx=0.5,hz=0.,ap=0.,model='tr'):
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    mag_values = np.array([0. for i in range(len(g_values))])
    kk_values = np.array([0. for i in range(len(g_values))])

    for i,g in enumerate(tqdm(g_values)):
        
        if model == 'tr':
            result = get_result(graph=graph,method='EXACT',hx=g,model=model)
    
        elif model == 'un':
            result = get_result(graph=graph,method='EXACT',hz=g,model=model)
        
        elif model == 'ap':
            result = get_result(graph=graph,method='EXACT',ap=g,model=model)
        
        #quantities evaluated
        E_values[i] = result.eigenvalue
        mag_values[i] = result.aux_operators_evaluated[0][0]
        kk_values[i] = np.real(result.aux_operators_evaluated[1][0])

    return E_values,mag_values,kk_values


def get_vqe_values_ap_hea(graph,
                          g_values,
                          p=1,
                          hx=0.5,
                          optimizer=COBYLA(),
                          shots=20000,
                          reps=1,
                          init_tol=1e-2):
    """Runs the vqe to simulate the antiparallel model in
    the hardware efficient ansatz for different values of
    the antiparallel field"""
    
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    mag_values = np.array([0. for i in range(len(g_values))])
    kk_values = np.array([0. for i in range(len(g_values))])
    angles_dict = {}
    mag_op = get_mag_op(graph)
    kk_op = get_kk_op(graph)
    aux_operators = [mag_op,kk_op]
    
    rev_g_values = g_values[::-1]
    for i,g in enumerate(tqdm(rev_g_values)):
        estimator = Estimator(options={'shots':shots,'seed':170})
        ansatz_hea = EfficientSU2(n_qubits,su2_gates='ry',reps=p)
        cost_operator = get_h_op(graph,hx=hx,ap=g)
        
        if i == 0:
            resampled_energy = 0.
            for j in range(reps):
                initial_point = np.random.random(ansatz_hea.num_parameters)
                vqe = VQE(estimator=estimator,ansatz=ansatz_hea,
                            optimizer=COBYLA(tol=init_tol),initial_point=initial_point)
                sample_result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                            aux_operators=aux_operators)
                eigenvalue = sample_result.eigenvalue
                
                if eigenvalue < resampled_energy:
                    resampled_energy = eigenvalue
                    result = sample_result
            
            initial_point = result.optimal_point    

        else:
            vqe = VQE(estimator=estimator,ansatz=ansatz_hea,
                            optimizer=optimizer,initial_point=initial_point)
            result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                        aux_operators=aux_operators)
        
        E_values[i] = result.eigenvalue
        mag_values[i] = result.aux_operators_evaluated[0][0]
        kk_values[i] = np.real_if_close(result.aux_operators_evaluated[1][0])
        #optimal angles storage
        angles = list(result.optimal_point)
        angles_dict[str(round(g,14))] = angles


    return E_values,mag_values,kk_values,angles_dict

def get_vqe_values_ap_hva(graph,
                          g_values,
                          p=1,
                          hx=0.5,
                          optimizer=COBYLA(),
                          shots=20000,
                          reps=1,
                          init_iter=800):
    """Runs the vqe to simulate the antiparallel model in
    the hardware efficient ansatz for different values of
    the antiparallel field"""
    
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    mag_values = np.array([0. for i in range(len(g_values))])
    kk_values = np.array([0. for i in range(len(g_values))])
    angles_dict = {}
    mag_op = get_mag_op(graph)
    kk_op = get_kk_op(graph)
    aux_operators = [mag_op,kk_op]
    
    rev_g_values = g_values[::-1]
    for i,g in enumerate(tqdm(rev_g_values)):
        estimator = Estimator(options={'shots':shots,'seed':170})
        theta_list = ParameterVector('theta',length=3*p)
        ansatz_hva = get_ansatz_antiparallel(graph,theta_list)
        cost_operator = get_h_op(graph,hx=hx,ap=g)
        
        if i == 0:
            resampled_energy = 0.
            for j in range(reps):
                initial_point = np.random.random(ansatz_hva.num_parameters)
                vqe = VQE(estimator=estimator,ansatz=ansatz_hva,
                            optimizer=COBYLA(init_iter),initial_point=initial_point)
                sample_result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                            aux_operators=aux_operators)
                eigenvalue = sample_result.eigenvalue
                
                if eigenvalue < resampled_energy:
                    resampled_energy = eigenvalue
                    result = sample_result
            
            initial_point = result.optimal_point    

        else:
            vqe = VQE(estimator=estimator,ansatz=ansatz_hva,
                            optimizer=optimizer,initial_point=initial_point)
            result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                        aux_operators=aux_operators)
        
        E_values[i] = result.eigenvalue
        mag_values[i] = result.aux_operators_evaluated[0][0]
        kk_values[i] = np.real_if_close(result.aux_operators_evaluated[1][0])
        #optimal angles storage
        angles = list(result.optimal_point)
        angles_dict[str(round(g,14))] = angles


    return E_values,mag_values,kk_values,angles_dict

def get_vqe_values_ap_hva2(graph,
                          g_values,
                          p=1,
                          hx=0.5,
                          optimizer=COBYLA(),
                          shots=20000,
                          reps=1):
    """Runs the vqe to simulate the antiparallel model in
    the hardware efficient ansatz for different values of
    the antiparallel field (alternative ansatz with only
    two parameters per layer)"""
    
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    mag_values = np.array([0. for i in range(len(g_values))])
    kk_values = np.array([0. for i in range(len(g_values))])
    angles_dict = {}
    mag_op = get_mag_op(graph)
    kk_op = get_kk_op(graph)
    aux_operators = [mag_op,kk_op]
    
    rev_g_values = g_values[::-1]
    for i,g in enumerate(tqdm(rev_g_values)):
        estimator = Estimator(options={'shots':shots,'seed':170})
        theta_list = ParameterVector('theta',length=2*p)
        ansatz_hva = get_ansatz_antiparallel_alt(graph,theta_list)
        cost_operator = get_h_op(graph,hx=hx,ap=g)
        
        if i == 0:
            resampled_energy = 0.
            for j in range(reps):
                initial_point = np.random.random(ansatz_hva.num_parameters)
                vqe = VQE(estimator=estimator,ansatz=ansatz_hva,
                            optimizer=COBYLA(500),initial_point=initial_point)
                sample_result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                            aux_operators=aux_operators)
                eigenvalue = sample_result.eigenvalue
                
                if eigenvalue < resampled_energy:
                    resampled_energy = eigenvalue
                    result = sample_result
            
            initial_point = result.optimal_point    

        else:
            vqe = VQE(estimator=estimator,ansatz=ansatz_hva,
                            optimizer=optimizer,initial_point=initial_point)
            result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
                                                        aux_operators=aux_operators)
        
        E_values[i] = result.eigenvalue
        mag_values[i] = result.aux_operators_evaluated[0][0]
        kk_values[i] = np.real_if_close(result.aux_operators_evaluated[1][0])
        #optimal angles storage
        angles = list(result.optimal_point)
        angles_dict[str(round(g,14))] = angles


    return E_values,mag_values,kk_values,angles_dict

def get_vqe_values_ap_hea(graph,
                          optimizer,
                          init_optimizer,
                          g_values,
                          initial_point,
                          angles_dict={},
                          p=1,
                          hx=0.5,
                          shots=20000,
                          reps=1):
    """Runs the vqe to simulate the antiparallel model in
    the hardware efficient ansatz for different values of
    the antiparallel field"""
    
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    mag_values = np.array([0. for i in range(len(g_values))])
    kk_values = np.array([0. for i in range(len(g_values))])

    mag_op = get_mag_op(graph)
    kk_op = get_kk_op(graph)
    aux_operators = [mag_op,kk_op]
    
    estimator = Estimator(options={'shots':shots,'seed':170})
    theta_list = ParameterVector('θ',2*n_qubits*p)
    ansatz_hea = get_ansatz_hea_ibm(graph,theta_list)
    
    rev_g_values = g_values[::-1]
    
    for i,g in enumerate(tqdm(rev_g_values)):
        cost_operator = get_h_op(graph,hx=hx,ap=g)
        def cost_function_vqe(theta):
            job = estimator.run(ansatz_hea, cost_operator, theta)
            values = job.result().values
            return values[0]
        
        if i == 0:
            # vqe = VQE(estimator=estimator,
            #             ansatz=ansatz_hea,
            #             optimizer=init_optimizer,
            #             initial_point=initial_point)
            # result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
            #                                             aux_operators=aux_operators)
            result = init_optimizer.minimize(fun=cost_function_vqe,
              x0=initial_point)

            initial_point = result.x   

        else:
            # vqe = VQE(estimator=estimator,
            #           ansatz=ansatz_hea,
            #           optimizer=optimizer,
            #           initial_point=initial_point)
            # result = vqe.compute_minimum_eigenvalue(operator=cost_operator,
            #                                             aux_operators=aux_operators)
            result = optimizer.minimize(fun=cost_function_vqe,
              x0=initial_point)
            initial_point = result.x   
        
        E_values[i] = result.fun
        # mag_values[i] = result.aux_operators_evaluated[0][0]
        # kk_values[i] = np.real_if_close(result.aux_operators_evaluated[1][0])
        #optimal angles storage
        angles = list(result.x)
        angles_dict[str(round(g,14))] = angles


    return E_values,mag_values,kk_values,angles_dict

def get_extrapolation(value_k1,value_k2,extrap='lin'):
    """Returns the exponential extrapolation given the 
    values for k=1 and k=2 noise factors"""
    k_values = [1.,2.]
    
    if extrap =='lin':
        y_values = [value_k1,value_k2]

        # Fit a linear regression model (polynomial of degree 1)
        coefficients = np.polyfit(k_values, y_values, 1)

        # The coefficients represent the slope (m) and y-intercept (b) of the line
        slope, intercept = coefficients

        extrapolation = intercept

    if extrap == 'exp':
        y_values = [value_k1/value_k2,1.]
        ln_y = np.log(y_values)

        # Fit a linear regression model (polynomial of degree 1)
        coefficients_exp = np.polyfit(k_values, ln_y, 1)

        # The coefficients represent the slope (m) and y-intercept (b) of the line
        slope_exp, intercept_exp = coefficients_exp

        extrapolation = np.exp(intercept_exp)*value_k2

    # Define the exponential function
    # def exponential_func(x, a, b):
    #     return a * np.exp(b * x)
    
    # params, covariance = curve_fit(exponential_func, k_values, y_values)
    # a_opt, b_opt = params

    # exp_extrapolation = a_opt*value_k2
    

    
    # plt.plot([1.,2.],[value_k1,value_k2],'ro')
    # plt.plot([0,1.,2.],[linear_extrapolation,value_k1,value_k2],label='linear')
    # plt.plot([0,1.,2.],[exp_extrapolation,value_k1,value_k2],label='exponential')
    # plt.legend()
    
    return extrapolation

def get_estimator(server='qasm',
                  shots=2**14,
                  device_aer=FakeNairobi(),
                  session=Session(backend = service.backend("ibmq_qasm_simulator")),
                  options_rtm=Options(),
                  seed=170):
    if server =='qasm':
        estimator = Estimator(options={'shots':shots,'seed':seed})
    
    elif server == 'aer':
        coupling_map = device_aer.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device_aer)
        estimator = AerEstimator(
            backend_options={
                "method": "density_matrix",
                "coupling_map": coupling_map,
                "noise_model": noise_model,
            },
            run_options={"seed": seed, "shots": shots},
            transpile_options={"seed_transpiler": seed})
    elif server ==  'rtm':
        estimator = IBM_Estimator(session=session,options=options_rtm)
        
    return estimator

def vqe_ap_hea_zne(graph,
               g_values,
               optimizer,
               init_optimizer,
               service,
               backend,
               initial_point,
               resampling=False,
               server='qasm',
               device_aer=FakeNairobi(),
               angles_dict = {},
               layers=1,
               hx=0.5,
               options=Options(),
               zne=True,
               extrap='lin',
               reps=1,
               shots=2**14,
               ansatz_str='hea',
               path=''):
    """Runs the vqe to simulate the antiparallel model in
    the hardware efficient ansatz for different values of
    the antiparallel field"""
    
    n_qubits = len(graph.nodes())
    
    E_values = np.array([0. for i in range(len(g_values))])
    if ansatz_str == 'hea':
        theta_list = ParameterVector('θ',2*n_qubits*layers)
        ansatz = get_ansatz_hea_ibm(graph,theta_list)
        ansatz_k2 = get_ansatz_hea_ibm_ZNE(graph,theta_list)
    elif ansatz_str == 'hva':
        theta_list = ParameterVector('θ',3*layers)
        ansatz = get_ansatz_antip_ibm(graph,theta_list)
        ansatz_k2 = get_ansatz_antip_ibm(graph,theta_list)

    
    rev_g_values = g_values[::-1]
    for i,g in enumerate(tqdm(rev_g_values)):
        cost_operator = get_h_op(graph,hx=hx,ap=g) #Defining Hamiltonian
        
        # Now we set the cost function, with no mitigation, linear or exp extrapolation
        if zne == False:
            def cost_function_vqe(theta):
                job = estimator.run(ansatz, cost_operator, theta)
                values = job.result().values[0]
                return values
        if zne == True:
            def cost_function_vqe(theta):
                job = estimator.run([ansatz,ansatz_k2], 2*[cost_operator], 2*[theta])
                value_k1 = job.result().values[0]
                value_k2 = job.result().values[1]
                return get_extrapolation(value_k1=value_k1,value_k2=value_k2,extrap=extrap)              


        if i == 0:
            if resampling==False:
                with Session(service=service,backend=backend) as session:
                    estimator = get_estimator(server=server,
                                            shots=shots,
                                            device_aer=device_aer,
                                            session=session,
                                            options_rtm=options)
                    result = init_optimizer.minimize(fun=cost_function_vqe,
                    x0=initial_point)
                    session.close()
                initial_point = result.x 
            else:
                sample = 0.
                for j in range(reps):
                    initial_point = np.random.random(ansatz.num_parameters)
                    with Session(service=service,backend=backend) as session:
                        estimator = get_estimator(server=server,
                                                shots=shots,
                                                device_aer=device_aer,
                                                session=session,
                                                options_rtm=options)
                        result_sample = init_optimizer.minimize(fun=cost_function_vqe,
                        x0=initial_point)
                        session.close()
                    if result_sample.fun < sample:
                        sample = result_sample.fun
                        result = result_sample
                initial_point = result.x
                        

        else:
            with Session(service=service,backend=backend) as session:
                estimator = get_estimator(server=server,
                                          shots=shots,
                                          device_aer=device_aer,
                                          session=session,
                                          options_rtm=options)
                result = optimizer.minimize(fun=cost_function_vqe,
                x0=initial_point)
                session.close()
    
        E_values[i] = result.fun
        np.savetxt(path+f'E_{len(graph.nodes)}qubits_{ansatz_str}{layers}_zne_{zne}_{extrap}_{server}.json',E_values)
        #optimal angles storage
        angles = list(result.x)
        angles_dict[str(round(g,14))] = angles
        with open(path+f'angles_{len(graph.nodes)}qubits_{ansatz_str}{layers}_zne_{zne}_{extrap}_{server}.json', 'w') as outfile:
            json.dump(angles_dict, outfile)


    return E_values,angles_dict

def vqe_recycling(graph,
               optimizer,
               service,
               backend,
               angles_opt,
               resampling=False,
               server='qasm',
               device_aer=FakeNairobi(),
               angles_dict = {},
               layers=1,
               hx=0.5,
               options=Options(),
               zne=False,
               extrap='lin',
               reps=1,
               shots=2**14,
               ansatz_str='hea',
               path=''):
    """Runs the vqe to simulate the antiparallel model in
    the hardware efficient ansatz for different values of
    the antiparallel field"""
    
    n_qubits = len(graph.nodes())
    g_values = [float(k)for k in angles_opt.keys()]
    
    E_values = np.array([0. for i in range(len(g_values))])
    if ansatz_str == 'hea':
        theta_list = ParameterVector('θ',2*n_qubits*layers)
        ansatz = get_ansatz_hea_ibm(graph,theta_list)
        ansatz_k2 = get_ansatz_hea_ibm_ZNE(graph,theta_list)
    elif ansatz_str == 'hva':
        theta_list = ParameterVector('θ',3*layers)
        ansatz = get_ansatz_antip_ibm(graph,theta_list)
        ansatz_k2 = get_ansatz_antip_ibm(graph,theta_list)

    
    i = 0
    for g_str,angles_opt in tqdm(angles_opt.items()):
        g = float(g_str)
        cost_operator = get_h_op(graph,hx=hx,ap=g) #Defining Hamiltonian
        
        # Now we set the cost function, with no mitigation, linear or exp extrapolation
        if zne == False:
            def cost_function_vqe(theta):
                job = estimator.run(ansatz, cost_operator, theta)
                values = job.result().values[0]
                return values
        if zne == True:
            def cost_function_vqe(theta):
                job = estimator.run([ansatz,ansatz_k2], 2*[cost_operator], 2*[theta])
                value_k1 = job.result().values[0]
                value_k2 = job.result().values[1]
                return get_extrapolation(value_k1=value_k1,value_k2=value_k2,extrap=extrap) 
                  
        #Setting the resamplings
        if resampling==False:
            with Session(service=service,backend=backend) as session:
                estimator = get_estimator(server=server,
                                        shots=shots,
                                        device_aer=device_aer,
                                        session=session,
                                        options_rtm=options)
                result = optimizer.minimize(fun=cost_function_vqe,
                x0=angles_opt)
                session.close()
        else:
            best_sample = 0.
            avg = 0.
            for j in range(reps):
                with Session(service=service,backend=backend) as session:
                    estimator = get_estimator(server=server,
                                            shots=shots,
                                            device_aer=device_aer,
                                            session=session,
                                            options_rtm=options)
                    result = optimizer.minimize(fun=cost_function_vqe,
                    x0=angles_opt)
                    session.close()
                
                if result.fun < best_sample:
                    best_sample = result.fun
                    best_result = result
                avg += (1/reps)*result.fun
            result.fun = avg
            result.x = best_result.x
    
        E_values[i] = result.fun
        np.savetxt(path+f'Erec_{len(graph.nodes)}qubits_{ansatz_str}{layers}_zne_{zne}_{extrap}_{server}.json',E_values)
        #optimal angles storage
        angles = list(result.x)
        angles_dict[str(round(g,14))] = angles
        with open(path+f'anglesrec_{len(graph.nodes)}qubits_{ansatz_str}{layers}_zne_{zne}_{extrap}_{server}.json', 'w') as outfile:
            json.dump(angles_dict, outfile)
        i+=1


    return E_values,angles_dict
    

########################################################
# OPTIMAL
########################################################
def get_E_values_opt(graph,angles_dict,shots=2**12):
    circuit_list = []
    hamiltonian_list = []
    p = len(angles_dict['0.0'])//3
    for g,angles in angles_dict.items():
        theta_list = ParameterVector('theta',length=3*p)
        ansatz = get_ansatz_antiparallel(graph,theta_list)
        optimal_circuit = ansatz.bind_parameters(angles)
        circuit_list.append(optimal_circuit)
        hamiltonian= get_h_op(graph,ap=float(g))
        hamiltonian_list.append(hamiltonian)
    estimator = Estimator(options={'shots':shots})
    job = estimator.run(circuit_list,hamiltonian_list)
    energies = job.result().values
    return energies

def get_kk_values_opt(graph,angles_dict,shots=2**12):
    circuit_list = []
    kk_op_list = [get_kk_op(graph)]*len(angles_dict.keys())
    p = len(angles_dict['0.0'])//3
    for g,angles in angles_dict.items():
        theta_list = ParameterVector('theta',length=3*p)
        ansatz = get_ansatz_antiparallel(graph,theta_list)
        optimal_circuit = ansatz.bind_parameters(angles)
        circuit_list.append(optimal_circuit)
    estimator = Estimator(options={'shots':shots})
    job = estimator.run(circuit_list,kk_op_list)
    kk_values = job.result().values
    return kk_values
    
def get_E_values_opt_ibm(graph,angles_dict,options,backend_str='ibmq_qasm_simulator'):
    circuit_list = []
    hamiltonian_list = []
    p = len(angles_dict['0.0'])//3
    for g,angles in angles_dict.items():
        theta_list = ParameterVector('theta',length=3*p)
        ansatz = get_ansatz_antip_ibm(graph,theta_list)
        optimal_circuit = ansatz.bind_parameters(angles)
        circuit_list.append(optimal_circuit)
        hamiltonian= get_h_op(graph,ap=float(g))
        hamiltonian_list.append(hamiltonian)
    #Setting the estimator
    backend = service.backend(backend_str)
    with Session(service=QiskitRuntimeService(), backend=backend) as session:
        estimator = IBM_Estimator(session=session,options=options)
        job = estimator.run(circuit_list,hamiltonian_list)
        energies = job.result().values
        session.close()
    return energies
