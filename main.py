import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.visualization import plot_histogram
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import matplotlib.pyplot as plt

def verificacion(counts):
    """
    Verifica entre las llaves de la función si esta es constante
    (Dict) -> Bool
    """
    if '0000' in counts.keys():
        print("Funcion Constante")
    else:
        print("Funcion Balanceada")


def main():
    #######0->1, 1->0, FUNCION 1########
    print("========================= Pruebas Deutsch =========================")
    print("===Funcion (0->1,1->0)===")
    # Prueba 1
    print("====Prueba 0,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.barrier(0, 1)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 2
    print("====Prueba 0,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 3
    print("====Prueba 1,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.barrier(0, 1)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 4
    print("====Prueba 1,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("=========================Prueba Algoritmo Deutsch=========================")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 1)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.h(0)
    circuit.h(1)
    circuit.barrier(0, 1)
    circuit.x(0)
    circuit.cx(0, 1)
    circuit.x(0)
    circuit.barrier(0, 1)
    circuit.h(0)

    circuit.measure([0], [0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("=========================Next=========================")

    #######0->0, 1->1, FUNCION 2########
    print("===Funcion (0->1,1->0)===")
    # Prueba 1
    print("====Prueba 0,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.barrier(0, 1)
    circuit.cx(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 2
    print("====Prueba 0,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.cx(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 3
    print("====Prueba 1,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.barrier(0, 1)
    circuit.cx(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 4
    print("====Prueba 1,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.cx(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # ALGORITMO DEUTSCH
    print("=========================Prueba Algoritmo Deutsch=========================")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 1)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.h(0)
    circuit.h(1)
    circuit.barrier(0, 1)
    circuit.cx(0, 1)
    circuit.barrier(0, 1)
    circuit.h(0)

    circuit.measure([0], [0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("=========================Next=========================")

    #######0->1, 1->1, FUNCION 3########
    print("===Funcion (0->1,1->0)===")
    # Prueba 1
    print("====Prueba 0,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.barrier(0, 1)
    circuit.x(1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 2
    print("====Prueba 0,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.x(1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 3
    print("====Prueba 1,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.barrier(0, 1)
    circuit.x(1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 4
    print("====Prueba 1,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.x(1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("=========================Prueba Algoritmo Deutsch=========================")

    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 1)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.h(0)
    circuit.h(1)
    circuit.barrier(0, 1)
    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.h(0)

    circuit.measure([0], [0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("=========================Last=========================")
    #######0->1, 1->1, FUNCION 4########
    print("===Funcion (0->1,1->0)===")
    # Prueba 1
    print("====Prueba 0,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.barrier(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 2
    print("====Prueba 0,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 3
    print("====Prueba 1,0====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.barrier(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    # Prueba 4
    print("====Prueba 1,1====")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 2)

    circuit.x(0)
    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.barrier(0, 1)

    circuit.measure([0, 1], [1, 0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("=========================Prueba Algoritmo Deutsch=========================")
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(2, 1)

    circuit.x(1)
    circuit.barrier(0, 1)
    circuit.h(0)
    circuit.h(1)
    circuit.barrier(0, 1)
    circuit.barrier(0, 1)
    circuit.h(0)

    circuit.measure([0], [0])
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()
    print()

    print("===================================================================================")
    print("=========================Pruebas Circuitos Deutsch - Jozsa=========================")
    # 1
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.cx(0, 4)
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Next=========================")
    # 2
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.cx(1, 4)
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Next=========================")
    # 3
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.cx(2, 4)
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    verificacion(counts)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Last=========================")
    # 4
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    for i in range(0, 4):
        circuit.i(i)
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Pruebas Algoritmo Deutsch - Jozsa=========================")
    # 1
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.x(4)
    circuit.barrier()
    for i in range(0, 5):
        circuit.h(i)
    circuit.barrier()
    circuit.cx(0, 4)
    circuit.barrier()
    for i in range(0, 4):
        circuit.h(i)
    circuit.barrier()
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    verificacion(counts)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Next=========================")
    # 2
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.x(4)
    circuit.barrier()
    for i in range(0, 5):
        circuit.h(i)
    circuit.barrier()
    circuit.cx(1, 4)
    circuit.barrier()
    for i in range(0, 4):
        circuit.h(i)
    circuit.barrier()
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    verificacion(counts)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Next=========================")
    # 3
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.x(4)
    circuit.barrier()
    for i in range(0, 5):
        circuit.h(i)
    circuit.barrier()
    circuit.cx(2, 4)
    circuit.barrier()
    for i in range(0, 4):
        circuit.h(i)
    circuit.barrier()
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    verificacion(counts)
    print(circuit)
    plot_histogram(counts)
    plt.show()

    print("=========================Next=========================")
    # 4
    simulator = Aer.get_backend('qasm_simulator')
    circuit = QuantumCircuit(5, 4)
    circuit.x(4)
    circuit.barrier()
    for i in range(0, 5):
        circuit.h(i)
    circuit.barrier()
    for i in range(0, 4):
        circuit.i(i)
    circuit.barrier()
    for i in range(0, 4):
        circuit.h(i)
    circuit.barrier()
    circuit.measure([0, 1, 2, 3], [3, 2, 1, 0])

    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1000)
    result = job.result()
    counts = result.get_counts(circuit)
    verificacion(counts)
    print(circuit)
    plot_histogram(counts)
    plt.show()


main()
