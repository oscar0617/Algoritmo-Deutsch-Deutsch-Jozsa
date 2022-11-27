# Algoritmo de Deutsch y Deutsch Jozsa:

_En este repositorio encontraremos la implementación de los algoritmos cuanticos de Deutsch y Deutsch Jozsa._ \
_Además de contener los circuitos de los algoritmos y sus respectivas pruebas._


### Pre-requisitos
_Para poder correr nuestra libreria necesitaremos un iDLE cualquiera de python._\
_Para poder obtener un resultado exitoso debemos tener la libreria de matplotlib y qiskit instalados en python, de lo contrario no podremos ejecutar satisfactoriamente nuestras funciones._ 
### Ejemplos
_A continuación, tendremos el codigo utilizado para construir un circuito:_
```
simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(2, 2)

circuit.barrier(0, 1)
circuit.x(0)
circuit.cx(0, 1)
circuit.x(0)
circuit.barrier(0, 1)


```
_¿Cómo podemos verificar estos resultados?_\
_Para poder correr este circuito, debemos ejecutar las siguientes lineas de codigo:_
```
circuit.measure([0, 1], [1, 0])
compiled_circuit = transpile(circuit, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print(circuit)
plot_histogram(counts)
plt.show()
print()
```
_En la terminal de PyCharm nos va a arrojar el circuito que se plantió y en una ventana emergente tendremos el resultado de esa prueba._

## ¿Como lo construimos?
* [Pycharm](https://www.jetbrains.com/es-es/pycharm/) -_El iDLE usado_

## Autor
* **Oscar Lesmes** - *Repositorio* - [GitHub](https://github.com/villanuevand)

