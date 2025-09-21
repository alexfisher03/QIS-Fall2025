# oracle
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from qiskit.circuit.library import MCXGate
import os
'''
    spec(z) = (z2 ∧ z1 ∧ ¬z0)  OR  (z2 ∧ ¬z1 ∧ z0)
    qubit mapping: q[0]=z0, q[1]=z1, q[2]=z2, q[3]=ancilla

    multiply |z2 z1 z0⟩ by -1 iff spec(z)=1.
'''
def makeOracle() -> QuantumCircuit:
    qc = QuantumCircuit(4, name="O_spec")

    """
    phase 'kickback' ancilla: |-> = H X |0>
    X|-> = -|->, a CCX targeting this ancilla implements a phase flip (−1) exactly when its controls are satisfied
    """
    
    qc.x(3)
    qc.h(3)

    mcx3 = MCXGate(3)  # 3-control X gate

    qc.x(0)                    
    qc.append(mcx3, [2, 1, 0, 3])
    qc.x(0)                    

    qc.x(1)                    
    qc.append(mcx3, [2, 1, 0, 3])
    qc.x(1)                   

    qc.h(3)
    qc.x(3)

    return qc

def save_png_mpl(filename: str = "circuit_models/oracle.png") -> None:
    circ = makeOracle()
    circ.draw(output="mpl")
    fig = circ.draw(output="mpl")
    fig.savefig(filename, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    if not os.path.exists("circuit_models/oracle.png"):
        os.makedirs("circuit_models", exist_ok=True)
        save_png_mpl("circuit_models/oracle.png")
