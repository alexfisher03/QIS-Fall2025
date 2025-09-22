from __future__ import annotations
import os
import math
from typing import Dict, List, Tuple

from rich.progress import track
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import Aer

from oracle import makeOracle

def int_to_bits_le(n: int, width: int) -> List[int]:
    """Little-endian list [b0..b_{width-1}] with b0 as LSB to match IBM format"""
    return [(n >> i) & 1 for i in range(width)]

def estimate_K_classical(oracle: QuantumCircuit, n_data: int, log_file) -> int:
    """
    count exactly how many z in {0,1}^n cause a phase flip (spec(z)=1)
    Uses statevector simulator and inspect the ancilla=0 slice amplitude sign
    """
    N = 1 << n_data
    backend = Aer.get_backend("statevector_simulator")
    K = 0

    desc = f"[K] enumerate {N} basis inputs"
    for z in track(range(N), description=desc):
        # Prepare |z> on data, ancilla remains |0>
        qc = QuantumCircuit(n_data + 1)
        bits_le = int_to_bits_le(z, n_data)  # [z0,z1,...,z_{n-1}]
        for i, bit in enumerate(bits_le):
            if bit == 1:
                # apply x to if 1
                qc.x(i)

        # add in the oracle
        qc.compose(oracle, inplace=True)

        # get psi vector of final state
        sv = backend.run(transpile(qc, backend)).result().get_statevector(qc)

        # Qiskit index format 
        idx = z
        # get amplitude of |z> in final state
        amp = sv[idx].real
        unsafe = (amp < -0.5)
        if unsafe:
            K += 1

        # log each z with its amp and safety
        z_str = "".join(str(b) for b in reversed(bits_le))
        log_file.write(
            f"[K] z={z_str}  amp≈{amp:+.3f}  {'UNSAFE(-)' if unsafe else 'safe(+)' }\n"
        )

    return K

def make_diffuser(n_data: int) -> QuantumCircuit:
    """
    Grover diffuser on n_data qubits: H^⊗ X^⊗ (multi-controlled Z) X^⊗ H^⊗
    MCZ is implemented by H on target, MCX, then H back
    """
    if n_data < 1:
        raise ValueError("n_data must be >= 1")

    qr = QuantumRegister(n_data, "d")
    diff = QuantumCircuit(qr, name="Us")

    # h on all data qubits
    for i in range(n_data):
        diff.h(i)
    # X on all data qubits
    for i in range(n_data):
        diff.x(i)

    # multi-controlled Z about |0...0>
    target = n_data - 1
    controls = list(range(n_data - 1))

    diff.h(target)
    if n_data == 1:
        diff.z(target)  # reflection about |0> for 1 qubit
    else:
        mcx = MCXGate(len(controls))
        diff.append(mcx, controls + [target])
    diff.h(target)

    # uncompute X then H
    for i in range(n_data):
        diff.x(i)
    for i in range(n_data):
        diff.h(i)

    return diff


def grover_search(
    oracle: QuantumCircuit,
    n_data: int,
    K: int,
    shots: int,
    log_file,
) -> Dict[str, int]:
    """
    Run Grover for r = floor((π/4)*sqrt(N/K)) iterations
    Returns counts dict over data bitstrings (msb->lsb)
    """
    if K <= 0:
        raise ValueError("K must be >= 1 for Grover search.")
    N = 1 << n_data
    r = max(1, int(math.floor((math.pi / 4.0) * math.sqrt(N / K))))

    qr_data = QuantumRegister(n_data, "d")
    qr_anc = QuantumRegister(1, "a")
    cr = ClassicalRegister(n_data, "cd")
    qc = QuantumCircuit(qr_data, qr_anc, cr, name="Grover")

    # Init data to uniform superposition; ancilla stays |0>
    for i in range(n_data):
        qc.h(qr_data[i])

    diffuser = make_diffuser(n_data)

    desc = f"[Grover] r={r} iterations"
    for _ in track(range(r), description=desc):
        # Append oracle over [data..., ancilla]
        qc.compose(oracle, qubits=[*qr_data, qr_anc[0]], inplace=True)
        # Diffuser over data only
        qc.compose(diffuser, qubits=qr_data, inplace=True)

    # measure data only
    for i in range(n_data):
        qc.measure(qr_data[i], cr[i])

    backend = Aer.get_backend("qasm_simulator")
    tqc = transpile(qc, backend)
    res = backend.run(tqc, shots=shots).result()
    counts = res.get_counts(qc)

    total = sum(counts.values())
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]
    log_file.write(f"[Grover] N={N} K={K} r={r} shots={shots}\n")
    for s, c in top:
        frac = c / total if total else 0.0
        log_file.write(f"[Grover] {s} : {c} ({frac:.3f})\n")

    return counts

def main():
    n_data = 3        
    shots = 4096        # qasm shots for Grover
    os.makedirs("logs", exist_ok=True)

    oracle = makeOracle()

    with open("logs/counting_grover.log", "w") as log:
        log.write("[Init] counting_grover start\n")

        K = estimate_K_classical(oracle, n_data, log)
        N = 1 << n_data
        r = max(1, int(math.floor((math.pi / 4.0) * math.sqrt(N / K)))) if K > 0 else 0
        log.write(f"[K] N={N} n={n_data} K={K} r={r}\n")

        if K == 0:
            log.write("[K] All inputs safe (K=0). Exiting.\n")
            print("All inputs safe (K=0). Nothing to search for.")
            return

        counts = grover_search(oracle, n_data, K, shots, log)

        items: List[Tuple[str,int]] = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        log.write(f"N={N}  n={n_data}  K={K}  r≈{r}  shots={shots}\n")
        log.write("Top outcomes (bitstrings as z2 z1 z0):\n")
        for s, c in items[:8]:
            log.write(f"  {s} : {c}\n")

        expected_unsafe = {"110", "101"}
        top_set = set(s for s,_ in items[:4])
        log.write(f"Expected unsafe: {expected_unsafe}\n")
        log.write(f"Top-4 found    : {top_set}\n")

        log.write("[Done] counting_grover end\n")


if __name__ == "__main__":
    main()
