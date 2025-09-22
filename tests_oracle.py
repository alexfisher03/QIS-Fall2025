from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
from oracle import makeOracle
import os
from rich.progress import track

def bits(n, w): 
    return format(n, f"0{w}b")

def main():
    # Build oracle and backend
    O = makeOracle()
    sv_backend = Aer.get_backend("statevector_simulator")

    # Ensure logs/ exists
    os.makedirs("logs", exist_ok=True)

    unsafe = {"110","101"}  # ground-truth
    trackDesc = f"Testing on: {unsafe}"

    with open("logs/tests_oracle.log", "w") as log_file:
            # 2^3
            for z in track(range(8), description=trackDesc):
                # 3 chars for z2 z1 z0
                zstr = bits(z, 3)
                qc = QuantumCircuit(4)

                # Prepare |z2 z1 z0>
                if zstr[2] == "1": qc.x(0)  # q0=z0
                if zstr[1] == "1": qc.x(1)  # q1=z1
                if zstr[0] == "1": qc.x(2)  # q2=z2

                # append oracle.py circuit into this local qc
                qc.compose(O, inplace=True)

                # transpile and run, store final |psi>
                sv = sv_backend.run(
                     transpile(qc, sv_backend)
                     ).result().get_statevector(qc)

                # index = q0 * 2^0 + q1 * 2^1 + q2 * 2^2 + q3 * 2^3
                idx = (z & 0b111)
                amp = sv[idx].real
                sign = "-" if amp < -0.5 else "+"

                log_file.write(
                    f"z={zstr}  expected={'unsafe' if zstr in unsafe else 'safe'}  "
                    f"amp≈{amp:+.3f}  sign={sign}\n"
                )

if __name__ == "__main__":
    main()
