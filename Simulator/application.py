import sys

from netqasm.sdk.classical_communication.socket import Socket
from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.epr_socket import EPRSocket
# from netqasm.sdk.qubit import Qubit
from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta

import random
import hashlib
import json
import unireedsolomon as rs
import math
from sympy import nextprime

import time

def generate_random_bitstring(length):
    str = [random.choice(range(2)) for _ in range(length)]
    return str


def xor(var, key, byteorder=sys.byteorder):
    key, var = key[:len(var)], var[:len(key)]
    int_var = int.from_bytes(var, byteorder)
    int_key = int.from_bytes(key, byteorder)
    int_enc = int_var ^ int_key
    return int_enc.to_bytes(len(var), byteorder)


def encrypt(hash_function, key, nonce, coded_message, h_len):
    h = hash_function.create_hash(nonce, key, h_len)
    prg = hashlib.shake_256()
    prg.update(h.to_bytes(math.ceil(h_len / 8)))
    generated = prg.hexdigest(256).encode()
    encrypted = xor(coded_message, generated)
    return encrypted


def create_nonce(length):
    """Creates a random bitstring {0,1} with a given length"""
    nonce = ''.join([random.choice(('0', '1')) for _ in range(length)])
    return nonce


def generate_commitment_strings(bits, basis, lam_HS):
    total_bit_commit = []
    total_base_commit = []

    for i in range(len(bits)):
        bit_commitment = Commitment.generate_commitment(bits[i], create_nonce(lam_HS)) # generate commitment
        base_commitment = Commitment.generate_commitment(basis[i], create_nonce(lam_HS)) # generate commitment

        total_bit_commit.append(bit_commitment)
        total_base_commit.append(base_commitment)

    return total_bit_commit, total_base_commit


class Commitment():
    nonce: str
    bit: int
    t: str

    def __init__(self, nonce=None, bit=None, t=None):
        self.nonce = nonce
        self.bit = bit
        if t is not None:
            self.t = t

    def generate_hash(self, bit, nonce):
        # Generate commitment with nonce
        commitment = hashlib.sha256()
        commitment.update(str(bit).encode() + nonce.encode())
        return commitment.hexdigest()

    def json_commitment(self):
        return json.dumps({
            "t": self.t
        })

    def verify_values(self):
        return {
            "bit": self.bit,
            "nonce": self.nonce
        }

    @classmethod
    def init_from_json(self, json_str):
        data = json.loads(json_str)
        t = None
        bit = None
        nonce = None
        if "t" in data:
            t = data["t"]
        if "bit" in data:
            bit = data["bit"]
        if "nonce" in data:
            nonce = data["nonce"]
        return self(nonce, bit, t)

    @classmethod
    def generate_commitment(self, bit, nonce):
        result = self.generate_hash(self, bit, nonce)
        return self(nonce, bit, result)

    def check_commitment(self):
        if self.bit is None or self.nonce is None or self.t is None:
            return -1

        if self.generate_hash(self.bit, self.nonce) == self.t:
            return 1
        else:
            return 0


class AliceProgram(Program):
    PEER_NAME = "Bob"

    def __init__(self, messages, lam_OT, lam_HS, lam_PQS, Q_tol, error_correction, hash_function, leakage_rate=0.2):
        self._lam_OT = lam_OT
        self._lam_HS = lam_HS
        self._lam_PQS = lam_PQS
        self._Q_tol = Q_tol
        self._leakage_rate = leakage_rate
        self.m0 = messages[0]
        self.m1 = messages[1]
        self.error_corr = error_correction
        self.hash_function = hash_function

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="test_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 0: Startup  @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@
        """
        logger = LogManager.get_stack_logger("AliceProgram")
        # Get classical and epr sockets. And connection to quantum network processing unit
        c_socket = context.csockets[self.PEER_NAME]
        epr_socket = context.epr_sockets[self.PEER_NAME]
        connection = context.connection

        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 1: State distrubution @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        logger.info("Phase 1: State Distribution")
        x = []
        # Generate a random basis to apply to the EPR qubits
        a_basis = generate_random_bitstring(2*self._lam_OT)
        # Send bitstring based on basis to Bob
        for i in range(2 * self._lam_OT):
            # Register a request to create an EPR pair
            q = epr_socket.create_keep(1)[0]
            if a_basis[i] == 1:
                q.H()
            m = q.measure()

            yield from connection.flush()
            x.append(m.value)

        logger.info(f"Alice measures local EPR qubits: {x} with basis {a_basis}")

        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 2: Commitment @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@
        1. Bob commits to his basis and measurement results (x_b, b_basis) using a commitment scheme
        2. Alice sends T with random indexes to open
        3. Bob opens commitment for indexes in T and Alice checks if commitment is correct
        """
        logger.info("Phase 2: Commitment")

        # Receive bobs commitments
        json_commitments = yield from c_socket.recv()

        # Split commitment up
        commitments = json.loads(json_commitments)
        comm_bits = [Commitment.init_from_json(bit_json) for bit_json in commitments["bits"]]
        comm_basis = [Commitment.init_from_json(basis_json) for basis_json in commitments["basis"]]

        # Pick random indexes to verify for the commitment and send to Bob
        T = random.sample(range(2*self._lam_OT), self._lam_OT)
        c_socket.send(T)

        # Receive the values to verify for the random indexes and check if commitment is correct
        verif_json = yield from c_socket.recv()
        verif = json.loads(verif_json)

        base_score = []
        bit_score = []
        for i in T:
            comm_bits[i].bit = verif["bits"][str(i)]["bit"]
            comm_bits[i].nonce = verif["bits"][str(i)]["nonce"]
            comm_basis[i].bit = verif["basis"][str(i)]["bit"]
            comm_basis[i].nonce = verif["basis"][str(i)]["nonce"]

            bit_score.append(comm_bits[i].check_commitment())
            base_score.append(comm_basis[i].check_commitment())

        # Calculate error score for commitment and check if below tolerance
        q_bit = 1 - (bit_score.count(1)/self._lam_OT)
        q_base = 1 - (base_score.count(1)/self._lam_OT)
        logger.info(f"Error score for bit commitment qubits: {q_bit} and basis {q_base}")
        if (q_bit > self._Q_tol or q_base > self._Q_tol):
            logger.error("Too much error in bit commtiment, aborting..")

        # Calculate the Qber
        alice_bits = [x[i] for i in T if a_basis[i] == comm_basis[i].bit]
        bob_bits = [comm_bits[i].bit for i in T if a_basis[i] == comm_basis[i].bit]
        not_matching = [i for i in range(len(alice_bits)) if alice_bits[i] != bob_bits[i]]
        Qber = (len(not_matching) / (len(alice_bits))) * 100
        self.error_corr.error_rate = Qber / 100


        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 3: Basis Reconciliation @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        1. Alice sends her basis(theta) to Bob
        2. Bob looks where their basis match (I0) and where they don't match (I1)
        3. Bob sends I0 and I1 to Alice
        4. Alice splits her measurement results based on I0 and I1
        """
        logger.info("Phase 3: Basis Reconciliation")
        c_socket.send(a_basis)
        I0 = yield from c_socket.recv()
        I1 = yield from c_socket.recv()

        logger.info(f"Alice receives from Bob: I0 {I0} and I1 {I1}")
        x0 = ''.join([str(x[i]) for i in I0])
        x1 = ''.join([str(x[i]) for i in I1])
        logger.info(f"Alice splits her measurement results based on I0 and I1 to x0: {x0} and x1: {x1}")
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 4: Error Correction + Enryption @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        1. Alice sends nonce and error correction code (Syn) for both x0 and x1
        2. Bob corrects his matching swapped qubit measurement results (x) with the error correction code
        3. Alice encrypts messages with a 2-universal hash consisting of the nonce
        and x, which is fed trough a PRG.
        The resulting value is XOR'ed with the key and send to bob.
        4. Bob decrypts using his error corrected x and the received values from alice.
        """
        # Pick random bits to add as nonce, and send to Bob
        nonce0 = create_nonce(self._lam_PQS + len(x0))
        nonce1 = create_nonce(self._lam_PQS + len(x1))
        c_socket.send(nonce0)
        c_socket.send(nonce1)

        # Send error correction codes from x0 and x1 to bob
        c_socket.send(self.error_corr.create_syndrome(x0, self._leakage_rate))
        c_socket.send(self.error_corr.create_syndrome(x1, self._leakage_rate))

        # Encrypt messages with x0 and x1 as key and send to Bob
        c_socket.send(encrypt(self.hash_function, x0, nonce0, self.m0.encode(), self._lam_PQS))
        c_socket.send(encrypt(self.hash_function, x1, nonce1, self.m1.encode(), self._lam_PQS))

        return {'qber': Qber, 'm0': self.m0, 'm1': self.m1}


class BobProgram(Program):
    PEER_NAME = "Alice"

    def __init__(self, lam_OT, lam_HS, lam_PQS, chosen_message, error_correction, hash_function=None, leakage_rate=0.2):
        self._lam_OT = lam_OT
        self._lam_HS = lam_HS
        self.chosen_message = chosen_message
        self._lam_PQS = lam_PQS
        self.error_corr = error_correction
        self.hash_function = hash_function
        self._leakage_rate = leakage_rate

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name="test_program",
            csockets=[self.PEER_NAME],
            epr_sockets=[self.PEER_NAME],
            max_qubits=2,
        )

    def run(self, context: ProgramContext):
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 0: Startup  @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@
        """
        starttime = time.time()
        logger = LogManager.get_stack_logger("BobProgram")
        # Get classical and epr sockets. And connection to quantum network processing unit
        c_socket: Socket = context.csockets[self.PEER_NAME]
        epr_socket: EPRSocket = context.epr_sockets[self.PEER_NAME]
        connection: BaseNetQASMConnection = context.connection

        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 1: State distrubution @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        logger.info("Phase 1: State Distribution")
        x_b = []
        # Bob has his own random basis to apply to the EPR qubits
        b_basis = generate_random_bitstring(2*self._lam_OT)
        # Recieve bitstring from Alice, and apply basis
        for i in range(2 * self._lam_OT):
            # Register a request to receive an EPR pair
            q = epr_socket.recv_keep(1)[0]
            if b_basis[i] == 1:
                q.H()
            m = q.measure()

            yield from connection.flush()
            x_b.append(m.value)

        logger.info(f"Bob measures local EPR qubits: {x_b} with basis {b_basis}")

        measurement_time = time.time()
        logger.info(f"Time taken for state distribution: {measurement_time - starttime} seconds")
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 2: Commitment @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@
        1. Bob commits to his basis and measurement results (x_b, b_basis) using a commitment scheme
        2. Alice sends T with random indexes to open
        3. Bob opens commitment for indexes in T and Alice checks if commitment is correct
        """
        logger.info("Phase 2: Commitment")
        # Generate commitment strings for bits and basis and send to Alice
        bits_commits, basis_commits = generate_commitment_strings(x_b, b_basis, self._lam_HS)
        commitments = {
            "bits": [i.json_commitment() for i in bits_commits],
            "basis": [i.json_commitment() for i in basis_commits]
        }
        c_socket.send(json.dumps(commitments))

        # Get random indexes to open from Alice
        T = yield from c_socket.recv()
        verif_bits = {}
        verif_base = {}
        for i in T:
            verif_bits.update({i: bits_commits[i].verify_values()})
            verif_base.update({i: basis_commits[i].verify_values()})

        verif = {
            "bits": verif_bits,
            "basis": verif_base
        }
        c_socket.send(json.dumps(verif))

        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 3: Basis Reconciliation @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        1. Alice sends her basis(theta) to Bob
        2. Bob looks where their basis match (I0) and where they don't match (I1)
        3. Depending on the message Bob wants to receive, he swaps I0 and I1
        4. Bob sends I0 and I1 to Alice
        """
        logger.info("Phase 3: Basis Reconciliation")
        # Look where basis matches(I1) and where it doesn't match(I0) and send to Alice
        a_basis = yield from c_socket.recv()
        I0 = [i for i in range(len(a_basis)) if a_basis[i] == b_basis[i] and i not in T]
        I1 = [i for i in range(len(a_basis)) if a_basis[i] != b_basis[i] and i not in T]

        # Swap I0 and I1 to receive m1 instead of m0, and send to Alice
        if self.chosen_message == 1:
            I0, I1 = I1, I0

        c_socket.send(I0)
        c_socket.send(I1)
        x0_bob = ''.join([str(x_b[i]) for i in I0])
        x1_bob = ''.join([str(x_b[i]) for i in I1])
        logger.info(f"Bob splits his measurement results based on I0 and I1 to x0: {x0_bob} and x1: {x1_bob}")
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        @@@ Phase 4: Error Correction + Decryption @@@
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        1. Alice sends nonce and error correction code (Syn) for both x0 and x1
        2. Bob corrects his matching swapped qubit measurement results (x) with the error correction code
        3. Alice encrypts messages with a 2-universal hash consisting of the nonce
        and x, which is fed trough a PRG.
        The resulting value is XOR'ed with the key and send to bob.
        4. Bob decrypts using his error corrected x and the received values from alice.
        """
        nonce0 = yield from c_socket.recv()
        nonce1 = yield from c_socket.recv()
        corr_code_0 = yield from c_socket.recv()
        corr_code_1 = yield from c_socket.recv()

        # Correct received qubits with the syndrome from Alice
        x0_corrected = self.error_corr.apply_syndrome(x0_bob, corr_code_0, self._leakage_rate)
        x1_corrected = self.error_corr.apply_syndrome(x1_bob, corr_code_1, self._leakage_rate)

        # Receive encrypted message from Alice and decrypt with corrected qubits as key
        enc_m0 = yield from c_socket.recv()
        enc_m1 = yield from c_socket.recv()
        m0 = ""
        m1 = ""
        if (x0_corrected != ""):
            decrypted = encrypt(self.hash_function, x0_corrected, nonce0, enc_m0, self._lam_PQS)
            m0 = decrypted.decode()
            logger.info(f"Bob decodes messages to: m0: {decrypted.decode()}")
        if (x1_corrected != ""):
            decrypted = encrypt(self.hash_function, x1_corrected, nonce1, enc_m1, self._lam_PQS)
            m1 = decrypted.decode()
            logger.info(f"Bob decodes messages to: m1: {decrypted.decode()}")

        endtime = time.time()
        logger.info(f"Total time taken for Bob's program: {endtime - starttime} seconds")
        logger.info(f"post process time: {endtime - measurement_time} seconds")
        return {"m0": m0, "m1": m1, "post_process_time": endtime - measurement_time, 'total_time': endtime - starttime}
