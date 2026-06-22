import numpy as np
import secrets
from Crypto.Cipher import AES
import threading


# ot
from application import AliceProgram, BobProgram
from squidasm.run.stack.config import (StackNetworkConfig, DepolariseLinkConfig, LinkConfig)
from squidasm.run.stack.run import run

from error_correction import LDPCErrorCorrection
from hashing import CarterWegmanHash
import numpy as np


# ---------- Helper functions ----------

def int_to_bits(x, length):
    """Return 'length' bits of integer x as np.array([0,1,...])."""
    bitstring = format(x, f"0{length}b")
    return np.fromiter((int(b) for b in bitstring), dtype=np.uint8)

def aes_prf(key: bytes, I: int, m: int) -> np.ndarray:
    """
    PRF F_K(I) using AES-ECB.
    key: 16-byte AES key
    I: integer input
    m: number of output bits
    """
    print(f"PRF called with key: {key.hex()}, I: {I}, m: {m}")
    cipher = AES.new(key, AES.MODE_ECB)
    # encode I as 16 bytes (128-bit block)
    block = I.to_bytes(16, byteorder="big", signed=False)
    out = cipher.encrypt(block)           # 16 bytes = 128 bits
    # convert to bits
    bits = np.unpackbits(np.frombuffer(out, dtype=np.uint8))
    return bits[:m].astype(np.uint8)

def xor_bits(a, b):
    return np.bitwise_xor(a, b)


def single_ot(m0, m1, bob_choice):
    # import network configuration from file
    cfg = StackNetworkConfig.from_file("./params/qdevice_params_optimistic.yaml")
    depolarise_config = DepolariseLinkConfig.from_file("./params/depolarise_link_config.yaml")
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
    # Replace link from YAML file with new depolarise link
    cfg.links = [link]


    # Security Parameters for OT
    multiplier = 5
    lam_OT  = 256 * multiplier     # length of bitstring
    lam_HS  = 20 * multiplier     # Hash size
    lam_BS  = 20 * multiplier      # Block size for error correction
    lam_PQS = 100 * multiplier     # Size of nonce added
    Q_tol   = 0.025     # Tolerance for fault in bit commitment
    chosen_message = 0     # Bob's choice of message, either 0 or 1
    error_correction = LDPCErrorCorrection()
    hash_function = CarterWegmanHash()

    cfg = StackNetworkConfig.from_file("./params/qdevice_params_current.yaml")
    depolarise_config = DepolariseLinkConfig.from_file("./params/depolarise_link_config.yaml")
    link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
    cfg.links = [link]

    alice_program = AliceProgram(messages=[m0, m1], lam_OT=lam_OT, lam_HS=lam_HS, Q_tol=Q_tol, lam_PQS=lam_PQS, error_correction=error_correction, hash_function=hash_function, leakage_rate=0.5)
    bob_program = BobProgram(lam_OT=lam_OT, lam_HS=lam_HS, lam_PQS=lam_PQS, chosen_message=bob_choice, error_correction=error_correction, hash_function=hash_function, leakage_rate=0.5)
    stats, res = run(config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1, measure_stats=True)
    res_alice, res_bob = res
    return res_bob[0][f'm{bob_choice}']

# ---------- Base node class ----------
# class OTBridge():




class node():
    def __init__(self, name, l, m):
        self.name = name
        self._peers = {}
        self._waiting_for_message = False
        self._message_queue = []
        self.l = l  # number of bits for index (log2 N)
        self.m = m  # message length in bits
        self.OT_input = []

    def set_peer(self, peer):
        self._peers[peer.name] = peer

    def send_message(self, message, peer_name):
        if peer_name in self._peers:
            peer = self._peers[peer_name]
            # spin until peer is ready to receive
            while not peer._waiting_for_message:
                pass
            peer._message_queue.append((message, self.name))
            # print(f"{self.name} sent message to {peer_name}: {message['type']}")
        else:
            print(f"Peer {peer_name} not found in {self.name}'s peers.")

    def receive_message(self):
        self._waiting_for_message = True
        # print(f"{self.name} waiting for message...")
        while len(self._message_queue) == 0:
            pass
        queued_message, queued_sender = self._message_queue.pop(0)
        # print(f"{self.name} received from {queued_sender}: {queued_message['type']}")
        self._waiting_for_message = False
        return queued_message, queued_sender

    def send_OT_bits(self, m0, m1):
        while len(self.OT_input) > 0:
            pass
        self.OT_input = [m0, m1]

    # ot is being handed off to the 1-2ot implementation.
    def recieve_OT_bits(self, peer_name, choice_bit):
        peer = self._peers[peer_name]
        while not len(peer.OT_input) == 2:
            pass

        m0, m1 = peer.OT_input
        res = single_ot(m0, m1, choice_bit)
        peer.OT_input = []  # clear the input after use
        return res

    def run(self):
        pass

# ---------- Alice (receiver) ----------

class Alice(node):
    def __init__(self, name, l, m):
        super().__init__(name, l, m)
        self.choice_index = None        # I
        self.chosen_keys = []           # [K_1^{i1}, ..., K_l^{i_l}]
        self.reconstructed_XI = None

    def run(self):
        """
        Alice wants to learn X_I for some I in {0,...,2^l-1}
        """
        N = 2 ** self.l
        # Choose a random index I
        self.choice_index = np.random.randint(0, N)
        print(f"Alice chose index I = {self.choice_index}")
        I = self.choice_index
        i_bits = int_to_bits(I, self.l)

        # 2. For each j, run a (simulated) 1-out-of-2 OT to obtain K_j^{i_j}
        self.chosen_keys = []
        for j in range(self.l):
            choice_bit = int(i_bits[j])
            response_ot = self.recieve_OT_bits("Bob", choice_bit)  # This will block until Bob has sent the OT response
            # translate response back to bytes
            print(f"Alice received OT response for j={j}, choice_bit={choice_bit}: {response_ot}")
            if len([c for c in response_ot if c in ['0', '1']]) != 128:
                self.send_message("Invalid", "Bob")
                j = j-1  # retry this OT
                
            else:
                key_bytes = int(response_ot, 2).to_bytes(16, byteorder="big")
                self.chosen_keys.append(key_bytes)
                self.send_message("Success", "Bob")
            
            # # Send OT request to Bob (insecure: reveal choice_bit)
            # msg = {
            #     "type": "OT_REQUEST",
            #     "j": j,
            #     "choice": choice_bit
            # }
            # self.send_message(msg, "Bob")

            # # Receive OT response with key K_j^{i_j}
            # response, _ = self.receive_message()
            # assert response["type"] == "OT_RESPONSE"
            # assert response["j"] == j
            # self.chosen_keys.append(response["key"])

        # 3. Receive all Y_I from Bob
        msg, _ = self.receive_message()
        assert msg["type"] == "Y_VALUES"
        Y_list = msg["Y"]  # list of numpy arrays of length m

        # 4. Reconstruct X_I = Y_I xor ⊕_j F_{K_j^{i_j}}(I)
        mask = np.zeros(self.m, dtype=np.uint8)
        for j, key in enumerate(self.chosen_keys):
            print(f"Key {j}: {key.hex()}, Index {I}, m: {self.m}")
            mask = xor_bits(mask, aes_prf(key, I, self.m))

        Y_I = Y_list[I]
        self.reconstructed_XI = xor_bits(Y_I, mask)

        print(f"Alice chose index I = {I}")
        print(f"Alice reconstructed X_I (first {min(32, self.m)} bits): {self.reconstructed_XI[:min(32, self.m)]}")

# ---------- Bob (sender) ----------

class Bob(node):
    def __init__(self, name, l, m, messages = None):
        super().__init__(name, l, m)
        self.X = None       # list of messages X_0,...,X_{N-1}
        self.keys = None    # keys[j][b] = K_j^b
        self.Y = None       # list of Y_I
        self.messages = messages  # Optional: Bob's messages to encode in X_I

    def setup(self):
        """
        Step 1 of the protocol:
        - sample X_1,...,X_N
        - sample keys (K_j^0, K_j^1) for j=1..l
        - compute Y_I = X_I xor ⊕_j F_{K_j^{i_j}}(I)
        """
        N = 2 ** self.l
        print(f"Bob is setting up with N={N} messages, each of length m={self.m} bits.")
        # Sender's messages X_I in {0,1}^m
        if self.messages is not None:
            assert len(self.messages) == N
            assert all(len(msg) == self.m for msg in self.messages)
            self.X = self.messages
        else:
            self.X = [np.random.randint(0, 2, self.m, dtype=np.uint8) for _ in range(N)]

        # l random pairs of AES keys
        self.keys = [
            [secrets.token_bytes(16), secrets.token_bytes(16)]  # (K_j^0, K_j^1)
            for _ in range(self.l)
        ]

        # Precompute all Y_I
        self.Y = []
        for I in range(N):
            i_bits = int_to_bits(I, self.l)
            mask = np.zeros(self.m, dtype=np.uint8)
            for j, bit in enumerate(i_bits):
                key = self.keys[j][bit]  # K_j^{i_j}
                mask = xor_bits(mask, aes_prf(key, I, self.m))
            Y_I = xor_bits(self.X[I], mask)
            self.Y.append(Y_I)

    def run(self):
        """
        Bob runs the sender side of Protocol 2.1
        """
        # Step 1: prepare X, keys, and Y
        self.setup()

        # Step 2: For each j, perform (simulated) 1-out-of-2 OT with Alice
        print(self.l)
        for j in range(self.l):
            input_0, input_1 = self.keys[j][0], self.keys[j][1]
            # convert inputs to strings for single_ot
            input_0_str = ''.join(format(byte, '08b') for byte in input_0)
            input_1_str = ''.join(format(byte, '08b') for byte in input_1)
            print(f"Bob is sending OT bits for j={j}, input_0: {input_0_str}, input_1: {input_1_str}")
            self.send_OT_bits(input_0_str, input_1_str)  # Provide keys for OT simulation
            result = self.receive_message()  # Wait for Alice's response about OT result
            if result[0] == "Invalid":
                print(f"Bob received invalid OT response for j={j}.")
            elif result[0] == "Success":
                print(f"Bob received success OT response for j={j}.")
                j = j-1
            else:
                print(f"Bob received unexpected OT response for j={j}: {result[0]}")
                return

        # Step 3: send Y_1,...,Y_N to Alice
        msg = {
            "type": "Y_VALUES",
            "Y": self.Y
        }
        self.send_message(msg, "Alice")

        print("Bob finished sending Y values.")

# ---------- Simulation driver ----------

def run_simulation(l=4, m=32):
    """
    Run a single 1-out-of-N OT simulation.
    l: number of bits in index (N = 2^l)
    m: length of each X_I in bits
    """
    alice = Alice("Alice", l=l, m=m)
    bob = Bob("Bob", l=l, m=m)

    alice.set_peer(bob)
    bob.set_peer(alice)

    alice_thread = threading.Thread(target=alice.run)
    bob_thread = threading.Thread(target=bob.run)
    alice_thread.start()
    bob_thread.start()
    alice_thread.join()
    bob_thread.join()

    # Verify correctness (in the simulator we can peek at Bob's X)
    I = alice.choice_index
    X_I_true = bob.X[I]
    ok = np.array_equal(alice.reconstructed_XI, X_I_true)
    print(f"Verification: Alice's X_I {'matches' if ok else 'DOES NOT MATCH'} Bob's X_I.")
    print(f"True X_I (first {min(32, m)} bits):      {X_I_true[:min(32, m)]}")

if __name__ == "__main__":
    run_simulation(l=6, m=16)