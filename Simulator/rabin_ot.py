
# Import the two programs to form the application ran
import itertools

from application import AliceProgram, BobProgram
from squidasm.run.stack.config import (StackNetworkConfig, DepolariseLinkConfig, LinkConfig)
from squidasm.run.stack.run import run
from error_correction import LDPCErrorCorrection
from hashing import CarterWegmanHash
import numpy as np
import math


def encoding_scheme(w, n, x_b):
    K = math.comb(n, int(x_b*n))
    w_int = int("".join(map(str, w)), 2)
    subset_index = w_int % K
    # Now we need to find the subset of size x_b*n that corresponds to subset_index
    # We can do this by iterating through all subsets of size x_b*n and counting until we reach subset_index
    count = 0
    for subset in itertools.combinations(range(n), int(x_b*n)):
        if count == subset_index:
            return subset
        count += 1
    return None


# import network configuration from file
cfg = StackNetworkConfig.from_file("./params/qdevice_params_optimistic.yaml")
depolarise_config = DepolariseLinkConfig.from_file("./params/depolarise_link_config.yaml")
link = LinkConfig(stack1="Alice", stack2="Bob", typ="depolarise", cfg=depolarise_config)
# Replace link from YAML file with new depolarise link
cfg.links = [link]

# Security Parameters for OT
multiplier = 1
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
# # Replace link from YAML file with new depolarise link

cfg.links = [link]


def single_ot(m0, m1, bob_choice):
    alice_program = AliceProgram(messages=[m0, m1], lam_OT=lam_OT, lam_HS=lam_HS, Q_tol=Q_tol, lam_PQS=lam_PQS, error_correction=error_correction, hash_function=hash_function, leakage_rate=0.5)
    bob_program = BobProgram(lam_OT=lam_OT, lam_HS=lam_HS, lam_PQS=lam_PQS, chosen_message=bob_choice, error_correction=error_correction, hash_function=hash_function, leakage_rate=0.5)
    stats, res = run(config=cfg, programs={"Alice": alice_program, "Bob": bob_program}, num_times=1, measure_stats=True)
    res_alice, res_bob = res
    return res_bob[0][f'm{bob_choice}']


n = 1000
n_ot = 10
# 1. Alice and Bob select x to be a (very small) positive constant less than 1.
x_a = np.random.uniform(0, 1)
x_b = np.random.uniform(0, 1)

# 2. Alice chooses two random strings T0, T1 ∈R {0, 1}n.
T0 = np.random.randint(0, 2, size=n).tolist()
T1 = np.random.randint(0, 2, size=n).tolist()

# Bob chooses a random c ∈R {0, 1}. Let m = ⌈log (( n  xn  ))⌉. Bob selects w ∈R {0, 1}m uniformly at random and decodes w into a subset s ⊂ I of cardinality xn according to the encoding/decoding scheme of Section 3.1.
c = np.random.randint(0, 2)
m = int(np.ceil(np.log2(math.comb(n, int(x_b*n)))))
w = np.random.randint(0, 2, size=m).tolist()
s = encoding_scheme(w, n, x_b)
print(f"Bob's choice of c: {c}, w: {w}, subset s: {s}")