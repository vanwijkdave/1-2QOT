# import numpy as np
import ldpc.codes as codes
from ldpc import BpOsdDecoder
from ldpc import BpDecoder
import math
import unireedsolomon as rs
import numpy as np
from scipy.io import mmread
from pathlib import Path


class ErrorCorrection:
    def create_syndrome(self, data, size):
        raise NotImplementedError("Subclasses must implement this method")

    def apply_syndrome(self, data, parity, size):
        raise NotImplementedError("Subclasses must implement this method")


class UniReedSolomonCorrection(ErrorCorrection):
    # Becomes quite slow for large sizes, but works well for small sizes. 
    # Optimal parity size is determined experimentally, and is around 5% of the data size. 
    # For larger sizes, the encoding and decoding time becomes too large.

    # Future improvement: improve RS in chunks, so we can handle larger sizes without running into performance issues.
    def parity_size(self, data_size, leakage_rate):
        max_leakage = data_size * leakage_rate
        return math.ceil(max_leakage / 8)

    def create_syndrome(self, data, leakage_rate):
        parity_size = self.parity_size(len(data), leakage_rate)
        parity = ""
        try:
            coder = rs.RSCoder(n=len(data) + parity_size, k=len(data))
            parity = coder.encode(data)[parity_size * -1:]
        except rs.RSCodecError:
            pass

        return parity

    def apply_syndrome(self, data, parity, leakage_rate):
        parity_size = self.parity_size(len(data), leakage_rate)
        concat = data+parity
        res = ""
        try:
            coder = rs.RSCoder(n=len(data) + parity_size, k=len(data))
            res = coder.decode(concat)[0]
        except rs.RSCodecError:
            pass

        return res


class LDPCErrorCorrection(ErrorCorrection):
    def __init__(
        self,
        error_rate: float = 0.033,
        max_iter: int = 50,
        leakage_fraction: float = 0.5,  # fraction of bits leaked as syndrome
        row_weight: int = 3, 
    ):
        self.error_rate = error_rate
        self.max_iter = max_iter
        self.leakage_fraction = leakage_fraction
        self.row_weight = row_weight

    # def read_LDPC_H(self):
    #     path = "codes_ldpc/rate_0.5/block_4096_proto_2x4_12131025.qccsc.mtx"
    #     pathpairs_csv = "codes_ldpc/rate_adaptation/rate_adaption_2x4_block_4096.csv"
    #     if not Path(path).exists():
    #         print(f"[Matrix] file {path} not found.")
    #         exit(1)
    #     if not Path(pathpairs_csv).exists():
    #         print(f"[Matrix] file {pathpairs_csv} not found.")
    #         exit(1)
    #     H = mmread(path)
    #     # convert coo_matrix to regular numpy array
    #     H = H.tocsr()
    #     print(H[:10, :10])
    #     return H.toarray()



    def build_ldpc_H(self, n, m, dv=3, seed=None):
        rng = np.random.default_rng(seed)
        H = np.zeros((m, n), dtype=int)
        for j in range(n):
            # choose dv distinct rows for column j
            rows = rng.choice(m, size=min(dv, m), replace=False)
            H[rows, j] = 1
        return H

    def create_syndrome(self, data: str, leakage_rate: int) -> bytes:
        bits = np.fromiter((int(b) for b in data), dtype=np.uint8)
        n = len(bits)
        m = int(round(leakage_rate * n))
        H = self.build_ldpc_H(n, m, dv=self.row_weight, seed=m)
        H = np.array(H, dtype=int)
        syn = (H @ bits) % 2
        arr = [int(b) for b in syn]
        return arr

    def apply_syndrome(self, data: str, syndrome: bytes, leakage_rate: int) -> str:
        bits = np.fromiter((int(b) for b in data), dtype=np.uint8)
        n = len(bits)
        m = int(round(leakage_rate * n))
        H = self.build_ldpc_H(n, m, dv=self.row_weight, seed=m)

        s_A = np.array(syndrome, dtype=int)
        # Bob's own syndrome
        s_B = (H @ bits) % 2
        # Syndrome of error e = a XOR b
        s_err = (s_A ^ s_B).astype(np.uint8)

        # BP decoder on the error pattern
        m, n = H.shape
        bpd = BpDecoder(
            H,
            error_rate=self.error_rate,        # channel error rate (QBER)
            max_iter= round(leakage_rate * n * 4),  # max iterations for BP decoding
            input_vector_type="syndrome",
            bp_method="product_sum"
        )

        # In syndrome mode, decode(syndrome) returns an estimate of the error vector
        e_hat = bpd.decode(s_err)
        e_hat = np.array(e_hat, dtype=int)

        # Correct Bob's bits
        b_corr = (bits ^ e_hat).astype(int)

        # Success check: Bob's corrected bits should satisfy Alice's syndrome
        s_corr = (H @ b_corr) % 2
        success = np.all(s_corr == s_A)

        if success:
            return ''.join(str(int(b)) for b in b_corr)
        else:
            return ""
