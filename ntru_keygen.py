# Code adapted from:
# https://github.com/kpatsakis/NTRU_Sage/blob/master/ntru.sage
# Under GPL 2.0 license.
from numpy import array, zeros, identity, block
from scipy.linalg import circulant
from numpy.random import shuffle
from numpy import random
import numpy as np

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
        raise ZeroDivisionError
    else:
        return x % m


def modinvMat(M, q):
    n, m = M.shape
    assert m==n

    invs = q * [None]
    for i in range(1, q):
        try:
            invs[i] = modinv(i, q)
            assert((i*invs[i]) % q == 1)
        except ZeroDivisionError:
            pass

    R = block([[M, identity(n, dtype="long")]])
    #print(R)

    # i-th column
    for i in range(n):
        #print(i, q, R)
        # Find a row with an invertible i-th coordinate
        for j in range(i, n+1):
            if j == n:
                raise ZeroDivisionError

            # Normalize the row and swap it with row j
            if invs[R[j,i]] is not None:
                R[j] = (R[j] * invs[R[j,i]]) % q

                if j > i:
                    R[i], R[j] = R[j], R[i]
                break

        # Kill all coordinates of that column except at row j
        for j in range(n):
            if i==j: continue
            R[j] = (R[j] -  R[i] * R[j, i]) % q

    #print(i, R)

    Minv = R[:,n:]
    return Minv

def DiscreteGaussian(shape, sigmasq):
    sz = int(np.ceil(10*np.sqrt(sigmasq)))
    interval = range(-sz, sz+1)
    p = [np.exp(-x*x/(2*sigmasq)) for x in interval]
    p /= np.sum(p)
    return np.random.choice(interval, shape, p=p)


class NTRUEncrypt_Matrix:

    def gen_keys(self):
        while True:
            F = DiscreteGaussian((self.n,self.n), self.sigmasq)
            try:
                Finv = modinvMat(F, self.q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue
        G = DiscreteGaussian((self.n,self.n), self.sigmasq)
        H = Finv.dot(G) % self.q
        return H, F, G

    def __init__(self, n, q, sigmasq):
        self.n = n
        self.q = q
        self.sigmasq = sigmasq


class NTRUEncrypt_Circulant:

    def gen_keys(self):
        while True:
            f = DiscreteGaussian(self.n, self.sigmasq)
            F = circulant(f)
            try:
                Finv = modinvMat(F, self.q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue
        g = DiscreteGaussian(self.n, self.sigmasq)
        G = circulant(g)
        H = Finv.dot(G) % self.q
        return H, F, G

    def __init__(self, n, q, sigmasq):
        self.n = n
        self.q = q
        self.sigmasq = sigmasq

class NTRUEncrypt:

    def sample_ternary(self, ones, minus_ones):
        s = [1]*ones + [-1]*minus_ones + [0]*(self.n - ones - minus_ones)
        shuffle(s)
        return s


    def gen_keys(self):
        while True:
            f = self.sample_ternary(self.Df, self.Df-1)
            F = circulant(f)
            try:
                Finv = modinvMat(F, self.q)
                break
            except ZeroDivisionError:
                # print("failed inverse")
                continue

        g = self.sample_ternary(self.Dg, self.Dg)
        G = circulant(g)
        H = G.dot(Finv) % self.q
        return H, F, G

    def __init__(self, n, q, Df, Dg):
        self.n = n
        self.q = q
        self.q = q
        self.Df = Df
        self.Dg = Dg

def build_ntru_lattice(n, q, H):

    lambd = block([[q * identity(n, dtype="long") , zeros((n, n), dtype="long") ],
                   [     H            , identity(n, dtype="long") ] ])
    return lambd

def gen_ntru_instance_matrix(n, q, sigmasq, seed=None):
    random.seed(np.uint32(seed))
    ntru = NTRUEncrypt_Matrix(n, q, sigmasq)
    H, F, G = ntru.gen_keys()
    B = build_ntru_lattice(n, q, H)
    return B, F, G

def gen_ntru_instance_circulant(n, q, sigmasq, seed=None):
    random.seed(np.uint32(seed))
    ntru = NTRUEncrypt_Circulant(n, q, sigmasq)
    H, F, G = ntru.gen_keys()
    B = build_ntru_lattice(n, q, H)
    return B, F, G

def gen_ntru_instance(n, q, Df=None, Dg=None, seed=None):
    random.seed(np.uint32(seed))
    if Df is None:
        Df = n//3
    if Dg is None:
        Dg = n//3

    ntru = NTRUEncrypt(n, q, Dg, Df)
    H, F, G = ntru.gen_keys()
    B = build_ntru_lattice(n, q, H.transpose())
    return B, [F[0], G[0]]
