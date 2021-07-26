#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
import copy
from collections import OrderedDict
from math import sqrt, log
from random import randint

import six
from six.moves import range

import numpy as np
from numpy import array, zeros, block, transpose
from numpy.linalg import slogdet
from scipy.linalg import circulant
from scipy.stats import linregress

from fpylll import IntegerMatrix, BKZ, GSO
from fpylll.fplll.lll import LLLReduction
from bkz2_callback import BKZReduction

from cli import parse_args, run_all, pretty_dict
from ntru_keygen import gen_ntru_instance_matrix, gen_ntru_instance_circulant



def is_prime(x):
    return all(x % i for i in range(2, int(sqrt(x))+1 ))

def next_prime(x):
    return min([a for a in range(x+1, 2*x) if is_prime(a)])


class DenseSubLatticeFound(Exception):
    def __init__(self, b, kappa, lf, vcan, vloc):
        self.b = b
        self.kappa = kappa
        self.lf = lf
        self.vcan = vcan
        self.vloc = vloc

def sqnorm(a):
    return sum([x**2 for x in a])

def one_experiment(n, q, sigmasq, float_type, circ, tours, seed):

    if circ:
        B, F, G = gen_ntru_instance_circulant(n, q, sigmasq, seed)
    else:
        B, F, G = gen_ntru_instance_matrix(n, q, sigmasq, seed)
    A = IntegerMatrix.from_matrix([[int(x) for x in v] for v in B])
    M = GSO.Mat(A, float_type=float_type)
    lll = LLLReduction(M)
    lll()
    bkz = BKZReduction(M)
    M.update_gso()

    sk_norms = [sqnorm(F[i])+sqnorm(G[i]) for i in range(n)]
    sk_norm_min = min(sk_norms)
    sk_norm_max = max(sk_norms)
    if circ:
        Tfg = block([[F], [-G]])
    else:
        Tfg = block([[F], [-np.linalg.inv(F).dot(G).dot(F)]])

    for i in range(2*n):
        x = array(A[i], dtype='int').dot(Tfg)
        if not any(np.abs(x)>0.001):
            if sqnorm(A[i]) < sk_norm_min:
                continue
            return sqnorm(A[i])/sk_norm_max, 2


    def insert_callback(call_stack, solution):
        kappa, b = call_stack[-1]
        assert b==len(solution)
        # Write in cannonical basis
        v = (bkz.M.B[kappa:kappa+b]).multiply_left(solution)
        # babai-reduce it
        lift_fix = bkz.M.babai(v, 0, kappa)
        lift_can = (bkz.M.B[0:kappa]).multiply_left(lift_fix)
        v = array(v) - array(lift_can)

        # Test if in dense sublattice
        x = v.dot(Tfg)
        if any(np.abs(x)>0.001): return
        if sqnorm(v) < sk_norm_min: return
        lf = sqnorm(v) / sk_norm_max

        raise DenseSubLatticeFound(b, kappa, lf, v, solution)

    bkz.insert_callback = insert_callback

    if tours==None:
        tours = 8

    for blocksize in list(range(3, 2*n)):
        #print("Starting a BKZ tour, b=%d"%blocksize)

        par = BKZ.Param(blocksize,
                              strategies=BKZ.DEFAULT_STRATEGY,
                              #flags=BKZ.BOUNDED_LLL,
                              max_loops=tours)
        try:
            bkz(par)
        except DenseSubLatticeFound as err:
            return err.lf, err.b

    raise ValueError("Reached maximal blocksize")


def ntru_kernel(params=None, seed=None):
    if seed is None:
        params, seed = params
    # Pool.map only supports a single parameter
    params = copy.copy(params)

    n = params["n"]
    float_type = params["float_type"]
    circ = params["circulant"]
    tours = params["tours"]
    sigmasq = params["sigmasq"]
    verbose = params["verbose"]

    qmax = n**2+randint(0, n**2)
    qmin = n

    while True:
        q = next_prime(int((qmax+qmin)/2))
        if q >= qmax:
            q = next_prime(int(qmin)+1)
            if q >= qmax:
                break

        lf, b = one_experiment(n, q, sigmasq, float_type, circ, tours, 13371337*seed+q)
        dense_found = int(lf > 1.01)
        if verbose:
            print("%4d \t%4d \t%4d \t%.1f \t%5s"%(n, q, b, lf, str(dense_found)))
        # ignore large q that succeed with very small blocksize
        if b<=3 or dense_found:
            qmax = (qmax+q)/2 - 1
        else:
            qmin = (qmin+q)/2 + 1

    stats = {"logq": log((qmax+qmin)/2), "logn": log(n)}
    if verbose:
        print(stats)
    return stats


def ntru():
    """
    Attempt to solve an ntru challenge.

    """
    description = ntru.__doc__

    args, all_params = parse_args(description,
                                  float_type = "double",
                                  circulant = True,
                                  tours=8,
                                  sigmasq=0.667,
                                  verbose=False
                                  )

    stats = run_all(ntru_kernel, list(all_params.values()), # noqa
                    trials=args.trials,
                    workers=args.workers)

    # aggregate data for summary
    print("\n\n AVERAGE DATA\n\n ")
    for x,y in six.iteritems(stats):
        avg = OrderedDict()
        for z in y:
            for k in z:
                avg[k] = avg.get(k, 0) + z[k]/args.trials

        print(pretty_dict(eval(x)))
        keys = (y[0].keys())
        for k in keys:
            print("%14s, "%k, end="")
        print()
        for k in keys:
            print("%14s, "%("%.4f"%avg[k]), end="")
        print()

    if args.full_data:
        print("\n\n FULL DATA (csv format)\n\n ")

        for x,y in six.iteritems(stats):
            print(pretty_dict(eval(x)))
            keys = (y[0].keys())
            for k in keys:
                print("%14s, "%k, end="")
            print()
            for z in y:
                for k in keys:
                    print("%14s, "%("%.4f"%z[k]), end="")
                print()


if __name__ == "__main__":
    ntru()
