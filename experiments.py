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


def sqnorm(a):
    return sum([x**2 for x in a])


class DenseSubLatticeFound(Exception):
    def __init__(self, call_stack, lf, vcan, vgs, vloc, gso):
        self.call_stack = call_stack
        self.lf = lf
        self.vcan = vcan
        self.vgs = vgs
        self.vloc = vloc
        self.gso = gso



def ntru_kernel(params, seed=None):
    if seed is None:
        params, seed = params
    # Pool.map only supports a single parameter
    params = copy.copy(params)

    n = params["n"]
    q = params["q"]
    float_type = params["float_type"]
    circ = params["circulant"]
    tours = params["tours"]
    sigmasq = params["sigmasq"]

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
    DS_vol = slogdet(transpose(Tfg).dot(Tfg))[1]/2.

    def insert_callback(call_stack, solution):
        kappa, b = call_stack[-1]
        assert b==len(solution)
        # Write in cannonical basis
        v = (bkz.M.B[kappa:kappa+b]).multiply_left(solution)
        # babai-reduce it
        lift_fix = bkz.M.babai(v, 0, kappa)
        lift_can = (bkz.M.B[0:kappa]).multiply_left(lift_fix)
        v = array(v) - array(lift_can)

        vg = bkz.M.from_canonical(v, start=0, dimension=kappa+b)
        vgs = [vg[i]**2 * bkz.M.r()[i] for i in range(kappa+b)]

        # Test if in dense sublattice
        x = v.dot(Tfg)
        if any(np.abs(x)>0.001): return
        if sqnorm(v) < sk_norm_min: return
        lf = sqnorm(v) / sk_norm_max

        raise DenseSubLatticeFound(call_stack, lf, v, vgs, solution, bkz.M.r())

    bkz.insert_callback = insert_callback

    for blocksize in list(range(2, n+1)):

        if tours==None:
            tours = 8

        par = BKZ.Param(blocksize,
                              strategies=BKZ.DEFAULT_STRATEGY,
                              flags=BKZ.BOUNDED_LLL,
                              max_loops=tours)
        try:
            bkz(par)
        except DenseSubLatticeFound as err:
            kappa, b = err.call_stack[0]
            assert (b==blocksize) or (kappa+b == 2*n)
            subkappa, subb = err.call_stack[-1]
            vsz = np.sum(np.abs(err.vloc))
            logr = [log(x)/2. for x in err.gso]
            d=len(err.gso)

            slope_part = min(30, n)
            l = n-slope_part
            r = n+slope_part
            slope=-linregress(range(l, r), logr[l:r]).slope
            byLLL = vsz<1.5

            sq_proj_sz = np.sum(err.vgs[kappa:kappa+b])/np.sum(err.vgs[:kappa+b])

            if (err.lf>1.):
                stats = {"DSD": True,   "DSD_lf": err.lf,  "kappa": kappa, "beta":blocksize, "DS_vol":DS_vol, "foundbyLLL": byLLL, "slope": slope, "sqproj_rel": sq_proj_sz}
            else:
                stats = {"DSD": False,   "DSD_lf": 1., "kappa": kappa, "beta":blocksize, "DS_vol": DS_vol, "foundbyLLL": byLLL, "slope": slope, "sqproj_rel": sq_proj_sz}

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
                                  sigmasq=0.667
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
