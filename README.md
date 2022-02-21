# NTRUFatigue
Experiments and Predictions for attacks on NTRU in the overstretched Regime


This repository contains the artifacts associated to the article

[DvW21] **NTRU Fatigue: How Stretched is Overstretched ?**
by _Léo Ducas and Wessel van Woerden_
https://eprint.iacr.org/2021/999

# Contributers

* Léo Ducas
* Wessel van Woerden

# Requirements

* [SageMath 9.3+](https://www.sagemath.org/)

Some older versions of SageMath contain a faulty FPLLL version which contains a bug that prevents our experiments from running successfully. Either use SageMath 9.3+ or manually install [fpylll](https://github.com/fplll/fpylll).

# Description of files
Short description of the files:
* bkz2_callback.py (BKZ2.0 including a hook to detect SKR and DSD events)
* estimator.sage (Estimator for SKR and DSD events)
* experiment.py (To run progressive BKZ until a SKR or DSD event is detected)
* find_fatigue.py (A soft binary search to find the fatigue point)
* ntru_keygen.py (NTRU instance generator)
* cli.py (helper file)
* paper/lucky_lift.sage (Preliminary analysis of an anecdotal Lucky Lift event)
* paper/claim3_5.ipynb (Symbolic algebra proof of Claim 3.5) 

# Estimator
The estimator estimator.sage requires sage. An example with parameters q=257, n=73, sigma^2=2/3 on a matrix NTRU instance, assuming progressive BKZ with 8 tours.
```
load("estimator.sage")
res=combined_attack_prob(257, 73, 2/3., ntru="matrix", fixed_tours=8)
print(res[0]) # average beta
print(res[1]) # probability of SKR event
print(res[2]) # probability of DSD event
print(res[3]) # Distribution of detection position kappa
```

# Experiments
The experiments use fpylll, and can be ran using sage or after installing fpylll manually.
For manual installation follow these [instructions](https://github.com/fplll/fpylll).
In addition to fpylll dependencies, the package scipy is also required.
And don't forget to active the fpylll environment by running `source ./activate' in the fpylll dir.

Additionally make sure that the following environment variables are set to 1, to prevent numpy from taking over all threads.

```
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Parameters:
```
--n X / -n X         # length of ntru secret, ntru lattice has dimension 2*n
--q X / -q X         # modulus
--circulant X        # keys have circulant structure (default 1)
--trials X / -t X    # do X trials per parameter set (default 1)
--workers X / -w X   # use X parallel threads (default 1)
--float_type X       # use floating point type X in {double, ld, dd, qd} for GSO in fplll. (Default "double")
--full_data X / -f X # show full data in CSV format (default 0)
```
Precision needs to be increased with n and q. Increase it if you encounter the infamous "infinte loop in Babai" error message. Using "dd" and "qd" requires the library `libqd` before compilation and installation of fplll/fpylll.

Parameters n and q can be given a single value (`-n 51`) , a list of values (e.g. `-n 73 89`), or an interval of prime integers (`-q 1300~1400p`). Key generation may be extremely slow if q or n isn't prime.


Example of an NTRU attack with n=127, and q ranging over all primes from 1300 to 1400
```
[sage/python] experiments.py --n 127 --q 1300~1400p --trials 2 --workers 2 --float_type dd --circulant 1 --tours 8 -f 1
```

Explanation of results:
```
DSD                # DSD (1) or SKR (0) event
DSD_lf             # Squared length ratio between detected dense vector and secret key
kappa              # Detection position in basis
beta               # Successful blocksize
DS_vol             # Log-volume of dense sublattice
foundbyLLL         # If the dense vector was inserted by intermediate LLL calls
slope              # Slope of log-profile on block [n-30:n+30) (or smaller for n<30) at moment of detection
sqproj_rel         # Squared length ratio between projection pi_kappa(v) and the detected dense vector v.
```

Example of finding the fatigue point for n=73 and 89
```
[sage/python] find_fatigue.py --n 73 89 --trials 2 --workers 2 --tours 8 --circulant 0 --float_type dd -f 1
```
