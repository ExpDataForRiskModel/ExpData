import numpy as np
import time

from factor import log_likelihood_day
from factor import ols_solver

R = np.fromfile('Y.dat', dtype=np.float64)
R.shape = 1000, 4200
R = R.T

LF = np.fromfile('C1.dat', dtype=np.float64)
LF.shape = 1000, 10
LF = LF.T

half_life = 60

time_len = 4100
len_test = 100
len_prepare = 0
d = np.power(0.5, 1/half_life)
d_sqrt = np.sqrt(d)

hood = {1:[], 7:[], 10:[], 13:[], 14:[], 1003:[], 1005:[], 1007:[]}
steps = {1:[], 7:[], 10:[], 13:[], 14:[], 1003:[], 1005:[], 1007:[]}
total_t = {1:0, 7:0, 10:0, 13:0,14:0, 1003:0, 1005:0, 1007:0}
name = {1:'HFA_13', 10: 'FA_10', 13:'FA_13', 1003:'OLS_10', 1005:'FFA_10', 1007:'FFA_13'}
Y = R[ 0 : time_len, :]
k = np.ones(time_len)*d_sqrt
e = np.power(k, np.arange(time_len-1, -1, -1))
Y = np.dot(np.diag(e), Y)
S = np.dot(Y.T, Y)
sum_w = 1
for i in range(time_len):
    sum_w = sum_w*d + 1


for time_begin in range(len_prepare + len_test):
    Y = np.vstack((Y*d_sqrt, R[time_begin + time_len, :]))
    SSl = S/sum_w
    sum_w = sum_w*d + 1
    v = R[time_begin+time_len, :]
    v.shape = LF.shape[1],1
    S = S*d + np.dot(v, v.T)    
    SS = S/sum_w
    www = np.diag(LF.dot(SSl).dot(LF.T)) / np.diag(LF.dot(LF.T))
    
    t0 = time.time()
    from factor import factor_analysis_mod_new as HFA
    CC, RR, _, _ = HFA(SS, 13, 10, LF.T)
    t1 = time.time() - t0
    total_t[1] = total_t[1] + t1
    Cov0 = np.dot(CC, CC.T) + np.diag(RR)
    print('HFA factors 13: log_likelihood %lf' 
        % (log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))
    hood[1].append((log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))


    t0 = time.time()
    beta, res = ols_solver(Y, LF.T)
    Lb = np.dot(LF.T, beta.T)
    Cov0 = np.dot(Lb, Lb.T)/sum_w + np.diag(res/(sum_w))

    t1 = time.time() - t0
    total_t[1003] = total_t[1003] + t1
    print('OLS factor 10: log_likelihood %lf' 
        % (log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))
    hood[1003].append((log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))
    
    t0 = time.time()
    from factor import factor_analysis as FFA
    CC, RR, _, _, _ = FFA(SS, 10)
    t1 = time.time() - t0
    total_t[1005] = total_t[1005] + t1
    Cov0 = np.dot(CC, CC.T) + np.diag(RR)
    print('FFA factors 10: log_likelihood %lf' 
        % (log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))
    hood[1005].append((log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))

    t0 = time.time()
    from factor import factor_analysis as FFA
    CC, RR, _, _, _ = FFA(SS, 13)
    t1 = time.time() - t0
    total_t[1007] = total_t[1007] + t1
    Cov0 = np.dot(CC, CC.T) + np.diag(RR)
    print('FFA factors 13: log_likelihood %lf' 
        % (log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))
    hood[1007].append((log_likelihood_day(Cov0, R[time_begin + time_len + 1, :].T)))

    ntf = [10,13]
    from factor import factor_analysis_o as FA
    for i in ntf:
        t0 = time.time()
        c, r, l1, l2, s1 = FA(Y.T, SS, sum_w, i)
        cov = np.dot(c, c.T) + np.diag(r)
        t1 = time.time() - t0
        total_t[i] = total_t[i] + t1
        print('FA factors %d: log_likelihood %lf'
            % (i, log_likelihood_day(cov, R[time_begin + time_len + 1, :].T)))
        hood[i].append(log_likelihood_day(cov, R[time_begin + time_len + 1, :].T))

print("Over:")

for i in [1,10,13,1003,1005,1007]:
    print(name[i],np.mean(hood[i]), np.std(hood[i])/np.sqrt(len_test),total_t[i]/len(hood[i]))
