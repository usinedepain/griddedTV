#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:27:14 2022

@author: degourna
"""
from ada_refine_dD import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cvxpy as cp




    
directories = ['1dtest', '1dtest_grad']
use_grads=[False,True]    

ks=[0,1]
for k in ks :
    directory=directories[k]
    use_grad=use_grads[k]
    
    
    if not os.path.isdir(directory):
        os.mkdir(directory)
    J = 20
    M = 20
    sigma = 2./M
    eps_measure = 1e-5 #weights below this threshold are considered 0
    
    #QP solver options
    verbose = False #For the solver to tell us more
    max_iters = 10000
    solver =  "SCS" #CVXOPT" #""OSQP" # 
    print(cp.installed_solvers())
    eps_solver = 1e-12
    
    Xs = np.array([1./3,4./6])
    alphas = np.array([8,-9])
    
    Pk = Cell_partition()
    A = sampling_operator(M, sigma)
    y = A.apply_A(Xs, alphas)
    
    #it, |Vk|, primal, dual, distH 
    Table = np.zeros((100,8))
    
    #%% The core of the algorithm
    fine_grid = np.linspace(0,1,10000)
    qk = np.zeros(M)
    it = 0
    
    Xopt= np.load('optimal_vector_1d.npy')
    Xopt=Xopt[2:].reshape(Xs.shape)
    while (Pk.get_min_edge_length()>2**(-J)):
        tic = time.time()
        
        Vk = Pk.get_vertices()      # the vertices
        Ak = A.A(np.array(Vk))                # matrix Mx|Vk| 
    
        #%% Solver for the dual problem
        qk = cp.Variable(M) 
        dual = cp.Problem(cp.Minimize((1/2)*cp.norm2(qk)**2 + y.T @ qk),
                         [Ak.T @ qk <= 1,
                          -Ak.T @ qk <= 1])
        dual.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
        qk = qk.value
        
        #%% Solver for the primal problem
        muk = cp.Variable(len(Vk)) 
        primal = cp.Problem(cp.Minimize((1/2)*cp.norm2(Ak@muk - y)**2 + cp.norm1(muk)))
        primal.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
        muk = muk.value
        Xks, alphaks = sparsify_measure(Vk,muk,eps_measure)
        # plt.plot(Xks,alphaks,'o')
        # plt.xlim([0, 1])
        # plt.title("Iteration %i" % (it))
        # plt.show()
    
        #%% Defining Omega_k^star    
        Omegas = []
        max_length = 0
        for ind, omega in enumerate(Pk.Omega):
            assert not omega.to_refine
            is_active_snd_order, bound_snd_order = A.Sampler.Second_order_bnd(qk, omega)
            is_active_grad, bound_grad = A.Sampler.Grad_first_order_bnd(qk, omega)
            
            omega.bound = bound_snd_order

            if is_active_snd_order and (is_active_grad or not use_grad):
                Omegas.append(omega)
                max_length = np.maximum(max_length, omega.length)
        
        
        
        #%% Specific display for the paper
        Pk.display_partition_bound(Omegas)
        ax = plt.gca()
        Aqk = np.abs(A.apply_AT(qk, fine_grid))
        plt.plot(fine_grid, Aqk, 'r--', linewidth=2)  
        plt.rc('font', size = 20)           
        name_save = directory+"/adarefine1D_%i.pdf"%it
        plt.savefig(name_save)  
        plt.show()
        
        #%% Refining the largest cells
        for omega in Omegas:
            if omega.length == max_length:
                omega.to_refine = True
        Pk.refine_all_active()
    
        #%% Final displays
        toc = time.time()
        Table[it,:] = np.array([it,len(Vk),primal.value,dual.value, distH(Vk,Xopt), dist_Dirac(Xks,Xopt),toc - tic,Pk.get_min_edge_length()])
        print("Iteration %i -- #vertex: %i -- primal: %1.4e dual: %1.4e -- dist: %1.1e -- time: %1.1e --min_length:%1.4e"  % (it,len(Vk),primal.value,dual.value,distH(Vk,Xs),toc - tic,Pk.get_min_edge_length()))
        it += 1
        
    #%% Finally solving the primal
    Vk = np.array(Pk.get_vertices())      # the vertices
    Ak = A.A(Vk)                          # matrix Mx|Vk| 
    
    np.savetxt(directory+"/table_test.csv", Table[0:it-1,[0,1,2,4]], delimiter=' & ', fmt=['%i','%i','%1.5e','%1.1e'], newline=' \\\\\n')
    
    # Solver for the primal problem
    muk = cp.Variable(len(Vk)) 
    primal = cp.Problem(cp.Minimize((1/2)*cp.norm2(Ak@muk - y)**2 + cp.norm1(muk)))
    primal.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
    muk = muk.value
    plt.plot(Vk,muk,'o')
    
    #%% Computing the exact solution 
    As = A.A(np.array(Xs))
    qs = cp.Variable(M) 
    dual_ex = cp.Problem(cp.Minimize((1/2)*cp.norm2(qs)**2 + y.T @ qs),
                      [As.T @ qs <= 1,
                       -As.T @ qs <= 1])
    dual_ex.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
    qs = qs.value
        
    mus = cp.Variable(len(Xs)) 
    primal_ex = cp.Problem(cp.Minimize((1/2)*cp.norm2(As@mus - y)**2 + cp.norm1(mus)))
    primal_ex.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
    mus = mus.value
    
    print("Exact  loc -- primal:  %1.6e -- dual:  %1.6e" %(primal_ex.value, dual_ex.value))
    print("Approx loc -- primal:  %1.6e -- dual:  %1.6e"%(primal.value, dual.value))