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
import osqp




    
directories = ['2dtest', '2dtest_grad']
use_grads=[False,True]    

ks=[0,1]
for k in ks :
    directory=directories[k]
    use_grad=use_grads[k]
    
    
    if not os.path.isdir(directory):
        os.mkdir(directory)
    J = 13
    M = 15
    sigma = 2./M
    eps_measure = 1e-5 #weights below this threshold are considered 0
   
   
    nb_variable_q=M**2
   
    #QP solver options
    verbose = False #For the solver to tell us more
    max_iters = 10000
    solver =  "SCS" #CVXOPT" #""OSQP" # print(cp.installed_solvers())
    eps_solver = 1e-8
   
    Xs =np.array([[1./3,4./6],[1./3,1./3],[4./6,4./6]])
    alphas = np.array([8,-9,5])
    
    Pk = Cell_partition(dim=2)
    A = sampling_operator(M, sigma,dim=2)
    y = A.apply_A(Xs, alphas)
    #Xopt= np.load(os.path.join(os.getcwd(),'optimal_vector_2d.npy'))
    #Xopt=Xopt[:3].reshape(Xs.shape)
    
    #it, |Vk|, primal, dual, distH 
    Table = []
    
    #%% The core of the algorithm
    n=1000
    xx = np.linspace(0,1,n)
    yy = np.linspace(0,1,n)
    fine_grid=np.zeros((n**2,2))
    X_grid,Y_grid=np.meshgrid(xx,yy)
    fine_grid[:,0],fine_grid[:,1] =X_grid.ravel(),Y_grid.ravel()
    grid_shape=X_grid.shape
    qk = np.zeros(nb_variable_q)
    it = 0
    
    while (Pk.get_min_edge_length()>2**(-J)):
        tic = time.time()
        
        Vk = Pk.get_vertices()      # the vertices
        Ak = A.A(np.array(Vk))                # matrix Mx|Vk| 
    
        #%% Solver for the dual problem
        qk = cp.Variable(M**2) 
        dual = cp.Problem(cp.Minimize((1/2)*cp.norm2(qk)**2 + y.T @ qk),
                         [Ak.T @ qk <= 1,
                          -Ak.T @ qk <= 1])
        dual.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
        qk = qk.value
        
        #%% Solver for the primal problem
        # muk = cp.Variable(len(Vk)) 
        # primal = cp.Problem(cp.Minimize((1/2)*cp.norm2(Ak@muk - y)**2 + cp.norm1(muk)))
        # primal.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
        # muk = muk.value
        # Xks, alphaks = sparsify_measure(Vk,muk,eps_measure)
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
        Pk.refine_all_active()
        plt.axis('off')
    #    ax = plt.gca()
        Aqk = np.abs(A.apply_AT(qk, fine_grid))
        plt.contour(X_grid,Y_grid,Aqk.reshape(grid_shape),levels=[0.75,0.9],colors='r',linestyles=['dotted','dashed'])#    plt.plot(fine_grid, Aqk, 'r--', linewidth=2)  
        plt.contourf(X_grid,Y_grid,Aqk.reshape(grid_shape),levels=[0.,1.],colors=[[0.,0.,0.,0.],[1.,0.,0.,0.5]],extend='max')
        name_save = directory+"/ada_%i.pdf"%it
        print(name_save)
        plt.savefig(name_save)  
        plt.show()
        
        
        #%% Refining the largest cells
        for omega in Omegas:
           if omega.length == max_length:
                omega.to_refine = True
        Pk.refine_all_active()
    
        #%% Final displays
        toc = time.time()
        to_save={'it':it,'nb_vert':len(Vk)}
        to_save['primal_value']=0.
        to_save['dual_value']=dual.value
        to_save['dists']= distH(Vk,Xs)
        to_save['time']=toc-tic
        Table.append(to_save)
        s="Iteration %i -- #vertex: %i -- primal: %1.4e dual: %1.4e -- dist: %1.1e -- time: %1.1e "
        s=s%(it,len(Vk),0.,dual.value,distH(Vk,Xs),toc - tic)
        print(s)
        it += 1
        
    #%% Finally solving the primal
    Vk = np.array(Pk.get_vertices())      # the vertices
    Ak = A.A(Vk)                          # matrix Mx|Vk| 
    
    to_csv=np.array([[s['it'],s['nb_vert'],s['dual_value'],s['dists']]  for s in Table])
    np.savetxt(directory+"/table_test.csv", to_csv, delimiter=' & ', fmt=['%i','%i','%1.5e','%1.1e'], newline=' \\\\\n')
    
    # Solver for the primal problem
    muk = cp.Variable(len(Vk)) 
    primal = cp.Problem(cp.Minimize((1/2)*cp.norm2(Ak@muk - y)**2 + cp.norm1(muk)))
    primal.solve(solver=solver, eps=eps_solver, verbose = verbose, max_iters = max_iters)
    muk = muk.value
    
    #%% Computing the exact solution 
    As = A.A(np.array(Xs))
    qs = cp.Variable(nb_variable_q) 
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