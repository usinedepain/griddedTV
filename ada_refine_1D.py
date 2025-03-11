#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:57:55 2022

@author: weiss
"""

#%% Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cvxpy as cp
import osqp

#%% Main structures for the program
class Cell1D: #The cell class (vertices, edge length, 2^D - children)
    def __init__(self, vertices = None, bound = 0):
        if vertices == None : 
            self.vertices = [0, 1]
        else:
            self.vertices = vertices
        self.length = np.abs(self.vertices[1] - self.vertices[0])
        self.is_active = 0
        self.bound = bound
        self.to_refine = False
    
    def get_vertices(self):
        return self.vertices

    def get_length(self):
        return self.length
        
    def get_children(self):
        midpoint = (self.vertices[0]+self.vertices[1])/2
        cell1 = Cell1D([self.vertices[0],midpoint], self.bound)
        cell2 = Cell1D([midpoint,self.vertices[1]], self.bound)
        return [cell1,cell2]
        
class Cell_partition: #The cell partition class (list of cells, vertices, refine, display)
    def __init__(self):
        omega = Cell1D()
        self.Omega = [omega]
    
    def get_vertices(self):
        vert_list = []
        for omega in self.Omega:
            vert_list.extend(omega.get_vertices())
        V = list(set(vert_list)) # to remove duplicates 
        V.sort()
        return V

    def refine_all_active(self):
        Omega_tmp = []
        for omega in self.Omega:
            if omega.to_refine:
                Omega_tmp.extend(omega.get_children())
            else:
                Omega_tmp.append(omega)
        self.Omega = Omega_tmp
            
    def refine(self, cell_index):
        omega = self.Omega[cell_index]
        children = omega.get_children() 
        self.Omega.pop(cell_index)
        self.Omega.extend(children)
        
    def get_min_edge_length(self):
        min = np.inf
        for omega in self.Omega:
            min = np.minimum(min, omega.length)
        return min
        
    def display_partition(self):
        plt.figure(0)
        V = self.get_vertices()
        for v in V:
            plt.plot(v,0,'bo')
        plt.show()
        
    def display_partition_bound(self):
        plt.figure(figsize=(20, 10))
        plt.clf()
        for omega in self.Omega:
            v = omega.vertices
            if omega.bound<1:
                plt.fill_between(v,omega.bound,color = [0.,0.,0.,0.3], edgecolor = [0.,0.,0.,0.4])
            else:
                plt.fill_between(v,omega.bound,color = [0.2,1.,0.2,0.5], edgecolor = [0.1,0.5,0.1])    
            plt.plot(v,[omega.bound,omega.bound],'k')
            plt.plot(v,[0,0],'b|')

class sampling_operator: # defines the operator and its upper-bound
    def __init__(self, M = 2, sigma = 0.1):
        self.M = M
        self.sigma = sigma
        self.Z = np.arange(0,M)/M
        self.c = 1/(self.sigma*np.sqrt(2*np.pi))

    def Gaussian(self, X):
        return self.c*np.exp(-X**2/(2*self.sigma**2))
    
    def A(self, X):
        return self.Gaussian(X[None,:]-self.Z[:,None])

    def A1(self, X):
        return -(X[None,:]-self.Z[:,None])/self.sigma**2 * self.A(X)

    def A2(self, X):
        return self.A(X)*(-1/self.sigma**2 + (X[None,:]-self.Z[:,None])**2/(self.sigma**4))
    
    def apply_A(self,X,alpha): #Evaluating Amu, with mu = sum alpha_s delta_{x_s}
        return self.A(X)@alpha

    def apply_AT(self,q,X): #Evaluating A^*q(X)
        return self.A(X).T@q

    def apply_A1T(self,q,X): #Evaluating A^*q(X)
        return self.A1(X).T@q

    def is_active(self,q,omega): #returns 1 if ||A^*q||_{Linf(omega)} > 1
        vert1 = omega.vertices[0]
        vert2 = omega.vertices[1]
        assert vert1<vert2
        dist = np.abs(self.Z-vert1)*(self.Z<vert1) + np.abs(self.Z-vert2)*(self.Z>vert2)
        Adist = self.Gaussian(dist)
        ub = self.sigma**2 + (dist+omega.length)**2
        kappa2 = 1./(self.sigma**4) * np.sum(np.abs(q) * Adist*ub)
        
        inf = np.inf
        for v1 in omega.vertices:
            A1=self.apply_AT(q,np.array([v1]))
            Ap1=self.apply_A1T(q,np.array([v1]))
            sup = 0
            for v2 in omega.vertices:   
                sup = np.maximum(sup, np.abs(A1+Ap1*(v2-v1)))
            inf = np.minimum(sup, inf)
        
        bound = inf + kappa2 * omega.length**2/2
        return bound>=1, bound
    
#This function detects pairs of Dirac masses with the same sign and create a single one from them
def sparsify_measure(X,alpha,eps):
    X = np.array(X)
    I = np.argsort(X)
    X = X[I]
    alpha = alpha[I]
    alpha[np.abs(alpha)<eps] = 0 #discarding weights that are too low
    Xs = []
    alphas = []
    n=0
    while (n<len(X)-1):
        if np.abs(alpha[n])>0:
            if alpha[n+1]*alpha[n]>0:
                Xs.append((X[n]*alpha[n]+X[n+1]*alpha[n+1])/(alpha[n]+alpha[n+1]))
                alphas.append(alpha[n]+alpha[n+1])
                n+=1
            else: 
                Xs.append(X[n])
                alphas.append(alpha[n])
        n+=1
    return Xs, alphas

def dist_Dirac(X1,X2):
    X1.sort()
    X2.sort()
    if len(X1) == len(X2):
        dist = np.max(np.abs(X1-X2))
    else:
        dist = np.inf
    return dist

def distH(X1,X2):
    d = 0
    for x2 in X2:
        dx2 = np.inf
        for x1 in X1:
            if dx2 > np.abs(x1-x2):
                dx2 = np.abs(x1-x2)
        if dx2 > d:
            d = dx2
    return d
        
#%% Initialization and problem definition
J = 10
M = 20
sigma = 2./M
eps_measure = 1e-5 #weights below this threshold are considered 0

#QP solver options
verbose = False #For the solver to tell us more
max_iters = 10000
solver =  "SCS" #CVXOPT" #""OSQP" # 
print(cp.installed_solvers())
eps_solver = 1e-8

Xs = np.array([1./3,4./6])
alphas = np.array([8,-9])

Pk = Cell_partition()
A = sampling_operator(M, sigma)
y = A.apply_A(Xs, alphas)

#it, |Vk|, primal, dual, distH 
Table = np.zeros((100,7))

#%% The core of the algorithm
fine_grid = np.linspace(0,1,10000)
qk = np.zeros(M)
it = 0
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
        is_active, bound = A.is_active(qk, omega)
        omega.bound = bound
        if is_active:
            Omegas.append(omega)
            max_length = np.maximum(max_length, omega.length)
    
    #%% Specific display for the paper
    Pk.display_partition_bound()
    ax = plt.gca()
    Aqk = np.abs(A.apply_AT(qk, fine_grid))
    plt.plot(fine_grid, Aqk, 'r--', linewidth=2)  
    plt.rc('font', size = 20)           
    name_save = "adarefine1D_%i.pdf"%it
    plt.savefig(name_save)  
    plt.show()
    bashCommand = "pdfcrop " + name_save + "; rm " + name_save 
    os.system(bashCommand)
    
    #%% Refining the largest cells
    for omega in Omegas:
        if omega.length == max_length:
            omega.to_refine = True
    Pk.refine_all_active()

    #%% Final displays
    toc = time.time()
    Table[it,:] = np.array([it,len(Vk),primal.value,dual.value, distH(Vk,Xs), dist_Dirac(Xks,Xs),toc - tic])
    print("Iteration %i -- #vertex: %i -- primal: %1.4e dual: %1.4e -- dist: %1.1e -- time: %1.1e" % (it,len(Vk),primal.value,dual.value,distH(Vk,Xs),toc - tic))
    it += 1
    
#%% Finally solving the primal
Vk = np.array(Pk.get_vertices())      # the vertices
Ak = A.A(Vk)                          # matrix Mx|Vk| 

np.savetxt("table_test.csv", Table[0:it-1,[0,1,2,4]], delimiter=' & ', fmt=['%i','%i','%1.5e','%1.1e'], newline=' \\\\\n')

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