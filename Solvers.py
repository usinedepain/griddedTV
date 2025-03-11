# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 12:24:14 2020

@author: flinth
"""

""" Methods for solving

    min_x ||x||_1 + tau/2*||Ax-b||_2^2

Minimal inputs: (A, b, tau,...)
"""

import numpy as np

""" FISTA

    Additional arguments: 
        tol, algorithm breaks when ||y-xk||<tol
        x, starting point for primal
""" 
def FISTA(A,b,tau,x=None,tol=1e-6):
    if x is None:
        x=np.zeros((A.shape[1],))
    
    xold=x
    xnew=np.inf* np.ones_like(x)
    y=np.copy(x)
    
    lamb= tau*np.linalg.norm(A,2)**2 # lipschitz-constant of tau*A*A
    ATA = np.transpose(A)@A
    ATb= np.transpose(A)@b
    while np.linalg.norm(y-xnew)>tol:
        xnew=softthres(y-tau/lamb*(ATA@y-ATb),1/lamb)
        y=xnew+ .99*(xnew-xold)
        xold=xnew
        
    return xnew

def softthres(x,alpha):
    return np.sign(x)*np.maximum(np.abs(x)-alpha,0)
"""
    ADMM
    
    Additional arguments: 
        x, starting point for primal
        tol, algorithm breaks when ||y-xk|| < tol
    
"""
def ADMM(A,b,tau,x=0,tol=1e-6):
    
    if x==0:
        x=np.zeros((A.shape[1],))

    
    xold=x
    xnew=np.inf* np.ones_like(x)
    y=np.copy(x)
    u=np.zeros_like(x)
    
    rho=1
    M=rho*np.eye(A.shape[1])+tau*np.transpose(A)@A
    ATB= np.transpose(A)@b
    
    
    while np.linalg.norm(y-xnew)>tol:
        y=np.linalg.solve(M,tau*ATB-rho*(u-xold))
        xnew=softthres(y+u,1/rho)
        u=u+y-xnew
        xold=xnew 
        
    return xnew


            
            
        
        
    
    
    
        