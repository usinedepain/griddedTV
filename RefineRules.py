# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:21:59 2020

@author: flinth
"""

""" 
Different RefinementMethods

Syntax: (Tree,cell,...)

"""

import numpy as np


"""
UB1

First order  upper bound method

Additional input
    kappaN
"""
def UB1(T,cell,kappaN):
    
    corners=cell.corners
    
    for k in range(4):
        val=T.Astarp(corners[:,k].reshape((2,1)))
        if np.abs(val)+kappaN*2**(-cell.level)*np.sqrt(2)>1:
            return True
        
    
    return False

"""
UB2

Second order upper bound method

Additional input
    kappaH
"""
def UB2(T,cell,kappaH):
    
    direction = np.array([[1,1,-1,-1],[1,-1,1,-1]])
    corners=cell.corners
 
    
    
    side=2**(-cell.level-1)
    # loop through corners
    for k in range(4):
        #evaluate function and derivative of |A*p|^2 in point
        val=T.Astarp(corners[:,k].reshape((2,1)))
        beta=2*val*T.nablaAstarp(corners[:,k])
        
        # find directions that help us
        betamod=np.maximum(direction[:,k]*beta,0)
        
        if np.abs(val)**2 + side*betamod.sum() + 2*side**2*kappaH>1:
            return True #max was bigger than 1
    
    return False #no maximum was greater than 1

"""
UB3

Third order upper bound method

Additional input
    kappat
"""

def UB3(T,cell,kappat):
    
    #feasible search direction
    direction = np.array([[1,1,-1,-1],[1,-1,1,-1]])
    
    
    corners=cell.corners
 
    side=2**(-cell.level-1)
    
    for k in range(4):
        #evaluate function, derivative and Hessian of |A*p|^2
        corner=corners[:,k].reshape((2,1))
        val = T.Astarp(corner)
        nabbet= T.nablaAstarp(corner)
        
        beta = 2*val*nabbet
        H = 2*val*T.hessianAstarp(corner) + 2*nabbet.transpose()*nabbet
        val=val**2-1
        
        #"rotate" according to feasible direcitons
        beta = beta*direction[:,k]
        H = np.diag(direction[:,k],0)@H@np.diag(direction[:,k],0)
        
        if secondDegreeOnSquare(val,beta,H,direction[:,k],side,kappat): #check if max >1
            return True
                
        #all tests failed, go to next corner
    
     
    #all corners failed
    return False       
        

def secondDegreeOnSquare(val,beta,H,dirr,side,kappa):
#check interior
        
        nu=np.linalg.solve(H,-beta)*dirr
        
        if all(nu[i]>=0 for i in range(2)) and all (nu[i]<=side for i in range (2)): 
            # corner + nu lies in the cell
            #print((val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2))
            if (val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2)>0:
                
                return True # max larger than 0 was found.
            
            
        #check sides
        
        for j in range(2):
            l=1-j
            nu[j]=-(beta[j])/H[j,j] 
            nu[l]=0
            if nu[j]>=0 and nu[j]<=side:
                #nu-vector on side is feasible
                #print(np.abs(val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2)>1)
                #print(val+beta.dot(nu)+.5*nu.dot(H@nu)+kappa*side**3*2*np.sqrt(2))
                if (val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2)>0:
                    
                    return True # max larger than 1 was found.
            nu[j]=-(H[j,l]*side+beta[j])/H[j,j]
            nu[l]= side
            if nu[j]>=0 and nu[j]<=side:
                #nu-vector on side is feasible
                #print(np.abs(val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2)>1)
                #print(val+beta.dot(nu)+.5*nu.dot(H@nu)+kappa*side**3*2*np.sqrt(2))
                if (val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2)>0:
                    
                    return True # max larger than 1 was found.
        
        #check corners
        for j in range(2):
            for l in range(2):
                nu[0]=j*side
                nu[1]=l*side
                #(val+beta.dot(nu)+.5*nu.dot(H@nu)+kappa*side**3*2*np.sqrt(2))
                if (val+beta.dot(nu)+.5*nu.dot(H@nu))+kappa*side**3*2*np.sqrt(2)>0:
                    return True # max larger than 1 was found.   
        return False          
            
"""
Nabla1 

First order gradient method

Additional input
kappaH
""" 

def Nabla1(T,cell,kappaH):
    
    direction = np.array([[1,1,-1,-1],[1,-1,1,-1]])
    corners=cell.corners

 
    side=2**(-cell.level)
    
    for k in range(4): # loop through corners
        
        
        # first check if function is small
        val=T.Astarp(corners[:,k].reshape((2,1)))
        
        grad= T.nablaAstarp(corners[:,k].reshape((2,1)))
        beta=2*val*grad
        
        if localMinimum(val,beta,side,direction[:,k]) + 4*side**2*kappaH<1: #
            return False # then infimum is also. Return False
        
        
        # then check gradient firs
        
        
        if np.linalg.norm(grad)<kappaH*side: #gradient was big
            return True
        
       



"""
Nabla2

Second order gradient method

Additional input
kappaH
kappaT
""" 

def Nabla2(T,cell,kappaH,kappaT):
    direction = np.array([[1,1,-1,-1],[1,-1,1,-1]])
    corners=cell.corners

 
    side=2**(-cell.level-1)
    
    for k in range(4): #loop through corners
    
        #first check gradient
        #evaluate gradient, Hessian of A*p
        corner=corners[:,k].reshape((2,1))
        
        
        beta= T.nablaAstarp(corner)
        H = T.hessianAstarp(corner)
        
        #convert to ||nabla A^*p||^2
        
        val=beta.dot(beta)
        beta=2*np.transpose(H)@beta
        H-2*np.transpose(H)@H
        
        #"rotate" according to feasible direcitons
        beta = beta*direction[:,k]
        H = np.diag(direction[:,k],0)@H@np.diag(direction[:,k],0)
        
        #
        
        if secondDegreeOnSquare(-val,-beta,-H,direction[:,k],side,kappaT*side): #check if maximum of - function is larger than 0
            return True #condition triggered

        #check function

        val = T.Astarp(corner)
        
        beta=2*val*beta

        # check if function was small
        if localMinimum(val,beta,side,direction[:,k]) + 16*side**2*kappaH<1: #
            return False # then infimum is also. Return False
            print('infimum was small')
            
    
    return False 
    
def localMinimum(alpha,beta,side,dirr):
    betamod=np.minimum(dirr*beta,0)
    # check if function was small
    return alpha**2 + side*betamod.sum()
            
            
def projGradAscent(beta,H,x,side,dirr):
    g=H@x+beta # gradient
    diff=1
    while diff>1e-9:
        y=x
        topt=1
        
        g=H@x+beta
        if np.linalg.norm(g,np.inf)>0: # exact line search
            topt=beta.dot(g)/(H@g).dot(g)
            topt=np.maximum(topt,0)
            topt=np.minimum(topt,1)
        
        x=x+topt*g
        
        #project
        x=x*dirr
        x=np.maximum(x,0)
        x=np.minimum(x,side)
        x=x*dirr
        
        diff=np.linalg.norm(y-x,np.inf)

    return x         
