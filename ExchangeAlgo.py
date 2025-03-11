# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
import numpy.lib.arraysetops as setop

import matplotlib.pyplot as plt
import Solvers as solve

"""
ExchangeAlgo 

module implementing the discretized Exchange algorithm

Classes : ExchangeAlgoTree - abstract class, handling the refinement etc.
          FistaTree - subclass of ExchangeAlgoTree, using FISTA for solution
                      of discretized l1-problems
          Node - class for handling the cell nodes

@author: flinth
"""

"""
# class ExchangeAlgoTree
# 
#    abstract class for handling the 
#    tree structure needed for applying the Exchange algo
#    
#    Attributes:
#        root - the root of the tree of nodes
#        A    - a matrix modeling the current discretized measurement operator
#        x    - the current primal solution
#        p    - the current dual solution
#        b    - the data term
#        points - list of the corners of the cells. Order is consistent with x!
#        
#    Methods
#        exchangeStep - make an ExchangeStep
#        refine - refines the tree. Updates the matrix A and embeds 
#                 current x accordingly
#        printTree - prints all cells of the tree
#        setData - set b after creation.
#        Astarp(self,x) - evaluates A*p on x
#        nablaAstarp(self,x) - evaluates nabla (A*p) on x
#        hessianAstarp(self,x) - evaluates (A*p)''(x) on x
#
#    Abstract Methods
#        func(self,x) - the function for evaluating the measurement functions in points
#        refineRule(self,x) - the rule used for refining the tree
#        solve(self) - method used for solving l1-problem
#        nablafunc(self,x) - function for evaluating the gradient.
#        hessianfunc(self,x) - function for evaluating the Hessian

##
"""
class ExchangeAlgoTree(object):
    
    def __init__(self,b=0):
        self.root=Node(np.array([[0],[0]]),None,0)
        self.A=self.func(self.root.corners)
        self.b=b
        self.x=np.zeros((4,))
        self.p=np.zeros((self.A.shape[0],))
        self.points=self.root.corners
    
    def exchangeStep(self):
        self.solve()
        self.refine()
    
    def refine(self):
         self.refineRec(self.root)
        
    def refineRec(self,node):
       
        if self.refineRule(node):
            if node.isLeaf():
                
                node.addChildren()
               
                # update matrix
                for k in range(0, len(node.children)):
                    newp=node.children[k].corners
                    self.points,ind=np.unique(np.hstack((self.points,newp)),
                                              return_index=True,axis=1)
                    self.A=np.hstack((self.A,self.func(newp)))
                    self.A=self.A[:,ind]
                    self.x=np.hstack((self.x,np.zeros([4,])))
                    self.x=self.x[ind]
            else: 
                for c in node.children:
                    self.refineRec(c)
                    
    def setData(self,b):
        self.b=b
    
    def printTree(self):

        plt.scatter(self.points[0][:],self.points[1][:],1)
        plt.show()
        
    
        
    #def func(self,x) :
        # abstract function method.
        #   input: (2,n)-array x
        #   output: (m,n) matrix A(x)
        # should be able to handle arrays
 
    
    """
    Astarp(self,x)
      input: (2,1)-array x
      output: scalar A*p(x)
    """
    def Astarp(self,x):
        alpha=self.func(x).squeeze(1) #(m,)-array
        return self.p.dot(alpha)
    
    #def nablafunc(self,x) :
        # abstract function method.
        #   input: (2,n)-array x
        #   output: (m,n,2) matrix A(x)
        # does not need to be able to handle arrays
        
    """Astarp
    nablaAstarp(self,x)
      input: (2,1)-array x
      output: (2,)-array nabla A*p(x)
    """
    def nablaAstarp(self,x):
        
        beta=self.nablafunc(x).squeeze(1) # (m,2)-array
        return self.p.dot(beta)
        # abstract method
        # evaluates nabla A*p on x
        # does not necessarily need to handle arrays
    
    #def hessianfunc(self,x)
    # abstract method
    #   input: (2,n)-array x
    #   output: (m,n,2,2) matrix A(x)
    # does not necessarily need to handle arrays
    #   
    """
    nablaAstarp(self,x)
      input: (2,1)-array x
      output: (2,)-array nabla A*p(x)
    """
    def hessianAstarp(self,x):
        gamma=self.hessianfunc(x).squeeze(1) #(m,2,2)-array
        
        return np.einsum('i,ikl->kl',self.p,gamma)
        
        
        
   # def refineRule(self,node) :
        # abstract refinement rule
        # returns true if cell should be refined
        
    
    #def solve(self):
        # abstract l1-solution routine
        # should set x and p

"""
 class FistaTree

    An ExchangeTree using FISTA as the l1-solution routine
    
    Attributes:
        tol - tolerance parameter for FISTA
        tau - regularity parameter of the l1-problem
"""  
    
class FistaTree(ExchangeAlgoTree):
    def __init__(self,b=0,tol=1e-9,tau=1e6):
        self.tol=tol
        self.tau=tau
        super(FistaTree,self).__init__(b)
        
    def solve(self):
        #solve FISTA initialized at current x
        self.x=solve.FISTA(self.A,self.b,self.tau,self.x,self.tol)
        
        #dual vector inferred from duality conditions
        self.p=(self.b-self.A@self.x)*self.tau
  
"""
# class Node 
#
#   class to handle the cell nodes   
#
#   Attributes: 
#       parent - parent node
#       pos - position of corresponding lower left corner of cell
#       children - list of nodes representing the children of the node
#       level - the level of the node in the tree
#       corners -  a numpyarray, where the columns are the corners of
#                   the cell
#                   [:,1] -- [:,3]
#                      .      .
#                   [:,0] -- [:,2]

#   Methods
#       addChildren - adds children to a leaf
#       isLeaf - checks if the node is a leaf
#       
#
#
"""

        
class Node:
    
    def __init__(self,pos, parent,level):
         self.parent=parent;
         self.pos=pos;
         self.level=level
         self.corners=self.makeCorners()
         self.children =[];
         
        
    def addChildren(self):
        if not self.isLeaf():
            print('Tried to add children to non-leaf')
            print('Children not added')
            return
        
        for k in range(0,2):
            for l in range(0,2):
                newPos=np.array([[self.pos[0][0]+2**(-self.level-1)*k], 
                                [self.pos[1][0]+2**(-self.level-1)*l]])         
                self.children.append(Node(newPos,self,self.level+1))

    
    def isLeaf(self):
        return (len(self.children)==0) 

    def makeCorners(self):
        corners=self.pos
        
        corners = np.hstack((corners,self.pos+ 2**(-self.level)*np.array([[0],[1]])))
        corners = np.hstack((corners,self.pos+ 2**(-self.level)*np.array([[1],[0]])))
        corners = np.hstack((corners,self.pos+ 2**(-self.level)*np.array([[1],[1]])))
  
        return corners