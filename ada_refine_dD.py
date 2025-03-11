#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:57:55 2022

@author: weiss
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import time

#%% Main structures for the program
class CelldD: #The cell class (vertices, edge length, 2^D - children)
    def __init__(self,x_0=None,length=None,cube = None,dim=1):
        self.info={} ## Empty dictionnary, used to store things for the end user of the library
        self.dim=dim
        self.to_refine = False

        if x_0 is None or length is None :
            self.x_0=np.zeros(self.dim)
            self.length=1
        else :
            self.x_0=x_0
            self.length=length
        if cube is None :
            # cube is the list of the vertices of the unit cube dimension d
            # it is of size 2**d x d
            from itertools import product
            self.cube=np.array(list(product((0, 1), repeat=self.dim))) 
        else :
            self.cube=cube
    
    def get_vertices(self):
        verts=self.x_0+self.length*self.cube
        return [tuple(v) for v in verts]

    def get_length(self):
        return self.length
        
    def get_children(self):
        children=[]
        half_len=0.5*self.length
        for v in self.cube :
            child=CelldD(x_0=self.x_0+half_len*v,length=half_len,cube=self.cube,dim=self.dim)
            children.append(child)
        return children
        
class Cell_partition: #The cell partition class (list of cells, vertices, refine, display)
    def __init__(self,dim=1):
        self.dim=dim
        omega = CelldD(dim=self.dim)
        self.Omega = [omega]
    
    def get_vertices(self):
        vert_list = []
        for omega in self.Omega:
            vert_list.extend(omega.get_vertices())
        V = list(set(vert_list)) # to remove duplicates 
        V.sort()
        return V
    
    def reset(self):
        for omega in self.Omega:
            omega.to_refine=False
            omega.bound = np.inf
    
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
        
    def display_partition(self,cells=False):
        if self.dim==1 :
            V = self.get_vertices()
            for v in V:
                plt.plot(v,0,'bo')
            if cells :
                for i,omega in enumerate(self.Omega):
                    middle=omega.x_0+0.5*omega.length
                    plt.text(middle[0],0.01,str(i))
        if self.dim==2 :
            V = self.get_vertices()
            for v in V:
                plt.plot(v[0],v[1],'bo')
            plt.axis('equal')
            if cells :
                for i,omega in enumerate(self.Omega):
                    middle=omega.x_0+0.5*omega.length
                    plt.text(middle[0],middle[1],str(i))
        plt.show()
        
    def display_partition_bound(self,Omegas):
        plt.figure(figsize=(20, 10))
        plt.clf()
        if self.dim==1 :
            for omega in self.Omega:
                v = np.array(omega.get_vertices()).ravel()
                if omega in Omegas :
                    plt.fill_between(v,omega.bound,color = [0.2,1.,0.2,0.5], edgecolor = [0.1,0.5,0.1])    
                else :
                    plt.fill_between(v,omega.bound,color = [0.,0.,0.,0.3], edgecolor = 'k')
                plt.plot(v,[omega.bound,omega.bound],'k')
                plt.plot(v,[0,0],'b|')
        if self.dim==2 :
            from matplotlib.patches import Polygon
            ind=[0,1,3,2]
            for omega in self.Omega:
                v = np.array(omega.get_vertices())[ind]
                if omega in Omegas:
                    p = Polygon(v, facecolor = [0.2,1.,.2,0.5],edgecolor=[0.1,0.5,0.,0.1])
                else:
                    p = Polygon(v, facecolor = [0.,0.,0.,0.1],edgecolor=[0.,0.,0.,0.1])
                ax=plt.gca()
                ax.add_patch(p)
            plt.axis('equal')
                    
class Gaussian_measurement :
    def __init__(self,Z,sigma,dim=1) :
        # On input, Z is a vector of size dim
        # On input, sigma is a real number
        self.dim=dim
        if not Z.shape==(self.dim,) :
            print('Warning expected shape ({:d},), found '.format(self.dim),Z.shape)
            assert False
        if not isinstance(sigma,float) :
            print('Expected a float, got something else')
        self.sigma = sigma
        self.Z = Z
        self.c = 1/(self.sigma*np.power(2*np.pi,self.dim/2))
    def Gaussian(self, X):
        return self.c*np.exp(-X**2/(2*self.sigma**2))
    
    def Value(self, X): 
        # X is assumed to be of size (nbpts,dim)
        # Z is assumed to be of size (dim,)
        # returns a vector of size  (nbpts,)

        if self.dim ==1 :
            return self.c*np.exp(-(X-self.Z)**2/(2*self.sigma**2))
        tmp=self.c*np.exp(-np.sum((X-self.Z)**2,axis=1)/(2*self.sigma**2))
        return tmp
    
    def Grad(self, X) :
        # X is assumed to be of size (nbpts,dim)
        # Z is assumed to be of size (dim)
        # returns a vector of size  (nbpts,dim)
        if self.dim ==1 :
            return -(X-self.Z)/self.sigma**2 * self.Value(X)
        return -(X-self.Z)/self.sigma**2 * self.Value(X)[:,None]
    
    def Hessian(self,X):
        # X is assumed to be of size (nbpts,dim)
        # Z is assumed to be of size (dim)
        # returns a vector of size  (nbpts,dim,dim)
        W= self.Z-X
        return self.Value(X)[:,None,None]*(-1/self.sigma**2*np.eye(self.dim)[None,:,:] + 1/self.sigma**4*W[:,:,None]*(W[:,None,:]))
        

    def Second_order_bnd(self, omega): 
        dist=np.abs(self.Z-omega.x_0-0.5*omega.length)-0.5*omega.length
        dist=np.sqrt(np.sum(np.maximum(0,dist)**2))
        Adist = self.Gaussian(dist)
        moddist  = dist+ np.sqrt(self.dim)*omega.length
        ub = np.maximum(self.sigma**2, moddist**2)
        kappa2 = 1./(self.sigma**4) * Adist*ub
        return kappa2 
    
    def Third_order_bnd(self,omega):
        dist=np.abs(self.Z-omega.x_0-0.5*omega.length)-0.5*omega.length
        dist=np.sqrt(np.sum(np.maximum(0,dist)**2))
        moddist  = dist+ np.sqrt(self.dim)*omega.length
        Adist = self.Gaussian(dist)
        
        ub = 3*moddist*self.sigma**2 + moddist**3 #Old version
        ub = moddist*np.maximum(3*self.sigma**2, moddist**2) # slightly improved version
        
        kappa3 = 1./self.sigma**6 * Adist*ub 
        
        return kappa3
        
    
class Sampling_operator :
    def __init__(self,list_measurements) :
        self.measures=list_measurements
        self.nb_measures=len(self.measures)
        
        
    def A(self, X):
        # Assume that X is (nb_points,dim)
        # returns an array of size (nb_measurements,nb_points)
        A=np.zeros((self.nb_measures,X.shape[0]))
        for (i,m) in enumerate(self.measures) :
            A[i,:]=m.Value(X).ravel()
        return A      
    def apply_A(self,X,alpha): #Evaluating Amu, with mu = sum alpha_s delta_{x_s}
          return self.A(X)@alpha
    def apply_AT(self,q,X): #Evaluating A^*q(X)
          return self.A(X).T@q
    def apply_grad_AT(self,q,X) :
        grad=np.zeros(X.shape)
        for (m,qm) in zip(self.measures,q) :
            grad+=qm*m.Grad(X).reshape(X.shape)
        return grad
    
    def apply_hess_AT(self,q,X):
        hess=np.zeros((X.shape[0],X.shape[1],X.shape[1]))
        for (m,qm) in zip(self.measures,q) :
            hess+=qm*m.Hessian(X)
        return hess
        
    def Second_order_bnd(self,q,omega): # returns value of second order upper bound
        # Computes kappa2
        kappa2=0
        for (m,qm) in zip(self.measures,q) :
            kappa2+=np.abs(qm)*m.Second_order_bnd(omega)
        
        inf = np.inf
        for v in omega.get_vertices():
            A=self.apply_AT(q,np.array([v]))
            Ap1=self.apply_grad_AT(q,np.array([v]))
            sup = 0
            for v2 in omega.get_vertices():   
                diff=np.array(v2)-np.array(v)
                sup = np.maximum(sup, np.abs(A+np.dot(Ap1,diff)) +kappa2*np.sum(diff**2)/2)
            inf = np.minimum(sup, inf)       
        return inf >=1, inf
    
    def Grad_first_order_bnd(self,q,omega): # returns value of second order upper bound
        # Computes kappa2
        kappa2=0
        for (m,qm) in zip(self.measures,q) :
            kappa2+=np.abs(qm)*m.Second_order_bnd(omega)
        sup = -np.inf
        for v in omega.get_vertices():
            Ap1=self.apply_grad_AT(q,np.array([v]))
            sup = np.maximum(sup, np.linalg.norm(Ap1) -kappa2*np.sqrt(omega.dim)*omega.length)
        return sup < 0.,sup
    
    def Third_order_info(self,q,omega): # returns value of third order upper bound
    
        kappa3 = 0 
        for (m,qm) in zip(self.measures,q):
            kappa3+=np.abs(qm)*m.Third_order_bnd(omega)
        
        X = np.array(omega.get_vertices())
        
        adjuster = 1 - 2*omega.cube # 2**d, d
        vals = self.apply_AT(q,X) # 2**d
        grads = self.apply_grad_AT(q,X) # (2**d,dim)
        hess = self.apply_hess_AT(q,X) # (2**d,dim,dim)
        
        return vals, grads * adjuster, hess*adjuster[:,:,None]*adjuster[:,None,:],kappa3
        
        
    
class sampling_operator :
    def __init__(self, M = 2, sigma = 1.,dim=1):
        if dim==1 :
            Z=np.arange(0,M)/M
        if dim==2 :
            x=np.arange(0,M)/M
            y=np.arange(0,M)/M
            X,Y=np.meshgrid(x,y)
            Z=np.zeros((M**2,2))
            Z[:,0]=X.ravel()
            Z[:,1]=Y.ravel()
        l_m=[]     
        for z in Z :
            l_m.append(Gaussian_measurement(np.array([z]).reshape(dim),sigma,dim=dim))
        self.Sampler=Sampling_operator(l_m)

    def A(self,X) :
        return self.Sampler.A(X)
    def apply_A(self,X,alpha): #Evaluating Amu, with mu = sum alpha_s delta_{x_s}
          return self.A(X)@alpha
    def apply_AT(self,q,X): #Evaluating A^*q(X)
          return self.A(X).T@q
    def is_active(self,q,omega) :
        is_active,bound= self.Sampler.Second_order_bnd(q,omega)
        return is_active, bound
    
    def project(self,x):

        return np.maximum(np.minimum(x,1),0)
    
    def deg_one_opt(self,beta,gamma):
        
        Sampler,dim,nb_measures,y
        # interior
        beta = beta.ravel()
        gamma = gamma.ravel()
        
        v = self.project(-beta/gamma)
        
        vals=np.array([0,beta*v+1/2*gamma*v**2, beta + 1/2*gamma])
        
        return (np.min(vals),np.max(vals))
        
        
        
    
    def deg_two_opt(self,beta,gamma):
        d = np.tile(np.array((0,1)),(beta.shape[0],1))
        zz = np.zeros(d.shape)
        oo = np.ones(d.shape)
        
        valgamma=gamma
        gamma = .5*(gamma+np.swapaxes(gamma,-2,-1))
        
        #interior
        beta = np.expand_dims(beta,-1)
        w = np.linalg.solve(gamma,-beta)
        v = self.project(w)
      
        m = ((beta*v + 1/2 *v*(valgamma@v)).sum(1)).squeeze(-1)
        M = ((beta*v + 1/2 *v*(valgamma@v)).sum(1)).squeeze(-1)
        
        # vertices
       
        gamma = np.expand_dims(gamma,-1)
        
        v = np.stack((np.concatenate((d,d),1),np.concatenate((zz,oo),1)))
        v = np.swapaxes(v,0,1)
        
        m = np.minimum(m,np.min((beta*v + 1/2 *v*(valgamma@v)).sum(1),1))
        M = np.maximum(M,np.max((beta*v + 1/2 *v*(valgamma@v)).sum(1),1))
        
        # boundary    
        
        s0 = (-beta[:,0]-d*gamma[:,0,1])/gamma[:,0,0]
        s1 = (-beta[:,1]-d*gamma[:,1,0])/gamma[:,1,1]
        
        v = np.stack((np.concatenate((s0,d),1),np.concatenate((d,s1),1)))
        v = self.project(np.swapaxes(v,0,1))
        
        m = np.minimum(m,np.min((beta*v + 1/2 *v*(valgamma@v)).sum(1),1))
        M = np.maximum(M,np.max((beta*v).sum(1) + 1/2 *(v*(valgamma@v)).sum(1),1))

        
        #print("============")
        return (m,M)
    

    
    def check_active_cells(self,q,part, grad = False,func=False):
        
        L = len(part.Omega)
        d = part.dim
        alpha = np.zeros((L,2**d))
        beta = np.zeros((L,2**d,d))
        dbeta = beta
        gamma = np.zeros((L, 2**d,d,d))
        dgamma = gamma
        kappa = np.zeros((L,1))
        gkappa = np.zeros((L,1))
        for (k,omega) in enumerate(part.Omega):
            v,g,h,dd = self.Sampler.Third_order_info(q,omega)
            alpha[k,:] = v
            beta[k,:,:] = g *omega.length
            gamma[k,:,:,:] = h * omega.length**2
            kappa[k] = dd*d**1.5*omega.length**3/6
            
            dbeta[k,:,:] = g
            dgamma[k,:,:,:] = h *omega.length 
            gkappa[k] = dd*d*omega.length**2/2
        
        if func:
        
            
            
            alpha = np.reshape(alpha,(L*2**d,))
            beta = np.reshape(beta,(L*2**d,d))
            gamma = np.reshape(gamma,(L*2**d,d,d))
        
            if d ==1:
                m,M = self.deg_one_opt(beta, gamma)
            elif d==2:
                    m, M = self.deg_two_opt(beta,gamma)
            else:
                        print('Warning, not implemented for this dimension')
                        assert False
        
            m+=alpha
            M+=alpha
            m = np.maximum(np.abs(m),np.abs(M))
            m = np.reshape(m,(L,2**part.dim))
            m = np.min(m+kappa,1)
        #print(m)
        #print(kappa)
            for (k,omega) in enumerate(part.Omega):
                
                if omega.bound>1 and m[k]>1:
                    omega.bound = m[k]
                    omega.to_refine = True
                if omega.bound<1 or m[k]<1:
                    omega.bound = np.minimum(omega.bound,m[k])
                    omega.to_refine=False
        
        if grad:
            galpha = (dbeta*dbeta).sum(-1)
            gbeta = 2*np.swapaxes(dgamma,-2,-1)@np.expand_dims(dbeta,-1)
            ggamma = 2* np.swapaxes(dgamma,-2,-1)@dgamma
            
            gbeta = np.reshape(gbeta,(L*2**d,d))
            ggamma = np.reshape(ggamma,(L*2**d,d,d))
            m,_ = self.deg_two_opt(gbeta,ggamma)
            m = np.reshape(m,(L,2**part.dim))
        
            m += np.reshape(galpha ,(L,2**part.dim))
        
            m = m.max(1)
        
            
            for (k,omega) in enumerate(part.Omega):
                if omega.to_refine:
                    omega.to_refine = False
                    omega.bound=.9
                    if m[k] <= gkappa[k]**2:
                        omega.bound=1.1
                        omega.to_refine = True
         
        # only refine the largest cells!            
        max_length = -np.inf  
        nbr = 0 
        for omega in part.Omega:
            if omega.to_refine:
                nbr += 1 # count the number of candidates
                max_length = np.maximum(max_length, omega.length)
                
        for omega in part.Omega:
            
            if omega.length<max_length:
                omega.to_refine = False
        
        return nbr

class Basis_Pursuit():
    def __init__(self,Sampler,dim,nb_pts,y) :
        self.Sampler=Sampler # Measurements
        self.dim=dim # dimension
        self.nb_pts=nb_pts # number of points
        self.y=y # target data
    def vec2meas(self,vec) : # transform a vector into a measure
        meas=(vec[:self.nb_pts],vec[self.nb_pts:].reshape(self.nb_pts,self.dim))
        return meas
    def meas2vec(self,meas) :
        (alpha,X)=meas        
        vec=np.zeros((1+self.dim)*self.nb_pts)
        vec[:self.nb_pts]=alpha
        vec[self.nb_pts:]=X.ravel()
        return vec
    def evaluate(self,meas) :
        (alpha,X)=meas
        y_tilde=self.Sampler.apply_A(X,alpha)
        cost=0.5*np.sum((self.y-y_tilde)**2)
        cost+=np.sum(np.abs(alpha))
        return cost
    def grad(self,meas) :
        (alpha,X)=meas
        AX=self.Sampler.A(X)
        y_tilde=AX@alpha-self.y
        galpha=AX.T@y_tilde+alpha/np.abs(alpha)
        gX=np.zeros(X.shape)
        for (m,mult) in zip(self.Sampler.measures,y_tilde) :
            gX+=m.Grad(X).reshape(X.shape)*mult*alpha[:,None]   
        return (galpha,gX)
    def first_order_search(self,v0,step=1.e-5,tol=1.e-6,nitermax=1e3) :
        v=np.copy(v0)
        grad=B.meas2vec(B.grad(B.vec2meas(v)))
        grad_l=[np.linalg.norm(grad)]
        cost_l=[B.evaluate(B.vec2meas(v))]
        niter=0
        while np.linalg.norm(grad)>tol and niter <nitermax :
            v=v-step*grad
            grad=B.meas2vec(B.grad(B.vec2meas(v)))
            normgrad=np.linalg.norm(grad)
            cost=B.evaluate(B.vec2meas(v))
            grad_l.append(normgrad)
            cost_l.append(cost)
            niter+=1
            if niter % 100 == 0 :
                print('iter %i --cost %1.3e --norm_grad %1.3e'%(niter,cost,normgrad))
        plt.semilogy(np.array(grad_l))
        plt.show()
        plt.plot(cost_l)
        plt.show()
        return v, niter
        
        
    
    
            
    
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
            d12=np.linalg.norm(x1-x2)
            if dx2 > d12:
                dx2 = d12
        if dx2 > d:
            d = dx2
    return d

    
if __name__ == '__main__' :
    ### TESTING A Gaussian measurement in 2D
    if False :
        P=Cell_partition(dim=2)
        P.refine(0)
        P.refine(3)
        P.refine(1)
        P.refine(0)
        P.display_partition(cells=True)
        Vk = np.array(P.get_vertices())
        Z=[0.4,0.3]
        sigma=0.2
        G=Gaussian_measurement(np.array(Z),sigma)
        V=G.Value(Vk)
        Grad=G.Grad(Vk)
        list_bnd=[]
        for omega in P.Omega :
            list_bnd.append(G.Second_order_bnd(omega))
        list_bnd=np.array(list_bnd)

    ### TESTING A Gaussian measurement in 1D
    if False:
        P=Cell_partition(dim=1)
        P.refine(0)
        P.refine(1)
        P.refine(1)
        P.refine(0)
        P.display_partition(cells=True)
        Vk = np.array(P.get_vertices())
        Z=[0.4]
        sigma=0.2
        G=Gaussian_measurement(np.array(Z),sigma)
        V=G.Value(Vk)
        Grad=G.Grad(Vk)
        list_bnd=[]
        for omega in P.Omega :
            list_bnd.append(G.Second_order_bnd(omega))
        list_bnd=np.array(list_bnd)
        

    ### TESTING THE SAMPLING OPERATOR IN 2D
    if False :
        J=12
        M=10
        Z=np.arange(0,M)/M
        sigma=2./M*np.ones_like(Z)
        A=Sampling_operator(Z,sigma)
        P=Cell_partition(dim=2)
        P.refine(0)
        P.refine(3)
        P.refine(1)
        P.refine(0)
        P.display_partition(cells=True)
        Vk = np.array(P.get_vertices())
        
        Ak=A.A(np.array(Vk))
    
    
    ### TESTING THE SAMPLING OPERATOR IN 1D
    if False :
        J=12
        M=10
        Z=np.arange(0,M)/M
        sigma=2./M*np.ones_like(Z)
        A=Sampling_operator(Z,sigma)
        P=Cell_partition(dim=1)
        P.refine(0)
        P.refine(0)
        P.refine(1)
        P.refine(0)
        P.display_partition(cells=True)
        Vk = np.array(P.get_vertices())
        
        Ak=A.A(np.array(Vk))
        
    ### TESTING THE SAMPLING OPERATOR IN 2D
    if False:
        M=10
        
        x = np.linspace(0,1)
        
        X,Y = np.meshgrid(x,x)
        
        samp = sampling_operator(dim=2,M=M,sigma=.4)
        
        q = 1 - 3*np.random.rand(100)/8
        
        
        
        Z = samp.apply_AT(q,np.array([X.flatten(),Y.flatten()]).T)
        
        plt.contourf(X,Y,Z.reshape(50,50))
        plt.colorbar()
        P=Cell_partition(dim=2)
        P.refine(0)
        P.refine(0)
        P.refine(1)
        P.refine(0)
        P.refine(1)
        
        
        P.refine(4)
        
        for k in range(5):
            P.display_partition(cells=True)
            Vk = np.array(P.get_vertices())
        
        
            samp.check_active_cells(q, P)
        
            P.refine_all_active()
            P.reset()


    ### TESTING THE CELL PARTITION IN 1D
    if False :
        P=Cell_partition(dim=1)
        P.display_partition(cells=True)
        P.refine(0)
        P.display_partition(cells=True)
        P.refine(0)
        P.display_partition(cells=True)
        P.refine(2)
        P.display_partition(cells=True)   
        P.refine(2)
        P.display_partition(cells=True)
        plt.show()
    
    ### TESTING THE CELL PARTITION IN 2D
    if False :
        P=Cell_partition(dim=2)
        P.display_partition(cells=True)
        P.refine(0)
        P.display_partition(cells=True)
        P.refine(0)
        P.display_partition(cells=True)
        P.refine(5)
        P.display_partition(cells=True)   
        P.refine(4)
        P.display_partition(cells=True)
        plt.show()
        
    ### TESTING THE BASIS PURSUIT in 2d
    if True :
        M = 15
        sigma = 2./M
        A = sampling_operator(M, sigma,dim=2)
        Xs =np.array([[1./3,4./6],[1./3,1./3],[4./6,4./6]])
        alphas = np.array([8,-9,5])
        y = A.apply_A(Xs, alphas)
        
        B=Basis_Pursuit(A.Sampler,Xs.shape[1],Xs.shape[0],y)
        v=1.2*B.meas2vec((alphas,Xs))
        grad=np.zeros(9)
        eps=1.e-8
        ref=B.evaluate(B.vec2meas(v))
        for i in range(9) :
            e=np.zeros(9)
            e[i]=1.
            grad[i]=(B.evaluate(B.vec2meas(v+eps*e))-ref)/eps
        gradth=B.meas2vec(B.grad(B.vec2meas(v)))
        print(grad,grad-gradth) #
        v=B.meas2vec((alphas,Xs))
        v=np.load('optimal_vector_2d.npy')
        v,niter=B.first_order_search(v,step=2.e-5,tol=1.e-8,nitermax=1e5)
        np.save('optimal_vector_2d.npy',v)
    
    ### TESTING THE BASIS PURSUIT in 1d
    if False :
        M = 20
        sigma = 2./M
        A = sampling_operator(M, sigma,dim=1)
        Xs = np.array([1./3,4./6]).reshape((2,1))
        alphas = np.array([8,-9])
        y = A.apply_A(Xs, alphas)
        
        B=Basis_Pursuit(A.Sampler,1,Xs.shape[0],y)
        v=1.2*B.meas2vec((alphas,Xs))
        grad=np.zeros(4)
        eps=1.e-8
        ref=B.evaluate(B.vec2meas(v))
        for i in range(4) :
            e=np.zeros(4)
            e[i]=1.
            grad[i]=(B.evaluate(B.vec2meas(v+eps*e))-ref)/eps
        gradth=B.meas2vec(B.grad(B.vec2meas(v)))
        print(grad,grad-gradth) #
        v=B.meas2vec((alphas,Xs))
        v=np.load('optimal_vector_1d.npy')
        v,niter=B.first_order_search(v,step=3.e-6,tol=4.e-10,nitermax=1e5)
        np.save('optimal_vector_1d.npy',v)
        
        
        
        
        
