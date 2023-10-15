"""
Collection of functions which implement different levels of approximation for the diagrammatic approach to the binary multispecies reaction A+B -> C with A,B,C <-> \nullset.
Also implements the EMRE self-consistent bubble resummation (SBR)!


Author: Moshir Harsh
btemoshir@gmail.com

"""

import numpy as np
import scipy as sc

def emre_ABC(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    eps         = np.zeros([num_species,len(time_grid)])
    var         = np.zeros([num_species,num_species,len(time_grid)])
    
    y[:,0]      = initial_values
    eps[:,0]    = 0
    var[0,0,0]  = initial_values[0]
    var[1,1,0]  = initial_values[1]
    var[2,2,0]  = initial_values[2]
    
    dydt   = np.zeros(num_species)
    depsdt = np.zeros(num_species)
    dvardt = np.zeros([num_species,num_species])
    
    S = np.zeros([num_species,2*num_species+1])
    S[0,0] = 1
    S[0,1] = -1
    S[1,2] = 1
    S[1,3] = -1
    S[2,4] = 1
    S[2,5] = -1
    S[0,6] = -1
    S[1,6] = -1
    S[2,6] = 1
    
    F = np.zeros([2*num_species+1,2*num_species+1])
    
    for i in range(len(time_grid)-1):

        dydt[0]    = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i]
        dydt[1]    = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i]
        dydt[2]    = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i]
        
        #J = np.array([[-alpha*k3[0]*y[1,i],-alpha*k3[0]*y[1,i],alpha*k3[0]*y[1,i]],\
        #             [-alpha*k3[0]*y[0,i],-alpha*k3[0]*y[0,i],alpha*k3[0]*y[0,i]],\
        #            [0.,0.,0.]])
        
        J = np.array([[-k2[0]-alpha*k3[0]*y[1,i],-alpha*k3[0]*y[0,i],0.],\
                     [-alpha*k3[0]*y[1,i],-k2[1]-alpha*k3[0]*y[0,i],0.],\
                    [alpha*k3[0]*y[1,i],alpha*k3[0]*y[0,i],-k2[2]]])
        
        F[0,0] = k1[0]
        F[1,1] = k2[0]*y[0,i]
        F[2,2] = k1[1]
        F[3,3] = k2[1]*y[1,i]
        F[4,4] = k1[2]
        F[5,5] = k2[2]*y[2,i]
        F[6,6] = alpha*k3[0]*y[0,i]*y[1,i]
        
        depsdt[0]  = -k2[0]*eps[0,i] -alpha*k3[0]*y[1,i]*eps[0,i] -alpha*k3[0]*y[0,i]*eps[1,i] -alpha*k3[0]*var[0,1,i]
        depsdt[1]  = -k2[1]*eps[1,i] -alpha*k3[0]*y[1,i]*eps[0,i] -alpha*k3[0]*y[0,i]*eps[1,i] -alpha*k3[0]*var[0,1,i]
        depsdt[2]  = -k2[2]*eps[2,i] +alpha*k3[0]*y[1,i]*eps[0,i] +alpha*k3[0]*y[0,i]*eps[1,i] +alpha*k3[0]*var[0,1,i]
        
        dvardt = np.matmul(J,var[:,:,i]) + np.matmul(var[:,:,i],J.T) + np.matmul(S,np.matmul(F,S.T))
        
        y[:,i+1]     = y[:,i] + dydt*dt
        eps[:,i+1]   = eps[:,i] + depsdt*dt
        var[:,:,i+1]   = var[:,:,i] + dvardt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0
            
    return y,time_grid,eps,var

# Till order $\alpha^2$:

def integrate_mak_ABC(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):

        dydt[0]    = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i]
        dydt[1]    = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i]
        dydt[2]    = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i]
        
        y[:,i+1]     = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[0,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[1,0] = -alpha*k3[0]*y[1,i]
        local_Sigma[2,0] = +alpha*k3[0]*y[1,i]
        local_Sigma[2,1] = +alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] #+local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
        
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    
# Upto Order alpha^2 in \mu and Sigma:

def integrate_alpha2_singleR_ABC(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):

        dydt[0] = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i] +dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1])*y[0,:i]*y[1,:i])
        
        dydt[1] = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i] +dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1])*y[0,:i]*y[1,:i])

        dydt[2] = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i] -dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1])*y[0,:i]*y[1,:i])
        
        y[:,i+1]     = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] +local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
            if j<i:
                #(resp[i,1:i+1,0,0]*resp[i,1:i+1,1,1] + resp[i,1:i+1,0,1]*resp[i,1:i+1,1,0])*y[0,:i]*y[1,:i]
                dRdt[0,0,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[1,1,i,j+1:i+1]*y[0,i])*y[1,j:i]*resp[0,0,j:i,j])
                dRdt[1,1,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,0,i,j+1:i+1]*y[1,i])*y[0,j:i]*resp[1,1,j:i,j])
                
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    
    
# Upto Order alpha^2 in \mu and Sigma:

def integrate_alpha2_bare(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):

        dydt[0] = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i] +dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1])*y[0,:i]*y[1,:i])
        
        dydt[1] = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i] +dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1])*y[0,:i]*y[1,:i])

        dydt[2] = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i] -dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1])*y[0,:i]*y[1,:i])
        
        y[:,i+1]     = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] #+local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
            #if j<i:
            #    #(resp[i,1:i+1,0,0]*resp[i,1:i+1,1,1] + resp[i,1:i+1,0,1]*resp[i,1:i+1,1,0])*y[0,:i]*y[1,:i]
            #    dRdt[0,0,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1])*y[1,j:i]*resp[0,0,j:i,j])
            #    dRdt[1,1,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1])*y[0,j:i]*resp[1,1,j:i,j])
                
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    

# ALL corrections with mixed Response

def integrate_All_bare(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):
        
        L = np.diag(np.ones(i),k=-1)
        T = resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1] #+ resp[0,1,:i+1,:i+1]*resp[1,0,:i+1,:i+1]
        
        temp_mat = np.matmul(T,sc.linalg.solve_triangular(a=np.identity(i+1)+ alpha*k3[0]*dt*np.matmul(L,T),b=np.identity(i+1),lower=True,overwrite_b=True))
        

        dydt[0]    = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i] + dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])
        
        dydt[1]    = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i] + dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])

        dydt[2]    = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i] - dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])
        
        y[:,i+1]     = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[0,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[1,0] = -alpha*k3[0]*y[1,i]
        local_Sigma[2,0] = +alpha*k3[0]*y[1,i]
        local_Sigma[2,1] = +alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] #+local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
            #if j<i:
                #(resp[i,1:i+1,0,0]*resp[i,1:i+1,1,1] + resp[i,1:i+1,0,1]*resp[i,1:i+1,1,0])*y[0,:i]*y[1,:i]
            #    dRdt[0,0,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j])
            #    dRdt[1,1,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j])
            #    dRdt[0,1,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j])
            #    dRdt[1,0,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j])
            #    dRdt[2,0,j] += -dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j])
            #    dRdt[2,1,j] += -dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j])
        
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    
    
# Upto Order alpha^2 in \mu and Sigma:

def integrate_alpha2_ABC(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):

        dydt[0]    = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i] +dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1] + resp[0,1,i,1:i+1]*resp[1,0,i,1:i+1])*y[0,:i]*y[1,:i])
        
        dydt[1]    = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i] +dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1] + resp[0,1,i,1:i+1]*resp[1,0,i,1:i+1])*y[0,:i]*y[1,:i])

        dydt[2]    = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i] -dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,1:i+1]*resp[1,1,i,1:i+1] + resp[0,1,i,1:i+1]*resp[1,0,i,1:i+1])*y[0,:i]*y[1,:i])
        
        y[:,i+1]     = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[0,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[1,0] = -alpha*k3[0]*y[1,i]
        local_Sigma[2,0] = +alpha*k3[0]*y[1,i]
        local_Sigma[2,1] = +alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] +local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
            if j<i:
                #(resp[i,1:i+1,0,0]*resp[i,1:i+1,1,1] + resp[i,1:i+1,0,1]*resp[i,1:i+1,1,0])*y[0,:i]*y[1,:i]
                dRdt[0,0,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[1,j:i]*resp[0,0,j:i,j]+\
                                                          (resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[0,j:i]*resp[1,0,j:i,j])
                
                dRdt[1,1,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[0,j:i]*resp[1,1,j:i,j]+\
                                                          (resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[1,j:i]*resp[0,1,j:i,j])
                
                dRdt[0,1,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[0,j:i]*resp[1,1,j:i,j]+\
                                                          (resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[1,j:i]*resp[0,1,j:i,j])
                
                dRdt[1,0,j] +=  dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[1,j:i]*resp[0,0,j:i,j]+\
                                                          (resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[0,j:i]*resp[1,0,j:i,j])
                
                dRdt[2,0,j] += -dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[1,j:i]*resp[0,0,j:i,j]+\
                                                          (resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[0,j:i]*resp[1,0,j:i,j])
                
                dRdt[2,1,j] += -dt*(alpha*k3[0])**2*np.sum((resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[0,j:i]*resp[1,1,j:i,j]+\
                                                          (resp[0,0,i,j+1:i+1]*resp[1,1,i,j+1:i+1] + resp[0,1,i,j+1:i+1]*resp[1,0,i,j+1:i+1])*y[1,j:i]*resp[0,1,j:i,j])
                        
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    
    
# ALL corrections with mixed Response

def integrate_All_ABC(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):
        
        L = np.diag(np.ones(i),k=-1)
        T = resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1] + resp[0,1,:i+1,:i+1]*resp[1,0,:i+1,:i+1]
        
        temp_mat = np.matmul(T,sc.linalg.solve_triangular(a=np.identity(i+1)+ alpha*k3[0]*dt*np.matmul(L,T),b=np.identity(i+1),lower=True,overwrite_b=True))
        

        dydt[0]    = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i] + dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])
        
        dydt[1]    = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i] + dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])

        dydt[2]    = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i] - dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])
        
        y[:,i+1]     = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[0,1] = -alpha*k3[0]*y[0,i]
        local_Sigma[1,0] = -alpha*k3[0]*y[1,i]
        local_Sigma[2,0] = +alpha*k3[0]*y[1,i]
        local_Sigma[2,1] = +alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] + np.sum(local_Sigma[sp1,:]*resp[:,sp2,i,j])
                    #dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] +local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
            if j<i:
                #(resp[i,1:i+1,0,0]*resp[i,1:i+1,1,1] + resp[i,1:i+1,0,1]*resp[i,1:i+1,1,0])*y[0,:i]*y[1,:i]
                dRdt[0,0,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j] + temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,0,j:i,j])
                dRdt[1,1,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j] + temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,1,j:i,j])
                dRdt[0,1,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j] + temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,1,j:i,j])
                dRdt[1,0,j] +=  dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j] + temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,0,j:i,j])
                dRdt[2,0,j] += -dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j] + temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,0,j:i,j])
                dRdt[2,1,j] += -dt*(alpha*k3[0])**2*np.sum(temp_mat[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j] + temp_mat[i,j+1:i+1]*y[1,j:i]*resp[0,1,j:i,j])
        
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    
    
# ALL corrections with single Response

def integrate_All_singleR_ABC(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid   = np.arange(init_time,final_time,dt)
    y           = np.zeros([num_species,len(time_grid)])
    y[:,0]      = initial_values
    resp        = np.zeros([num_species,num_species,len(time_grid),len(time_grid)])
    
    for sp in range(num_species):
        resp[sp,sp,0,0] = 1.
    
    dydt = np.zeros(num_species)
    for i in range(len(time_grid)-1):
        
        L = np.diag(np.ones(i),k=-1)
        T = resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1]
        
        temp_mat = np.matmul(T,sc.linalg.solve_triangular(a=np.identity(i+1)+ alpha*k3[0]*dt*np.matmul(L,T),b=np.identity(i+1),lower=True,overwrite_b=True))
        

        dydt[0]    = k1[0] - k2[0]*y[0,i] -alpha*k3[0]*y[0,i]*y[1,i] + dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])
        
        dydt[1]    = k1[1] - k2[1]*y[1,i] -alpha*k3[0]*y[0,i]*y[1,i] + dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])

        dydt[2]    = k1[2] - k2[2]*y[2,i] +alpha*k3[0]*y[0,i]*y[1,i] - dt*(alpha*k3[0])**2*np.sum(temp_mat[i,1:i+1]*y[0,:i]*y[1,:i])
        
        y[:,i+1]   = y[:,i] + dydt*dt
        
        for sp in range(num_species):
            if y[sp,i+1]  < 0:
                y[sp,i+1] = 0

        #Define local Sigma
        local_Sigma      = np.zeros([num_species,num_species])
        local_Sigma[0,0] = -alpha*k3[0]*y[1,i] 
        local_Sigma[1,1] = -alpha*k3[0]*y[0,i]
        
        dRdt = np.zeros([num_species,num_species,i+1])
        
        T2 = np.zeros([i+1,i+1])
        T3 = np.zeros([i+1,i+1])
        
        for k in range(i+1):
            T2[k,:] = y[0,k]*resp[1,1,k,:i+1]
            T3[k,:] = y[1,k]*resp[0,0,k,:i+1]        
        
        #T2 = y[0,k]*resp[1,1,:i+1,:i+1] + resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1]
        #T3 = y[1,k]*resp[0,0,:i+1,:i+1] + resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1]
        
        T2 += resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1]
        T3 += resp[0,0,:i+1,:i+1]*resp[1,1,:i+1,:i+1]  
        
        
        #T2 = y[0,:i+1]*resp[1,1,:i+1,:i+1]
        temp_mat2 = np.matmul(T2,sc.linalg.solve_triangular(a=np.identity(i+1)+ alpha*k3[0]*dt*np.matmul(L,T2),b=np.identity(i+1),lower=True,overwrite_b=True))
        
        #T3 = y[1,:i+1]*resp[0,0,:i+1,:i+1]
        temp_mat3 = np.matmul(T3,sc.linalg.solve_triangular(a=np.identity(i+1)+ alpha*k3[0]*dt*np.matmul(L,T3),b=np.identity(i+1),lower=True,overwrite_b=True))
        
        for j in range(i+1):
            for sp1 in range(num_species):
                for sp2 in range(num_species):
                    dRdt[sp1,sp2,j] = -k2[sp1]*resp[sp1,sp2,i,j] +local_Sigma[sp1,sp2]*resp[sp2,sp2,i,j]
            if j<i:
                #(resp[i,1:i+1,0,0]*resp[i,1:i+1,1,1] + resp[i,1:i+1,0,1]*resp[i,1:i+1,1,0])*y[0,:i]*y[1,:i]
                dRdt[0,0,j] +=  dt*(alpha*k3[0])**2*np.sum((temp_mat2)[i,j+1:i+1]*y[1,j:i]*resp[0,0,j:i,j])
                dRdt[1,1,j] +=  dt*(alpha*k3[0])**2*np.sum((temp_mat3)[i,j+1:i+1]*y[0,j:i]*resp[1,1,j:i,j])
        
        for sp in range(num_species):
            resp[sp,sp,i+1,i+1] = 1.
        
        for j in range(i+1):
            resp[:,:,i+1,j] = resp[:,:,i,j] + dt*dRdt[:,:,j]
            
    return y,resp,time_grid
    
    

    
    