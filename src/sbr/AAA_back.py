"""
Collection of functions which implement different levels of approximation for the diagrammatic approach to the coagulation reaction and branching A+A <-> A with A <-> \nullset.
Also implements the EMRE and the self-consistent bubble resummation (SBR)!


Author: Moshir Harsh
btemoshir@gmail.com

"""


import numpy as np
import scipy as sc

def integrate_alpha2_alpha_2(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.

    for i in range(len(time_grid)-1):

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + 2*(alpha*k3[0])**2*dt*np.sum(resp[i,1:i+1]**2*y[0:i]**2) -alpha*k3[1]*y[i]  -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(resp[i,1:i+1]**2*y[0:i])
        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] -2*alpha*k3[0]*y[i]*resp[i,j] + alpha*k3[1]*resp[i,j]
            if j < i:
                dRdt[j] += 4*(alpha*k3[0])**2*dt*np.sum(resp[i,j+1:i+1]**2*y[j:i]*resp[j:i,j]) -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(resp[i,j+1:i+1]**2*resp[j:i,j])

        resp[i+1,i+1] = 1.
        # Update it using the dt derivative for R:
        #for j in range(i,-1,-1):
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]
            
    return y,resp,time_grid


def integrate_alpha2_alpha_1(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.
    

    for i in range(len(time_grid)-1):

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + 2*(alpha*k3[0])**2*dt*np.sum(resp[i,1:i+1]**2*y[0:i]**2) -alpha*k3[1]*y[i]  -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(resp[i,1:i+1]**2*y[0:i])
        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] -2*alpha*k3[0]*y[i]*resp[i,j] + alpha*k3[1]*resp[i,j]
            #if j < i:
            #    dRdt[j] += 4*(alpha*k3[0])**2*dt*np.sum(resp[i,j+1:i+1]**2*y[j:i]*resp[j:i,j])

        resp[i+1,i+1] = 1.
        # Update it using the dt derivative for R:
        #for j in range(i,-1,-1):
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]
            
    return y,resp,time_grid
    

def integrate_all_alpha_0(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.
    

    for i in range(len(time_grid)-1):
        
        L = np.diag(np.ones(i),k=-1)
        temp_mat = np.matmul(resp[:i+1,:i+1]**2,sc.linalg.solve_triangular(a=np.identity(i+1)+ 2*alpha*k3[0]*dt*np.matmul(L,resp[:i+1,:i+1]**2),b=np.identity(i+1),lower=True,overwrite_b=True))

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + 2*(alpha*k3[0])**2*dt*np.sum(temp_mat[i,1:i+1]*y[0:i]**2) + alpha*k3[1]*y[i] -2*(alpha**2*k3[1]*k3[0])*dt*np.sum(temp_mat[i,1:i+1]*y[0:i])

        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] #-2*alpha*k3[0]*y[i]*resp[i,j]
            #if j < i:
            #    dRdt[j] += 4*(alpha*k3[0])**2*dt*np.sum(resp[i,j+1:i+1]**2*y[j:i]*resp[j:i,j])

        resp[i+1,i+1] = 1.
        # Update it using the dt derivative for R:
        #for j in range(i,-1,-1):
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]
            
    return y,resp,time_grid


def integrate_alpha2_alpha_all(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.
    

    for i in range(len(time_grid)-1):
        
        L = np.diag(np.ones(i),k=-1)
        temp_mat = np.matmul(resp[:i+1,:i+1]**2,sc.linalg.solve_triangular(a=np.identity(i+1)+ 2*alpha*k3[0]*dt*np.matmul(L,resp[:i+1,:i+1]**2),b=np.identity(i+1),lower=True,overwrite_b=True))

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + 2*(alpha*k3[0])**2*dt*np.sum(resp[i,1:i+1]**2*y[0:i]**2) +alpha*k3[1]*y[i]  -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(resp[i,1:i+1]**2*y[0:i])

        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] -2*alpha*k3[0]*y[i]*resp[i,j] + alpha*k3[1]*resp[i,j]
            if j < i:
                dRdt[j] += 4*(alpha*k3[0])**2*dt*np.sum(temp_mat[i,j+1:i+1]*y[j:i]*resp[j:i,j]) -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(temp_mat[i,j+1:i+1]*resp[j:i,j])

        resp[i+1,i+1] = 1.
        # Update it using the dt derivative for R:
        #for j in range(i,-1,-1):
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]
            
    return y,resp,time_grid
    

def integrate_alpha2_alpha_0(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.
    

    for i in range(len(time_grid)-1):

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + 2*(alpha*k3[0])**2*dt*np.sum(resp[i,1:i+1]**2*y[0:i]**2) +alpha*k3[1]*y[i]  -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(resp[i,1:i+1]**2*y[0:i])
        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] #-2*alpha*k3[0]*y[i]*resp[i,j]
            #if j < i:
            #    dRdt[j] += 4*(alpha*k3[0])**2*dt*np.sum(resp[i,j+1:i+1]**2*y[j:i]*resp[j:i,j])

        resp[i+1,i+1] = 1.
        # Update it using the dt derivative for R:
        #for j in range(i,-1,-1):
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]
            
    return y,resp,time_grid
    

def integrate_all_corr(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.
    
    for i in range(len(time_grid)-1):
    
        L = np.diag(np.ones(i),k=-1)
        temp_mat = np.matmul(resp[:i+1,:i+1]**2,sc.linalg.solve_triangular(a=np.identity(i+1)+ 2*alpha*k3[0]*dt*np.matmul(L,resp[:i+1,:i+1]**2),b=np.identity(i+1),lower=True,overwrite_b=True))

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + 2*(alpha*k3[0])**2*dt*np.sum(temp_mat[i,1:i+1]*y[0:i]**2) + alpha*k3[1]*y[i] -2*(alpha**2*k3[1]*k3[0])*dt*np.sum(temp_mat[i,1:i+1]*y[0:i])
        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] -2*alpha*k3[0]*y[i]*resp[i,j] +alpha*k3[1]*resp[i,j]
            if j < i:
                dRdt[j] += 4*(alpha*k3[0])**2*dt*np.sum(temp_mat[i,j+1:i+1]*y[j:i]*resp[j:i,j]) -2*(alpha**2*k3[0]*k3[1])*dt*np.sum(temp_mat[i,j+1:i+1]*resp[j:i,j])

        resp[i+1,i+1] = 1.
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]

    return y,resp,time_grid


def integrate_mak(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    y[0]      = initial_values
    resp      = np.zeros([len(time_grid),len(time_grid)])
    resp[0,0] = 1.

    for i in range(len(time_grid)-1):

        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 +alpha*k3[1]*y[i]
        y[i+1]     = y[i] + dydt*dt

        if y[i+1]  < 0:
            y[i+1] = 0

        dRdt = np.zeros(i+1)

        #for j in range(i,-1,-1):
        for j in range(i+1):
            dRdt[j] = -k2*resp[i,j] #-2*alpha*k3[0]*y[i]*resp[i,j]

        resp[i+1,i+1] = 1.
        # Update it using the dt derivative for R:
        #for j in range(i,-1,-1):
        for j in range(i+1):
            resp[i+1,j] = resp[i,j] + dt*dRdt[j]
            
    return y,resp,time_grid
    
def emre(k1,k2,k3,alpha=1.,init_time=0.,final_time=1.,dt=0.001,initial_values=1.):
    
    time_grid  = np.arange(init_time,final_time,dt)
    y         = np.zeros(len(time_grid))
    eps       = np.zeros(len(time_grid))
    var       = np.zeros(len(time_grid))
    
    y[0]      = initial_values
    eps[0]    = 0
    var[0]    = initial_values
    
    for i in range(len(time_grid)-1):
        
        dydt       = k1[0] - k2[0]*y[i] -alpha*k3[0]*y[i]**2 + alpha*k3[1]*y[i]
        depsdt     = (-k2[0] -2*alpha*k3[0]*y[i] + alpha*k3[1])*eps[i] -alpha*k3[0]*(var[i] - y[i])
        dvardt     = 2*(-k2[0] -2*alpha*k3[0]*y[i] + alpha*k3[1])*var[i] + k1 + k2*y[i] +alpha*k3[0]*y[i]**2 + alpha*k3[1]*y[i]
        
        y[i+1]     = y[i] + dydt*dt
        eps[i+1]   = eps[i] + depsdt*dt
        var[i+1]   = var[i] + dvardt*dt
        
    return y,time_grid,eps,var