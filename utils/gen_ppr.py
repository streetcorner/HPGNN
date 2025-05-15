from numba import njit,float64,int32,jit
import numpy as np

import argparse
import multiprocessing
import scipy.sparse as sp
from tqdm import tqdm
import random
import json
import time
import sys

def _power_push_sor(adj_matrix,indptr,indices,degree,clique,cliqueNum,s,eps,alpha,omega,maxCliqueSize):
    n = cliqueNum[0] 
    m = cliqueNum[1] 
    gpr_vec ={}
    for i in range(1, maxCliqueSize + 1): 
        if((degree[i][s]==0) or ((degree[i][s]==1) and (indices[i][indptr[i][s]:indptr[i][s + 1]][0] == s))):
            gpr_vec[i] = np.zeros(n)
            break 
        if(i==1):
            queue_size=n
        else:
            queue_size=n + cliqueNum[i] 

            high_degree = np.array([i + 1] * cliqueNum[i], dtype=np.int64)
            degree[i]=np.concatenate((degree[i],high_degree))
        queue = np.zeros(queue_size, dtype=np.int64) 
        front, rear = np.int64(0), np.int64(1)  
        gpr_vec[i], res = np.zeros(queue_size), np.zeros(queue_size) 
        switch_size = np.int64( queue_size / 4)      
        queue[rear] = s  
        q_mark = np.zeros(queue_size)  
        q_mark[s] = 1 
        res[s] = 1  
        
        eps_cur=eps / indptr[i][-1] 
        r_max = eps_cur / cliqueNum[i] 
        r_sum = 1      
        eps_vec = r_max * degree[i]   

        num_oper = 0. 
        l1_error = []  
        num_oper_list = []  
        step = 1e5 
        threshold = step
       
        
        while front != rear and ((rear - front) <= switch_size): 
            front = (front + 1) % (queue_size) 
            u = queue[front] 
            if not( (i>1) & (u>=n)):
                q_mark[u] = False  
            if np.abs(res[u]) > eps_vec[u]: 
                residual = omega * alpha * res[u]  
                gpr_vec[i][u] += residual 
                r_sum -= residual  
                if (degree[i][u]==0): 
                    if(front==rear): 
                        rear = (rear + 1) % queue_size 
                        queue[rear] = s 
                    continue
                increment = omega * (1. - alpha) * res[u] / degree[i][u]                  
                res[u] -= omega * res[u] 
                num_oper += degree[i][u]   
                
                if num_oper > threshold: 
                    l1_error.append(r_sum) 
                    num_oper_list.append(num_oper)
                    threshold += step 
                    if threshold > 3e7: 
                        step=1e5 
                '''
                    邻居节点
                '''
                if((i > 1) & (u >= n)): 
                    u_real= u % n 
                    for v in clique[i][u_real]:                       
                        res[v] += increment 
                        if not q_mark[v]: 
                            rear = (rear + 1) % queue_size
                            queue[rear] = v
                            q_mark[v] = True  

                else: 
                    for v in indices[i][indptr[i][u]:indptr[i][u + 1]]:
                        if(i>1): 
                            v = n + v 
                        res[v] += increment 
                        if not q_mark[v]: 
                            rear = (rear + 1) % queue_size
                            queue[rear] = v
                            q_mark[v] = True  
        jump = False
        if r_sum <= eps_cur: 
            jump = True  
            
        num_epoch = 8 
        
        r_max_prime1 = np.power(eps_cur, 2 / num_epoch)   
        r_max_prime2 = np.power(eps_cur, 2 / num_epoch) / sum(degree[i])
        
        if not jump:
            for epoch in np.arange(1, num_epoch + 1):
                
                while r_sum > r_max_prime1: 
                    for u in range(queue_size): 
                        if(r_sum <= r_max_prime1): 
                                jump=True
                                break
                        if (np.abs(res[u]) > (r_max_prime2 * degree[i][u])): 
                            residual = omega * alpha * res[u]
                            gpr_vec[i][u] += residual
                            r_sum -= residual
                            if (degree[i][u]==0): 
                                continue
                            increment = omega * (1. - alpha) * res[u] /  degree[i][u]   
                            res[u] -= omega * res[u]
                            num_oper += degree[i][u]
                            if num_oper>threshold:
                                l1_error.append(r_sum)
                                num_oper_list.append(num_oper)
                                threshold += step
                                if threshold>3e7:
                                    step=1e5
                            '''
                            邻居节点
                            '''
                            if((i > 1) & (u >= n)): 
                                u_real= u % n
                                for v in clique[i][u_real]: 
                                    res[v] += increment 
                            else:
                                for index in indices[i][indptr[i][u]:indptr[i][u + 1]]: 
                                    if(i>1): 
                                        index =index + n      
                                    res[index] += increment
    for k in range(1, len(gpr_vec) + 1): 
        gpr_vec[k]=gpr_vec[k][:n]
    return gpr_vec 

#PWR_PUSH_SOR
def power_push_sor(adj_matrix,clique,cliqueNum,s,eps,alpha,omega=None,maxCliqueSize=2): 
    if not omega:
        omega = 1. + ((1. - alpha) / (1. + np.sqrt(1 - (1. - alpha) ** 2.))) ** 2
    indices,indptr,degree={},{},{}
    for i in range(1, maxCliqueSize + 1):
        indices[i]=adj_matrix[i].indices
        indptr[i]=adj_matrix[i].indptr
        degree[i]=np.int64(adj_matrix[i].sum(1).A.flatten())
    vec_num_op=_power_push_sor(adj_matrix,indptr,indices,degree,clique,cliqueNum,s,eps,alpha,omega,maxCliqueSize)
    return vec_num_op

