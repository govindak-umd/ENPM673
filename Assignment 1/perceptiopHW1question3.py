#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


# In[61]:


x1,x2,x3,x4,y1,y2,y3,y4,xp1,xp2,xp3,xp4,yp1,yp2,yp3,yp4 = 5,150,150,5,5,5,150,150,100,200,220,100,100,80,80,200 


# In[62]:


#define input matrix
A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],[0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],[-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],[0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],[-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],[0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],[-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],[0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])


# In[63]:


def compute_svd(A):                              #function definition for svd calculation
    AT=A.T                                       #transpose
    ATA=AT.dot(A)                                #dot product of AT and A
    eigenvalue_V,eigenvector_V=LA.eig(ATA)      #return the eigen values and vectors
    sort_val = eigenvalue_V.argsort()[::-1]
    new_eigenvalue_V = eigenvalue_V[sort_val]
    new_eigenvector_V = eigenvector_V[:,sort_val] #sort the eigenvectors based on the largest eigen values
    new_eigenvector_VT = new_eigenvector_V.T
    AAT=A.dot(AT)
    eigenvalue_U,eigenvector_U=LA.eig(AAT)
    sort_val1 = eigenvalue_U.argsort()[::-1]
    new_eigenvalue_U = eigenvalue_U[sort_val1]
    new_eigenvector_U = eigenvector_U[:,sort_val1]
    temp = np.diag((np.sqrt(new_eigenvalue_U)))  #compute sigma matrix as a diagonal matrix with elements as square root of eigen values of U
    sigma = np.zeros_like(A).astype(np.float64)
    sigma[:temp.shape[0],:temp.shape[1]]=temp
    H = new_eigenvector_V[:,8]                   #Take last column of V as the homography matrix
    H = np.reshape(H,(3,3))
    return new_eigenvector_VT,new_eigenvector_U,sigma,H


# In[64]:


#return decomposed values of A
VT,U,S,H = compute_svd(A)


# In[69]:


np.set_printoptions(suppress=True)  #suppress the use of scientific notation for small numbers
print("V_Transpose=",VT)


# In[70]:


print("U=",U)


# In[71]:


print("sigma=",S)


# In[72]:


print("Homography matrix=",H)


# In[ ]:




