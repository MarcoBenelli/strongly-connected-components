#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import random
from timeit import default_timer
import sys
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


sys.setrecursionlimit(2**16)


# In[3]:


class Vertex():
    def __init__(self, num):
        self.adj = []
        self.num = num
        self.f = 0

class Graph():
    def __init__(self, num_vertices, edge_prob=0):
        self.vertices = [Vertex(i) for i in range(num_vertices)]
        if edge_prob > 0:
            for u in self.vertices:
                for v in self.vertices:
                    if random() < edge_prob:
                        u.adj.append(v)
    
    def depth_first_search(self):
        
        def DFS_visit(g, u):
            nonlocal time
            time += 1
            u.d = time
            u.color = "GRAY"
            for v in u.adj:
                if v.color == "WHITE":
                    v.parent = u
                    DFS_visit(g, v)
            u.color = "BLACK"
            time += 1
            u.f = time
        
        for u in self.vertices:
            u.color = "WHITE"
            u.parent = None
        time = 0
        num_trees = 0
        for u in self.vertices:
            if u.color == "WHITE":
                DFS_visit(self, u)
                num_trees += 1
        return num_trees
    
    def transpose(self):
        trans = Graph(len(self.vertices))
        for u in self.vertices:
            for v in u.adj:
                trans.vertices[v.num].adj.append(trans.vertices[u.num])
        return trans
    
    def strongly_connected_components(self):
        self.depth_first_search()
        trans = self.transpose()
        trans.vertices.sort(key=lambda v: self.vertices[v.num].f, reverse=True)
        for u in trans.vertices:
            u.adj.sort(key=lambda v: self.vertices[v.num].f, reverse=True)
        return trans.depth_first_search()
    
    def printg(self):
        for u in self.vertices:
            print(f'{u.num} ({u.f}): ', end='')
            for v in u.adj:
                print(v.num, end=' ')
            print()


# In[4]:


f = 2
x = np.array([])
y = np.empty((f, 0))
y_time = np.empty((f, 0))

n = 1
time_n = 0
num_tests = 16
while time_n < 16:
    n *= 2
    x.resize(x.size+1)
    x[-1] = n
    y = np.hstack((y, np.zeros((f, 1))))
    y_time = np.hstack((y_time, np.zeros((f, 1))))
    print(n, 'vertices')
    time_n = 0
    
    for i in range(num_tests):
        time_n -= default_timer()
        
        g = Graph(n, 1/n)
        y_time[0, -1] -= default_timer()
        y[0, -1] += g.strongly_connected_components()
        y_time[0, -1] += default_timer()
    
        g = Graph(n, 1/2)
        y_time[1, -1] -= default_timer()
        y[1, -1] += g.strongly_connected_components()
        y_time[1, -1] += default_timer()
        
        time_n += default_timer()
    
    y_time[0, -1] /= num_tests
    y_time[1, -1] /= num_tests
    y[0, -1] /= num_tests
    y[1, -1] /= num_tests
    print(f'    V edges: {y[0, -1]} SCC')
    print(f'    V**2/2 edges: {y[1, -1]} SCC')
    time_n /= num_tests


# In[5]:


plt.figure(1)
plt.xscale('log')
plt.yscale('log')

plt.plot(x, y_time[0], label='linear')
xcl = np.logspace(1, np.log2(x[-1]), num=np.log2(x[-1]), base=2)
coeff = y_time[0][-1] / xcl[-1]
ycl = xcl * coeff
plt.plot(xcl, ycl, 'k--')

plt.plot(x, y_time[1], label='quadratic')
xcq = np.logspace(1, np.log2(x[-1]), num=np.log2(x[-1]), base=2)
coeff = y_time[1][-1] / xcq[-1]**2
ycq = xcq**2 * coeff
plt.plot(xcq, ycq, 'k--')

plt.legend()


# In[ ]:




