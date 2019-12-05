#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage import data, segmentation, filters, color
from skimage.future import graph
from plot_rag_merge import _weight_mean_color, merge_mean_color


# In[2]:


p1 = np.load('0153MR0008490000201265E01_DRLX.npy',allow_pickle=True)


# In[3]:


p1 = np.delete(p1, 0, 0)


# In[4]:


p1.shape


# In[5]:


imgplot = plt.imshow(p1)


# In[6]:


segments = felzenszwalb(p1, scale=32, sigma=12.5, min_size=15)


# In[7]:


imgplot = plt.imshow(mark_boundaries(p1, segments))


# In[8]:


g = graph.rag_mean_color(p1, segments)


# In[9]:


labels2 = graph.merge_hierarchical(segments, g, thresh=0.05, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[10]:


graph.show_rag(segments, g, p1)


# In[11]:


plt.figure()
out = color.label2rgb(labels2, p1)
plt.imshow(out)


# In[12]:


p2 = np.load('0172ML0009240000104879E01_DRLX.npy',allow_pickle=True)
p2 = np.delete(p2, 0, 0)
p2.shape


# In[13]:


imgplot = plt.imshow(p2)


# In[14]:


segments = felzenszwalb(p2, scale=20.0, sigma=10.95, min_size=2)


# In[15]:


imgplot = plt.imshow(mark_boundaries(p2, segments))


# In[16]:


g = graph.rag_mean_color(p2, segments)
labels2 = graph.merge_hierarchical(segments, g, thresh=0.095, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[17]:


graph.show_rag(segments, g, p2)


# In[18]:


plt.figure()
out = color.label2rgb(labels2, p2)
plt.imshow(out)


# In[19]:


p3 = np.load('0172ML0009240340104913E01_DRLX.npy',allow_pickle=True)
p3 = np.delete(p3, 0, 0)
p3.shape


# In[20]:


imgplot = plt.imshow(p3)


# In[21]:


segments = felzenszwalb(p3, scale=50.0, sigma=8, min_size=20)


# In[22]:


imgplot = plt.imshow(mark_boundaries(p3, segments))


# In[23]:


g = graph.rag_mean_color(p3, segments)
labels2 = graph.merge_hierarchical(segments, g, thresh=0.055, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[24]:


graph.show_rag(segments, g, p3)


# In[25]:


plt.figure()
out = color.label2rgb(labels2, p3)
plt.imshow(out)


# In[26]:


p4 = np.load('0270MR0011860360203259E01_DRLX.npy',allow_pickle=True)
p4 = np.delete(p4, 0, 0)
p4.shape


# In[27]:


imgplot = plt.imshow(p4)


# In[28]:


segments = felzenszwalb(p4, scale=20.75, sigma=18.95, min_size=3)


# In[29]:


imgplot = plt.imshow(mark_boundaries(p4, segments))


# In[30]:


g = graph.rag_mean_color(p4, segments)
labels2 = graph.merge_hierarchical(segments, g, thresh=0.095, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[31]:


graph.show_rag(segments, g, p4)


# In[32]:


plt.figure()
out = color.label2rgb(labels2, p4)
plt.imshow(out)


# In[ ]:




