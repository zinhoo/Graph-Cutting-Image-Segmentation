#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage import data, segmentation, filters, color
from skimage.future import graph
from plot_rag_merge import _weight_mean_color, merge_mean_color


# In[3]:


p1 = np.load('0153MR0008490000201265E01_DRLX.npy',allow_pickle=True)


# In[4]:


p1 = np.delete(p1, 0, 0)


# In[5]:


p1.shape


# In[6]:


imgplot = plt.imshow(p1)


# In[24]:


labels = segmentation.slic(p1, compactness=40, n_segments=3)


# In[25]:


imgplot = plt.imshow(mark_boundaries(p1, labels))


# In[26]:


g = graph.rag_mean_color(p1, labels)


# In[33]:


labels2 = graph.merge_hierarchical(labels, g, thresh=0.025, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[34]:


graph.show_rag(labels, g, p1)


# In[35]:


plt.figure()
out = color.label2rgb(labels2, p1)
plt.imshow(out)


# In[38]:


p2 = np.load('0172ML0009240000104879E01_DRLX.npy',allow_pickle=True)
p2 = np.delete(p2, 0, 0)
p2.shape


# In[39]:


imgplot = plt.imshow(p2)


# In[48]:


labels = segmentation.slic(p2, compactness=40, n_segments=9)


# In[49]:


imgplot = plt.imshow(mark_boundaries(p2, labels))


# In[53]:


g = graph.rag_mean_color(p2, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=0.15, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[54]:


graph.show_rag(labels, g, p2)


# In[55]:


plt.figure()
out = color.label2rgb(labels2, p2)
plt.imshow(out)


# In[56]:


p3 = np.load('0172ML0009240340104913E01_DRLX.npy',allow_pickle=True)
p3 = np.delete(p3, 0, 0)
p3.shape


# In[57]:


imgplot = plt.imshow(p3)


# In[91]:


labels = segmentation.slic(p3, compactness=60, n_segments=15)


# In[92]:


imgplot = plt.imshow(mark_boundaries(p3, labels))


# In[93]:


g = graph.rag_mean_color(p3, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=0.09, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[94]:


graph.show_rag(labels, g, p3)


# In[95]:


plt.figure()
out = color.label2rgb(labels2, p3)
plt.imshow(out)


# In[96]:


p4 = np.load('0270MR0011860360203259E01_DRLX.npy',allow_pickle=True)
p4 = np.delete(p4, 0, 0)
p4.shape


# In[97]:


imgplot = plt.imshow(p4)


# In[161]:


labels = segmentation.slic(p1, compactness=25, n_segments=200)


# In[162]:


imgplot = plt.imshow(mark_boundaries(p4, labels))


# In[169]:


g = graph.rag_mean_color(p4, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=0.075, rag_copy=True,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)


# In[170]:


graph.show_rag(labels, g, p4)


# In[171]:


plt.figure()
out = color.label2rgb(labels2, p4)
plt.imshow(out)


# In[ ]:




