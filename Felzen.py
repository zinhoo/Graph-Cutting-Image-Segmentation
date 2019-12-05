#!/usr/bin/env python
# coding: utf-8

# In[23]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, mark_boundaries


# In[16]:


p1 = np.load('0153MR0008490000201265E01_DRLX.npy',allow_pickle=True)


# In[17]:


p1 = np.delete(p1, 0, 0)


# In[18]:


p1.shape


# In[33]:


imgplot = plt.imshow(p1)


# In[185]:


segments = felzenszwalb(p1, scale=32, sigma=12.5, min_size=15)


# In[186]:


imgplot = plt.imshow(mark_boundaries(p1, segments))


# In[35]:


p2 = np.load('0172ML0009240000104879E01_DRLX.npy',allow_pickle=True)
p2 = np.delete(p2, 0, 0)
p2.shape


# In[36]:


imgplot = plt.imshow(p2)


# In[165]:


segments = felzenszwalb(p2, scale=20.0, sigma=10.95, min_size=2)


# In[166]:


imgplot = plt.imshow(mark_boundaries(p2, segments))


# In[87]:


p3 = np.load('0172ML0009240340104913E01_DRLX.npy',allow_pickle=True)
p3 = np.delete(p3, 0, 0)
p3.shape


# In[90]:


imgplot = plt.imshow(p3)


# In[159]:


segments = felzenszwalb(p3, scale=50.0, sigma=8, min_size=20)


# In[160]:


imgplot = plt.imshow(mark_boundaries(p3, segments))


# In[93]:


p4 = np.load('0270MR0011860360203259E01_DRLX.npy',allow_pickle=True)
p4 = np.delete(p4, 0, 0)
p4.shape


# In[94]:


imgplot = plt.imshow(p4)


# In[121]:


segments = felzenszwalb(p4, scale=20.75, sigma=18.95, min_size=3)


# In[122]:


imgplot = plt.imshow(mark_boundaries(p4, segments))


# In[ ]:




