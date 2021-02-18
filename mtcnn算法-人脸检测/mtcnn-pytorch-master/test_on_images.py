#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[2]:


from src import detect_faces, show_bboxes
from PIL import Image


# In[3]:


img = Image.open('images/1.jpg')
bounding_boxes, landmarks = detect_faces(img)
print("bounding_boxes:",bounding_boxes)
print("landmarks:",landmarks)
show_bboxes(img, bounding_boxes, landmarks).show()

'''
# In[4]:


img = Image.open('images/office2.jpg')
bounding_boxes, landmarks = detect_faces(img)
show_bboxes(img, bounding_boxes, landmarks).show()


# In[5]:


img = Image.open('images/office3.jpg')
bounding_boxes, landmarks = detect_faces(img)
show_bboxes(img, bounding_boxes, landmarks).show()


# In[6]:


img = Image.open('images/office4.jpg')
bounding_boxes, landmarks = detect_faces(img, thresholds=[0.6, 0.7, 0.85])
show_bboxes(img, bounding_boxes, landmarks).show()


# In[7]:


img = Image.open('images/office5.jpg')
bounding_boxes, landmarks = detect_faces(img, min_face_size=10.0)
show_bboxes(img, bounding_boxes, landmarks).show()
'''

