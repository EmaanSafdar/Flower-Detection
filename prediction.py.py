#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


import tensorflow as tf


# In[4]:


from keras.preprocessing.image import ImageDataGenerator


# In[5]:


from keras.preprocessing import image


# In[6]:


flower_categories = ['daisy', 'dandelion' , 'rose', 'sunflower' , 'tulip']


# In[8]:


model = tf.keras.models.load_model('C:\\Users\xyz\\flowers\\flower.h5')


# In[9]:


print(model.summary())


# In[30]:


img_path='C:\\Users\\xyz\flowers\\daisy2.jpg'


# In[31]:


test_image = image.load_img(img_path,target_size=(224,224))


# In[32]:


test_image = image.img_to_array(test_image)
print(test_image.shape)


# In[33]:


test_image = np.expand_dims(test_image, axis=0)
print(test_image.shape)


# In[34]:


result = model.predict(test_image)
print(result)


# In[29]:


if result[0][0]==1:
    print('Flower is Daisy')
elif result[0][1]==1:
    print('Flower is Dandelion')
elif result[0][2]==1:
    print('Flower is Rose')
elif result[0][3]==1:
    print('Flower is SunFlower')
elif result[0][4]==1:
    print("Flower is Tulip")


# In[ ]:




