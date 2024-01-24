


import numpy as np


import tensorflow as tf 





from keras.preprocessing.image import ImageDataGenerator





import matplotlib.pyplot as plt





TRAIN_DIR = "C:\\Users\\your path to the training folder \\flowers\\Train"


# In[7]:


TEST_DIR="C:\\Users\\your path to the test folder\\flowers\\Test"


# In[8]:


VAL_DIR="C:\\Usersyour path to the validate  folder\\flowers\\Validate"


# In[9]:


train_datagen = ImageDataGenerator(
                    rescale = 1. / 255,
                    shear_range = 0.2,
                    zoom_range=0.2 ,
                    horizontal_flip=True)


# In[10]:


train_set = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(224,224), batch_size=32 , class_mode='categorical')


# In[11]:


val_datagen = ImageDataGenerator(rescale = 1. / 255)


# In[12]:


val_set = val_datagen.flow_from_directory(VAL_DIR, target_size=(224,224), batch_size=32 , class_mode='categorical')



# In[15]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32 , kernel_size = (5,5) , padding='Same', activation='relu', input_shape=[224,224,3])  )
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))

model.add(tf.keras.layers.Conv2D(filters=64 , kernel_size = (5,5) , padding='Same', activation='relu'  ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=96 , kernel_size = (5,5) , padding='Same', activation='relu'  ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(filters=96 , kernel_size = (5,5) , padding='Same', activation='relu'  ))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2) ))
model.add(tf.keras.layers.Dropout(0.5))


# In[16]:


model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense (units=512 , activation='relu'))


# In[17]:


model.add(tf.keras.layers.Dense(units=5 , activation='softmax'))

print( model.summary())


# In[19]:


model.compile(optimizer='rmsprop' , loss='categorical_crossentropy' , metrics=['accuracy']   )


# In[21]:


history = model.fit (x=train_set, validation_data=val_set, batch_size=32 , epochs=20)


# In[22]:


acc = history.history['accuracy']


# In[23]:


val_acc = history.history['val_accuracy']


# In[24]:


loss = history.history['loss']


# In[25]:


val_loss = history.history['val_loss']


# In[26]:


print(acc)


# In[27]:


print(val_acc)


# In[34]:


epochs_range = range(20) 


# In[35]:


plt.figure(figsize=(8,8))


# In[36]:


plt.subplot(1,2,1)


# In[38]:


plt.plot(epochs_range, acc, label = "Training Accuracy")
plt.plot(epochs_range, val_acc, label = "Validation Accuracy")
plt.legend(loc='lower right')
plt.title('Training and validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss ,label = "Training Loss")
plt.plot(epochs_range, val_loss, label = "Validation Loss")
plt.legend(loc='upper right')
plt.title('Training and validation Loss')

plt.show()


# In[43]:


model.save('C:\\Users\\adnan\\Desktop\\flowers\\flower.h5')


# In[ ]:




