import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

#encodeing
from sklearn.preprocessing import LabelEncoder

#deep model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

#dataset load
print("Loading dataset.....")

with open("array/images.p","rb") as f:
    images = pickle.load(f)
with open("array/labels.p","rb") as f:
    labels = pickle.load(f)

print("image_shape:",images.shape)
print("Label_shape",labels.shape)

#encoding
l = LabelEncoder()
labels = l.fit_transform(labels)

print("\nClasses Found:")
print(set(labels))

p = len(set(labels))
print("\nTotal Number of Person: ",p)

#plt
plt.imshow(images[60],cmap='gray')
plt.title("sample image")
plt.show()

#img preprocessing
def preprocess(img):
    img = cv2.equalizeHist(img)
    img = img.reshape(100,100,1)
    img = img / 255.0
    return img

images = np.array(list(map(preprocess,images)))
print("\nPreprocessed image shape:",images.shape)

#to categorical conver
labels = to_categorical(labels)
print("Output label shape:",labels.shape)

#model create
model = Sequential()
#fir con or input
model.add(Conv2D(30,(5,5),activation='relu',input_shape=(100,100,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(15,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#flatten
model.add(Flatten())
#Fully conn
#hidden
model.add(Dense(50,activation='relu'))
#output
model.add(Dense(p,activation='softmax'))
#compile
model.compile(optimizer=Adam(learning_rate=0.01),
              loss="categorical_crossentropy",
              metrics=['accuracy'])

#model.fit
model.fit(images,labels,validation_split=0.1,
          epochs=10,batch_size=32)

#save model
model.save("Final_model.h5")
print("\nModel save Successfully")