import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Embedding,LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
data=pd.read_csv("Reviews.csv")

#print(data.shape)
data = data.replace({'label':{'positive':1,'negative':0}}).infer_objects(copy=False)
#print(data.head())
train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)
#print(train_data.shape)
#print(test_data.shape)
tokenizer=Tokenizer(num_words=500)
test=tokenizer.fit_on_texts(train_data['Text'])
X_train=pad_sequences(tokenizer.texts_to_sequences(train_data["Text"]),maxlen=100)
X_test=pad_sequences(tokenizer.texts_to_sequences(test_data["Text"]),maxlen=100)

Y_train=train_data['label']
Y_test=test_data['label']
#print(Y_train)
#print(Y_test)
'''
model=Sequential()
model.add(Embedding(input_dim=1000,output_dim=12,input_length=200))
model.add(LSTM(64,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation="sigmoid"))
model.build(input_shape=(None, 200))
#model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_test, Y_test))
#print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=5,batch_size=32,validation_split=0.2)
model.save("reviews.h5")
loss,accuracy=model.evaluate(X_test,Y_test)
'''
#import joblib
#joblib.dump(tokenizer,"tokenizer.pkl") 
model=load_model("reviews.h5")
#loss,accuracy=model.evaluate(X_test,Y_test)
def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    paddedSequence = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(paddedSequence)
    predicted_label = 1 if prediction[0][0] > 0.5 else 0
    print(f"Prediction Probability: {prediction[0][0]}")
    print(f"Predicted Label: {'Review is Positive' if predicted_label == 1 else 'Review is Negative'}")
    return predicted_label

result = predictive_system("The product had quality issues- it was torn and had dirt all over. It was very badly packaged")

