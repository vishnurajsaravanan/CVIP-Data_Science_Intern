import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load the Kaggle dataset (assuming you have the dataset CSV file in the same directory)
df = pd.read_csv('/content/spam.csv', encoding='latin-1')
# Assuming the dataset has columns 'v1' for labels and 'v2' for text
emails = df['v2'].values
labels = df['v1'].apply(lambda x: 1 if x == 'spam' else 0).values

# Create a Tokenizer to convert text data into sequences of numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(emails)

# Convert text data into sequences of numbers
sequences = tokenizer.texts_to_sequences(emails)

# Pad sequences to have the same length
max_sequence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predict whether a new email is spam or not spam
new_email = list(input("Enter the new mail: "))
new_email_sequence = tokenizer.texts_to_sequences(new_email)
new_email_sequence = pad_sequences(new_email_sequence, maxlen=max_sequence_length, padding='post')
prediction = model.predict(new_email_sequence)

if prediction[0][0] > 0.5:
    print("The new email is predicted as spam.")
else:
    print("The new email is predicted as not spam.")
