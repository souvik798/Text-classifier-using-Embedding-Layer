# Text-classifier-using-Embedding-Layer
## AIM
To create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D.

## PROBLEM STATEMENT AND DATASET
The program enables us to classify the given BBC dataset into its respective areas like different categories, for example buisness, sports and tech using Deep learning techniques, which includes loading and preprocessing the data, creating the neural network model, training and evaluation its performance. 
## Dataset

![image](https://github.com/user-attachments/assets/56eab5ae-7881-4bd2-ad4b-ac752d550686)


## DESIGN STEPS
### STEP 1: 
Unzip the zip file and load the BBC news dataset, split it into training and validation dataset.

### STEP 2:
Implement a function to convert the text into lower cases, remove the stop words and eliminate punctuation.

### STEP 3:
Create a TextVectorizer layer to tokenize and convert the dataset into sequences for model training.

### STEP 4:
Use TensorFlow's StringLookup layer to encode text labels into numerical format.

### STEP 5:
Build a model using an embedding layer, global average pooling, and dense layers for multi-class classification

### STEP 6:
Train the model for 30 epochs using the prepared training data and validate its performance on the validation set.

### STEP 7:
Evaluate the model's accuracy and loss, and plot the results to track performance over time.

## PROGRAM
### Name:Joyce Beulah R
### Register Number:212222230058
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import zipfile

with zipfile.ZipFile('/content/BBC News Train.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('extracted_data')
with open("extracted_data/BBC News Train.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"The second line (first data point) looks like this:\n\n{csvfile.readline()}")

# Define the global variables
VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRAINING_SPLIT = 0.8

data_dir = "/content/extracted_data/BBC News Train.csv"
data = np.loadtxt(data_dir, delimiter=',', skiprows=1, dtype='str', comments=None)
print(f"Shape of the data: {data.shape}")
print(f"{data[0]}\n{data[1]}")

# Test the function
print(f"There are {len(data)} sentence-label pairs in the dataset.\n")
print(f"First sentence has {len((data[0,1]).split())} words.\n")
print(f"The first 5 labels are {data[:5,2]}")

def train_val_datasets(data):
    # Define the training size (e.g., 80% of the total data)
    train_size = int(0.8 * len(data))
    
    # Slice the dataset to get texts and labels
    texts = data[:, 1]
    labels = data[:, 2]
    
    # Split the sentences and labels into train/validation sets
    train_texts = texts[:train_size]
    validation_texts = texts[train_size:]
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:]
    
    # Create the train and validation datasets from the splits
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_texts, validation_labels))
    
    
    return train_dataset, validation_dataset

# Create the datasets
train_dataset, validation_dataset = train_val_datasets(data)
print('Name:Joyce Beulah R       Register Number: 212222230058')
print(f"There are {train_dataset.cardinality()} sentence-label pairs for training.\n")
print(f"There are {validation_dataset.cardinality()} sentence-label pairs for validation.\n")

def standardize_func(sentence):
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here",  "hers", "herself", "him", "himself", "his", "how",  "i", "if", "in", "into", "is", "it", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",  "should", "so", "some", "such", "than", "that",  "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",  "were", "what",  "when", "where", "which", "while", "who", "whom", "why", "why", "with", "would", "you",  "your", "yours", "yourself", "yourselves", "'m",  "'d", "'ll", "'re", "'ve", "'s", "'d"]
 
    # Sentence converted to lowercase-only
    sentence = tf.strings.lower(sentence)
    
    # Remove stopwords
    for word in stopwords:
        if word[0] == "'":
            sentence = tf.strings.regex_replace(sentence, rf"{word}\b", "")
        else:
            sentence = tf.strings.regex_replace(sentence, rf"\b{word}\b", "")
    
    # Remove punctuation
    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']', "")
    
    return sentence


def fit_vectorizer(train_sentences, standardize_func):
    
    # Instantiate the TextVectorization class, passing in the correct values for the parameters
    vectorizer = tf.keras.layers.TextVectorization(
        standardize=standardize_func,            # Custom standardization function
        max_tokens=VOCAB_SIZE,                   # Maximum vocabulary size
        output_sequence_length=MAX_LENGTH        # Truncate sequences to this length
    )
    
    # Adapt the vectorizer to the training sentences
    vectorizer.adapt(train_sentences)
    
    
    return vectorizer


# Create the vectorizer
text_only_dataset = train_dataset.map(lambda text, label: text)
vectorizer = fit_vectorizer(text_only_dataset, standardize_func)
vocab_size = vectorizer.vocabulary_size()
print('Name:Joyce Beulah R       Register Number: 212222230058  ')
print(f"Vocabulary contains {vocab_size} words\n")
def fit_label_encoder(train_labels, validation_labels):
  
    # Concatenate the training and validation label datasets
    labels = train_labels.concatenate(validation_labels)
    
    # Instantiate the StringLookup layer without any OOV token
    label_encoder = tf.keras.layers.StringLookup(num_oov_indices=0)
    
    # Adapt the StringLookup layer on the combined labels dataset
    label_encoder.adapt(labels)
    
    
    return label_encoder

# Create the label encoder
train_labels_only = train_dataset.map(lambda text, label: label)
validation_labels_only = validation_dataset.map(lambda text, label: label)

label_encoder = fit_label_encoder(train_labels_only,validation_labels_only)
print('Name: Joyce Beulah R     Register Number: 212222230058    ')
print(f'Unique labels: {label_encoder.get_vocabulary()}')

def preprocess_dataset(dataset, vectorizer, label_encoder, batch_size=32):
    # Define a mapping function to preprocess each (text, label) pair
    def preprocess(text, label):
        text = vectorizer(text)                # Vectorize the text
        label = label_encoder(label)           # Encode the label
        return text, label
    
    # Map the preprocessing function to the dataset and batch it
    dataset = dataset.map(preprocess).batch(batch_size)
    
    return dataset
# Preprocess your dataset
train_proc_dataset = preprocess_dataset(train_dataset, vectorizer, label_encoder)
validation_proc_dataset = preprocess_dataset(validation_dataset, vectorizer, label_encoder)

train_batch = next(train_proc_dataset.as_numpy_iterator())
validation_batch = next(validation_proc_dataset.as_numpy_iterator())
print('Name: Joyce Beulah R     Register Number: 212222230058      ')
print(f"Shape of the train batch: {train_batch[0].shape}")
print(f"Shape of the validation batch: {validation_batch[0].shape}")

def create_model():
    
    # Define your model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    
    # Compile model. Set an appropriate loss, optimizer and metrics
    model.compile(
        loss='sparse_categorical_crossentropy',  # or 'categorical_crossentropy' if labels are one-hot encoded
        optimizer='adam',
        metrics=['accuracy'] 
    ) 

    return model

model = create_model()

example_batch = train_proc_dataset.take(1)

try:
	model.evaluate(example_batch, verbose=False)
except:
	print("Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another.")
else:
	predictions = model.predict(example_batch, verbose=False)
	print(f"predictions have shape: {predictions.shape}")

history = model.fit(train_proc_dataset, epochs=30, validation_data=validation_proc_dataset)

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()
print('Name: Joyce Beulah R   Register Number: 212222230058 ')
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")
```

## OUTPUT
### Loss, Accuracy Vs Iteration 
![image](https://github.com/user-attachments/assets/76fb0869-1507-4ab6-86a3-7d98effc9b8b)


![image](https://github.com/user-attachments/assets/d73510a4-b0f8-4033-b05b-bf0a099cb7e8)


## RESULT
Thus the program to create a classifier using specialized layers for text data such as Embedding and GlobalAveragePooling1D is implemented successfully.

