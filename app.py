import streamlit as st
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import nltk

# Download NLTK resources if not already downloaded
nltk.download("stopwords")

# Function for text preprocessing
def text_preprocessing(dataframe, dependent_var):
    # Normalizing Case Folding - Uppercase to Lowercase
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: x.lower())

    # Removing Punctuation
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: ''.join([char for char in x if char not in string.punctuation]))

    # Removing Numbers
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: ''.join([char for char in x if not char.isdigit()]))

    # StopWords
    sw = stopwords.words('english')
    dataframe[dependent_var] = dataframe[dependent_var].apply(lambda x: ' '.join([word for word in x.split() if word not in sw]))

    return dataframe

# Load dataset
df = pd.read_csv("reviews1.csv")

# Preprocess text
df = text_preprocessing(df, "Description")

# Define X and y
X = df["Description"]
# Assuming there's no sentiment label in the dataset, let's create a placeholder label
y = df["Stars"].apply(lambda x: "pos" if x >= 4 else "neg")

# Define CountVectorizer
count_vectorizer = CountVectorizer()
x_train_count_vectorizer = count_vectorizer.fit_transform(X)

# Define train_y
train_y = y

# Load the trained model
@st.cache_data()
def load_model(_x_train_count_vectorizer, train_y):
    model = RandomForestClassifier()
    model.fit(_x_train_count_vectorizer, train_y)
    return model

# Function to predict sentiment
def predict_sentiment(model, comment):
    comment = text_preprocessing(pd.DataFrame({"Description": [comment]}), "Description")["Description"][0]
    new_comment = pd.Series(comment)
    new_comment = count_vectorizer.transform(new_comment)
    result = model.predict(new_comment)
    return result

# Title
st.title('Sentiment Analysis App')

# Sidebar
st.sidebar.title('About')
st.sidebar.info(
    "This web app performs sentiment analysis using a Random Forest Classifier model trained on a dataset of reviews."
)

# User input
user_input = st.text_area("Enter your review:")

# Predict
if st.button('Predict'):
    model = load_model(x_train_count_vectorizer, train_y)
    result = predict_sentiment(model, user_input)
    if result == "pos":
        st.success("Positive sentiment!")
    else:
        st.error("Negative sentiment!")
