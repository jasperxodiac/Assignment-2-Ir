import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
from tensorflow.keras.applications import ResNet50
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os
from sklearn.metrics.pairwise import cosine_similarity
# Load the dataset
data = pd.read_csv('A.csv')

# 1. Image Feature Extraction
# a. Image preprocessing
def preprocess_image(img_url):
    response = requests.get(img_url)
    if response.status_code == 200:
        try:
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (224, 224))
            img = np.array(img) / 255.0
            return img
        except (PIL.UnidentifiedImageError, Exception) as e:
            print(f"Error: {e}")
            return None
    else:
        print(f"Error: Failed to fetch image, status code: {response.status_code}")
        return None

# b. Extract features using ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Loading saved image features if available 
if os.path.exists('image_features.pkl'):
    with open('image_features.pkl','rb') as f:
        image_features = pickle.load(f)
        print("Image features loaded from 'image_features.pkl'!")
else:
    image_features = {}
    for index, row in data.iterrows():
        img_urls = row['Image'].strip("[]").split(', ')  # Split the image URLs
        features = []
        for img_url in img_urls:
            img_url.replace("''","")
            img_url = img_url[1:-1]
            #print(img_url)
            img = preprocess_image(img_url)
            if img is not None:
                feature = resnet_model.predict(np.expand_dims(img, axis=0))
                features.append(feature.squeeze())
        if features:
            image_features[row['Review Text']] = np.mean(features, axis=0)  # Store the average of features

    print("Image Features Retrieved Using ResNEt50")

    # c. Normalize features
    image_features = {k: v / np.linalg.norm(v) for k, v in image_features.items()}

    # Save image features
    with open('image_features.pkl', 'wb') as f:
        pickle.dump(image_features, f)

# 2. Text Feature Extraction

# a. Text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Lowercasing
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Remove punctuations
        tokens = [token for token in tokens if token not in string.punctuation]

        # Remove blank space tokens
        tokens = [token for token in tokens if token.strip()]

        # Save preprocessed text
        ptext = ' '.join(tokens)

        return ptext
    else:
        return ''  # or any other default value you prefer


# b. Calculate TF-IDF tfidf_scores
if os.path.exists('tfidf_scores.pkl'):
    with open('tfidf_scores.pkl', 'rb') as f:
        vectorizer, tfidf_matrix = pickle.load(f)
        print("TF-IDF scores loaded from 'tfidf_scores.pkl'")
else:
    vectorizer = TfidfVectorizer()
    text_reviews = data['Review Text'].apply(preprocess_text)
    tfidf_matrix = vectorizer.fit_transform(text_reviews)

    # Save TF-IDF scores
    with open('tfidf_scores.pkl', 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix), f)

# 3. Image Retrieval and Text retrieval
# a. Image retrieval
def image_retrieval(query_img_urls, top_k=3):
    features = []
    for img_url in query_img_urls:
        img = preprocess_image(img_url.strip("'"))
        if img is not None:
            feature = resnet_model.predict(np.expand_dims(img, axis=0)).squeeze()
            features.append(feature.flatten()) # Flatten feature vectors

    if not features:
        return []
    
    query_feature = np.mean(features, axis=0)
    query_feature /= np.linalg.norm(query_feature)

    similarities = {}
    for k, v in image_features.items():
        similarities[k] = cosine(query_feature, v.flatten()) # Flatten image vectors

    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])
    top_k_images = sorted_similarities[:top_k]
    return top_k_images

# b. Text retrieval
def text_retrieval(query_text, top_k=3):
    query_text = preprocess_text(query_text)
    query_tfidf = vectorizer.transform([query_text])

    similarities = cosine_similarity(query_tfidf, tfidf_matrix).squeeze()
    top_k_indices = similarities.argsort()[-top_k:][::-1]
    top_k_reviews = [(data.iloc[idx]['Review Text'], similarities[idx]) for idx in top_k_indices]
    return top_k_reviews

# 4. Combined Retrieval
def combined_retrieval(query_img_urls, query_text, top_k=3):
    image_results = image_retrieval(query_img_urls, top_k)
    text_results = text_retrieval(query_text, top_k)

    combined_results = []
    for review, img_sim in image_results:
        for review_text, text_sim in text_results:
            if review == review_text:
                combined_sim = (img_sim + text_sim) / 2
                combined_results.append((review, combined_sim))

    combined_results = sorted(combined_results, key=lambda x: x[1], reverse=True)[:top_k]
    return combined_results

# 5. Results and Analysis

def analysingquery(query_img_urls, query_text):

    top_k = 3
    
    # Image retrieval
    print('Image Retrieval:')
    try:
        image_results = image_retrieval(query_img_urls, top_k)
        if not image_results:
            print("No similar images found.") 
        else:
            for review, img_sim in image_results:
                print(f'Review: {review}')
                print(f'Cosine Similarity: {1 - img_sim}')
                print('---')
    except Exception as e:
        print("Error in image retrieval:", e)
        
    # Text retrieval
    print('\nText Retrieval:')
    try: 
        text_results = text_retrieval(query_text, top_k)
        if not text_results:
            print("No similar reviews found.")
        else:
            for review_text, text_sim in text_results:
                print(f'Review: {review_text}')
                print(f'Cosine Similarity: {text_sim}')
                print('---')
    except Exception as e:
        print("Error in text retrieval:", e)

    # Combined retrieval
    print('\nCombined Retrieval:')
    try:
        combined_results = combined_retrieval(query_img_urls, query_text, top_k)
        if not combined_results:
            print("No similar (image, review) pairs found.")
        else:
            for review, combined_sim in combined_results:
                print(f'Review: {review}')
                print(f'Combined Similarity Score: {combined_sim}')
                print('---')
    except Exception as e:
        print("Error in combined retrieval:", e)

if __name__ == "__main__":

    n = int(input("Enter the number of queries: "))
    
    for i in range(n):
        image_links = input("Enter the links of the images (comma-separated): ").split(',') 
        text_review = input("Enter the review: ")
        
        analysingquery(image_links, text_review)

