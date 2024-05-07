import pandas as pd

url_df = pd.read_excel('Input.xlsx')

url_df.head()

import requests
from bs4 import BeautifulSoup
import pandas as pd

extracted_data = []

for index, row in url_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        article_title = soup.find('h1').text.strip()
        article_text = soup.find('div', class_='td-post-content')

        for tag in article_text.find_all('pre', class_='wp-block-preformatted'):
            tag.decompose()

        article_text = article_text.text.strip()

        extracted_data.append({'URL_ID': url_id, 'Title': article_title, 'Text': article_text})

        pass
    else:
        print(f"Failed to fetch URL: {url}")

extracted_data_df = pd.DataFrame(extracted_data)

extracted_data_df.head()

import os

# Define the path to the folder containing stop words text files
stopwords_folder_path = r'C:\Users\LENOVO\Downloads\BlackCoffer\StopWords-20240507T120351Z-001\StopWords'

# Function to extract stop words from a text file
def extract_stopwords(file_path):
    stopwords = set()  # Use a set to store unique stop words
    encodings = ['utf-8', 'latin1', 'windows-1252']  # Add additional encodings as needed
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    # Split the line based on the pipe character (|)
                    stop_word = line.split('|')[0].strip()  # Extract the stop word and remove leading/trailing spaces
                    stopwords.add(stop_word)
            return stopwords
        except UnicodeDecodeError:
            pass
    # If all encodings fail
    raise Exception("Unable to decode file using any of the specified encodings")

# Initialize a dictionary to store stop words from each file
stopwords_dict = {}

# Iterate over each text file in the folder
for file_name in os.listdir(stopwords_folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(stopwords_folder_path, file_name)
        stopwords = extract_stopwords(file_path)
        stopwords_dict[file_name] = stopwords

# Now stopwords_dict will contain stop words extracted from each text file

# Convert all words in the 'Title' and 'Text' columns to lowercase
extracted_data_df['Title'] = extracted_data_df['Title'].str.lower()
extracted_data_df['Text'] = extracted_data_df['Text'].str.lower()

# Convert all stopwords in the stopwords dictionary to lowercase
for file_name, stopwords in stopwords_dict.items():
    stopwords_dict[file_name] = {word.lower() for word in stopwords}

# Function to remove stopwords from text using the dictionary of stopwords
def remove_stopwords(text, stopwords_dict):
    # Tokenize the text into words
    words = text.split()
    # Initialize an empty list to store non-stopwords
    filtered_words = []
    # Iterate over each word in the text
    for word in words:
        # Check if the word is not in any of the stopwords sets in the stopwords_dict
        if not any(word in stopwords_set for stopwords_set in stopwords_dict.values()):
            # If the word is not a stopword, add it to the filtered list
            filtered_words.append(word)
    # Join the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Remove stopwords from the 'Title' column
extracted_data_df['Title'] = extracted_data_df['Title'].apply(lambda x: remove_stopwords(x, stopwords_dict))

# Remove stopwords from the 'Text' column
extracted_data_df['Text'] = extracted_data_df['Text'].apply(lambda x: remove_stopwords(x, stopwords_dict))

extracted_data_df['Text'][0]

import os

# Define the path to the folder containing stop words text files
stopwords_folder_path = r'C:\Users\LENOVO\Downloads\BlackCoffer\MasterDictionary-20240507T120347Z-001\MasterDictionary'

# Function to extract stop words from a text file
def extract_stopwords(file_path):
    stopwords = set()  # Use a set to store unique stop words
    encodings = ['utf-8', 'latin1', 'windows-1252']  # Add additional encodings as needed
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    # Split the line based on the pipe character (|)
                    stop_word = line.split('|')[0].strip()  # Extract the stop word and remove leading/trailing spaces
                    stopwords.add(stop_word)
            return stopwords
        except UnicodeDecodeError:
            pass
    # If all encodings fail
    raise Exception("Unable to decode file using any of the specified encodings")

# Initialize a dictionary to store stop words from each file
positive_and_negative_words_dict = {}

# Iterate over each text file in the folder
for file_name in os.listdir(stopwords_folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(stopwords_folder_path, file_name)
        stopwords = extract_stopwords(file_path)
        positive_and_negative_words_dict[file_name] = stopwords

# Now positive_and_negative_words_dict will contain words extracted from each text file

for file_name, stopwords in positive_and_negative_words_dict.items():
    stopwords_dict[file_name] = {word.lower() for word in stopwords}

import nltk

# Download the nltk tokenizer if not already downloaded
nltk.download('punkt')

# Function to calculate Positive Score
def calculate_positive_score(text):
    tokens = nltk.word_tokenize(text)
    positive_words = positive_and_negative_words_dict.get('positive-words.txt', set())
    positive_score = sum(1 for token in tokens if token.lower() in positive_words)
    return positive_score

# Function to calculate Negative Score
def calculate_negative_score(text):
    tokens = nltk.word_tokenize(text)
    negative_words = positive_and_negative_words_dict.get('negative-words.txt', set())
    negative_score = sum(1 for token in tokens if token.lower() in negative_words)
    return negative_score

# Function to calculate Polarity Score
def calculate_polarity_score(positive_score, negative_score):
    denominator = positive_score + negative_score + 0.000001  # Add a small value to avoid division by zero
    polarity_score = (positive_score - negative_score) / denominator
    return polarity_score

# Function to calculate Subjectivity Score
def calculate_subjectivity_score(positive_score, negative_score, total_words):
    denominator = total_words + 0.000001  # Add a small value to avoid division by zero
    subjectivity_score = (positive_score + negative_score) / denominator
    return subjectivity_score

# Iterate over each row in the dataframe and calculate variables
for index, row in extracted_data_df.iterrows():
    text = row['Text']
    total_words = len(nltk.word_tokenize(text))
    positive_score = calculate_positive_score(text)
    negative_score = calculate_negative_score(text)
    polarity_score = calculate_polarity_score(positive_score, negative_score)
    subjectivity_score = calculate_subjectivity_score(positive_score, negative_score, total_words)

    # Update the dataframe with the calculated values
    extracted_data_df.at[index, 'Positive Score'] = positive_score
    extracted_data_df.at[index, 'Negative Score'] = negative_score
    extracted_data_df.at[index, 'Polarity Score'] = polarity_score
    extracted_data_df.at[index, 'Subjectivity Score'] = subjectivity_score

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download NLTK resources
nltk.download('punkt')

# Function to calculate Average Sentence Length
def average_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences)

# Function to calculate Percentage of Complex Words
def percentage_complex_words(text, complex_words):
    words = word_tokenize(text)
    complex_word_count = sum(1 for word in words if word in complex_words)
    return complex_word_count / len(words)

# Define a set of complex words
# You can use a list of complex words obtained from a dictionary or any other reliable source
complex_words = set(['complex', 'difficult', 'advanced', 'technical', 'sophisticated'])  # Example set of complex words

# Function to calculate Fog Index
def fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Apply the functions to the DataFrame
extracted_data_df['Average Sentence Length'] = extracted_data_df['Text'].apply(average_sentence_length)
extracted_data_df['Percentage of Complex Words'] = extracted_data_df['Text'].apply(lambda x: percentage_complex_words(x, complex_words))
extracted_data_df['Fog Index'] = fog_index(extracted_data_df['Average Sentence Length'], extracted_data_df['Percentage of Complex Words'])

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from textstat import textstat

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define a set of stopwords
stop_words = set(stopwords.words('english'))

# Function to calculate Average Sentence Length, Percentage of Complex Words, Fog Index, Word Count, Syllable Count Per Word, Personal Pronouns Count, Average Word Length
def calculate_metrics(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words_lower = [word.lower() for word in words]
    words_no_stopwords = [word for word in words_lower if word not in stop_words and word not in string.punctuation]
    complex_words = [word for word in words_no_stopwords if textstat.syllable_count(word) > 2]
    personal_pronouns = len(re.findall(r'\b(?:I|we|my|ours|us)\b', text, flags=re.IGNORECASE))

    avg_sentence_length = len(words) / len(sentences)
    percentage_complex_words = len(complex_words) / len(words)
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    word_count = len(words_no_stopwords)
    syllable_count_per_word = sum(textstat.syllable_count(word) for word in words) / len(words)
    avg_word_length = sum(len(word) for word in words) / len(words)

    return avg_sentence_length, percentage_complex_words, fog_index, word_count, syllable_count_per_word, personal_pronouns, avg_word_length

# Apply the function to the DataFrame
extracted_data_df[['Average Sentence Length', 'Percentage of Complex Words', 'Fog Index', 'Word Count', 'Syllable Count Per Word', 'Personal Pronouns', 'Average Word Length']] = extracted_data_df['Text'].apply(calculate_metrics).apply(pd.Series)

extracted_data_df.head(1)

extracted_data_df.insert(1, 'URL', url_df['URL'])
extracted_data_df.drop(columns=['Text', 'Title'], inplace=True)
extracted_data_df.to_excel('Output.xlsx', index=False)

