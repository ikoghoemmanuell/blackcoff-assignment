# Importing necessary libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string
import re
from textstat import textstat

# Downloading NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define a set of stopwords downloaded from nltk
nltk_stop_words = set(stopwords.words('english'))

# Reading input data from Excel file
url_df = pd.read_excel('Input.xlsx')

# Initializing a list to store extracted data
extracted_data = []

# Iterating over each row in the DataFrame to extract data from URLs
for index, row in url_df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']

    # Sending a GET request to the URL
    response = requests.get(url)

    # Checking if the response is successful (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        article_title = soup.find('h1').text.strip()
        article_text = soup.find('div', class_='td-post-content')

        # Removing preformatted tags from the article text
        for tag in article_text.find_all('pre', class_='wp-block-preformatted'):
            tag.decompose()

        article_text = article_text.text.strip()

        # Storing extracted data in a dictionary
        extracted_data.append({'URL_ID': url_id, 'Title': article_title, 'Text': article_text})
    else:
        print(f"Failed to fetch URL: {url}")

# Creating a DataFrame from the extracted data
extracted_data_df = pd.DataFrame(extracted_data)

# Defining the path to the folder containing stop words text files
stopwords_folder_path = 'StopWords-20240507T120351Z-001\StopWords'

# Function to extract stop words from a text file
def extract_stopwords(file_path):
    stop_words = set()  # Use a set to store unique stop words
    encodings = ['utf-8', 'latin1', 'windows-1252']  # Add additional encodings as needed
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                for line in file:
                    # Split the line based on the pipe character (|)
                    stop_word = line.split('|')[0].strip()  # Extract the stop word and remove leading/trailing spaces
                    stop_words.add(stop_word)
            return stop_words
        except UnicodeDecodeError:
            pass
    # If all encodings fail
    raise Exception("Unable to decode file using any of the specified encodings")

# Initializing a dictionary to store stop words from each file
stopwords_dict = {}

# Iterating over each text file in the folder
for file_name in os.listdir(stopwords_folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(stopwords_folder_path, file_name)
        stop_words = extract_stopwords(file_path)
        stopwords_dict[file_name] = stop_words

# merge stop_words with nltk_stop_words
# stopwords_dict['nltk_stop_words'] = nltk_stop_words

# Now stopwords_dict will contain stop words extracted from each text file

# Converting all words in the 'Title' and 'Text' columns to lowercase
extracted_data_df['Title'] = extracted_data_df['Title'].str.lower()
extracted_data_df['Text'] = extracted_data_df['Text'].str.lower()

# Converting all stopwords in the stopwords dictionary to lowercase
for file_name, stop_words in stopwords_dict.items():
    stopwords_dict[file_name] = {word.lower() for word in stop_words}

# Function to remove stopwords from text using the dictionary of stopwords
def remove_stopwords(text, stopwords_dict):
    # Tokenizing the text into words
    words = text.split()
    # Initializing an empty list to store non-stopwords
    filtered_words = []
    # Iterating over each word in the text
    for word in words:
        # Checking if the word is not in any of the stopwords sets in the stopwords_dict
        if not any(word in stopwords_set for stopwords_set in stopwords_dict.values()):
            # If the word is not a stopword, add it to the filtered list
            filtered_words.append(word)
    # Joining the filtered words back into a single string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# Removing stopwords from the 'Title' column
extracted_data_df['Title'] = extracted_data_df['Title'].apply(lambda x: remove_stopwords(x, stopwords_dict))

# Removing stopwords from the 'Text' column
extracted_data_df['Text'] = extracted_data_df['Text'].apply(lambda x: remove_stopwords(x, stopwords_dict))

# Defining the path to the folder containing positive and negative words text files
master_folder_path = 'MasterDictionary-20240507T120347Z-001\MasterDictionary'

# Initializing a dictionary to store words from each file
positive_and_negative_words_dict = {}

# Iterating over each text file in the folder
for file_name in os.listdir(master_folder_path):
    if file_name.endswith('.txt'):
        file_path = os.path.join(master_folder_path, file_name)
        words = extract_stopwords(file_path)
        positive_and_negative_words_dict[file_name] = words

# Now positive_and_negative_words_dict will contain words extracted from each text file

for file_name, words in positive_and_negative_words_dict.items():
    stopwords_dict[file_name] = {word.lower() for word in words}

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

# Iterating over each row in the dataframe and calculate variables
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

# Function to calculate Average Sentence Length
def average_sentence_length(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    return len(words) / len(sentences)

# Function to calculate Percentage of Complex Words
def percentage_complex_words(text):
    words = word_tokenize(text)
    words_no_stopwords = [word for word in words if word not in stop_words and word not in string.punctuation and word not in nltk_stop_words]
    complex_words = [word for word in words_no_stopwords if textstat.syllable_count(word) > 2]

    complex_word_count = sum(1 for word in words if word in complex_words)
    return complex_word_count / len(words)

# Function to calculate Fog Index
def fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)

# Apply the functions to the DataFrame
extracted_data_df['Average Sentence Length'] = extracted_data_df['Text'].apply(average_sentence_length)
extracted_data_df['Percentage of Complex Words'] = extracted_data_df['Text'].apply(lambda x: percentage_complex_words(x))
extracted_data_df['Fog Index'] = fog_index(extracted_data_df['Average Sentence Length'], extracted_data_df['Percentage of Complex Words'])

# Function to calculate Average Sentence Length, Percentage of Complex Words, Fog Index, Word Count, Syllable Count Per Word, Personal Pronouns Count, Average Word Length
def calculate_metrics(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    words_no_stopwords = [word for word in words if word not in stop_words and word not in string.punctuation and word not in nltk_stop_words]
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

# Inserting the 'URL' column from the original DataFrame into the extracted DataFrame
extracted_data_df.insert(1, 'URL', url_df['URL'])

# Dropping unnecessary columns ('Text' and 'Title') from the extracted DataFrame
extracted_data_df.drop(columns=['Text', 'Title'], inplace=True)

# Saving the extracted data to an Excel file
extracted_data_df.to_excel('Output_cleaned.xlsx', index=False)
