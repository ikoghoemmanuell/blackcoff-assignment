# Blackcoff Internship Assignment

# Part 1. How I approached the solution

## My Initial Approach:

When approaching the solution, my first step was to break it down into actionable steps. For instance, I tackled the initial challenge of data extraction by employing Python code. This allowed me to systematically solve each problem. A crucial tool I utilized from the outset was a Jupyter notebook, which enabled me to run Python code and monitor progress step-by-step, observing results at each stage.

![image](https://github.com/ikoghoemmanuell/blackcoff-assignment/assets/102419217/de601c61-aff7-405c-9b6b-79c13a54c749)


Although I later converted the Jupyter notebook into a Python .py file, ensuring I could submit my code as a Python file as requested.
 
After completing the solution and converting it into a Python file, my focus shifted to refining the code. I concentrated on cleaning it up to enhance readability, reusability, and efficiency. This involved removing any redundant or unnecessary repetitions, ensuring that the code was streamlined and optimized for future use.

![image](https://github.com/ikoghoemmanuell/blackcoff-assignment/assets/102419217/0cf63713-ee1e-40eb-b78d-a7e3acf71775)


One important observation arose during the extraction process: while retrieving the article text from each URL, I noticed additional text was appended. To investigate further, I inspected the HTML structure of each URL using Google Chrome. This inspection revealed the presence of a "pre" tag with a class named 'wp-block-preformatted', nestled within the "div" tag containing the article text. Consequently, I adjusted the Python code to exclude this portion from the article text during extraction.
 

# Code Overview:

Here's a concise overview of the python code I wrote to solve the problem and the logic behind it:

The script starts by importing necessary libraries and downloading NLTK resources. It then reads input data from an Excel file, containing URL IDs and URLs. Next, it iterates over each URL, extracts article titles and texts using BeautifulSoup, and removes preformatted tags from the text.

The extracted data is stored in a DataFrame and saved to individual text files. Stop words are extracted from text files and combined with NLTK stop words. These stop words are then removed from the article titles and texts.

Sentiment scores (positive, negative, polarity, and subjectivity) are calculated for each article text. Additionally, text metrics such as average sentence length, percentage of complex words, fog index, word count, syllable count per word, personal pronouns count, and average word length are computed.

The script inserts the 'URL' column from the original DataFrame into the extracted DataFrame and drops unnecessary columns ('Text' and 'Title'). Finally, the cleaned extracted data is saved to an Excel file named 'Output_cleaned.xlsx'.
 

# Part 2: How to run the .py file to generate output

Firstly, you'll need to clone this repository, which contains all the necessary files. Then, in your terminal, execute the following command:

**Setup**
Install the required packages to be able to run the evaluation locally.
You need to have Python 3 on your system (a Python version lower than 3.10). After cloning the repository and being at the repo's root:

**Windows:**
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

**Linux & macOS:**
python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt
Both long command-lines have the same structure; they pipe multiple commands using the symbol “**;”**. However, you may manually execute them one after another.

1. Create Python's virtual environment to isolate the required libraries of the project to avoid conflicts.
2. Activate the Python's virtual environment so that the Python kernel & libraries will be those of the isolated environment.
3. Upgrade Pip, the installed libraries/packages manager, to have the up-to-date version that will work correctly.
4. Install the required libraries/packages listed in the **requirements.txt** file so that it will allow importing them into the Python scripts and notebooks without any issues.

**NB:** For macOS users, please install Xcode if you encounter any issues.


# Part 3: Dependencies required

Now that you have created your virtual environment with all the ***requirements***, you can run the Python file using the created virtual environment in your terminal. Once it's done running, you'll have the output Excel file format, as well as the extracted articles for each URL, all within the cloned repository's location.
