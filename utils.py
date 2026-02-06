import re
import string

def clean_resume_text(text):
    """
    Cleans resume text by removing URLs, RT, hashtags, mentions, 
    punctuations, and extra whitespaces.
    """
    text = text.lower()
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]', r' ', text) # remove non-ascii characters
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text.strip()
