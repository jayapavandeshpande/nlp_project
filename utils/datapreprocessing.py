# Data Preprocessing Util
import re
from rake_nltk import Rake
from gensim.summarization.summarizer import summarize

def removeString(data, regex):
    return data.str.lower().str.replace(regex.lower(), ' ')


def getRegexList():
    """
     Adding regex list as per the given data set to flush off the unnecessary text
    :return:
    """
    regex_list = []
    regex_list += ['From:(.*)\r\n']  # from line
    regex_list += ['Sent:(.*)\r\n']  # sent to line
    regex_list += ['received from:(.*)\r\n']  # received data line
    regex_list += ['received']  # received data line
    regex_list += ['To:(.*)\r\n']  # to line
    regex_list += ['CC:(.*)\r\n']  # cc line
    regex_list += ['https?:[^\]\n\r]+']  # https & http
    regex_list += ['[\w\d\-\_\.]+@[\w\d\-\_\.]+']  # emails are not required
    regex_list += ['[0-9][\-0–90-9 ]+']  # phones are not required
    regex_list += ['[0-9]']  # numbers not needed
    regex_list += ['[^a-zA-z 0-9]+']  # anything that is not a letter
    regex_list += ['[\r\n]']  # \r\n
    regex_list += [' [a-zA-Z] ']  # single letters makes no sense
    regex_list += [' [a-zA-Z][a-zA-Z] ']  # two-letter words makes no sense
    regex_list += ["  "]  # double spaces

    regex_list += ['^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$']
    regex_list += ['[\w\d\-\_\.]+ @ [\w\d\-\_\.]+']
    regex_list += ['Subject:']
    regex_list += ['[^a-zA-Z]']

    return regex_list


def cleanDataset(dataset, column, regex_list):
    for regex in regex_list:
        dataset[column] = removeString(dataset[column], regex)
    return dataset


def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols&  pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"log in", "login", text)
    text = re.sub(r"log on", "login", text)
    text = re.sub(r"logon", "login", text)
    text = re.sub(r"logged", "login", text)
    text = re.sub(r"logging", "login", text)
    text = re.sub(r"tologin", "login", text)

    text = re.sub(r"Job_", "job ", text)
    text = re.sub(r"mm_", "mm ", text)

    # 6th June Edit suggested by Charan/Vaishakh
    text = re.sub(r"time cards", "timecards", text)
    text = re.sub(r"engg", "engineer", text)
    text = re.sub(r"nx 9", "nx9", text)
    text = re.sub(r"installing", "install", text)
    text = re.sub(r"installation", "install", text)

    text = re.sub(r"us time", "US time", text)  

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    text = re.sub(r"ain't", "am not", text)
    text = re.sub(r"aren't", "are not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"can't've", "cannot have", text)
    text = re.sub(r"'cause", "because", text)
    text = re.sub(r"could've", "could have", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"couldn't", "could not",  text)
    text = re.sub(r"couldn't've", "could not have", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"doesn't", "does not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"hadn't", "had not", text)
    text = re.sub(r"hadn't've", "had not have", text)
    text = re.sub(r"hasn't", "has not", text)
    text = re.sub(r"haven't", "have not", text)
    text = re.sub(r"he'd", "he would", text)
    text = re.sub(r"he'd've", "he would have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"how'd", "how did", text)
    text = re.sub(r"how'll", "how will", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"i'd", "i would", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"i've", "i have", text)
    text = re.sub(r"isn't", "is not", text)
    text = re.sub(r"it'd", "it would", text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r"ma'am", "madam", text)
    text = re.sub(r"mayn't", "may not", text)
    text = re.sub(r"might've", "might have", text)
    text = re.sub(r"mightn't", "might not", text)
    text = re.sub(r"must've", "must have", text)
    text = re.sub(r"mustn't", "must not", text)
    text = re.sub(r"needn't", "need not", text)
    text = re.sub(r"oughtn't", "ought not", text)
    text = re.sub(r"shan't", "shall not", text)
    text = re.sub(r"sha'n't", "shall not", text)
    text = re.sub(r"she'd", "she would", text)
    text = re.sub(r"she'll", "she will", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"that'd", "that would", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there'd", "there had", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they're", "they are", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"we're", "we are", text)
    text = re.sub(r"we've", "we have", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"what'll", "what will", text)
    text = re.sub(r"what're", "what are", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"what've", "what have", text)
    text = re.sub(r"where'd", "where did", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"who'll", "who will", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"wouldn't", "would not", text)
    text = re.sub(r"you'd", "you would", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"you're", "you are", text)

    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

r = Rake()

def rake_implement(x) :
    r.extract_keywords_from_text(x)
    return r.get_ranked_phrases()


def create_summarized_feature(x):
    str_local = ""
    try :
            if len(x.split()) > 200:
                str_local = summarize(x, word_count = 200)
            else:
                str_local = x
                
    except ValueError:
        str_local_Error = ". ".join(rake_implement(x))
        str_local = summarize(str_local_Error, word_count = 200)
        print("Can't Summarize this sentence as input has only one sentence. Hence, replacing with (Rake + Summarized Value)" )
    return str_local