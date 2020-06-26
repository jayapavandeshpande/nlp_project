
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_cell_magic('capture', '', '!pip install ftfy fasttext pycountry regex translate')


import pandas as pd
import matplotlib.pyplot as plt
from ftfy import fix_text,badness
from fasttext import FastText
from pycountry import languages
from translate import Translator

def mojifix(df,col_num):
    j=0
#    df['mojibake'] = ''
    for i in range(df.shape[0]):
        if badness.sequence_weirdness(str(df[df.columns[col_num]][i]))!=0:
            j = j+1
            try:
                df[df.columns[col_num]][i] = fix_text(str(df[df.columns[col_num]][i]))
                df[df.columns[col_num]][i] = df[df.columns[col_num]][i].encode('sloppy-windows-1252')
                df[df.columns[col_num]][i] = fix_text(str(df[df.columns[col_num]][i]))
#                df['mojibake'][i] = badness.sequence_weirdness(df[df.columns[col_num]][i])
                
            except: Exception
        else:
            df[df.columns[col_num]][i] = df[df.columns[col_num]][i]
#            df['mojibake'][i] = badness.sequence_weirdness(df[df.columns[col_num]][i])

    return df,j

def detect_lang(df,col_1,col_2,summary=True):
    lid = fasttext.load_model('/lid/lid.176.ftz')
    
    df['iso_Short'] = ''
    df['iso_Desc'] = ''
    df['lang_Short'] = ''
    df['lang_Desc'] = ''

    
    for i in range(0,df.shape[0]):
        df['iso_Short'][i] = lid.predict(df[df.columns[col_1]][i])[0][0].split("__label__")[1]
        df['iso_Desc'][i] = lid.predict(df[df.columns[col_2]][i])[0][0].split("__label__")[1]
    
    for i in range(0,df.shape[0]):
        if len(df['iso_Short'][i])==3:
            df['lang_Short'][i] = languages.get(alpha_3=df['iso_Short'][i]).name
        else:
            df['lang_Short'][i] = languages.get(alpha_2=df['iso_Short'][i]).name 
        if len(df['iso_Desc'][i])==3:
            df['lang_Desc'][i] = languages.get(alpha_3=df['iso_Desc'][i]).name
        else:
            df['lang_Desc'][i] = languages.get(alpha_2=df['iso_Desc'][i]).name
    
    if summary==True:
        print("Count of 'Short description' in foreign language: ",df[df['iso_Short']!='en'].shape[0])
        print("No: of foreign languages in 'Short description': ",df['lang_Short'].value_counts()[1:].shape)
        print('\nTop-5 foreign languages:')
        print(df['lang_Short'].value_counts()[1:6])
        print('\n========================================================')
        print("\nCount of 'Description' in foreign language: ",df[df['iso_Desc']!='en'].shape[0])
        print("No: of foreign languages in 'Description': ",df['lang_Desc'].value_counts()[1:].shape)
        print('\nTop-5 foreign languages:')
        print(df['lang_Desc'].value_counts()[1:6])
    
    return df


def plot_lang(df,col_1,col_2):
    df = detect_lang(df,col_1,col_2,summary=False)
    fig, axarr = plt.subplots(1, 2, figsize=(12, 12))
    axarr[0].pie(df['lang_Short'].value_counts()[1:],autopct='%1.1f%%',pctdistance=1.1,startangle=180)
    axarr[0].axis('equal')
    axarr[0].legend(loc='right', labels=df['lang_Short'].value_counts()[1:].index.tolist())
    axarr[0].title.set_text('Distribution of languages other than English for "Short description')
    axarr[1].pie(df['lang_Desc'].value_counts()[1:],autopct='%1.1f%%',pctdistance=1.1,startangle=180)
    axarr[1].axis('equal')
    axarr[1].legend(loc='right', labels=df['lang_Desc'].value_counts()[1:].index.tolist())
    axarr[1].title.set_text('Distribution of languages other than English for "Description')
    fig.tight_layout()
    plt.show()
#    df.drop(['iso_Short','iso_Desc','lang_Short','lang_Desc'],axis=1,inplace=True)

def lang_translator(df,col_1,col_2):
    df['translated_Short'] = ''
    df['translated_Desc'] = ''
    lid = FastText.load_model('lid/lid.176.ftz')
    for i in range(df.shape[0]):
        try:
            if (lid.predict(df[df.columns[col_1]][i])[0][0].split("__label__")[1]!='en') & (lid.predict(df[df.columns[col_2]][i])[0][0].split("__label__")[1]!='en'):
                if (lid.predict(df[df.columns[col_1]][i])[0][0].split("__label__")[1]) == (lid.predict(df[df.columns[col_2]][i])[0][0].split("__label__")[1]):
                    lang = lid.predict(df[df.columns[col_1]][i])[0][0].split("__label__")[1]
                else:
                    lang = None
                df['translated_Short'][i] = Translator(provider='microsoft',from_lang=lang,to_lang='en',secret_access_key='884ec9c6954d4facb689de54383a912a').translate(str(df[df.columns[col_1]][i]))
                df['translated_Desc'][i] = Translator(provider='microsoft',from_lang=lang,to_lang='en',secret_access_key='884ec9c6954d4facb689de54383a912a').translate(str(df[df.columns[col_2]][i]))
            else:
                df['translated_Short'][i] = df[df.columns[col_1]][i]
                df['translated_Desc'][i] = df[df.columns[col_2]][i]
          
        except: Exception
        
    return df
    
def plot_lang(df,col_1,col_2):
    df = detect_lang(df,col_1,col_2,summary=False)
    fig, axarr = plt.subplots(1, 2, figsize=(12, 12))
    axarr[0].pie(df['lang_Short'].value_counts()[1:],autopct='%1.1f%%',pctdistance=1.1,startangle=180)
    axarr[0].axis('equal')
    axarr[0].legend(loc='right', labels=df['lang_Short'].value_counts()[1:].index.tolist())
    axarr[0].title.set_text('Distribution of languages other than English for "Short description')
    axarr[1].pie(df['lang_Desc'].value_counts()[1:],autopct='%1.1f%%',pctdistance=1.1,startangle=180)
    axarr[1].axis('equal')
    axarr[1].legend(loc='right', labels=df['lang_Desc'].value_counts()[1:].index.tolist())
    axarr[1].title.set_text('Distribution of languages other than English for "Description')
    fig.tight_layout()
    plt.show()
#    df.drop(['iso_Short','iso_Desc','lang_Short','lang_Desc'],axis=1,inplace=True)

def lang_translator_full(df,col_1,col_2):
    for i in range(df.shape[0]):
        try:
            df['Short description'][i] = Translator(provider='microsoft',from_lang=None,to_lang='en',secret_access_key='884ec9c6954d4facb689de54383a912a').translate(str(df[df.columns[col_1]][i]))
            df['Description'][i] = Translator(provider='microsoft',from_lang=None,to_lang='en',secret_access_key='884ec9c6954d4facb689de54383a912a').translate(str(df[df.columns[col_2]][i]))
        except: Exception
    
    return df