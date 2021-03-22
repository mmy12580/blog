---
title: "NlP预处理常用"
date: 2019-09-19
tags: ['natural language processing']
categories: ['nlp', 'python']
---

NLP的下游任务(downstream)，需要对应的预处理工作。在不同的语言之间，也有不同的处理方式。在我的一些工作中，我能发现，一个灵活可拓展的预处理方案，可以在调节模型的情况下，增加很多的效率。在这里我会列举一些常用的预处理方案，感兴趣的童鞋，可以直接从对应的code section中获取，以便于你们设计自己的NLP项目。



## 去除非文本部分

这里要$\color{red}{\text{特意}}$说一句，如果你们在做的任务是$\color{blue}{\text{语言模型（language model)}}$, 或者是利用$\color{blue}{\text{预训练模型（pre-training)}}$, e.g., Bert, Xlnet, ERNIE, Ulmfit, Elmo, etc.，可能有些非文本部分是需要保留的，首先我们来看看哪些是非文本类型数据

1. 数字 (digit/number)
2. 括号内的内容 (content in brackets)
3. 标点符号 (punctuations)
4. 特殊符号（special symbols)

````python
import re
import sys
import unicodedata

# number 

````python
number_regex = re.compile(r"(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))")

# puncuation with unicode
punct_regex = dict.fromkeys(
    (i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")),"")
r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
````


## 引号转换
由于输入的问题，很多文字在非英语的一些情况下会出现不同的引号。比如中文输入法里，会出现$\color{red}{\text{全角}}$和$\color{red}{\text{半角}}$的两种选择。一种是跟英文一样，另一种会出现不同的类型，这里也全部概括了。可用于多类型的处理。
````python
# double quotes
double_quotes = ["«", "‹", "»", "›", "„", "“", "‟", "”", "❝", "❞",
"❮", "❯", "〝", "〞", "〟","＂",]

# single quotes
single_quotes = ["‘", "‛", "’", "❛", "❜", "`", "´", "‘", "’"]

# define related regex
double_quote_regex = re.compile("|".join(double_quotes))
single_quote_regex = re.compile("|".join(single_quotes))
````


## Unicode修复
简单的unicode修复，这里使用了第三方package `ftfy`。感兴趣的童鞋可以去[ftfy pypi](https://pypi.org/project/ftfy/) 查询一些对你有用的预处理方案，这里就不详细介绍了。

````python
from ftfy import fix_text

def fix_unicode(text, method='NFC'):
	"""Available methods: ['NFC', 'NFKC', 'NFD', 'NFKD']"""
	try:
		text = text.encode().decode('unicode-escape')
	except:
		pass

	return fix_text(text, normalization=method)
````

## ascii-unicode转换
这里需要注意的事情是，每一种语言有一些自己的错误。我大部分做的是中文和英文，所以很多时候不需要额外的处理一些太多的语言问题。但是，对于很多欧洲的语种，例如德语，是需要进行额外的特殊例的catch和修复.

````python
def to_ascii_unicode(text, lang="en"):
	"""A wrapper for unicode converting"""
	return unidecode(text)
````

## 空格清理
空格清理是非常常见的。大概源于本身数据读取的特点，以及清理完数据后，将不需要的特殊符号换成空白，网络文件中多空格和无空格是一个我们经常需要处理的情况。这里也特地写了三种类型。需要$\color{red}{\text{注意}}$的是，这里只是给通用的数据处理并进行下游模型训练。如果你关注最新的NLP动态，其实可以参考一下**BERT**的预处理python[脚本](https://github.com/google-research/bert/blob/master/tokenization.py)。

1. 分行处理 (linebreak)
2. 多空格处理 (mul_whitespace)
3. 不间断（增加空格）处理 (nonbreaking_space)


````python
linebreak_regex = re.compile(r"((\r\n)|[\n\v])+")
multi_whitespace_regex = re.compile(r"\s+")
nonbreaking_space_regex = re.compile(r"(?!\n)\s+")
````

## emoij清理

同样，这里需要有一个$\color{red}{\text{提醒}}$，对于正在研究$\color{blue}{\text{情感分析}}$的童鞋。Emoji，在帮助判断情感类别的时候是可以起到一定量的作用的，甚至是在Aspect-based sentiment analysis (ABSA), $\color{blue}{\text{细粒度情感分析}}$也是有一定作用的。

````python
import emoji
import unicodedata

# 第一种
def remove_emoji(text):
	return emoji.demojize(text)

# 第二种, build-in unicode data
def replace_emoji(self, input_string):
    for character in input_string:
        try:
            character.encode("ascii")
            return_string += character
        except UnicodeEncodeError:
            replaced = str(character)
            if replaced != '':
                return_string += replaced
            else:
                try:
                    return_string += "[" + unicodedata.name(character) + "]"
                except ValueError:
                    return_string += "[x]"
    return return_string
````

## 常见的标签化清理

1. 链接 (url)
2. 邮件地址 (email address)
3. 电话号码 (phone numbers)
4. 货币转换 (currency exchange)


````python
# 1. url, source: https://gist.github.com/dperini/729294
url_regex= re.compile(
    r"(?:^|(?<![\w\/\.]))"
    # protocol identifier
    # r"(?:(?:https?|ftp)://)"  <-- alt?
    r"(?:(?:https?:\/\/|ftp:\/\/|www\d{0,3}\.))"
    # user:pass authentication
    r"(?:\S+(?::\S*)?@)?" r"(?:"
    # IP address exclusion
    # private & local networks
    r"(?!(?:10|127)(?:\.\d{1,3}){3})"
    r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    r"|"
    # host name
    r"(?:(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)"
    # domain name
    r"(?:\.(?:[a-z\\u00a1-\\uffff0-9]-?)*[a-z\\u00a1-\\uffff0-9]+)*"
    # TLD identifier
    r"(?:\.(?:[a-z\\u00a1-\\uffff]{2,}))" r")"
    # port number
    r"(?::\d{2,5})?"
    # resource path
    r"(?:\/[^\)\]\}\s]*)?",
    flags=re.UNICODE | re.IGNORECASE)

# 2. email address
email_regex = re.compile(r"(?:^|(?<=[^\w@.)]))([\w+-](\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(\.([a-z]{2,})){1,3}(?:$|(?=\b))", flags=re.IGNORECASE | re.UNICODE)

# 3. phone number 
phone_regex = re.compile(r"(?:^|(?<=[^\w)]))(\+?1[ .-]?)?(\(?\d{3}\)?[ .-]?)?(\d{3}[ .-]?\d{4})(\s?(?:ext\.?|[#x-])\s?\d{2,6})?(?:$|(?=\W))"）

# 4 currency exchange
currencies = {"$": "USD", "zł": "PLN", "£": "GBP", "¥": "JPY", "฿": "THB",
    		  "₡": "CRC", "₦": "NGN", "₩": "KRW", "₪": "ILS", "₫": "VND",
    		  "€": "EUR", "₱": "PHP", "₲": "PYG", "₴": "UAH","₹": "INR"}

currency_regex = re.compile("({})+".format("|".join(re.escape(c) for c in CURRENCIES.keys())))
````

## 停用词处理 

在做text mining和一些visualization, e.g., word cloud（云图）会出现需要处理一些停用词。我目前做过的发现有三个NLP语言包通用很强，也想对比较丰富。当然，在domain-specific的情况下，还是需要你们自己整理，再加入到这个stopwords这个类里。

````python
# 1. NLTK
import nltk
from nltk.corpus import stopwords
nltk_stop_words = set(stopwords.words('english'))

# 2. Spacy 
from spacy.lang.en.stop_words import STOP_WORDS
spacy_stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)

# 3. gensim
from gensim.parsing.preprocessing import STOPWORDS
gensim_stopwords = STOPWORDS

# combine 
all_stopwords = gensim_stopwords.union([nltk_top_words, spacy_stopwords])
````

# 还有什么？
这里，我只介绍了，在很多NLP任务中常会需要的处理手段。还有一些，这里没包括但是对有些任务也有意义的方法，我在此给一个小的总结以及对应的code链接，需要的朋友可以去那边查找。

1. 词干提取 ([Stemmning](https://www.cnblogs.com/no-tears-girl/p/6964910.html))
2. 词干还原 ([Lemmatization](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)) 
3. [N-gram](http://www.albertauyeung.com/post/generating-ngrams-python/)
4. [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
5. 稀缺词处理: 小噪音，可以根据词频直接过滤掉

