# Topic Modeling with Latent Dirichlet Allocation (LDA)using Gensim and NLP techniques


In this tutorial (Part I), I will present the process of generating the interest tags of users based on their tweets using LDA and NLP techniques.

I had already covered this topic using BERTopic and BERTweet transformer in a previous tutorial (https://medium.com/@hajar.zankadi/using-bertopic-and-bertweet-transformer-to-predict-interest-tag-from-tweets-67189f11b992).

The process starts from data cleaning, data preprocessing using NLP pipeline, generating text features and then training the LDA model.

For this tutorial, I will cover the different techniques in more details so that you can follow up and understand well how they work and then, I will present the code used to clean, preprocess and train the model.

First, I will explain what is LDA, how it works, what are the hyper-parameters of the model and then I will talk about some of the techniques used in NLP and in text feature extraction.

So, let’s start:
1. What is LDA?

LDA is an unsupervised clustering technique used for text analysis and used as a topic modelling technique. Topic modeling is simply trying to predict the subject of a document.

LDA assumes that:

    Each document is considered as a distribution of topics.
    Each topic is considered as a distribution of words.

2. How LDA works?

The graphical model of LDA is presented in the figure below:
Generative process of LDA
![lda generative process](https://user-images.githubusercontent.com/44951948/210504323-2d0b2eb0-58ef-4f84-81eb-20b917c2bdc3.png)




Where:

    Θm is the topic mix for document m where Θm ~ Dirichlet(α).
    Zm,n is a topic assignment for word n in document m.
    Wm,n is word n in document m.
    N is the total number of words in all documents (the size of the vocabulary).
    M is the number of documents.
    K is the number of topics.
    Φk is the distribution of words for topic k where Φk ~ Dirichlet(β)

LDA model has two Dirichlet distributions:

    Alpha (ɑ) that controls the document-topic distribution.
    Beta (β) that controls the topic-word distribution.

Here is a useful link that explains in details what Dirichlet stands for in LDA, why it is used and how the algorithm work (https://medium.com/analytics-vidhya/the-intuition-behind-latent-dirichlet-allocation-lda-fb1e1fb01543).

3. LDA hyper-parameters:

A hyperparameter is a parameter whose value is used to control the learning process.

LDA has three hyper-parameters:

    Alpha (ɑ)
    Beta (β)
    The number of topics K


4. NLP pipeline:

NLP constitutes a core interest in the field of artificial intelligence and computer science. NLP studies comprise theories and methods that enable effective communication between humans and computers in natural language.

Several NLP techniques include the following:

- **Tokenization**: is one of the most foundational NLP tasks. It is the process of breaking down a piece of text into small units called tokens. A token may be a word, part of a word or just characters like punctuation.

    Example: The sentence “I love Data Science” becomes after tokenization: ‘I’, ‘love’, ’Data’, ‘Science’.

- **Removing stop words**: Stop word is a word which has less significant meaning than other tokens. Stop words are most common words found in any natural language which carries very little or no significant semantic context in a sentence.

    Example of stop words like “a”, “an”, “the”, etc.

- **N-grams implementation**: N-gram based techniques are predominant in modern NLP and its applications.
    N-grams are continuous sequences of n-items in a sentence. N can be 1, 2 or any other positive integers.
 
- **POS / Speech of tag selection**: also called grammatical tagging, is the process of determining the part of speech of a particular word or piece of text based on its use and context.

    Example: Part of speech identifies ‘make’ as a verb in ‘I can make a paper plane,’ and as a noun in ‘What make of car do you own?’.
		
- **Lemmatization**: involves identifying for each word of the text its associated canonical word. The lemma is the minimum form of the word that bears its main meaning and represents the inputs of dictionaries.

    Example: the word “meet” is the lemma of the original word “meeting”.

**TIP: Always convert your text to lowercase before performing any NLP task including lemmatizing.**

5. Text feature extraction:

Text feature extraction is based on vector space model, where a text is viewed as a dot in a N-dimensional space. Each dimension of the dot represents one feature of the text in digital form. the process is also called text vectorization.

Among feature extraction techniques, there is: Bag Of Words (BOW) and TF-IDF that I am going to explain just now.

- **BOW:**

    Bag Of Words is a representation model of document data, which simply counts how many times a word appears in a document.
    It involves two things: A vocabulary of known words and a measure of the presence of known words.
		
    I highly recommend you to check this link (https://machinelearningmastery.com/gentle-introduction-bag-words-model/). It explains in details and with examples what is bag of words and how it works.

- **TFIDF:**

    The TF-IDF is used to weigh a keyword in any content and assign importance to that keyword based on the number of times it appears in the document. More importantly, it checks how relevant the keyword is throughout the web, which is referred to as corpus.
    TF-IDF is scored between 0 and 1. The higher the numerical weight value, the rarer the term. The smaller the weight, the more common the term.

    TF-IDF is calculated as the following: **TF-IDF = TF * IDF**, where:

    -- **Term frequency (TF)** is the ratio of the count of a particular word present in a sentence to the total count of words in the same sentence.

    TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).

    For example, let’s say we have the following sentence:

    “I love coffee, coffee is my favorite drink. I drink coffee everyday”.

    if we take the word “coffee”, it is present in the sentence 3 times and the total number of words in this sentence is 12. ( we do not take into consideration punctuation, spaces since it is assumed that our data is already cleaned and preprocessed).

    The TF for the word ‘coffee’ is:

    TFcoffee = 3/12=0.25

    --**Inverse document frequency (IDF)** is a log of the ratio of the total number of rows to the number of rows in a particular document in which a word is present:

    IDF = log(N/n), where N is the total number of rows, and n is the number of rows in which the word was present. IDF measures the rareness of a term.

    For example: Let’s say the size of our corpus is 10000 documents. If we assume there are 200 documents that contain the term “coffee”, then:

    IDFcoffee = log(1000/200) = 0.69

**TF-IDF Calculation**:

    TF-IDFcoffee = 0.25*0.69 = 0.17

Now, you have gained an understanding of how LDA works and discovered some of the NLP and text features extraction techniques.
