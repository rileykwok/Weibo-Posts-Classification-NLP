# Weibo-Posts-Classification-NLP-
NLP Classification for UBS Challenge 1 in BigDatathon

## Background
There is a lot of content generated from social media channels such as Weibo in China every day. For marketing purpose, UBS would like to know what different interests amongst different cities of China. With the results, UBS could focus then host relevant client outreach events based on these interests. For this question, you must demonstrate the capability to add value to the business for a marketing solution based on the findings.

Based on the selected posts' content (from 1st Feb 2018 to 30th Apr 2018) from KOLs',  these posts can be categorized into different interests. 

## Aim 
- Classify the Weibo posts into 13 groups of interest:
    <br>Stock, Bond, Oil, Gold, Real Estate, Chinese Art (Painting/drawing, calligraphy), Western Art (Painting/drawing, calligraphy), Jewellery, Artefacts, Golf, Car, Overseas Education, Young Children education
- Minimise F1 Score

## Exploratory Data Analysis



## Data Cleaning Tools Used

To segment chinese sentences into meaningful and useful phrases, chinese segmentation tool [Jieba](https://github.com/fxsjy/jieba) is used for words segmentation.
```python
seg_list = jieba.cut_for_search(text1)
```
Removing meaningless words in phrases would increase the accuracies for classification, Stopwords [stopwords-zh](https://github.com/stopwords-iso/stopwords-zh) dictionary is used for cleaning.
```python
def cleanstop(x):
    stop = pd.read_csv('stopwords-zh.txt').iloc[:,0].tolist()
    x1 = (','.join(jieba.cut_for_search(x))).split(',')
    x2 = [i for i in x1 if i not in stop]
    return ''.join(x2)
```

## Approach
Approach used 
![alt text][https://drive.google.com/file/d/1YpezbDJfjTYMVKw3gZB5yN-a3_u2XNbQ/view?usp=sharing]

## Results
The final F1 score reached 0.85 for the given data sets.

