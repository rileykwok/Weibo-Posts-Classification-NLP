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
Some very simple visualisation for a general sense of data given.

Posts distribution by labels [Left] and the length distribution of posts [Right].
<img src="https://github.com/rileykwok/Weibo-Posts-Classification-NLP-/blob/master/images/01%20label%20distribution.png" width="400">  <img src="https://github.com/rileykwok/Weibo-Posts-Classification-NLP-/blob/master/images/post%20length%20dist.png" width="400">

[Left] Post distribution by each KOL.<br>
[Right] Total number of likes for each KOL.

<img src="https://github.com/rileykwok/Weibo-Posts-Classification-NLP-/blob/master/images/user%20post%20dist.png" width="400">  <img src="https://github.com/rileykwok/Weibo-Posts-Classification-NLP-/blob/master/images/user%20id%20likes.png" width="400">

## Results
The final F1 score reached 0.85 using Cat-Boost for the given data sets.

