# Vietnamese Punctuation Prediction Using Deep Neural Networks
(link paper)

In this paper, we have studied the punctuation prediction problem for the Vietnamese language. We collect two large-scale datasets and conduct extensive experiments with both traditional method (using CRF models) and a deep learning approach. We address the class imbalance problem in this task and show promising results using the focal loss on the Vietnamese Newspapers data.

# Requirements
This code has been tested with 

python 3.6.8

tensorflow 1.13.1

fasttext (https://fasttext.cc/docs/en/crawl-vectors.html)

# Dataset

In this work, we collect over 40,000 articles from the Vietnamese news (https://baomoi.com/) and 86 novels (https://gacsach.com/tac-gia/nguyen-nhat-anh.html) to build two datasets with a total of over 900000 sentences.

### Data Preprocessing
We label each word by its immediately following punctuation, where label O denotes a space. Example:
```
Biển tạo ra 1/2 lượng oxy con người hít thở, giúp lưu chuyển nhiệt quanh Trái Đất và hấp thụ một lượng lớn CO2.
(The ocean produces a half of the amount of oxygen that humans can breathe, and help to circulate heat around the Earth and absorb large amounts of CO2.)
```
This paragraph can be labeled as follows:
```
biển tạo ra 1/2 lượng oxy con người  hít  thở   giúp lưu chuyển nhiệt quanh trái đất và hấp thụ một lượng lớn co2
 O    O  O   O    O    O   O    O    O   Comma    O    O     O     O     O     O   O   O  O   O   O    O    O  Period 
```
# 

