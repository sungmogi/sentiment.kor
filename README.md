# sentiment.kor
## Introduction / Inspirations

During spring quarter this year, I was introduced into computational linguistics / NLP through a class called Introduction to Computational Linguistics (Ling 334) taught by Prof. Rob Voigt. 

One of the most intriguing tasks was sentiment analysis, despite its (relative) simplicity. For my final project for this class, I wanted to do something similar to what we were taught in class, but with a dataset in Korean which is my mother tongue. 

![Screen Shot 2023-07-14 at 2.53.27 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/21f8a32e-7d94-47b2-ab49-b391147bbac7/Screen_Shot_2023-07-14_at_2.53.27_PM.png)

(Slide from class)

Rob gave us an example of how we might extract features from a corpus, train the classifier model using neural networks, and predict the sentiment of unseen documents. 

I searched for a Korean sentiment analysis dataset, and found this:

https://huggingface.co/datasets/sepidmnorozy/Korean_sentiment

It’s a film review dataset, where a positive review is labeled “1”, and a negative review “0”. For instance, “날 믿어봐 이건 쓰레기야 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ” (”Trust me this is garbage lolololol”) was labeled 0, while “재미있었습니다.잼잼잼” (”It was fun. Funfunfun”) was labeled 1.

## Implementation

I followed the sentiment features outlined above and implemented a linear layer followed by a Sigmoid function. Stochastic Gradient Descent was used for optimization. 

I did not have to make big changes for my Korean sentiment analysis, except for feature 3. To identify the equivalent of “no”, I used a mecab.pos tagger to search for “VCN” (Verb, Copula, Negative). 

mecab.morphs is a morpheme-level tokenizer.

```python
def feature_extraction(line):
    wc = 0.0
    x = [0.0,0.0,0.0,0.0,0.0,0.0]
    for w in mecab.morphs(line):
        if w in wc_pos and w in wc_neg:
            if wc_pos[w] > wc_neg[w]:
                x[0] += 1.0 # x1
            else:
                x[1] += 1.0 # x2
        if mecab.pos(w) == 'VCN':
            x[2] = 1 # x3 do we need this feature?
        if w in pron:
            x[3] += 1.0
        if w == "!":
            x[4] = 1.0
        wc += 1
    x[5] = math.log(wc)

    return x
```

I used the pytorch library which I am just starting to get used to. Nothing special here. 

```python
class SentimentClassifier(nn.Module):
    def __init__(self, num_inputs):
        super(SentimentClassifier, self).__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.linear(x)
        out = self.sigmoid(out)
        return out
```

I chose 1e-3 as the learning rate, and took 300k steps. After training, the loss was 0.4852 (4s.f.) for training data and 0.5547 for dev data. 

![v1 result.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46ff2f8d-73c0-41f0-a315-7d41f1e0a545/v1_result.png)

When I changed the learning rate to 1e-2, I had the same result, but the loss converged much quickly. I decided to decrease the learning rate to 1e-2 and also the number of iterations to 100k. 

![v1 lr 1e-2.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e23d1ab3-df19-440b-8fed-dc1e5c37f7fc/v1_lr_1e-2.png)

I wanted to see how a two-layer network would work, but the losses were exactly the same (training 0.4852; dev 0.5547).

I also wanted to see how the Adam optimizer would perform instead of SGD, but there was no improvement either, apart from the fact that the loss converged much faster. The training loss started to converge after 5k steps.  

## Evaluation

I loaded the training, dev, and test datasets and calculated the accuracies:

**Training data** accuracy: 0.7764

**Dev data** accuracy: 0.7284

**Test data** accuracy: 0.7518
