# Word2Vec With Negative Sampling

### Installation
---
#### Clone and install requirements
```bash
git clone https://github.com/ranok92/ml_collab.git
cd ml_collab/word2vec
pip3 install -r requirements.txt
```

#### Download Gutenberg Dataset
```bash
mkdir data
cd data
gdown https://drive.google.com/uc?id=0B2Mzhc7popBga2RkcWZNcjlRTGM
unzip Gutenberg.zip
```
### Train
---
#### Generate dataset
```bash
python data_utils.py -h show this help message and exit
  --parent_folder path to the parent folder
  --data_store_path path to store the dataset
  --freq_threshold frequecy threshold cutoff
  --context_window context window size
```
**Example**
```bash
python data_utils.py --parent_folder ./data/Gutenberg/txt --data_store_path ./data/ --freq_threshold 100 --context_window 2
```

#### Train the model

```bash
python train.py -h show this help message and exit
  --model_type model type vanilla/sgns
  --epochs number of epochs
  --hidden_dim size of the hidden layer dimension
  --dataset_type cbow or skipgram model
  --batch_size size of each word batch
  --lr learning rate of the model
  --checkpoint_interval interval between saving model weights
  --dataset_path path to the JSON dataset
  --neg_sample_size Number of negative samples
  --test_file path to the file used in the testing part
  --checkpoint_path path to the training checkpoint
```
**Example**
```bash
python train.py --model_type sgns --epochs 10 --dataset_type sgram --batch_size 512 --checkpoint_interval 2 --dataset_path ./data/skipgram_style_training_dataset.json  --neg_sample_size 15
```
After the training, the model checkpoints will be saved in the ``` checkpoint ``` directory and trained embeddings will be saved as ```embeddings.json```.

#### Get pretrained skipgram negative sampling embeddings
```bash
cd data
gdown https://drive.google.com/uc?id=1ND-ED42ciT5QuTSyzFQVQXJpfNqvANmH
```
## Introduction
This is a repository that contains an implementation of the Word2Vec.

<strong>Word2Vec</strong> is a process(combination of a network archiecture and a cleverly designed supervised learning task) that helps to create word embeddings.

<strong>Word embeddings</strong> are projection of words (of any language) in a n-dimensional vector space, in a way that these n-dimensional vectors somewhat capture the essence of the words. Word2Vec at its heart is a supervised training technique that generates these fancy vectors and any supervised learning consists of two main components.

- The model to train.
- The dataset to train on.

## The Dataset
Let's start with the dataset, as we know the architecture of a neural network depends on the task at hand. The brillance of the [paper](https://arxiv.org/abs/1301.3781) lie in the deign of the task that as a byproduct creates the word embeddings. The authors propose two such tasks:

- Given a single word from a sentence, train a neural network to output a word in its viscinity. (skipgram model)
- Given a bag of words from a sentence, train a neural network to output a single word whose context contains the input words. (Continuous bag of words (cbow) model)

Before delving into the two models, lets try to understand one key concept that is extensively used in Word2Vec training: <strong>context</strong>.

<strong>Context</strong> of a word in sentence is the set of words that are atmost <em>N</em> positions away from the word, where <em>N</em> is the size of the <strong>context window</strong>. Simply put, context is the defined neighbourhood of a word in a given sentence and the context window controls the size of the neighbourhood we would like to have for the training.

Eg:
In the given sentence-
<strong>This is a sample sentence to help understand the concept of context and context window.</strong>
For the word <strong>‘understand’</strong>, if the context window is set as 2,then the set of words in its context would be: <strong>(to, help, the, concept)</strong>

For the word <strong>‘sample’</strong>, for a context window of size 4 the words in its context would be: <strong>( this, is, a, sentence, to, help, understand, the)</strong>.

<p class="has-line-data" data-line-start="21" data-line-end="22">Now, that we have an idea of what the context of a word stands for, we move on to the datasets.</p>
<h3 class="code-line" data-line-start=22 data-line-end=23><a id="Skipgram_model_dataset_22"></a>Skipgram model dataset:</h3>
<p class="has-line-data" data-line-start="23" data-line-end="27">In the skipgram model, the input is a single word from a sentence, while its corresponding output would be a word from its context.<br>
Eg:<br>
For the given sentence-<br>
<strong>A simple sentence for Skipgram training.</strong> (context window size =2)</p>
<table class="table table-striped table-bordered">
<thead>
<tr>
<th>Word</th>
<th>Context</th>
<th>Training tuples {input, output}</th>
</tr>
</thead>
<tbody>
<tr>
<td>a</td>
<td>simple, sentence</td>
<td>{a, simple} , {a, sentence}</td>
</tr>
<tr>
<td>simple</td>
<td>a, sentence, for</td>
<td>{simple, a}, {simple, sentence}, {simple, for}</td>
</tr>
<tr>
<td>sentence</td>
<td>a, simple, for, Skipgram</td>
<td>{sentence, a}, {sentence, simple}, {sentence, for}, {sentence, Skipgram}</td>
</tr>
<tr>
<td>for</td>
<td>simple, sentence, Skipgram, training</td>
<td>{for, simple}, {for, sentence}, {for, Skipgram}, {for, training}</td>
</tr>
<tr>
<td>Skipgram</td>
<td>sentence, for, training</td>
<td>{Skipgram, sentence}, {Skipgram, for}, {Skipgram, training}</td>
</tr>
<tr>
<td>training</td>
<td>for, Skipgram</td>
<td>{training, for}, {training, Skipgram}</td>
</tr>
</tbody>
</table>

<h3 class="code-line" data-line-start=35 data-line-end=36><a id="Continuous_bag_of_words_CBOW_model_dataset_35"></a>Continuous bag of words (CBOW) model dataset:</h3>
<p class="has-line-data" data-line-start="36" data-line-end="39">For CBOW, the input is all the words in the context of a particular word and the output is the word.<br>
Eg sentence:<br>
<strong>Switching things up for CBOW training</strong> (context window size=3)</p>
<table class="table table-striped table-bordered">
<thead>
<tr>
<th>Word</th>
<th>Context</th>
<th>Traning tuples {<strong>input</strong>, output}</th>
</tr>
</thead>
<tbody>
<tr>
<td>Switching</td>
<td>things, up, for</td>
<td>{<strong>things, up, for</strong>, Switching}</td>
</tr>
<tr>
<td>things</td>
<td>Switching, up, for, CBOW</td>
<td>{<strong>Switching, up, for, CBOW</strong>, things}</td>
</tr>
<tr>
<td>up</td>
<td>Switching, things, for, CBOW, training</td>
<td>{<strong>Switching, things, for, CBOW, training</strong>, up}</td>
</tr>
<tr>
<td>for</td>
<td>Switching, things, up, CBOW, training</td>
<td>{<strong>Switching, things, up, CBOW, training</strong>, for}</td>
</tr>
<tr>
<td>CBOW</td>
<td>things, up, for, training</td>
<td>{<strong>things, up, for, training</strong>, CBOW}</td>
</tr>
<tr>
<td>training</td>
<td>up, for, CBOW</td>
<td>{<strong>up, for, CBOW</strong>, training}</td>
</tr>
</tbody>
</table>
<p class="has-line-data" data-line-start="48" data-line-end="49">For both the cases, the words are represented as one-hot vectors, where the dimensionality of each vector is equal to the size of the vocabulary of the training corpus. So, for CBOW, where the input consists of multiple words, the one-hot vectors are summed up to get the representation of the input context.</p>
<h2 class="code-line" data-line-start=50 data-line-end=51><a id="The_model_50"></a>The model</h2>
<p class="has-line-data" data-line-start="51" data-line-end="53">The [original paper](https://arxiv.org/abs/1301.3781) proposes a number of network archietctures of varying complexity and capacity that can be used to for the task. In this work, we follow the feedforward neural net language model[cite the paper].<br>
The neural network consists of 3 fully connected layer, absent of any non linear activation within the layers. The sizes (number of nodes) of the different layers depend on the size of the vocabulary and the desired dimensionality of the word embeddings.</p>
<p class="has-line-data" data-line-start="54" data-line-end="59">For example:<br>
With <strong>V</strong> as the size of the vocabulary, and <strong>D</strong> as the desired dimensionality of the word embeddings, the network dimensions we use are as follows:<br>
layer 1 : <strong>V x D</strong><br>
layer 2 : <strong>D x H_1</strong><br>
layer 3 : <strong>H_1 x V</strong></p>
<h2 class="code-line" data-line-start=60 data-line-end=61><a id="Training_the_model_60"></a>Training the model</h2>
<p class="has-line-data" data-line-start="61" data-line-end="63">Training the network is straight forward, the output from the final layer is passed through a softmax function and the result is compared with the one-hot vector of the output word to obtain the loss, which is then backpropagated through the network.<br>
We use cross entropy loss and Adam optimizer to update the weights.</p>
<h2 class="code-line" data-line-start=64 data-line-end=65><a id="Negative_sampling_64"></a>Negative sampling</h2>
<p class="has-line-data" data-line-start="65" data-line-end="67">The calculation of the softmax over all the words in the vocabulary is an expensive operation, especially for large datasets with thousands of words. To avoid this, the authors introduce a modification in the form of negative sampling.<br>
Negative sampling converts the problem of <strong>V</strong> way classification problem to a binary classification problem. Instead of outputting the word itself ( <strong>V</strong> way classification), the problem is switched the prediction of whether the words appear in context or not (binary classifictation). That means changing the training data, again.</p>
<h3 class="code-line" data-line-start=68 data-line-end=69><a id="Skipgram_with_negative_sampling_68"></a>Skipgram with negative sampling</h3>
<p class="has-line-data" data-line-start="69" data-line-end="70">To make the switch all you need to do is instead of having an input and an output word from its context, select words that appear in context and label them as 1. We have a problem now. Getting the positive samples is easy. We already had that, all we need to do is combine the words in a tuple and slap a 1 for the output. But how to get word pairs that are not in context, or in other words, the negative samples? The authors propose different ways to sample negative examples.</p>

<img src="https://latex.codecogs.com/gif.latex?P(word_i)&space;=&space;\frac{f_{wi}^{a}}{\sum_{i=1}^{V}f_{wi}^{a}}\\&space;\\&space;f_{wi}\&space;=&space;frequency\&space;of\&space;Word\&space;w_i\&space;and&space;\\&space;a&space;=&space;Sampling\&space;control\&space;parameter" />

Eg.
For, a=1, the sampling is equal to the frequency of each word. a=0 results to a uniform sampling.

### Training Details
---
**Dataset** - [Gutenberg](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html)
**Raw Dataset Size** - 1.1GB
**Batch Size** - 512
**Dataloader Num Workers** - 4
**Vocabulary Size** - 41383

``` python3
Model Type - Two Layer Neural Network
Dataset Type - Skipgram
Dataset Size - 1.1GB
Total Samples - 47868117
Training Time - 02 Hours 10 Minutes/Epoch

Epoch 1/10
----------
Loss = 9.75381: 5%|##               |5039/93493 [07:18/2:03:12, 12.10it/s]
```

``` python3
Model Type - Two Layer Neural Network
Dataset Type - CBOW
Dataset Size - 2.8GB
Total Samples - 66399672
Training Time - 03 Hours 37 Minutes/Epoch

Epoch 1/10
----------
Loss = 7.80813: 1%|#                 |1803/129687 [03:20/3:34:54, 9.96it/s]
```

``` python3
Model Type - Skipgram Negative Sampling
Negative Sampling Size - 15
Training Time - 32 Minutes/Epoch

Epoch 1/10
----------
Loss = 2.59145: 100%|################| 93493/93493 [32:34<00:00, 50.07it/s]
Loss after epoch 1 : 1.10383967
==================
Epoch 2/10
----------
Loss = 0.25489: 20%|###              |18470/93493 [07:20/25:12, 52.10it/s]
```
### Results
---
** Results for Word2Vec Negative Sampling **
```python
>>> get_k_most_similar("fruit", "embeddings_sgns.json", k=10)
vegetables, tomatoes, vegetable, oranges, fruits, bread, chocolate, herbs, cheese, organic

>>> get_k_most_similar("positive", "embeddings_sgns.json", k=10)
results, improvement, consistent, optimism, outlook, disappointing, disappointment, expectations, confidence, confident

>>> get_k_most_similar("speak", "embeddings_sgns.json", k=10)
spoke, spoken, talked, contacted, responded, interviewed, addressed, expressed, quoted, replied

>>> get_k_most_similar("laptop", "embeddings_sgns.json", k=10)
laptops, keyboard, computer, smartphone, computers, desktop, ipads, device, touchscreen, desk

>>> get_k_most_similar("lion", "embeddings_sgns.json", k=10)
leopard, elephants, cats, elephant, snake, dog, bison, fur, snakes, rhinoceros
```

** Results for Word2Vec Two layer neural Net **
```python
>>> get_k_most_similar("fruit", "embeddings.json", k=10)
cakes, eggs, seed, apples, olive, unc, blossom, poison, stretching, der

>>> get_k_most_similar("positive", "embeddings.json", k=10)
improper, discomfort, reverse, iti, afresh, matrimony, doubting, liable, kilburn, mei

>>> get_k_most_similar("speak", "embeddings.json", k=10)
answered, ask, tell, wish, done, talk, let, question, so, mother

>>> get_k_most_similar("device", "embeddings.json", k=10)
wad, medicine, treason, accusation, bernard, sarpent, distorted, understands, wills, scratch

>>> get_k_most_similar("lion", "embeddings.json", k=10)
warrior, ribs, stead, arbitrary, thompson, lions, theseus, cat, judges, dot,
```

#### Comparison to Google's Word2Vec model with our SGNS model
```python
>>> compare_with_word2vec(['woman', 'fruit', 'bicycle', 'school'], "embeddings_sgns.json",
                          'data/GoogleNews-vectors-negative300.bin', k=10)
Given Word woman

  Google Word2Vec similarity              Our model Word2Vec similarity
--------------------------------------------------------------------------------
             man              0.766 |           daughter           0.440
             girl             0.749 |            mother            0.437
         teenage_girl         0.734 |             girl             0.423
           teenager           0.632 |            sister            0.415
             lady             0.629 |             she              0.403
        teenaged_girl         0.614 |             wife             0.397
            mother            0.608 |           husband            0.394
         policewoman          0.607 |           teenager           0.392
             boy              0.598 |             boy              0.384
            Woman             0.577 |           parents            0.380

Given Word fruit

  Google Word2Vec similarity              Our model Word2Vec similarity
--------------------------------------------------------------------------------
            fruits            0.774 |          vegetables          0.513
           cherries           0.690 |           tomatoes           0.484
           berries            0.685 |          vegetable           0.483
            pears             0.683 |           oranges            0.468
         citrus_fruit         0.669 |            fruits            0.459
            mango             0.663 |            bread             0.455
            grapes            0.651 |          chocolate           0.455
            berry             0.650 |            herbs             0.451
           peaches            0.643 |            cheese            0.447
            apple             0.641 |           organic            0.446

Given Word bicycle

  Google Word2Vec similarity              Our model Word2Vec similarity
--------------------------------------------------------------------------------
             bike             0.852 |             bike             0.524
           scooter            0.751 |            bikes             0.405
           bicycles           0.736 |          motorcycle          0.388
          motorcycle          0.697 |           backpack           0.385
          bicycling           0.696 |           bicycles           0.379
            bikes             0.693 |             gear             0.370
        mountain_bike         0.650 |           luggage            0.366
          skateboard          0.647 |          motorbikes          0.362
            biking            0.642 |           offroad            0.362
            moped             0.641 |           buggies            0.361

Given Word school

  Google Word2Vec similarity              Our model Word2Vec similarity
--------------------------------------------------------------------------------
          elementary          0.787 |           students           0.394
           schools            0.741 |           college            0.389
            shool             0.669 |            police            0.389
      elementary_schools      0.660 |             home             0.383
         kindergarten         0.653 |          university          0.376
         eighth_grade         0.649 |           parents            0.367
            School            0.648 |             two              0.366
           teacher            0.638 |          elementary          0.343
           students           0.630 |           schools            0.332
          classroom           0.628 |            people            0.331

```

### Testing
---
To test our trained model, we test it against the Google Word2Vec trained model. We choose some set of words and show ***k most similar words*** chosen by the models. To get the Google's pretrained model:
```bash
cd data
gdown https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM
gunzip GoogleNews-vectors-negative300.bin.gz
```

After this, run test_embeddings.py
```python
python test_embeddings.py
```
This shows the comparison between Google's word2vec trained model and our trained model for the words *'woman', 'fruit', 'bicycle', 'school'*. It also shows the k most similar words for the word *apple* achieved by our trained model. These words can be changed in the test_embeddings.py file.

### Credit and References
---
#### Papers
#### [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
*Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean*

#### [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
*Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean*

#### Blogs
- [**The Illustrated Word2vec**](http://jalammar.github.io/illustrated-word2vec/) by Jay Alammar
- [**Word2Vec Tutorial - The Skip-Gram Model**](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) by Chris McCormick
- [**Github Negative Sampling**](https://github.com/topics/negative-sampling) topic in GitHub
- [**On word embeddings - Part 1**](https://ruder.io/word-embeddings-1/index.html) by Sebastian Ruder
</body></html>
