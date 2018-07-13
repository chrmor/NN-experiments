
# coding: utf-8

# In[57]:

from sklearn.utils import shuffle

import pickle
import json
import gzip
import os
import re
#import nltk

#stemmer = nltk.stem.PorterStemmer()

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"[0-9]", " ", string)
    string = string.replace("'","")
    string = string.replace("`","")
    
    return string.strip()

class Loader():

    def read_WikipediaEventsDataset(self, vector_source, start, end, minConf):
        listEvents = []
        data_dir='data/WE'
        for year in range(start,end+1):
            filename='wiki-events-' + str(year) + '_data.json.gz'
            print("loading file " + filename + " ...")
            with gzip.open(os.path.join(data_dir, filename), "rb") as f:
                events = json.loads(f.read().decode("utf8"))
            print("found " + str(len(events['results'])) + " events...")
            listEvents = listEvents + events['results']
            print("total: " + str(len(listEvents)) + " events...")
        
        classes = ['armed conflicts and attacks', 'politics and elections', 
           'law and crime', 'disasters and accidents', 'international relations', 
           'sport', 'business and economy', 'arts and culture', 'science and technology']
        
        data = {}
        x, y = [], []

        for event in listEvents:
            #check is the keys are present
            if 'event-type' in event and 'full-text' in event:
                #keep only events with non empty full-text and event-type
                if event['full-text'] and event['event-type']:
                    label = event['event-type']
                    if (label in classes):
                        entities = event['entities']
                        if vector_source=='entities':
                            vector = []
                            for entity in entities:
                                avgconf = sum(float(i) for i in entity['confidence'])/len(entity['confidence'])
                                if avgconf>minConf:
                                    vector.append(entity['label'])
                        elif vector_source=='short-text':
                            vector = event['event'].split()
                        elif vector_source=='full-text':
                            vector = clean_str(event['full-text']).split()
                            #for i,x in vector:
                                #vector[i] = stemmer.stem(x)
                        y.append(label)
                        x.append(vector)
        x, y = shuffle(x, y)

        print("number of selected events: " + str(len(x)))

        dev_idx = len(x) // 10 * 8
        test_idx = len(x) // 10 * 9

        data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
        data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
        data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

        print("# train: " + str(len(data["train_x"])))
        print("# dev: " + str(len(data["dev_x"])))
        print("# test: " + str(len(data["test_x"])))
        
        return data
    
    
    def read_TREC(self):
        print("Reading TREC dataset...")
        data = {}

        def read(mode):
            x, y = [], []

            with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
                for line in f:
                    if line[-1] == "\n":
                        line = line[:-1]
                    y.append(line.split()[0].split(":")[0])
                    x.append(line.split()[1:])

            x, y = shuffle(x, y)

            if mode == "train":
                dev_idx = len(x) // 10
                data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
                data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
            else:
                data["test_x"], data["test_y"] = x, y

        read("train")
        read("test")

        return data


    def read_MR(self):
        data = {}
        x, y = [], []

        with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                x.append(line.split())
                y.append(1)

        with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                x.append(line.split())
                y.append(0)

        x, y = shuffle(x, y)
        dev_idx = len(x) // 10 * 8
        test_idx = len(x) // 10 * 9

        data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
        data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
        data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

        return data


    def save_model(self, model, params):
        path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}_{params['LEARNING_RATE']}_{params['VECTORS_FROM']}.pkl"
        pickle.dump(model, open(path, "wb"))
        print(f"A model is saved successfully as {path}!")


    def load_model(self, params):
        path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}_{params['LEARNING_RATE']}_{params['VECTORS_FROM']}.pkl"

        try:
            model = pickle.load(open(path, "rb"))
            print(f"Model in {path} loaded successfully!")

            return model
        except:
            print(f"No available model such as {path}.")
            exit()

myloader = Loader()


# In[58]:

# Load Data
get_ipython().magic("time data = myloader.read_WikipediaEventsDataset('short-text',2012,2013,0.6)")


# In[60]:

data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
data["classes"] = sorted(list(set(data["train_y"])))
data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}


# In[109]:

#set global parmas and load data
import sys
import argparse

sys.argv = ['-h']
parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
parser.add_argument("--architecture", default="CNN", help="available achitectures: CBOW, CNN")
parser.add_argument("--vectors-from", default="short-text", help="available: full-text, short-text")
parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
parser.add_argument("--dataset", default="WE", help="available datasets: MR, TREC, WE")
parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
parser.add_argument("--epoch", default=1, type=int, help="number of max epoch")
parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")

options = parser.parse_args()
#data = getattr(myloader, f"read_{options.dataset}")()

params = {
    "MODEL": 'non-static',#options.model,
    "ARCHITECTURE": 'CBOW',#options.architecture,
    "DATASET": "WE",#options.dataset,
    "EMBEDDINGS_FILE": "embeddings/GoogleNews-vectors-negative300.bin",
    "VECTORS_FROM": "short-text",
    "SAVE_MODEL": True, #options.save_model,
    "EARLY_STOPPING": options.early_stopping,
    "EPOCH": 1, #options.epoch,
    "LEARNING_RATE": 0.1, #options.learning_rate,
    "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
    "BATCH_SIZE": 50,
    "WORD_DIM": 300,
    "VOCAB_SIZE": len(data["vocab"]),
    "CLASS_SIZE": len(data["classes"]),
    "FILTERS": [3, 4, 5],
    "FILTER_NUM": [100, 100, 100],
    "DROPOUT_PROB": 0.1,
    "NORM_LIMIT": 3,
    "GPU": 0
}


# In[106]:

#look at data samples
#data['train_x'][20]


# In[63]:

import torch
torch.manual_seed(0)


# In[107]:

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x


# In[110]:

class CBOW(nn.Module):

    def __init__(self, **kwargs):
        super(CBOW, self).__init__()
        
        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        
        self.embeddings = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.WV_MATRIX = kwargs["WV_MATRIX"]
        print("WV matrix size: " + str(self.WV_MATRIX.shape))
        self.embeddings.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        if self.MODEL == "static":
                self.embeddings.weight.requires_grad = False
        self.linear1 = nn.Linear(self.WORD_DIM, 128)
        self.linear2 = nn.Linear(128, self.CLASS_SIZE)     

    def forward(self, inputs):
        #embeds = terch.tensor(50,300)
        #for inp in self.embeddings(inputs):
        #    embeds.append(sum(in))
        #embeds = sum(self.embeddings(inputs)).view((1, -1))
        embeds = torch.sum(self.embeddings(inputs), dim=1)
        #print("Inputs size: " + str(self.embeddings(inputs).size()))
        #print("Embeds size: " + str(embeds.size()))
        out = F.relu(self.linear1(embeds))
        out = F.dropout(out, p=self.DROPOUT_PROB, training=self.training)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


# In[66]:

# load word2vec
from gensim.models.keyedvectors import KeyedVectors
print("loading word2vec...")
get_ipython().magic('time word_vectors = KeyedVectors.load_word2vec_format(params["EMBEDDINGS_FILE"], binary=True)')


# # PREPARE THE EXPERIMENT

# In[111]:

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import copy

#The Embeddings:
wv_matrix = []

def train(data, params):

    wv_matrix = []
    if params["MODEL"] != "rand":
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix
  
    if params["ARCHITECTURE"] == "CBOW":
        print("Initializing CBOW module...")
        model = CBOW(**params)
    else:
        print("Initializing CNN module...")
        #model = CNN(**params).cuda(params["GPU"])
        model = CNN(**params)
    
    #Force CPU usage to avoid probelms with parallel computation
    device = torch.device("cpu")
    model.to(device)
    
    #Parallel CPU
    #torch.nn.DataParallel(model)
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_acc = 0
    max_dev_acc = 0
    max_test_acc = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            #batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            #batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])
            batch_x = Variable(torch.LongTensor(batch_x))
            batch_y = Variable(torch.LongTensor(batch_y))

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()
            if (i%1000==0):
                print(str(i) + " steps: Loss = " + str(loss));
            

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params)
        print("epoch:", e + 1, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= pre_dev_acc:
            print("early stopping by dev_acc!")
            break
        else:
            pre_dev_acc = dev_acc

        if dev_acc > max_dev_acc:
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            #best_model = copy.deepcopy(model)
            best_model = model

    print("max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    #x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    x = Variable(torch.LongTensor(x))

    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc

def testSamples(data, model, params, start, end):
    model.eval()
    x, y = data["test_x"][:start], data["test_y"][:end]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    #x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    x = Variable(torch.LongTensor(x))

    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    for i in range(0,9):
        print("\n" + str(data["test_x"][i]) + "\nClass: " + data["test_y"][i])
        print("PredictedClass: " + str(data["classes"][pred[i]])) 
    #acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)


# # RUN THE EXPERIMENT

# In[112]:

#Do the training and evaluation
get_ipython().magic('time')

print("=" * 20 + "INFORMATION" + "=" * 20)
print("MODEL:", params["MODEL"])
print("ARCHITECTURE:", params["ARCHITECTURE"])
print("DATASET:", params["DATASET"])
print("VECTORS_FROM:", params["VECTORS_FROM"])
print("VOCAB_SIZE:", params["VOCAB_SIZE"])
print("MAX_SENT_LEN:", params["MAX_SENT_LEN"])
print("EPOCH:", params["EPOCH"])
print("LEARNING_RATE:", params["LEARNING_RATE"])
print("EARLY_STOPPING:", params["EARLY_STOPPING"])
print("SAVE_MODEL:", params["SAVE_MODEL"])
print("=" * 20 + "INFORMATION" + "=" * 20)

model = None

if options.mode == "train":
    print("=" * 20 + "TRAINING STARTED" + "=" * 20)
    model = train(data, params)
    if params["SAVE_MODEL"]:
        myloader.save_model(model, params)
    print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
else:
    #model = load_model(params).cuda(params["GPU"])
    model = myloader.load_model(params)

    test_acc = test(data, model, params)
    print("test acc:", test_acc)


# In[104]:

testSamples(data, model, params, 110 , 100)


# In[91]:

idx = data["word_to_idx"]['Carnival']
print(str(idx) + "\nVECTOR: " + str(len(wv_matrix)))


# In[ ]:



