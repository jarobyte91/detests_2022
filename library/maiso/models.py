# import spacy
import numpy as np
from sklearn.preprocessing import normalize
#####################################################
#                    Glove MODEL                    #
#####################################################
class GloveModel(object):   
#     import spacy
#     nlp = spacy.load('es_core_news_lg')
    targets = ['xenophobia',
               'suffering',
               'economic',
               'migration',
               'culture',
               'benefits',
               'health',
               'security',
               'dehumanisation',
               'others',
              ]
    def __init__(self, sklearn_model):
        self.model = sklearn_model
    def entrenar(self,X,y):
        self.model.fit(X,y)
    def predecir(self,X):
        return self.model.predict(X)
    def get_X(text_list):
        return np.array([normalize([doc.vector])[0,:] for doc in GloveModel.nlp.pipe(text_list)])
    
    
#####################################################
#                    BERT  MODEL                    #
#####################################################
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np 
class BERTMultilingualUncased(object):
    def __init__(self, training_args):
        self.training_args = training_args
        self.trained=False

    def entrenar(self, X, y):
        model_name = '../../../../../../home/maiso/.cache/huggingface/bert-base-multilingual-uncased/'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
#         assert all([type(elem)==str for elem in X])
        data = {'sentence':[text for text in X],
                'label': [label for label in y]
               }
        train_data = Dataset.from_dict(data)
        def tokenize_function(example):
            return self.tokenizer(example["sentence"], truncation=True)
        
        tokenized_dataset = train_data.map(tokenize_function, batched=True)
        from transformers import Trainer
        from transformers import TrainingArguments



        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model,
            self.training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        output = trainer.train()
        self.trainer = trainer
        self.model = model
        self.trained=True
        return output
    def get_X(text_list):
        return np.array(text_list)
    def predecir(self, X):
        assert self.trained
        data = {'sentence':[text for text in X]
               }
        dataset = Dataset.from_dict(data)

        def tokenize_function(example):
            return self.tokenizer(example["sentence"], truncation=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        predictions = self.trainer.predict(tokenized_dataset)

        
        return np.argmax(predictions.predictions, axis=-1)
    
    
#####################################################
#                  ROBERTA  MODEL                   #
#####################################################   
import torch
import torch.nn as nn
import torch.utils.data as tud
class ROBERTABasic(nn.Module):
    def __init__(
        self, 
        tokenizer, 
        model,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = model
        
    def tokenize_data(self, example):
        return self.tokenizer(example['features'], padding = 'max_length')
        
    def forward(self, input_ids, attention_mask):
        hidden = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask
        ).last_hidden_state[:, 0, :]
        return self.output_layer(hidden)
    
    def predecir_base(self, X, batch_size = 2, progress_bar = False):
        with torch.no_grad():
            self.eval()
#             print("    Generating predictions...")
            tokens = self.tokenizer(
                X.tolist(), 
                padding = "longest", 
                truncation = True
            )
            p = next(self.parameters())
            input_ids = torch.tensor(tokens["input_ids"]).long()
            attention_mask = torch.tensor(tokens["attention_mask"]).long()
            dataset = tud.TensorDataset(
                input_ids, 
                attention_mask,
            )
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            output = []
            iterator = iter(loader)
            if progress_bar:
                iterator = tqdm(iterator)
            for batch in iterator:
                i, a = batch
                predictions = self.forward(
                    input_ids = i.to(p.device),
                    attention_mask = a.to(p.device)
                )
                output.append(predictions)
            return torch.cat(output, axis = 0)
        
#######################################################
#                  BETO SHELL MODEL                   #
#######################################################
# from tqdm import tqdm
# import pandas as pd     
# from transformers import AutoModel
# import pickle
# class BETOShell(object):
#     def __init__(self,model_name='bert-base-spanish-wwm-uncased', epochs=15):
#         self.epochs=epochs
#         print("Loading tokenizer...")
#         tokenizer = AutoTokenizer.from_pretrained(f"/home/maiso/detests_2022/assets/{model_name}/tokenizer")
#         print("Loading model...")
#         model = AutoModel.from_pretrained(f"/home/maiso/detests_2022/assets/{model_name}/model")
#         self.beto = BetoMTL(tokenizer,model)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.trained=False
#     def entrenar(self, X, y):
#         self.trained=True
#         self.columns = list(y.columns)
#         self.beto.to(self.device)
#         self.beto.entrenar(X.values, 
#                            y.values,
#                            epochs = self.epochs, 
#                            batch_size = 2,
#                            learning_rate = 10**-5,
#                            progress_bar = True,
#                            refresh = True, 
#                            weight_decay = 0, 
#                            freeze_encoder = False, 
#                            )
#     def get_columns(self):
#         return self.columns
#     def predecir(self, X):
#         return self.beto.predecir(X)
#     def predecir_proba(self, X):
#         return self.beto.predecir_proba(X)
#     def get_X(text_list):
#         return pd.Series(text_list)
    
#     def save_pretrained(self, file_path):
#         assert self.trained
#         pickle.dump(self.columns, open(file_path+'.columns.p','wb'))
#         torch.save(self.beto.state_dict(), file_path)
#     def load_from_pretrained(self, file_path):
#         self.columns = pickle.load(open(file_path+'.columns.p','rb'))
#         self.beto.load_state_dict(torch.load(file_path))
        
        
##############################################################
#                  TRANSFORMER SHELL MODEL                   #
##############################################################
from tqdm import tqdm
import pandas as pd     
from transformers import AutoModel
import pickle
import os
class TransformersShell(object):
    def __init__(self,model_name='roberta-base-bne', epochs=15):
        valid_models = {'roberta-base-bne', 'bert-base-spanish-wwm-uncased', 'bert-base-multilingual-uncased'}
        assert model_name in valid_models, f'Invalid model_name, valid names are: {str(valid_models)}'
        print(f'Working with model {model_name}')
        self.epochs=epochs
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(f"/home/maiso/detests_2022/assets/{model_name}/tokenizer")
        print("Loading model...")
        model = AutoModel.from_pretrained(f"/home/maiso/detests_2022/assets/{model_name}/model")
        if 'roberta' in model_name:
            self.transformer = ROBERTA(tokenizer,model)
        else:
            self.transformer = BetoMTL(tokenizer,model)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained=False
    def entrenar(self, X, y):
        self.trained=True
        self.columns = list(y.columns)
        self.transformer.to(self.device)
        self.transformer.entrenar(X.values, 
                              y.values,
                              epochs = self.epochs, 
                              batch_size = 2,
                              learning_rate = 10**-5,
                              progress_bar = True,
                              refresh = True, 
                              weight_decay = 0, 
                              freeze_encoder = False, 
                              )
    def get_columns(self):
        return self.columns
    def predecir(self, X):
        return self.transformer.predecir(X)
    def predecir_proba(self, X):
        return self.transformer.predecir_proba(X)
    def get_X(text_list):
        return pd.Series(text_list)
    
    def save_pretrained(self, file_path):
        assert self.trained
        pickle.dump(self.columns, open(file_path+'.columns.p','wb'))
        if os.path.isfile(file_path):
            os.remove(file_path)
        torch.save(self.transformer.state_dict(), file_path)
    def load_from_pretrained(self, file_path):
        assert os.path.isfile(file_path) and os.path.isfile(file_path+'.columns.p')
        self.columns = pickle.load(open(file_path+'.columns.p','rb'))
        self.transformer.load_state_dict(torch.load(file_path))
    
class ROBERTA(ROBERTABasic):
    def __init__(
        self, 
        tokenizer, 
        model,
    ):
        super().__init__(tokenizer, model)
        self.output_layer = nn.Linear(768, 10)
        torch.save(self.state_dict(), "clf.pt")      
        
    def predecir_proba(self, X, **kwargs):
        return self.predecir_base(X, **kwargs).sigmoid().cpu().numpy()
    
    def predecir(self, X, **kwargs):
        return self.predecir_proba(X, **kwargs) > 0.5

    def entrenar(
        self, 
        X_train, 
        Y_train,
        epochs = 2, 
        batch_size = 2,
        learning_rate = 10**-5,
        progress_bar = True,
        refresh = True, 
        weight_decay = 0, 
        freeze_encoder = False, 
    ):
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)

        print("Training model...")
        if refresh:
            self.load_state_dict(torch.load("clf.pt"))
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        tokens = self.tokenizer(
            X_train.tolist(), 
            padding = "longest", 
            truncation = True
        )
        p = next(self.parameters())
        input_ids = torch.tensor(tokens["input_ids"]).long()
        attention_mask = torch.tensor(tokens["attention_mask"]).long()
        label = torch.tensor(Y_train).float()
        dataset = tud.TensorDataset(
            input_ids, 
            attention_mask,
            label
        )
        loader = tud.DataLoader(
            dataset, 
            shuffle = True, 
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
        training_steps = epochs * len(loader)
        if progress_bar:
            bar = tqdm(range(training_steps))
        self.train()
        for epoch in range(1, epochs + 1):
            losses = []
            accuracies = []
            for j, batch in enumerate(loader):
                i, a, l = batch
                predictions = self.forward(
                    input_ids = i.to(p.device),
                    attention_mask = a.to(p.device)
                )
                l = l.to(p.device)
                loss = criterion(predictions, l)
                accuracy = ((predictions > 0) == l).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if progress_bar:
                    bar.update(1)
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                if epoch == 1 and j == 0:
                    print(f"    first step, train loss:{loss.item():.4f}")
            total_loss = sum(losses) / len(losses)
            total_accuracy = sum(accuracies) / len(X_train) / 10
            print(f"    epoch: {epoch}, train loss:{total_loss:.4f}, train accuracy: {total_accuracy:.4f}")

            
#####################################################
#                    BETO  MODEL                    #
#####################################################  
    
class BetoBasic(nn.Module):
    def __init__(
        self, 
        tokenizer, 
        model,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = model
        
    def tokenize_data(self, example):
        return self.tokenizer(example['features'], padding = 'max_length')
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden = self.bert(
            input_ids = input_ids, 
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        ).last_hidden_state[:, 0, :]
        return self.output_layer(hidden)
    
    def predecir_base(self, X, batch_size = 2, progress_bar = False):
        with torch.no_grad():
            self.eval()
#             print("    Generating predictions...")
            tokens = self.tokenizer(
                X.tolist(), 
                padding = "longest", 
                truncation = True
            )
            p = next(self.parameters())
            input_ids = torch.tensor(tokens["input_ids"]).long()
            token_type_ids = torch.tensor(tokens["token_type_ids"]).long()
            attention_mask = torch.tensor(tokens["attention_mask"]).long()
            dataset = tud.TensorDataset(
                input_ids, 
                token_type_ids,
                attention_mask,
            )
            loader = tud.DataLoader(dataset, batch_size = batch_size)
            output = []
            iterator = iter(loader)
            if progress_bar:
                iterator = tqdm(iterator)
            for batch in iterator:
                i, t, a = batch
                predictions = self.forward(
                    input_ids = i.to(p.device),
                    token_type_ids = t.to(p.device),
                    attention_mask = a.to(p.device)
                )
                output.append(predictions)
            return torch.cat(output, axis = 0)


class BetoMTL(BetoBasic):
    def __init__(
        self, 
        tokenizer, 
        model,
    ):
        super().__init__(tokenizer, model)
        self.output_layer = nn.Linear(768, 10)
        torch.save(self.state_dict(), "clf.pt")      
        
    def predecir_proba(self, X, **kwargs):
        return self.predecir_base(X, **kwargs).sigmoid().cpu().numpy()
    
    def predecir(self, X, **kwargs):
        return self.predecir_proba(X, **kwargs) > 0.5

    def entrenar(
        self, 
        X_train, 
        Y_train,
        epochs = 2, 
        batch_size = 2,
        learning_rate = 10**-5,
        progress_bar = True,
        refresh = True, 
        weight_decay = 0, 
        freeze_encoder = False, 
    ):
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)

        print("Training model...")
        if refresh:
            self.load_state_dict(torch.load("clf.pt"))
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        tokens = self.tokenizer(
            X_train.tolist(), 
            padding = "longest", 
            truncation = True
        )
        p = next(self.parameters())
        input_ids = torch.tensor(tokens["input_ids"]).long()
        token_type_ids = torch.tensor(tokens["token_type_ids"]).long()
        attention_mask = torch.tensor(tokens["attention_mask"]).long()
        label = torch.tensor(Y_train).float()
        dataset = tud.TensorDataset(
            input_ids, 
            token_type_ids,
            attention_mask,
            label
        )
        loader = tud.DataLoader(
            dataset, 
            shuffle = True, 
            batch_size = batch_size
        )
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
        training_steps = epochs * len(loader)
        if progress_bar:
            bar = tqdm(range(training_steps))
        self.train()
        for epoch in range(1, epochs + 1):
            losses = []
            accuracies = []
            for j, batch in enumerate(loader):
                i, t, a, l = batch
                predictions = self.forward(
                    input_ids = i.to(p.device),
                    token_type_ids = t.to(p.device),
                    attention_mask = a.to(p.device)
                )
                l = l.to(p.device)
                loss = criterion(predictions, l)
                accuracy = ((predictions > 0) == l).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if progress_bar:
                    bar.update(1)
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                if epoch == 1 and j == 0:
                    print(f"    first step, train loss:{loss.item():.4f}")
            total_loss = sum(losses) / len(losses)
            total_accuracy = sum(accuracies) / len(X_train) / 10
            print(f"    epoch: {epoch}, train loss:{total_loss:.4f}, train accuracy: {total_accuracy:.4f}")
