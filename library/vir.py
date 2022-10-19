import transformers
# import datasets
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm.auto import tqdm
import torch.utils.data as tud
import torch
import torch.nn as nn
import sys
# import psutil

def show(df):
    print(df.shape)
    return df.head()


class CustomLogisticRegression:
    def __init__(self, **kwargs):
        self.classifier = LogisticRegression(**kwargs)
    
    def entrenar(self, X, Y):
        self.classifier.fit(X, Y)
        
    def predecir(self, X):
        return self.classifier.predict(X)

    
class CustomRandomForestClassifier:
    def __init__(self, **kwargs):
        self.classifier = RandomForestClassifier(**kwargs)
    
    def entrenar(self, X, Y):
        self.classifier.fit(X, Y)
        
    def predecir(self, X):
        return self.classifier.predict(X)

    
class CustomSVC:
    def __init__(self, **kwargs):
        self.classifier = SVC(**kwargs)
    
    def entrenar(self, X, Y):
        self.classifier.fit(X, Y)
        
    def predecir(self, X):
        return self.classifier.predict(X)

    
class Ensemble:
    def __init__(self):
        self.lr = LogisticRegression(class_weight = "balanced", C = 2**-3)
        self.svc = SVC(class_weight = "balanced", probability = True, C = 2**-1)
    
    def entrenar(self, X, Y):
        self.lr.fit(X, Y)
        self.svc.fit(X, Y)
        
    def predecir(self, X):
        predicted_proba = (self.lr.predict_proba(X) + self.svc.predict_proba(X)) / 2
        return predicted_proba[:, 1] > 0.5
    
    
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
    
class Beto(BetoBasic):
    def __init__(
        self, 
        tokenizer, 
        model,
    ):
        super().__init__(tokenizer, model)
        self.output_layer = nn.Linear(768, 2)
        torch.save(self.state_dict(), "clf.pt") 
        
    def predecir_proba(self, X, **kwargs):
        return self.predecir_base(X, **kwargs).softmax(axis = 1).cpu().numpy()
    
    def predecir(self, X, **kwargs):
        return self.predecir_proba(X, **kwargs)[:, 1] > 0.5

    def entrenar(
        self, 
        X, 
        y, 
        epochs = 2, 
        refresh = True, 
        weight_decay = 0, 
        freeze_encoder = False, 
        batch_size = 2,
        learning_rate = 10**-5,
        class_weights = None,
        progress_bar = True
    ):
        if refresh:
            self.load_state_dict(torch.load("clf.pt"))
        for param in self.bert.parameters():
            if freeze_encoder:
                param.requires_grad = False
            else:
                param.requires_grad = True
        tokens = self.tokenizer(
            X.tolist(), 
            padding = "longest", 
            truncation = True
        )
        p = next(self.parameters())
        input_ids = torch.tensor(tokens["input_ids"]).long()
        token_type_ids = torch.tensor(tokens["token_type_ids"]).long()
        attention_mask = torch.tensor(tokens["attention_mask"]).long()
        label = torch.tensor(y.values).long()
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
        if class_weights == "balanced":
            cw = torch.tensor(
                1 / y.value_counts(normalize = True), 
                device = p.device
            ).float()
            print(cw)
            criterion = nn.CrossEntropyLoss(weight = cw)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr = learning_rate, 
            weight_decay = weight_decay
        )
        training_steps = int(epochs * len(loader))
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
                loss = criterion(predictions, l.to(p.device))
                max_values, max_indices = predictions.max(axis = 1)
                accuracy = (max_indices == l.to(p.device)).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
                accuracies.append(accuracy.item())
                if progress_bar:
                    bar.update(1)
                if epoch == 1 and j == 0:
                    print(f"      first step, train loss:{loss.item():.4f}")
            total_loss = sum(losses) / len(losses)
            total_accuracy = sum(accuracies) / len(X)
            print(f"      epoch: {epoch:>3}, train loss:{total_loss:.4f}, train accuracy: {total_accuracy:.4f}")

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
        X_test,
        Y_test,
        epochs = 2, 
        batch_size = 20,
        learning_rate = 10**-5,
        progress_bar = True,
        refresh = True, 
        weight_decay = 0, 
        freeze_encoder = False, 
    ):
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(Y_test, np.ndarray)
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
        # --- VIR----
        # prueba de pesos por clase
        #*******************************
        # pos_weight = torch.tensor(2946/871)
        # pos_weight = torch.ones([1])
        # weights_per_batch = torch.ones(batch_size)
        weights_per_batch = torch.tensor(1)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = nn.BCEWithLogitsLoss(weight = weights_per_batch)
        #*******************************
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
            
            
# class BetoMTL_HF(object):
#     def __init__(self, model, tokenizer):
#         # model_name = '/home/vsabando/detests2022/assets/bert-base-spanish-wwm-uncased/model/'
#         # tokenizer_name = '/home/vsabando/detests2022/assets/bert-base-spanish-wwm-uncased/tokenizer/'
#         self.tokenizer = tokenizer
#         self.model = model
#         torch.save(self.state_dict(), "clf.pt") 
        
#     def tokenize_data(self, example):
#         return self.tokenizer(example['features'], padding = 'max_length')
        
#     def forward(self, input_ids, token_type_ids, attention_mask):
#         hidden = self.bert(
#             input_ids = input_ids, 
#             token_type_ids = token_type_ids,
#             attention_mask = attention_mask
#         ).last_hidden_state[:, 0, :]
#         return self.output_layer(hidden)
    
#     def predecir_base(self, X, batch_size = 2, progress_bar = False):
#         with torch.no_grad():
#             self.model.eval()
#             tokens = self.tokenizer(
#                 X.tolist(), 
#                 padding = "longest", 
#                 truncation = True
#             )
#             p = next(self.model.parameters())
#             input_ids = torch.tensor(tokens["input_ids"]).long()
#             token_type_ids = torch.tensor(tokens["token_type_ids"]).long()
#             attention_mask = torch.tensor(tokens["attention_mask"]).long()
#             dataset = tud.TensorDataset(
#                 input_ids, 
#                 token_type_ids,
#                 attention_mask,
#             )
#             loader = tud.DataLoader(dataset, batch_size = batch_size)
#             output = []
#             iterator = iter(loader)
#             if progress_bar:
#                 iterator = tqdm(iterator)
#             for batch in iterator:
#                 i, t, a = batch
#                 predictions = self.forward(
#                     input_ids = i.to(p.device),
#                     token_type_ids = t.to(p.device),
#                     attention_mask = a.to(p.device)
#                 )
#                 output.append(predictions)
#             return torch.cat(output, axis = 0)
        
#     def predecir_proba(self, X, **kwargs):
#         return self.predecir_base(X, **kwargs).sigmoid().cpu().numpy()
    
#     def predecir(self, X, **kwargs):
#         return self.predecir_proba(X, **kwargs) > 0.5

#     def entrenar(
#         self, 
#         X_train, 
#         Y_train,
#         X_test,
#         Y_test,
#         epochs = 2, 
#         batch_size = 20,
#         learning_rate = 10**-5,
#         progress_bar = True,
#         refresh = True, 
#         weight_decay = 0, 
#         freeze_encoder = False, 
#     ):
#         assert isinstance(X_train, np.ndarray)
#         assert isinstance(Y_train, np.ndarray)
#         assert isinstance(X_test, np.ndarray)
#         assert isinstance(Y_test, np.ndarray)
#         print("Training model...")
#         if refresh:
#             self.load_state_dict(torch.load("clf.pt"))
#         if freeze_encoder:
#             for param in self.bert.parameters():
#                 param.requires_grad = False
#         tokens = self.tokenizer(
#             X_train.tolist(), 
#             padding = "longest", 
#             truncation = True
#         )
#         p = next(self.parameters())
#         input_ids = torch.tensor(tokens["input_ids"]).long()
#         token_type_ids = torch.tensor(tokens["token_type_ids"]).long()
#         attention_mask = torch.tensor(tokens["attention_mask"]).long()
#         label = torch.tensor(Y_train).float()
#         dataset = tud.TensorDataset(
#             input_ids, 
#             token_type_ids,
#             attention_mask,
#             label
#         )
#         loader = tud.DataLoader(
#             dataset, 
#             shuffle = True, 
#             batch_size = batch_size
#         )
#         #  *********************************************************************************************
#         pos_weight = torch.rand(10)
#         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         #  *********************************************************************************************
#         optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate, weight_decay = weight_decay)
#         training_steps = epochs * len(loader)
#         if progress_bar:
#             bar = tqdm(range(training_steps))
#         self.train()
#         for epoch in range(1, epochs + 1):
#             losses = []
#             accuracies = []
#             for j, batch in enumerate(loader):
#                 i, t, a, l = batch
#                 predictions = self.forward(
#                     input_ids = i.to(p.device),
#                     token_type_ids = t.to(p.device),
#                     attention_mask = a.to(p.device)
#                 )
#                 l = l.to(p.device)
#                 loss = criterion(predictions, l)
#                 accuracy = ((predictions > 0) == l).sum()
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 if progress_bar:
#                     bar.update(1)
#                 losses.append(loss.item())
#                 accuracies.append(accuracy.item())
#                 if epoch == 1 and j == 0:
#                     print(f"    first step, train loss:{loss.item():.4f}")
#             total_loss = sum(losses) / len(losses)
#             total_accuracy = sum(accuracies) / len(X_train) / 10
#             print(f"    epoch: {epoch}, train loss:{total_loss:.4f}, train accuracy: {total_accuracy:.4f}")