from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden = self.bert(
            input_ids = input_ids, 
            token_type_ids = token_type_ids,
            attention_mask = attention_mask
        ).last_hidden_state[:, 0, :]
        return self.output_layer(hidden)
    
    
    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    
    
    
    
    
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