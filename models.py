# models.py
import torch.nn as nn
from transformers import PreTrainedModel, BertModel, BertConfig

class ActorCriticBERT(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.policy_head = nn.Linear(config.hidden_size, num_labels)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.policy_head(pooled_output)
        value = self.value_head(pooled_output)
        return {"logits": logits, "value": value}