import torch
from typing import Optional
from transformers import BertModel,BertPreTrainedModel,PretrainedConfig


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig = None):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        #self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
    ):  
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooler_output = outputs[1]  
        # outputs[0] = last_hidden_state (batch_size, seq_len, hidden_size) 입력 문장의 각 토큰의 임베딩값
        # outputs[1] = pooler_output : (batch_size, hidden_size) 입력 문장 전체에 대한 [CLS]의 임베딩값
        return pooler_output
    
    
#class RobertaEncoder(RobertaPreTrainedModel):