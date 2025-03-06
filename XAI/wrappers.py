import torch
import torch.nn as nn


class BertWrapper(nn.Module):
    
    def __init__(self, model, method):
        super(BertWrapper, self).__init__()
        self.model = model
        self.method = method
    
    def forward(self, inputs_embeds,  aa_position, embedding_position, metric=False):
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        past_key_values_length = 0
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=inputs_embeds.device)
        extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.model.get_head_mask(None, self.model.config.num_hidden_layers)
        encoder_outputs = self.model.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            use_cache=False
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.pooler(sequence_output)
        if not metric and self.method == 'DeepLift':
            return torch.cat((sequence_output[0][aa_position][embedding_position].reshape((1, )), torch.tensor([0]).cuda()))
        return sequence_output[0][aa_position][embedding_position].reshape((1, ))


class T5Wrapper(nn.Module):
    
    def __init__(self, model, method):
        super(T5Wrapper, self).__init__()
        self.model = model
        self.method = method

    def forward(self, inputs_embeds, aa_position, embedding_position, metric=False):
        use_cache = self.model.config.use_cache
        output_attentions = self.model.config.output_attentions
        input_shape = inputs_embeds.size()[:-1]
        batch_size, seq_length = input_shape
        mask_seq_length = seq_length
        attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        past_key_values = [None] * len(self.model.block)
        extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, input_shape)
        encoder_extended_attention_mask = None
        head_mask = self.model.get_head_mask(None, self.model.config.num_layers)
        cross_attn_head_mask = self.model.get_head_mask(None, self.model.config.num_layers)
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.model.dropout(inputs_embeds)
        for i, (layer_module, past_key_value) in enumerate(zip(self.model.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=None,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, _ = layer_outputs[:2]
            position_bias = layer_outputs[2]
        hidden_states = self.model.final_layer_norm(hidden_states)
        hidden_states = self.model.dropout(hidden_states)
        if not metric and self.method == 'DeepLift':
            return torch.cat((hidden_states[0][aa_position][embedding_position].reshape((1, )), torch.tensor([0]).cuda()))
        return hidden_states[0][aa_position][embedding_position].reshape((1, ))