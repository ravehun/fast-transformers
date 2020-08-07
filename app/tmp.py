import torch
from transformers import GPT2Config, GPT2Model

batch_size = 14
seq_length = 7
is_training = True
use_token_type_ids = True
use_input_mask = True
use_labels = True
use_mc_token_ids = True
vocab_size = 2
hidden_size = 32
num_hidden_layers = 5
num_attention_heads = 4
intermediate_size = 37
hidden_act = "gelu"
hidden_dropout_prob = 0.1
attention_probs_dropout_prob = 0.1
max_position_embeddings = 512
type_vocab_size = 16
type_sequence_label_size = 2
initializer_range = 0.02
num_labels = 3
num_choices = 4

config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=hidden_size,
    n_layer=num_hidden_layers,
    n_head=num_attention_heads,
    # intermediate_size=intermediate_size,
    # hidden_act=hidden_act,
    # hidden_dropout_prob=hidden_dropout_prob,
    # attention_probs_dropout_prob=attention_probs_dropout_prob,
    n_positions=max_position_embeddings,
    n_ctx=max_position_embeddings,
    # type_vocab_size=type_vocab_size,
    # initializer_range=initializer_range,
    # bos_token_id=,
    # eos_token_id=eos_token_id,
    return_dict=True,
)

inputs_embeds = torch.zeros(2, 512, 32)
model = GPT2Model(config=config)
result = model(inputs_embeds=inputs_embeds)
print([x[0].shape for x in result])