import torch

from .transformer_decoder import TransformerBlock


class GSASRec(torch.nn.Module):
    def __init__(
        self,
        num_items,
        sequence_length=200,
        embedding_dim=256,
        num_heads=4,
        num_blocks=3,
        dropout_rate=0.5,
        reuse_item_embeddings=False,
    ):
        super(GSASRec, self).__init__()
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)

        self.num_heads = num_heads

        self.item_embedding = torch.nn.Embedding(
            self.num_items + 2, self.embedding_dim
        )  # items are enumerated from 1;  +1 for padding
        self.position_embedding = torch.nn.Embedding(
            self.sequence_length, self.embedding_dim
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate
                )
                for _ in range(num_blocks)
            ]
        )
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(
                self.num_items + 2, self.embedding_dim
            )

    def get_output_embeddings(self) -> torch.nn.Embedding:
        if self.reuse_item_embeddings:
            return self.item_embedding
        else:
            return self.output_embedding

    # returns last hidden state and the attention weights
    def forward(self, input):
        seq = self.item_embedding(input.long())
        if seq.dim() == 3:  # 이미 3차원인 경우
            pass  # 아무 작업도 하지 않음
        elif seq.dim() == 2:  # 2차원인 경우
            seq = seq.unsqueeze(
                0
            )  # [batch_size, embedding_dim] -> [1, batch_size, embedding_dim]
        mask = (input != self.num_items + 1).float().unsqueeze(-1)
        # print(seq.shape)
        bs = seq.size(0)
        positions = (
            torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        )
        pos_embeddings = self.position_embedding(positions)[: input.size(0)]
        # print(pos_embeddings.shape)
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_predictions(self, input, limit, rated=None):
        # print(input.shape)
        with torch.no_grad():
            model_out, _ = self.forward(input)
            # print(model_out.shape)
            seq_emb = model_out[:, -1, :]
            # print(seq_emb.shape)
            output_embeddings = self.get_output_embeddings()

            # print(output_embeddings.weight.shape)
            # print(seq_emb.shape)
            scores = torch.einsum("bd,nd->bn", seq_emb, output_embeddings.weight)
            scores[:, 0] = float("-inf")
            scores[:, self.num_items + 1 :] = float("-inf")
            # print(scores.shape)
            if rated is not None:
                for i in range(len(rated)):
                    scores[0, rated[i]] = float("-inf")

            result = torch.topk(scores, limit, dim=1)
            return result.indices, result.values
