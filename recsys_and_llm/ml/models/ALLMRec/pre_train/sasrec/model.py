import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(
            self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        )
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.kwargs = {"user_num": user_num, "item_num": item_num, "args": args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(
            self.item_num + 1, args.hidden_units, padding_idx=0
        )
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.args = args

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    # 모델의 시퀀스 입력 데이터를 특성으로 변환
    def log2feats(self, log_seqs):
        # [batch_size, seq_len, hidden_dim] 크기로 변환(고차원 벡터로 변환)
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        # 스케일링 기법 -> 큰 차원의 임베딩에서 숫자가 너무 작아지는 문제 방지
        seqs *= self.item_emb.embedding_dim**0.5
        # 0 ~ log_seqs의 길이 범위만큼 숫자로 나타내서 순서를 표시한다.
        # **단일 시퀀스로 수정됨**
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        # 두 가지 정보를 결합해서 self attention 학습 시에 아이템 자체의 의미와 순서의 의미를 학습한다.
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        # 패딩된 부분을 계산에서 제외하기 위해 -> log_seqs에서 item ID가 0이어도 임베딩 레이어를 거치면 값이 생길 수도 있다 -> mask 행렬과 곱해서 item ID가 0인 부분은 임베딩 값 0으로 만들어준다
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        # 어텐션 마스크 생성(하삼각 행렬: 아래쪽 삼각형 부분만 True 생성)
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):  # block 개수
            # Multi-Head Attention 레이어의 입력 형식은 (seq_len, batch_size, hidden_dim)과 같은 형태를 요구한다 -> transpose
            # 시퀀스의 각 위치 간 관계를 계산해야 하므로 시퀀스 길이를 첫 번째 차원으로 두는 것이 메모리와 연산 효율성 측면에서 유리
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            # Q, K, V
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs, attn_mask=attention_mask
            )
            # residual connection
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            # 어텐션 연산을 수행하면서 패딩 값이 잠재적으로 영향을 미칠 수 있음
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs, pos_seqs=None, neg_seqs=None, mode="default"):
        log_feats = self.log2feats(log_seqs)
        if mode == "log_only":
            log_feats = log_feats[:, -1, :]
            return log_feats

        # log_seqs는 사용자의 과거 상호작용 시퀀스 -> 순서, 상호작용 패턴을 학습한다
        # pos_embs, neg_embs는 log_feats와 달리 시퀀스의 관계를 학습할 필요가 없다. 이들의 역할은 단순히 유사도를 계산하는 데 필요한 비교 대상
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        # 유사도 계산
        pos_logits = (log_feats * pos_embs).sum(
            dim=-1
        )  # 마지막 차원을 전부 합산해 차원을 없앤다.
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        if mode == "item":
            return (
                log_feats.reshape(-1, log_feats.shape[2]),
                pos_embs.reshape(-1, log_feats.shape[2]),
                neg_embs.reshape(-1, log_feats.shape[2]),
            )
        else:
            return pos_logits, neg_logits

    def predict(self, item_indices, user_ids=None, log_seqs=None, log_emb=None):
        if log_emb is None:
            log_feats = self.log2feats(log_seqs)
            final_feat = log_feats[:, -1, :]
        else:
            final_feat = log_emb

        # log_feats는 [batch_size, seq_len, hidden_dim]으로 이루어져 있음 -> final_feat은 seq_len 마지막 시점에서의 사용자 특성 나타냄
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        return logits
