import transformers, random
from nn.prenet import *

class DecisionTransformer(nn.Module):
    def __init__(self, config:transformers.GPT2Config):
        super(DecisionTransformer, self).__init__()
        self.token_dim = config.n_embd
        self.GPT = transformers.GPT2Model(config)
        self.state_embed = StateEmbed(self.token_dim)
        self.reward_embed = RewardEmbed(self.token_dim)
        self.path_embed = PathEmbed(dim=self.token_dim)
        self.layernorm = nn.LayerNorm(self.token_dim)
        self.path_predict = PathPredict(dim=self.token_dim)

    def forward(self, states, reward, path, mask):
        batch_size, seq_length = reward.shape[0], reward.shape[1]
        state_tokens = self.state_embed(states)
        reward_token = self.reward_embed(reward)
        path_tokens = self.path_embed(path)
        state_len = len(state_tokens)
        path_len = len(path_tokens)
        tokens = torch.stack([reward_token,]+state_tokens+path_tokens, dim=1).permute(0, 2, 1, 3).reshape(batch_size, seq_length*(1+state_len+path_len), self.token_dim)
        mask = torch.stack([mask]*(1+state_len+path_len), dim=1).permute(0, 2, 1).reshape(batch_size, seq_length*(1+state_len+path_len))
        mask_pt = torch.argmax(mask.to(dtype=torch.int64), dim=1)
        position_ids = torch.stack([torch.cat([torch.zeros((mask_pt[i].item(),), dtype=torch.int64), torch.arange(1, mask.shape[1]-mask_pt[i].item()+1, dtype=torch.int64)], dim=0) for i in range(batch_size)], dim=0).to(device="cuda", dtype=torch.int64)
        mask_pt = random.randint(0, path_len)
        mask, _ = mask.split([mask.shape[1]-mask_pt, mask_pt], dim=1)
        mask = torch.cat([mask, torch.zeros((batch_size, mask_pt), device="cuda", dtype=torch.bool)], dim=1)
        tokens = self.layernorm(tokens)
        output_tokens = self.GPT(inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids)['last_hidden_state'].reshape(batch_size, seq_length, (1+state_len+path_len), self.token_dim).permute(0, 2, 1, 3)
        path = self.path_predict(output_tokens[:, state_len:state_len+path_len]).permute(1, 0, 2, 3)
        return [path[i] for i in range(path.shape[0])]

    def learn(self, states, reward, path, mask):
        batch_size, seq_length = reward.shape[0], reward.shape[1]
        path_pred = self.forward(states, reward, path, mask)
        loss = None
        for i in range(len(path)):
            path_i = path[i].reshape((batch_size*seq_length,))[mask.reshape(-1)==True]
            path_pred_i = path_pred[i].reshape((batch_size*seq_length,-1))[mask.reshape(-1)==True]
            loss_i = F.nll_loss(torch.log(path_pred_i), path_i)
            if loss == None:
                loss = loss_i
            else: loss = loss + loss_i
        return loss

    def pred_path(self, states, reward, path):
        with torch.no_grad():
            seq_length = reward.shape[0]
            for i in range(len(states)):
                states[i] = states[i].unsqueeze(0)
            state_tokens = self.state_embed(states)
            state_len = len(state_tokens)
            reward_token = self.reward_embed(reward.unsqueeze(0))
            for i in range(len(path)):
                path[i] = path[i].unsqueeze(0)
            radian = []
            for i in range(len(path)):
                if i > 0:
                    path[i-1][0][-1][0] = radian[-1]
                path_tokens = self.path_embed(path)
                path_len = len(path_tokens)
                tokens = torch.stack([reward_token,]+state_tokens+path_tokens, dim=1).permute(0, 2, 1, 3).reshape(1, seq_length*(1+state_len+path_len), self.token_dim)
                mask = torch.ones((1, seq_length)).to(dtype=torch.bool, device="cuda")
                mask = torch.stack([mask]*(1+state_len+path_len), dim=1).permute(0, 2, 1).reshape(1, seq_length*(1+state_len+path_len))
                mask, _ = mask.split([seq_length*(1+state_len+path_len)-path_len+i, path_len-i], dim=1)
                mask = torch.cat([mask, torch.zeros((1, path_len-i)).to(dtype=torch.bool, device="cuda")], dim=1)
                position_ids = torch.arange(1, mask.shape[1]+1, device="cuda", dtype=torch.int64).unsqueeze(0)
                tokens = self.layernorm(tokens)
                output_tokens = self.GPT(inputs_embeds=tokens, attention_mask=mask, position_ids=position_ids)['last_hidden_state'].reshape(1, seq_length, (1+state_len+path_len), self.token_dim).permute(0, 2, 1, 3)
                path_pred = self.path_predict(output_tokens[:, state_len:state_len+path_len]).permute(1, 0, 2, 3).reshape(path_len, seq_length, -1).permute(1, 0, 2)
                path_pred = torch.argmax(path_pred[-1], dim=1)
                radian.append(path_pred[i].item())
            torch.cuda.empty_cache()
        return radian
        

