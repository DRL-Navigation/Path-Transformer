from nn.prenet import *

class ReturnPred(nn.Module):
    def __init__(self, config):
        super(ReturnPred, self).__init__()
        self.token_dim = config.n_embd
        self.state_embed = StateEmbed_ReturnPred(self.token_dim)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.mlp = mlp([(self.token_dim, self.token_dim, "relu"), (self.token_dim, 1, "None")])
        self.loss_fn = lambda pred, true: torch.mean(torch.abs(pred-true))

    def forward(self, states):
        x = self.state_embed(states)
        x = self.dropout(x)
        x = self.mlp(x)
        return x

    def learn(self, states, reward):
        reward_pred = self.forward(states)
        loss = self.loss_fn(reward_pred, reward)
        return loss

    def pred_return(self, states):
        with torch.no_grad():
            return_pred = self.forward(states)
            torch.cuda.empty_cache()
        return return_pred