import torch, tqdm, os, random
import torch.utils.tensorboard as tensorboard

from data import Experience_Pool, exp_get_batch
from nn import DecisionTransformer, GPT2Config

class train:
    def __init__(self, model, optimizer, scheduler, epoch=10000, batch=256, grad_norm_clip=9.0, fine_tune=False):
        self.model = model
        if fine_tune == True:
            self.model.load_state_dict(torch.load('model/transformer.pt'))
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = epoch
        self.batch = batch
        self.grad_norm_clip = grad_norm_clip
        self.exp_pool = Experience_Pool()

    def __run_epoch(self, batch_data):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.model.learn(*batch_data)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        # print(total_norm, flush=True)
        self.optimizer.step()
        if self.scheduler != None:
            self.scheduler.step()
        return loss

    def __run(self, epoch:int, tb_writer:tensorboard.SummaryWriter)->None:
        batch_data = exp_get_batch(self.exp_pool, self.batch)
        loss = self.__run_epoch(batch_data)
        tb_writer.add_scalar('Loss', loss.item(), epoch)

    def Run(self)->None:
        print('--Training Process Running:', flush=True)
        if not os.path.exists('./log/model'): os.mkdir('./log/model')
        tb_writer = tensorboard.SummaryWriter(log_dir='./log/tfboard')
        for epoch in tqdm.tqdm(range(self.epoch), '', ncols=120, unit='Epoch'):
            self.__run(epoch, tb_writer)
            if (epoch+1) % 1000 == 0:
                torch.save(self.model.state_dict(), './log/model/'+str(int(epoch/1000))+'.pt')
        self.exp_pool.close()
        torch.save(self.model.state_dict(), './log/model/last_model.pt')
        print('--Finish: Models are saved in ./log/model .', flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--nlayer", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ninner", type=int, default=64*4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=10**-4)
    parser.add_argument("--grad_norm_clip", type=float, default=4.0)
    parser.add_argument("--weight_decay", type=float, default=10**-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parse = parser.parse_args()

    config = GPT2Config(n_embd=parse.token_dim, n_layer=parse.nlayer, n_head=parse.nhead, n_inner=parse.ninner, resid_pdrop=parse.dropout)
    model = DecisionTransformer(config).cuda().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parse.lr, weight_decay=parse.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps+1)/parse.warmup_steps, 1))
    Train = train(model, optimizer, scheduler, parse.epoch, parse.batch, parse.grad_norm_clip)
    Train.Run()
