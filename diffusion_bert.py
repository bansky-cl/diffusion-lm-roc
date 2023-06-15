import math
from torch.utils.data.dataset import Dataset
import csv
from transformers import AutoModelForPreTraining,AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F 
import random

# 数据集roc
class ROCstory(Dataset): 
    def __init__(self,csv_dir,init_model, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(init_model)
        print("create dataloader from " + csv_dir)
        with open(csv_dir,'r', encoding='UTF-8') as f:
            story_teller = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
            self.story = list(story_teller)
        self.max_len = max_len
        self.tokenizer.model_max_length = max_len
    def __getitem__(self, index):
        #index = 0
        #index = random.randint(0,9)
        story = "".join(self.story[index][2:])
        from_tokenizer = self.tokenizer(story,padding="max_length",truncation = True,return_tensors="pt")
        input_ids = from_tokenizer["input_ids"].squeeze_().long()
        token_type_ids = from_tokenizer["token_type_ids"].squeeze_().long()
        attention_mask = from_tokenizer["attention_mask"].squeeze_().long()
        return input_ids,token_type_ids,attention_mask
    def __len__(self):
        return len(self.story)
    
# 数据集e2e
class e2e(Dataset): 
    def __init__(self,csv_dir, init_model, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(init_model)
        with open(csv_dir,'r', encoding='UTF-8') as f:
            story_teller = f.readlines()
            self.story = list(story_teller)
        self.max_len = max_len
        self.tokenizer.model_max_length = max_len
    def __getitem__(self, index):
        #index = 0
        #index = random.randint(0,9)
        story = self.story[index].split("||")[-1].strip() # || 分割
        from_tokenizer = self.tokenizer(story,padding="max_length",truncation = True,return_tensors="pt")
        input_ids = from_tokenizer["input_ids"].squeeze_().long()
        token_type_ids = from_tokenizer["token_type_ids"].squeeze_().long()
        attention_mask = from_tokenizer["attention_mask"].squeeze_().long()
        return input_ids,token_type_ids,attention_mask
    def __len__(self):
        return len(self.story)

# diffusion主体
class diffusion_bert(nn.Module):
    
    def __init__(self, init_model, max_len, max_step) -> None:
        super().__init__()
        # freezed_w 获取冻结的参数
        if "bert-base" in init_model:
            self.model = AutoModelForMaskedLM.from_pretrained(init_model)
            freezed_w = [self.model.bert.embeddings.token_type_embeddings.weight,self.model.bert.embeddings.word_embeddings.weight] #self.model.bert.embeddings.LayerNorm.weight, self.model.bert.embeddings.LayerNorm.bias
        else:
            self.model = AutoModelForPreTraining.from_pretrained(init_model)
            freezed_w = [self.model.cls.seq_relationship.bias, self.model.cls.seq_relationship.weight, self.model.bert.pooler.dense.bias, self.model.bert.pooler.dense.weight, self.model.bert.embeddings.token_type_embeddings.weight,self.model.bert.embeddings.word_embeddings.weight] #self.model.bert.embeddings.LayerNorm.weight, self.model.bert.embeddings.LayerNorm.bias
        self.max_len = max_len # 最大长度
        self.max_step = max_step # 最大步数 T
        self.time_embed = nn.Embedding(max_step,self.model.config.hidden_size) # 时间步 embedding layer 768 维度
        # self.layernorm = nn.LayerNorm(self.model.config.hidden_size, eps=self.model.config.layer_norm_eps)
        for p in  freezed_w:  # 冻住不训这个embedding model
            p.requires_grad = False
        nn.init.constant_(self.time_embed.weight, 0)
        
    def forward(self,input_ids,token_type_ids,attention_mask,t =None):
        input_shape = input_ids.size()  # [bsz, seq_len]
        seq_length = input_shape[1]  # seq_len 因为每句话的长度不一样
         
        position_ids = self.model.bert.embeddings.position_ids[:, 0 : seq_length] # [1, seq_len]每句话的位置，从0开始到句子长度，最长不过512
        position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids) # [1, seq_len, 768]
       
        with torch.no_grad():
            word_emb = self.model.bert.embeddings.word_embeddings(input_ids)  # [seq_len, 768]
            #print(word_emb.shape)
            token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids) # [seq_len, 768]
            
            # 扩散步
            # input_shape[0] = bzs = 64
            # t ~ U[1,T] 
            if t is None:
                diffusion_steps = torch.randint(0,self.max_step,size = (input_shape[0],),device=input_ids.device) # [bzs] 一维向量，里面值是[1-T]间取得
            else:
                diffusion_steps = torch.ones(size = (input_shape[0],),device=input_ids.device).long()*t # [bzs] 一维向量，里面全是t

            # 噪声生成 randn_like 随机生成 数值会超过[-1,1]
            # 除了sqrt变小很多，保持在[-1,1]
            # 类似attention
            noise = torch.randn_like(word_emb) / math.sqrt(self.model.config.hidden_size) # [seq_len, 768] 和word_emb形状一样
            # +1 是因为0开始下标 类似t/T
            # [bzs].view(-1,1,1) to [bzs,1,1] 类似生了2个维度
            alpha = 1 - torch.sqrt((diffusion_steps+1)/self.max_step).view(-1,1,1)  # [bzs,1,1]
            # 加了噪声的x_t = sqrt(alpha) * x_t + sqrt(1-alpha)*noise
            # 但是因为bert的方法，所以词向量加了噪声后是直接把token_type_embeddings 也加上去的
            noisy_word = torch.sqrt(alpha) * word_emb + torch.sqrt(1 - alpha) * noise + token_type_embeddings # [seq_len, 768]
        
        # [bzs] -> time_embed(T,768) -> [bzs,768] -> unsqueeze(1) -> [bzs,1,768]
        time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)
        # [seq_len, 768] + [1, seq_len, 768] + [bzs,1,768] -> [bzs, seq_len, 768]
        noisy_word = noisy_word + position_embeddings + time_embedding # 对应位置加，得到噪声
        
        #noisy_word = self.layernorm(noisy_word)
        noisy_word = self.model.bert.embeddings.LayerNorm(noisy_word) # [bzs, seq_len, 768]
        # attention 函数
        # attention_scores = attention_scores + attention_mask
        # 因为这里的 attention_mask 已经【被动过手脚】
        # 将原本为 1 的部分变为 0，而原本为 0 的部分（即 padding）变为一个较大的负数
        # 这样相加就得到了一个较大的负值
        # 至于为什么要用【一个较大的负数】？因为这样一来经过 softmax 操作以后这一项就会变成接近 0 的小数。
        # 最终在 BertModel 的前向传播过程中找到了这一调用
        # attention_mask 参数传入，也就是tok后返回字典里面的attention mask
        extended_attention_mask = self.model.bert.get_extended_attention_mask(attention_mask, input_shape) # [5, 1, 1, 64]
        # 现在才对带噪声的x_t编码,得到一个768维的向量
        encoder_outputs = self.model.bert.encoder( # [[bsz, seq_len, 768]]
            noisy_word, # [bzs, seq_len, 768]
            attention_mask=extended_attention_mask, # 这个有点复杂
            head_mask=[None] * self.model.config.num_hidden_layers # 12
        )

        sequence_output = encoder_outputs[0] # [bsz, seq_len, 768]
          
        # 预测token，768 to 30522(词表)
        prediction_scores = self.model.cls.predictions(sequence_output) #  [bsz, seq_len, 30522]
        
        # loss 
        # [bsz, seq_len, 30522] to [bzs*seq_len, 30522]
        # [1, seq_len] to [seq_len] 作为类别标签
        loss = F.cross_entropy(prediction_scores.view(-1, self.model.config.vocab_size),input_ids.flatten(),ignore_index=0)
        
        #loss = F.smooth_l1_loss(sequence_output,word_emb)
        return loss,prediction_scores,diffusion_steps

    def test_pretrained(self,input_ids,token_type_ids,attention_mask):
        loss,prediction_scores,diffusion_steps = self.forward(input_ids,token_type_ids,attention_mask,0)
        return loss,prediction_scores,diffusion_steps

    # infer
    @torch.no_grad()
    def sampler(self, device, k=10, N=128):
        import time
        
        start_time = time.time()
        # mean, std = stats
        # mean = torch.tensor(mean).view(1,3,1,1)
        # std = torch.tensor(std).view(1,3,1,1)
        
        # N is the sample numbers, because is gen in parallel    
        noisy_word = torch.normal(0,1,(N, self.max_len,self.model.config.hidden_size)).to(device) / math.sqrt(self.model.config.hidden_size) # [bzs, seq_len, 768]
        token_type_ids = torch.zeros(N, self.max_len).long().to(device)
        attention_mask = torch.ones(N, self.max_len).long().to(device)
        extended_attention_mask = self.model.bert.get_extended_attention_mask(attention_mask, attention_mask.shape)

        position_ids = self.model.bert.embeddings.position_ids[:, 0 : self.max_len]
        position_embeddings = self.model.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.model.bert.embeddings.token_type_embeddings(token_type_ids)
        
        for t in range(self.max_step-1, 0, -k):
        # for t in range(1999,0,-1):
            # 1999 -> 1989 -> 1979 -> ... -> 19 -> 9 , k is control the interval
            # prepare time emb
            diffusion_steps = torch.ones(size = (N, ),device=device).long() * t
            time_embedding = self.time_embed(diffusion_steps).unsqueeze(1)
            
            # cat 4 embedding
            model_input = noisy_word + position_embeddings + token_type_embeddings + time_embedding
            model_input = self.model.bert.embeddings.LayerNorm(model_input)
            
            # denoise
            encoder_outputs = self.model.bert.encoder(
                model_input,
                attention_mask=extended_attention_mask,
                head_mask=[None] * self.model.config.num_hidden_layers
            )
            
            sequence_output = encoder_outputs[0] # [bsz, seq_len, 768]
            prediction_scores = self.model.cls.predictions(sequence_output) #  [bsz, seq_len, 30522]

            # clamp
            # pred = torch.argmax(prediction_scores,-1).long()
            # denoised_word = self.model.bert.embeddings.word_embeddings(pred)
            # [bsz, seq_len, 30522] 每个维度的行归一化 @ [1, 30522, 768] -> [bzs, seq_len, 768]
            # model's predict x_0
            denoised_word = prediction_scores.softmax(-1) @ self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0) 
            
            # DDIM
            # alpha_t and alpha_t_minus_delta_t should be enough close, here is 1e-5
            alpha_tk = 1 - math.sqrt( (t + 1 - k) / self.max_step) # + 1e-5 alpha_t_minus_delta_t
            alpha_t = 1 - math.sqrt( (t + 1) / self.max_step) + 1e-5 # alpha_t
            
            # formula(9) 
            # f^(t)_theta(x_t) = (x_t - sqrt(1 - alpha_t) * epsilon^t_theta(x_t)) / sqrt(alpha_t) # model's predict x_0
            # transposition
            # epsilon^t_theta(x_t) = (x_t - sqrt(alpha_t) * f^(t)_theta(x_t) ) / sqrt(1 - alpha_t) # is the model that attempt to predict epsilon_t from x_t without knowledge x_0
            noise = (noisy_word - math.sqrt(alpha_t) * denoised_word) / math.sqrt(1 - alpha_t)
            
            # formula (13)
            # x_t_minus_delta_t = sqrt(alpha_t_minus_delta_t) * (
            #   x_t / sqrt(alpha_t) + epsilon^t_theta(x_t) * (
            #       sqrt( (1 - alpha_t_minus_delta_t) / alpha_t_minus_delta_t ) - sqrt( (1 - alpha_t) / alpha_t )
            #       )
            #   )
            noisy_word = math.sqrt(alpha_tk) * ( \
                noisy_word / math.sqrt(alpha_t) + noise * ( \
                    math.sqrt((1 - alpha_tk) / alpha_tk)- math.sqrt((1 - alpha_t) / alpha_t) \
                    ) \
                )
            # origin code
            # noisy_word = math.sqrt(alpha_tk) * (noisy_word / math.sqrt(alpha_t) +  (math.sqrt((1 - alpha_tk) / alpha_tk)- math.sqrt((1 - alpha_t) / alpha_t)) * noise)
            # noisy_word = math.sqrt(alpha_tk) * denoised_word + math.sqrt(1 - alpha_tk) * noise
            # 1999 -> 9 , infer time  
            print(f"\rnoise level {t}  {time.time()-start_time:.2f} seconds",end='')
        

        pred = torch.argmax(prediction_scores, -1).long()
        return pred




if __name__ == "__main__":
    
    # TEST   
    import os,sys
    get_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(get_path)
    # initializing = get_path+'/bert-base-uncased'
    initializing = 'bert-base-uncased'
    max_len = 64
    diff_step = 2000
    device = torch.device('cuda')
    model = diffusion_bert(initializing,max_len,diff_step)
    # state = torch.load(get_path+'/'+sys.argv[1]) #"/Saved_Models/20220903bert_diffusion/bestloss.pkl")
    # model.load_state_dict(state,strict=True)
    model = model.to(device)
    model.eval()
    
    test_set = ROCstory(get_path+"/ROCstory_test.csv",init_model=initializing,max_len=max_len)
    # out = model.sampler(device,int(sys.argv[2]),int(sys.argv[3]))
    out = model.sampler(device)

    f = open(get_path+"/samples.txt",'w', encoding='UTF-8')
    for s in out:
        sample = test_set.tokenizer.decode(s.cpu().flatten())
        f.write(sample+"\n")  
    f.close()      
