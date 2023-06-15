# 获取主目录
import os,sys
get_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(get_path)

# 加载模型数据集
from diffusion_bert import diffusion_bert, ROCstory, e2e

# 加载torch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 加载 辅助库
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt


# 输出函数
# 把一个串写进log文件里面
def printLog(string, fileName):
    f = open(fileName, 'a', encoding='UTF-8')
    f.write(string+'\n')
    f.close()
    #print(string)
    return 0


# 模型评估
def evaluate(model, dataloader, device):
    model.eval()
    datagenerator = iter(dataloader)
    loss_arr = []
    for _ in range(len(dataloader)):
        input_ids,token_type_ids,attention_mask = next(datagenerator)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            loss,pred,t = model(input_ids,token_type_ids,attention_mask)
        loss_arr.append(loss.item())
    return np.array(loss_arr).mean() 


# 主训练函数
# 主要的参数都写在yaml文件里面
def main(
    initializing, # init model, bert-base-uncased
    amp, # mix fp16 fp32
    batch_size,
    epoch_num,
    num_gpus,
    lr,
    resume, # resume train
    datadir, #  where datasets in
    SavedDir, # where to save model
    log, # where to save log file
    CheckpointDir, # where to save ckpt
    max_len, # sen's max len 64
    diff_step, # default T = 2000
):
    # global para
    loss_rec = 10.0 # recall loss
    steps = 0
    best_loss = 10.0 
    
    # datasets
    if "ROC" in datadir[0]:  
        print("use roc dataset")
        train_set = ROCstory(datadir[0], init_model=initializing, max_len=max_len)
        test_set = ROCstory(datadir[1], init_model=initializing, max_len=max_len)
    elif "e2e" in datadir[0]:
        train_set = e2e(datadir[0], init_model=initializing, max_len=max_len)
        test_set = e2e(datadir[1], init_model=initializing, max_len=max_len)     
    else:
        raise NotImplementedError()   
    
    # # sampler
    # train_sampler = BatchSampler(train_set, batch_size=batch_size, drop_last=True)
    # test_sampler = BatchSampler(test_set, batch_size=batch_size, drop_last=False)
    
    # dataloader
    train_dataloader = DataLoader(train_set,batch_size=batch_size)
    test_dataloader = DataLoader(test_set,batch_size=batch_size)
    
    
    # batchs_len
    # 有多少个batch = number sample / bzs
    train_batchs = len(train_dataloader) # 79 = round(10000/128)
    test_batchs = len(test_dataloader)
    
    
    print(train_batchs)
    
    # cuda:0
    rank = 0
    device = torch.device('cuda:{:d}'.format(rank))
    torch.cuda.set_device(device)  
    
    # diffusion_model
    print("create diffusion model")
    model = diffusion_bert(initializing, max_len, diff_step) 
    model.to(device)
    
    # lr 
    print("create optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # amp default = "none" 
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    # lr schedule , similar to warmup and anneal
    # 注意这里的max_lr和你优化器中的lr并不是同一个
    # 注意,无论你optim中的lr设置是啥,最后起作用的还是max_lr
    # 学习率有点小 max_lr = base_lr = yaml.2e-05
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        lr, 
        # total_steps = epochs * steps_per_epoch
        epochs = epoch_num[1], # max_epoch
        steps_per_epoch = train_batchs, # how many batch
        verbose = True,  # my add
        pct_start = 5 / epoch_num[1], # 学习率上升部分占比
        div_factor = 1e4, # 初始学习率 = max_lr / div_factor
        cycle_momentum = False)   
    
    # print(scheduler.state_dict()) # debug
    
    # if need resume
    if not(resume == 'none'):
        ckpt = torch.load(resume)
        epoch_num[0] = ckpt['epoch'] + 1 # from breakpoint start
        loss_rec = ckpt['loss_rec']
        # write resume to log file
        printLog(f'resuming from epoch {epoch_num[0]:8d} of '+
                     resume,log)
        
        scheduler.load_state_dict(ckpt['scheduler'])
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        
        steps = ckpt['steps']
        best_loss = ckpt['metric']
        printLog(f'recovering best_loss {best_loss:4f}',log)
        
    loss_dict = []
     
    print("---------train start---------")
    time_start=time.time()
    for epoch in tqdm(range(*epoch_num)):
        model.train()
        # we don't need dist_train
        # train_sampler.set_epoch(epoch)
        train_generator = iter(train_dataloader)
        for _ in range(len(train_dataloader)):
            input_ids,token_type_ids,attention_mask = next(train_generator)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
        
        # if need mix fp  
        with torch.cuda.amp.autocast(enabled=amp):
            loss,pred,t = model(input_ids,token_type_ids,attention_mask) 
        
        # update recall loss
        loss_rec = loss_rec * 0.99 + loss.item() * 0.01  
        loss_dict.append(loss_rec)  
        
        # lr
        optimizer.zero_grad(set_to_none=False)
        
        # scaler is fp mix obj.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        steps += 1
        scheduler.step()
        
        # each 100 step to loh
        if steps % 100 == 0:
            time_end=time.time()
            printLog(f'steps: {steps:8d} loss: {loss_rec:.4f} time_cost: {time_end-time_start:.2f}',log) 
            time_start=time.time()
            
    print("---------train over---------")
    
    
    print("---------draw loss curve---------")
    plt.figure()
    plt.plot(range(epoch_num[1]), loss_dict, label="loss curve")
    plt.savefig(CheckpointDir + "loss_curve.png")
    
    # why del        
    del input_ids,token_type_ids,attention_mask,loss,pred  
    
    # test_sampler.set_epoch(epoch)
    
    # evaluate loss
    print("---------eval start---------")
    vloss = evaluate(model, test_dataloader, device)
    vloss = torch.tensor(vloss, device=device)
    vloss = vloss.cpu().numpy()
    
    # write evaluate loss to log 
    printLog(f'epoch: {epoch:4d}    loss: {vloss:.5f}    time:{time.asctime(time.localtime(time.time()))}',log) 

    # update evaluate loss
    if vloss < best_loss:
        best_loss = vloss
        torch.save(
            model.state_dict(),
            CheckpointDir + "bestloss.pkl")
        
    # save model
    torch.save({'epoch': epoch,
                'steps': steps,
                'loss_rec': loss_rec,
                'metric': best_loss,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, 
               CheckpointDir + "lastepoch.pkl")
    
    print("---------eavl over---------")


if __name__ == '__main__':
    
    import os,sys
    import yaml
    import shutil
    
    # get path
    current_path = os.path.dirname(os.path.abspath(__file__))  # D:\workspace\202306\diffusion-lm
    sys.path.append(current_path)
    
    # load train para
    # ExpName = "20230608"
    # ExpName = "20230609"
    # ExpName = "20230610"
    # ExpName = "20230610v2"
    # ExpName = "20230611"
    ExpName = "20230612"



    
    with open(current_path +'/' + ExpName + '.yaml', encoding='UTF-8') as f: # D:\workspace\202306\diffusion-lm\20230608.yaml
        training_parameters = yaml.full_load(f) 
    
    # load datasets which like [train_filename, test_filename]
    TrainDir = [ 
        current_path + training_parameters['dataStorage'][0],  # D:\workspace\202306\diffusion-lm\ROCstory_train.csv
        current_path + training_parameters['dataStorage'][1] # D:\workspace\202306\diffusion-lm\ROCstory_test.csv
        ]
    
    # assign train para
    # initializing = current_path + training_parameters["initializing"] # "bert-base-uncased" perhaps save in current dir rather than cachi dir
    initializing = training_parameters["initializing"]
    amp = training_parameters['AMP'] # 'none'
    modelName = training_parameters["framework"] # 'bert_diffusion'
    num_gpus = training_parameters["num_gpus"] # single gpu = 1
    resume = training_parameters["resume"] # 'none'
    epoch_num = training_parameters["epoch"] # list, like [0, max_epoch]
    max_len = training_parameters["max_len"] # 64
    diff_step = training_parameters["diff_step"] # 32
    
    # if need mix fp
    if amp:
        batch_size=training_parameters['batch_size'] * 2 # fp16 to fp32
    else:
        batch_size=training_parameters['batch_size'] # run follow this
    
    # lr
    # lr = training_parameters["base_lr"] * batch_size * num_gpus / 512 # why divide 512? 
    lr = training_parameters["base_lr"] 
    
    # where to save model 
    SavedDir= current_path + "/Saved_Models/" # # D:\workspace\202306\diffusion-lm\Saved_Models
    try:
        os.mkdir(SavedDir)
    except:
        pass
    
    # create ckpt dir
    CheckpointDir = SavedDir + ExpName + modelName + "/"
    try:
        os.mkdir(SavedDir + ExpName + modelName) # yaml + modelframe D:\workspace\202306\diffusion-lm\Saved_Models\20230608bert_diffusion
    except:
        print('Warning!Current folder already exist!')      
        
    # copy train para to ckpt dir   
    shutil.copy(current_path + '/' + ExpName + '.yaml', SavedDir + ExpName + modelName)
    
    # train log file
    log = SavedDir + ExpName + modelName + "/train.log" # D:\workspace\202306\diffusion-lm\Saved_Models\20230608bert_diffusion\train_log
    
    # run main must receive para in order
    main(initializing,amp,batch_size,epoch_num,num_gpus,lr,resume,TrainDir,SavedDir,log,CheckpointDir,max_len,diff_step)
    