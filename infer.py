from diffusion_bert import *

import os,sys


if __name__ == "__main__":
    
    get_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(get_path)
    
    initializing = 'bert-base-uncased'
    max_len = 64
    diff_step = 2000
    device = torch.device('cuda')

    print("\n------init model------")
    model = diffusion_bert(initializing, max_len, diff_step)
    
    yaml_name = "20230612"
    
    # ckpt ="Saved_Models/20230608bert_diffusion/bestloss.pkl"
    # ckpt ="Saved_Models/20230609bert_diffusion/bestloss.pkl"
    # ckpt ="Saved_Models/20230610bert_diffusion/bestloss.pkl"
    # ckpt ="Saved_Models/20230610v2bert_diffusion/bestloss.pkl"
    ckpt ="Saved_Models/"+ yaml_name + "bert_diffusion/bestloss.pkl" # 1w train 1k test in epoch 2k

    
    # load state dict
    print("\n------load ckpt------")
    state = torch.load(get_path + '/'+ ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    test_set = ROCstory(get_path + "/ROCstory_test.csv", init_model=initializing, max_len=max_len)
    
    print("\n------sampling------")
    out = model.sampler(device, k=10, N=128)
    
    print("\n------write to file------")
    with open(get_path + "/samples_" + yaml_name + "_bestepoch_test" +".txt",'w') as f:
        for s in out:
            sample = test_set.tokenizer.decode(s.cpu().flatten()) # 这里对 模型最后的 prediction_scores 做预测
            f.write(sample+"\n")  
        f.close()  