import sys
sys.path.append('/home/maiso/detests_2022/library/')

from juan import BetoMTL
import juan
import utils
import pandas as pd
import transformers
import torch
import importlib
import os
import argparse

_INVALID_ARGS=1

if __name__=='__main__':
    
    if len(sys.argv)<2:
        print('[ ERROR ] Name of the model required.')
        print('Usage python genericMTL.py <model_name>')
        sys.exit(_INVALID_ARGS)
#     else:
#         model_name = sys.argv[1]
        
    parser = argparse.ArgumentParser(description = "Launch experiments")
    parser.add_argument("--model", type = str)
    parser.add_argument("--epochs", type = int, default = 2)
    parser.add_argument("--weight_decay", type = float, default = 0)
    parser.add_argument('--full', action = "store_true", default = False)
    parser.add_argument('--freeze_encoder', action = "store_true", default = False)
    
    args = parser.parse_args()
    epochs = args.epochs
    weight_decay = args.weight_decay
    full = args.full
    freeze_encoder = args.freeze_encoder
    model_name=args.model
    print('Using model_name='+model_name)


    # DATA    
    data = pd.read_csv("/home/maiso/detests_2022/data/task_2.csv")
    labels = data.iloc[:, 1:]
    print('data.shape='+str(data.shape))


    # TOKENIZER 
    print('Loading tokenizer..')
    tokenizer_path = f"/home/maiso/detests_2022/assets/{model_name}/tokenizer"
    assert os.path.exists(tokenizer_path), tokenizer_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
    print('Tokenizer loaded.')

    # MODEL
    print('loading model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device='+str(device))
    model_path = f"/home/maiso/detests_2022/assets/{model_name}/model"
    assert os.path.exists(model_path), model_path
    model = transformers.AutoModel.from_pretrained(model_path, num_labels=10)
    # model.to(device)
    print('Model loaded.')

    # X & y
    X = data.sentence
    y = labels

    importlib.reload(juan)
    importlib.reload(utils)
    clf = juan.BetoMTL(tokenizer, model)
    # clf.to(device)
    clf.cuda()
    results = utils.validate_MTL_juan(
        X, 
        y, 
        clf, 
        verbose = True, 
        epochs = epochs,
        weight_decay = weight_decay,
        freeze_encoder = freeze_encoder,
        progress_bar = False
    )
    print('epoch='+str(epochs))
    print('weight_decay='+str(weight_decay))
    print('freeze_encoder='+str(freeze_encoder))
    print(results)
    output_path = f'/home/maiso/detests_2022/results/task_2/mm_{model_name}-MTL_epochs{epochs}.csv'
    print('output_path='+output_path)
    results.to_csv(output_path)
    print('Done')
