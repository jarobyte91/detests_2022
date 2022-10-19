import sys
sys.path.append('/home/maiso/detests_2022/library')

from maiso.task import get_texts, get_y
import numpy as np
from maiso.models import TransformersShell

import argparse
if __name__=='__main__': 
    parser = argparse.ArgumentParser(description = "Launch experiments")
    parser.add_argument("--model_name", type = str)
    parser.add_argument("--output_file", type = str)
    parser.add_argument("--epochs", type = int, default = 2)
    
    args = parser.parse_args()
    model_name = args.model_name
    output_file = args.output_file
    epochs = args.epochs
    
    assert isinstance(epochs,int)
    assert isinstance(output_file,str)
    assert isinstance(model_name,str)
    assert model_name in ['roberta-base-bne', 'bert-base-spanish-wwm-uncased', 'bert-base-multilingual-uncased']
    
    model = TransformersShell(model_name=model_name, epochs=epochs)
    
    X = TransformersShell.get_X(get_texts())
    y = get_y()
    
    print('Starting training')
    model.entrenar(X,y)
    model.save_pretrained(output_file)
