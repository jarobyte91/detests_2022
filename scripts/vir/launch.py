import pandas as pd
import transformers
import argparse
import sys
sys.path.append("/home/vsabando/detests2022/library")
# from juan import *
import utils
import juan

parser = argparse.ArgumentParser(description = "Launch experiments")
parser.add_argument("--epochs", type = int, default = 2)
parser.add_argument("--weight_decay", type = float, default = 0)
parser.add_argument('--full', action = "store_true", default = False)
parser.add_argument('--freeze_encoder', action = "store_true", default = False)
# parser.add_argument('--balanced', action = "store_true", default = False)
# parser.add_argument('--random', action = "store_true")
args = parser.parse_args()
epochs = args.epochs
weight_decay = args.weight_decay
full = args.full
freeze_encoder = args.freeze_encoder
# balanced = args.balanced

print("Loading data...")
data = pd.read_csv("/home/vsabando/detests2022/data/task_2.csv")
print("data", data.shape)
labels = data.iloc[:, 1:]
print("labels", labels.shape)
# print(labels.head())

print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained("/home/vsabando/detests2022/assets/beto/tokenizer")
print("Loading model...")
model = transformers.AutoModel.from_pretrained("/home/vsabando/detests2022/assets/beto/model")
print()


name = f"beto_mtl_e_{epochs}"
print("Epochs:", epochs)

if weight_decay > 0:
    name += f"_wd_{weight_decay}"
    print("Weight Decay:", weight_decay)
    
if freeze_encoder:
    name += "_frozen"
    print("Freeze Encoder:", freeze_encoder)
    
# if balanced:
#     name += "_balanced"
#     print("Balanced:", balanced)
    
if not full:
    data = data.head(10)
    labels = labels.head(10)
    name += "_test"

print()

# weight_decay = 0.5

clf = juan.BetoMTL(tokenizer, model)
clf.cuda()

results = utils.validate_MTL_juan(
    data.sentence, 
    labels, 
    clf, 
    verbose = True, 
    epochs = epochs,
    weight_decay = weight_decay,
    freeze_encoder = freeze_encoder,
    progress_bar = False
)

path_1 = f"/home/vsabando/detests2022/results/task_1/data/{name}.csv"
print("\nWriting to:", path_1)
results.query("column == 'task_1'").to_csv(path_1)

path_2 = f"/home/vsabando/detests2022/results/task_2/data/{name}.csv"
print("\nWriting to:", path_2)
results.to_csv(path_2)