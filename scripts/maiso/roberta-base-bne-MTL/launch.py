import pandas as pd
import transformers
import argparse
import sys
sys.path.append("/home/maiso/detests_2022/library")
# from juan import *
import utils
import juan
from maiso.models import ROBERTA

parser = argparse.ArgumentParser(description = "Launch experiments")
parser.add_argument("--model", type = str)
parser.add_argument("--epochs", type = int, default = 2)
parser.add_argument("--weight_decay", type = float, default = 0)
parser.add_argument('--full', action = "store_true", default = False)
parser.add_argument('--freeze_encoder', action = "store_true", default = False)
# parser.add_argument('--balanced', action = "store_true", default = False)
# parser.add_argument('--random', action = "store_true")
args = parser.parse_args()
model_name = args.model
epochs = args.epochs
weight_decay = args.weight_decay
full = args.full
freeze_encoder = args.freeze_encoder
# balanced = args.balanced

print("Loading data...")
data = pd.read_csv("/home/maiso/detests_2022/data/task_2.csv")
print("data", data.shape)
labels = data.iloc[:, 1:]
print("labels", labels.shape)
# print(labels.head())

print("Loading tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(f"/home/maiso/detests_2022/assets/{model_name}/tokenizer")
print("Loading model...")
model = transformers.AutoModel.from_pretrained(f"/home/maiso/detests_2022/assets/{model_name}/model")
print()


name = f"{model_name}_mtl_e_{epochs}"
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

clf = ROBERTA(tokenizer, model)
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

path_1 = f"/home/maiso/detests_2022/results/task_1/data/mm_{name}.csv"
print("\nWriting to:", path_1)
results.query("column == 'task_1'").to_csv(path_1)

path_2 = f"/home/maiso/detests_2022/results/task_2/data/mm_{name}.csv"
print("\nWriting to:", path_2)
results.to_csv(path_2)
