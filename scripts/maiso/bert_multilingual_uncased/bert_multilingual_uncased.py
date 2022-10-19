import sys
import warnings
import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)
sys.path.append('/home/maiso/detests_2022/library/')
from maiso.models import BERTMultilingualUncased
from maiso.task import get_texts,get_y
from utils import validate
results =   validate(
                BERTMultilingualUncased.get_X(get_texts()), 
                get_y(), 
                BERTMultilingualUncased()) 

results.to_csv("/home/maiso/detests_2022/results/task_2/data/BERT_multilingual_uncased.csv")
