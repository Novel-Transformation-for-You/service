# %%

from module.ner_utils import ner_inference


#%%
from utils.load_model import load_ner
from utils.input_process import make_ner_input
from utils.ner_utils import make_name_list, show_name_list, combine_similar_names
import torch

path ='model/NER.pth'
checkpoint = torch.load(path)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])

with open('test/test.txt', "r", encoding="utf-8") as f:
    file_content = f.read()

content = make_ner_input(file_content)
name_list, time, place = make_name_list(content, checkpoint)
name_dic = show_name_list(name_list)
similar_name = combine_similar_names(name_dic)

for i in similar_name:
    print(i)

# %% CSN 모델
import torch

from utils.fs_utils import get_alias2id, find_speak
from utils.ner_utils import make_name_list
from utils.input_process import make_ner_input, make_instance_list, input_data_loader

checkpoint = torch.load('./model/final.pth')
model = checkpoint['model']
model.to('cpu')
tokenizer = checkpoint['tokenizer']

check_name = './data/name.txt'
alias2id = get_alias2id(check_name)

with open('test/KoCSN_test.txt', "r", encoding="utf-8") as f:
    file_content = f.read()

instances, instance_num = make_instance_list(file_content)
inputs = input_data_loader(instances, alias2id)
output = find_speak(model, inputs, tokenizer, alias2id)



def make_script(texts, instance_num, output):
    script = []
    for idx, text in enumerate(texts):
        if idx in instance_num


# %%
