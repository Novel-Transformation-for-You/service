#%%
from utils.load_model import load_ner
from utils.input_process import make_ner_input
from utils.ner_utils import make_name_list, show_name_list, combine_similar_names
import torch

from utils.train_model import KCSN
from utils.arguments import get_train_args


args = get_train_args()
path ='model/model.ckpt'
model = KCSN(args)
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model'])

# model = checkpoint['model']


# %%
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


#%%

n = int(input())
num = list(map(int, input().split()))
ans = []

for i, j in enumerate(num):
    print(i, j)
    if len(ans) == 0:
        ans.append(i+1)
    else:
        ans.insert(len(ans)-j, i+1)

print(ans)
# %%
