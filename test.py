# %%
from module.load_model import load_ner
from module.ner_utils import make_name_list, show_name_list, combine_similar_names, ner_inference
from module.input_process import make_ner_input, make_instance_list, input_data_loader
from module.fs_utils import get_alias2id, find_speak

from collections import Counter

ner_model, checkpoint = load_ner('model/NER.pth')
# fs_model, fs_checkpoint = load_fs('model/FS.pth')

with open('test/test.txt', "r", encoding="utf-8") as f:
    file_content = f.read()


#%%
from module.load_model import load_ner
from module.input_process import make_ner_input
from module.ner_utils import make_name_list, show_name_list, combine_similar_names

ner_model, checkpoint = load_ner('model/NER.pth')

with open('test/test.txt', "r", encoding="utf-8") as f:
    file_content = f.read()

content = make_ner_input(file_content)
name_list, time, place = make_name_list(content, checkpoint)
name_dic = show_name_list(name_list)
similar_name = combine_similar_names(name_dic)



for i in similar_name:
    print(i)









# %%
check_name = 'data/names.txt'
alias2id = get_alias2id(check_name)


# %%
instances = make_instance_list(file_content)
inputs = input_data_loader(instances, alias2id)
output = find_speak(fs_model, inputs, fs_checkpoint, alias2id)