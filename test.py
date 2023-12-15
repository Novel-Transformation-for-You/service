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

content = make_ner_input(file_content)
name_list = make_name_list(content, checkpoint)

# %%
name_list = ['승호','승호','승호','하원','하원', '정원']
name = Counter(name_list)
# counter = Counter

# nn = dict(counter(name_list))

# cc = combine_similar_names(dict(nn))



# %%
check_name = 'data/names.txt'
alias2id = get_alias2id(check_name)


# %%
instances = make_instance_list(file_content)
inputs = input_data_loader(instances, alias2id)
output = find_speak(fs_model, inputs, fs_checkpoint, alias2id)