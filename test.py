# %%
from module.load_model import load_fs, load_ner
from module.input_process import make_ner_input
from module.ner_utils import make_name_list, show_name_list, combine_similar_names

from collections import Counter

ner_model, checkpoint = load_ner('model/NER.pth')

with open('test/test.txt', "r", encoding="utf-8") as f:
    file_content = f.read()

content = make_ner_input(file_content)
name_list = make_name_list(content, checkpoint)

# counter = Counter
# nn = dict(counter(name_list))

# cc = combine_similar_names(dict(nn))

# %%


# %%
