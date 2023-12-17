"""
화자 찾는 모델 유틸 파일들
"""
class InputFeatures:
    """
    Represents the inputs of the BERT model.
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, tokenizer):
    """
    Convert textual segments into word IDs.
    """
    features = []
    tokens_list = []

    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example)
        tokens_list.append(tokens)

        new_tokens = []
        input_type_ids = []

        new_tokens.append("[CLS]")
        input_type_ids.append(0)
        new_tokens = new_tokens + tokens
        input_type_ids = input_type_ids + [0] * len(tokens)
        new_tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        input_mask = [1] * len(input_ids)

        features.append(
            InputFeatures(
                tokens=new_tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    return features, tokens_list


def get_alias2id(name_list_path) -> dict:
    """
    name list 만들기
    """
    with open(name_list_path, 'r', encoding='utf-8') as fin:
        name_lines = fin.readlines()
    alias2id = {}

    for i, line in enumerate(name_lines):
        for alias in line.strip().split()[1:]:
            alias2id[alias] = i

    return alias2id


def find_speak(fs_model, input_data, tokenizer, alias2id):
    """
    화자 찾는거
    """
    model = fs_model
    check_data_iter = iter(input_data)

    names = []

    for _ in range(len(input_data)):

        seg_sents, css, scl, mp, qi, cut_css, name_list_index = next(check_data_iter)
        features, tokens_list = convert_examples_to_features(examples=css, tokenizer=tokenizer)

        try:
            predictions = model(features, scl, mp, qi, 0, "cuda:0", tokens_list, cut_css)
        except RuntimeError:
            predictions = model(features, scl, mp, qi, 0, "cpu", tokens_list, cut_css)

        scores, _, _ = predictions

        # 후처리
        try:
            scores_np = scores.detach().cpu().numpy()
            scores_list = scores_np.tolist()
            score_index = scores_list.index(max(scores_list))
            name_index = name_list_index[score_index]

            for key, val in alias2id.items():
                if val == name_index:
                    result_key = key

            names.append(result_key)
        except AttributeError:
            names.append('알 수 없음')

    return names



# """
# 사용자로부터 입력받은 text를 전처리 하는 모듈
# """
# import torch
# from transformers import AutoTokenizer

# # from ..model.ner.ner_utils import show_tokens
# from .input_process import make_instance_list, input_data_loader, making_script



# def user_input(text):
#     """
    
#     """
#     ins_list, ins_num = make_instance_list(text)
#     user_input = input_data_loader(ins_list, alias2id=alias2id)
#     user_input_iter = iter(user_input)

#     who = []
#     for i, _ in enumerate(user_input):
#         model_s.load_state_dict(model_state_dict)
#         seg_sents, css, sent_char_lens, mention_poses, quote_idxes, cut_css, name_list_index = next(
#             user_input_iter)
#         features, tokens_list = convert_examples_to_features(examples=css, tokenizer=tokenizer_s)

#         try:
#             predictions = model_s(features, sent_char_lens, mention_poses, quote_idxes, 0, "cpu",
#                                 tokens_list, cut_css)

#             # scores, scores_false, scores_true = predictions
#             scores, _, _ = predictions

#             # 후처리
#             scores_np = scores.detach().cpu().numpy()
#             scores_list = scores_np.tolist()
#             score_index = scores_list.index(max(scores_list))
#             name_index = name_list_index[score_index]

#             for key, val in alias2id.items():
#                 if val == name_index:
#                     result_key = key

#             # print(result_key, ins_list[i][10])
#             who.append(result_key)

#         except RuntimeError:
#             # print('UNK', ins_list[i][10])
#             who.append('UNK')

#     output = making_script(text, who, ins_num)

#     return output
