"""
화자 찾는 모델 유틸 파일들
"""

from .bert_features import convert_examples_to_features

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


def find_speak(fs_model, input_data, fs_checkpoint, alias2id):
    """
    화자 찾는거
    """
    model = fs_model
    tokenizer = fs_checkpoint['tokenizer']
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
        scores_np = scores.detach().cpu().numpy()
        scores_list = scores_np.tolist()
        score_index = scores_list.index(max(scores_list))
        name_index = name_list_index[score_index]

        for key, val in alias2id.items():
            if val == name_index:
                result_key = key

        names.append([name_index, result_key])
        
    return names
