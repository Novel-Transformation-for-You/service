"""
화자 찾는 모델 유틸 파일들
"""
class InputFeatures:
    """
    BERT 모델의 입력들
    """
    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, tokenizer):
    """
    텍스트 segment를 단어 ID로 변환합니다.
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
    주어진 이름 목록 파일에서 별칭(alias)을 ID로 매핑하는 사전을 생성.
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
    주어진 모델과 입력 데이터를 사용하여 각 입력에 대한 화자를 찾는 함수
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


def making_script(text, speaker:list, instance_num:list) -> str:
    """
    주어진 텍스트와 화자 목록, 해당하는 줄 번호를 사용하여 대화 스크립트를 생성하는 함수
    """
    lines = text.splitlines()
    for num, people in zip(instance_num, speaker):
        lines[num] = f'{people}: {lines[num]}'
    return lines
