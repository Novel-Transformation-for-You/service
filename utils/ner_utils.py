"""
NER 모델을 이용하여 작업
"""
import re
import torch
import numpy as np
from collections import Counter

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def make_ner_input(text, chunk_size=500) -> list:
    """
    문장을 New Lines 기준으로 나누어 줍니다.
    chunk size보다 문장이 길 경우, 마지막 문장은 뒤에서 chunk size 만큼 추가합니다.
    """
    count_text = chunk_size
    max_text = len(text)
    newline_position = []

    while count_text < max_text:
        sentence = text[:count_text]
        last_newline_position = sentence.rfind('\n')
        newline_position.append(last_newline_position)
        count_text = last_newline_position + chunk_size

    split_sentences = []
    start_num = 0

    for _, num in enumerate(newline_position):
        split_sentences.append(text[start_num:num])
        start_num = num

    if max_text % chunk_size != 0:
        f_sentence = text[max_text-500:]
        first_newline_position = max_text-500 + f_sentence.find('\n')
        split_sentences.append(text[first_newline_position:])

    return split_sentences


def ner_tokenizer(sent, max_seq_length, checkpoint):
    """
    NER 토크나이저
    """
    tokenizer = checkpoint['tokenizer']

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    pre_syllable = "_"
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length
    sent = sent[:max_seq_length-2]

    for i, syllable in enumerate(sent):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable
        pre_syllable = syllable

        input_ids[i] = tokenizer.convert_tokens_to_ids(syllable)
        attention_mask[i] = 1

    input_ids = [cls_token_id] + input_ids[:-1] + [sep_token_id]
    attention_mask = [1] + attention_mask[:-1] + [1]

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids}


def get_ner_predictions(text, checkpoint):
    """
    tokenized_sent, pred_tags 만들기
    """

    model = checkpoint['model']
    tag2id = checkpoint['tag2id']
    model.to(device)
    text = text.replace(' ', '_')

    predictions, true_labels = [], []

    tokenized_sent = ner_tokenizer(text, len(text) + 2, checkpoint)
    input_ids = torch.tensor(
        tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(
        tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(
        tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)

    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.append(label_ids)

    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    return tokenized_sent, pred_tags


def ner_inference(tokenized_sent, pred_tags, checkpoint, name_len=5) -> list:
    """
    Name에 한해서 inference
    """
    name_list = []
    speaker = ''
    tokenizer = checkpoint['tokenizer']
    scene = {'장소': [], '시간': []}
    target = ''
    c_tag = None

    for i, tag in enumerate(pred_tags):
        token = tokenizer.convert_ids_to_tokens(tokenized_sent['input_ids'][i]).replace('#', '')
        if 'PER' in tag:
            if 'B' in tag and speaker != '':
                name_list.append(speaker)
                speaker = ''
            speaker += token

        elif speaker != '' and tag != pred_tags[i-1]:
            if speaker in name_list:
                name_list.append(speaker)
            else:
                tmp = speaker
                found_name = False
                # print(f'{speaker}에 의문이 생겨 확인해봅니다.')
                for j in range(name_len):
                    if i + j < len(tokenized_sent['input_ids']):
                        token = tokenizer.convert_ids_to_tokens(
                            tokenized_sent['input_ids'][i+j]).replace('#', '')
                        tmp += token
                        # print(f'{speaker} 뒤로 나온 {j} 번째 까지 확인한결과, {tmp} 입니다')
                        if tmp in name_list:
                            name_list.append(tmp)
                            found_name = True
                            # print(f'명단에 {tmp} 가 존재하여, {speaker} 대신 추가하였습니다.')
                            break

                if not found_name:
                    name_list.append(speaker)
                    # print(f'찾지 못하여 {speaker} 를 추가하였습니다.')
                speaker = ''

        elif tag != 'O':
            if tag.startswith('B'):
                if c_tag in ['TIM', 'DAT']:
                    scene['시간'].append(target)
                elif c_tag =='LOC':
                    scene['장소'].append(target)
                c_tag = tag[2:]
                target = token
            else:
                target += token.replace('_', ' ')

    return name_list, scene


def make_name_list(ner_inputs, checkpoint):
    """
    문장들을 NER 돌려서 Name List 만들기.
    """
    name_list = []
    times = []
    places = []

    for ner_input in ner_inputs:
        tokenized_sent, pred_tags = get_ner_predictions(ner_input, checkpoint)
        names, scene = ner_inference(tokenized_sent, pred_tags, checkpoint)
        name_list.extend(names)
        times.extend(scene['시간'])
        places.extend(scene['장소'])

    return name_list, times, places


def show_name_list(name_list):
    """
    사용자 친화적으로 보여주기용
    """
    name = dict(Counter(name_list))

    return name


def compare_strings(str1, str2):
    """
    한국 배경일 경우 길이가 다른 경우와 같은 경우를 비교하여 데이터를 처리
    """
    if len(str1) != len(str2):
        # 더 짧은 문자열이 더 긴 문자열에 포함되는지 확인
        shorter, longer = (str1, str2) if len(str1) < len(str2) else (str2, str1)
        if shorter in longer:
            return True
    else:
        same_part = []
        for i in range(len(str1)):
            if str1[i] in str2:
                same_part += str1[i]
                continue
            else:
                break
        if len(same_part) >= 2:
            return True

    return False

def combine_similar_names(names_dict):
    """
    compare_strings 을 바탕으로 Name List 관련 데이터를 처리

    2글자는 이름일 확률이 높으니 일단 넣고 시작
    """
    names = names_dict.keys()
    similar_groups = [[name] for name in names if len(name) == 2]
    idx = 0
    # print(similar_groups, '\n',idx)

    for name in names:
        found = False
        for group in similar_groups:
            idx += 1
            for item in group:
                if compare_strings(name, item) and len(name)>1:
                    found = True
                    cleaned_text = re.sub(r'(아|이)$', '', item)
                    if len(name) == len(item):
                        same_part = ''
                        # 완전히 일치하는 부분이 있는지 확인
                        for i in range(len(name)):
                            if name[i] in item:
                                same_part += name[i]
                        if same_part not in group and cleaned_text not in group:
                            group.append(cleaned_text)
                            # print(similar_groups, '\n',idx, '문자열의 길이가 같을 때')
                    else:    
                        group.append(name)
                        # print(similar_groups, '\n',idx, '문자열의 길이가 다를 때')
                        break
            if found:
                break
        if not found:
            similar_groups.append([name])

    updated_names = {tuple(name for name in group if len(name) > 1): counts for group, counts in (
        (group, sum(names_dict[name] for name in group if name != '')) for group in similar_groups)
        if len([name for name in group if len(name) > 1]) > 0}

    return updated_names

def convert_name2codename(codename2name, text):
    """RE를 이용하여 이름을 코드네임으로 변경합니다."""
    # 우선 각 name을 길이 내림차순으로 정렬하고,
    import re
    for n_list in codename2name.values():
        n_list.sort(key=lambda x:(len(x), x), reverse=True)

    for codename, n_list in codename2name.items():
        for subname in n_list:
            text = re.sub(subname, codename, text)

    return text


def convert_codename2name(codename2name, text):
    """코드네임을 이름으로 변경해줍니다."""
    outputs = []
    for i in text:
        try:
            outputs.append(codename2name[i][0])
        except:
            outputs.append('알 수 없음')

    return outputs