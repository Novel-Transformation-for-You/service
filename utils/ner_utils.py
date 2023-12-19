"""
NER 모델을 이용하여 작업하는 코드입니다.
"""
import re
import torch
import numpy as np
from collections import Counter

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def ner_tokenizer(text, max_seq_length, checkpoint):
    """
    NER을 위해 텍스트를 토큰화합니다. 
    Args:
        sent: 처리하고자 하는 텍스트를 입력받습니다.
        max_seq_length: BERT의 config에서 처리 가능한 최대 문자열 길이는 512입니다. 최대 길이를 넘어서지 않도록, 텍스트 길이가 512를 넘어갈 경우 여러 개의 문자열로 분리합니다. 
                        문맥 정보를 고려하므로 가능한 긴 길이로 chunking하는 것이 좋은 성능을 보장할 수 있습니다.
        checkpoint: NER 모델에 대한 정보를 불러들입니다.
    Return:
        ner_tokenizer_dict: 아래 세 요소를 포함한 딕셔너리입니다.
            input_ids: 각 토큰의 모델 딕셔너리에서의 아이디값입니다.
            attention_mask: 각 토큰의 어탠션 마스크 활성화 여부입니다.
            token_type_ids: 개체명 인식 된 토큰의 경우 그 타입의 아이디(숫자 조합)를 반환합니다.
    """
    #저장된 모델의 토크나이저를 불러옵니다.
    tokenizer = checkpoint['tokenizer']

    #각각 패딩, 문장 시작, 문장 끝을 나타내는 특별한 토큰들의 ID 값들을 가져옵니다.
    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    #이전 음절을 저장하는 변수를 초기화합니다.
    pre_syllable = "_" 

    #토크나이징된 결과를 저장할 리스트들을 초기화합니다.
    input_ids = [pad_token_id] * (max_seq_length - 1)
    attention_mask = [0] * (max_seq_length - 1)
    token_type_ids = [0] * max_seq_length

    #입력된 텍스트를 최대 시퀀스 길이에 맞게 잘라냅니다.
    text = text[:max_seq_length-2]

    #텍스트의 각 음절에 대해 반복문을 실행합니다.
    for i, syllable in enumerate(text):
        if syllable == '_':
            pre_syllable = syllable
        if pre_syllable != "_":
            syllable = '##' + syllable
        pre_syllable = syllable

        #토큰을 모델의 단어 사전에 있는 ID 값으로 변환하여 input_ids 리스트에 저장합니다.
        input_ids[i] = tokenizer.convert_tokens_to_ids(syllable)
        #해당 위치의 토큰에 대한 어텐션 마스크를 활성화합니다.
        attention_mask[i] = 1

    #입력 시퀀스의 시작에는 cls_token_id를, 끝에는 sep_token_id를 추가합니다.
    input_ids = [cls_token_id] + input_ids[:-1] + [sep_token_id]
    #어텐션 마스크도 시작과 끝 토큰을 고려하여 수정합니다.
    attention_mask = [1] + attention_mask[:-1] + [1]

    ner_tokenizer_dict = {"input_ids": input_ids,
                          "attention_mask": attention_mask,
                          "token_type_ids": token_type_ids}
    
    return ner_tokenizer_dict

def get_ner_predictions(text, checkpoint):
    """
    토큰화한 문장(tokenized_sent)과 예측한 태그(pred_tags) 값을 만드는 함수입니다.
    Args:
        text: NER 예측을 필요로 하는 텍스트를 입력합니다. 
        checkpoint: 저장한 모델을 불러들입니다.
    Returns:
        tokenized_sent: 모델 입력을 위한 토큰화된 문장 정보입니다.
        pred_tags: 각 토큰에 대한 예측된 태그들을 포함합니다.
    """
    #저장한 모델을 불러들입니다.
    model = checkpoint['model']
    #태그와 해당 태그의 ID 매핑 정보를 가져옵니다.
    tag2id = checkpoint['tag2id']
    model.to(device)
    #입력된 텍스트에서 공백을 언더스코어(_)로 대체합니다.
    text = text.replace(' ', '_')

    #예측값과 실제 라벨을 저장할 빈 리스트를 생성합니다.
    predictions, true_labels = [], []

    #ner_tokenizer 함수를 사용하여 텍스트를 토큰화합니다.
    tokenized_sent = ner_tokenizer(text, len(text) + 2, checkpoint)

    #토큰화된 결과를 토대로 텐서로 변환하여 모델 입력 형식에 맞게 준비합니다.
    input_ids = torch.tensor(
        tokenized_sent['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(
        tokenized_sent['attention_mask']).unsqueeze(0).to(device)
    token_type_ids = torch.tensor(
        tokenized_sent['token_type_ids']).unsqueeze(0).to(device)

    #그래디언트 계산을 수행하지 않기 위해 torch.no_grad() 컨텍스트 내에서 다음을 실행합니다. (eval 영역이기 때문에 학습을 하지 않습니다)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        
    #모델 출력에서 로짓 값을 가져와 Numpy값으로 변환하고, 라벨 ID들을 CPU 상의 NumPy 배열로 가져옵니다.
    logits = outputs['logits']
    logits = logits.detach().cpu().numpy()
    label_ids = token_type_ids.cpu().numpy()

    #예측된 라벨 값을 가져와서 리스트에 추가합니다.
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    #실제 라벨을 리스트에 추가합니다.
    true_labels.append(label_ids)

    #예측된 라벨 ID를 실제 태그로 변환합니다.
    pred_tags = [list(tag2id.keys())[p_i] for p in predictions for p_i in p]

    return tokenized_sent, pred_tags


def ner_inference(tokenized_sent, pred_tags, checkpoint, name_len=5) -> list:
    """
    NER을 실행하고, 이름과 시간 및 공간 정보를 추출합니다.
    Args:
        tokenized_sent: 토큰화된 문장이 저장된 리스트
        pred_tags: 각 토큰에 대한 예측 태그값 (NER 결과)
        checkpoint: 저장해둔 모델을 불러옴
        name_len: 더 정확한 이름 인식을 위해 앞뒤로 몇 개의 음절을 더 검토할지 지정합니다.
    Returns:
        namelist: 추출한 이름(별칭 포함) 리스트입니다. 후처리를 통해 
        place: 추출한 장소 리스트입니다.
        time: 추출한 시간 리스트입니다.
    """
    name_list = []
    speaker = ''
    tokenizer = checkpoint['tokenizer']
    place = []
    time = []
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

                # 추출한 이름이 일반적인 이름 형식과 맞지 않을 경우 앞뒤 토큰을 재검토합니다.
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
                    time.extend(target)
                elif c_tag =='LOC':
                    place.extend(target)
                c_tag = tag[2:]
                target = token
            else:
                target += token.replace('_', ' ')

    return name_list, place, time


def show_name_list(name_list):
    """
    사용자 친화적으로 네임리스트를 보여줍니다.
    Arg:
        name_list: 추출한 이름 리스트
    Return:
        name: 동일한 이름이 몇 번 등장했는지 횟수를 함께 제공합니다.
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
