"""
Author: 
"""
import copy
from typing import Any
from ckonlpy.tag import Twitter
from tqdm import tqdm
import re

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 사용자가 사전에 단어 추가가 가능한 형태소 분석기를 이용(추후에 name_list에 등재된 이름을 등록하여 인식 및 분리하기 위함)
twitter = Twitter()


def load_data(filename) -> Any:
    """
    지정된 파일에서 데이터를 로드합니다.
    """
    return torch.load(filename)


def NML(seg_sents, mention_positions, ws):
    """
    Nearest Mention Location (특정 후보 발화자가 언급된 위치중, 인용문으로부터 가장 가까운 언급 위치를 찾는 함수)
    
    Parameters:
        - seg_sents: 문장을 분할한 리스트
        - mention_positions: 특정 후보 발화자가 언급된 위치를 모두 담은 리스트 [(sentence_index, word_index), ...]
        - ws: 인용문 앞/뒤로 고려할 문장의 수

    Returns:
        - 가장 가까운 언급 위치의 (sentence_index, word_index)
    """
    def word_dist(pos):
        """
        발화 후보자 이름이 언급된 위치와 인용문 사이의 거리를 단어 수준(word level)에서 반환합니다.

        Parameters:
            - pos: 발화 후보자가 언급된 위치 (sentence_index, word_index)
            
        Returns:
            - 발화 후보자와 언급된 위치 사이의 거리 (단어 수준)
        """
        if pos[0] == ws:
            w_d = ws * 2
        elif pos[0] < ws:
            w_d = sum(len(
                sent) for sent in seg_sents[pos[0] + 1:ws]) + len(seg_sents[pos[0]][pos[1] + 1:])
        else:
            w_d = sum(
                len(sent) for sent in seg_sents[ws + 1:pos[0]]) + len(seg_sents[pos[0]][:pos[1]])
        return w_d

    # 언급된 위치들과 인용문 사이의 거리를 가까운 순으로 정렬
    sorted_positions = sorted(mention_positions, key=lambda x: word_dist(x))

    # 가장 가까운 언급 위치(Nearest Mention Location) 반환
    return sorted_positions[0]


def max_len_cut(seg_sents, mention_pos, max_len):
    """
    주어진 문장을 모델에 입력 가능한 최대 길이(max_len)로 자르는 함수

    Parameters:
        - seg_sents: 문장을 분할한 리스트
        - mention_pos: 발화 후보자가 언급된 위치 (sentence_index, word_index)
        - max_len: 입력 가능한 최대 길이

    Returns:
        - seg_sents : 자르고 남은 문장 리스트
        - mention_pos : 조정된 언급된 위치
    """
    
    # 각 문장의 길이를 문자 단위로 계산한 리스트 생성
    sent_char_lens = [sum(len(word) for word in sent) for sent in seg_sents]

    # 전체 문자의 길이 합
    sum_char_len = sum(sent_char_lens)

    # 각 문장에서, cut을 실행할 문자의 위치(맨 마지막 문자)
    running_cut_idx = [len(sent) - 1 for sent in seg_sents]

    while sum_char_len > max_len:
        max_len_sent_idx = max(list(enumerate(sent_char_lens)), key=lambda x: x[1])[0]

        if max_len_sent_idx == mention_pos[0] and running_cut_idx[max_len_sent_idx] == mention_pos[1]:
            running_cut_idx[max_len_sent_idx] -= 1

        if max_len_sent_idx == mention_pos[0] and running_cut_idx[max_len_sent_idx] < mention_pos[1]:
            mention_pos[1] -= 1

        reduced_char_len = len(
            seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]])
        sent_char_lens[max_len_sent_idx] -= reduced_char_len
        sum_char_len -= reduced_char_len

        # 자를 위치 삭제
        del seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]]

        # 자를 위치 업데이트
        running_cut_idx[max_len_sent_idx] -= 1

    return seg_sents, mention_pos


def seg_and_mention_location(raw_sents_in_list, alias2id):
    """
    주어진 문장을 분할하고 발화자 이름이 언급된 위치를 찾는 함수

    Parameters:
        - raw_sents_in_list: 분할할 원본 문장 리스트
        - alias2id: 캐릭터 별 이름(및 별칭)과 ID를 매핑한 딕셔너리

    Returns:
        - seg_sents: 문장을 단어로 분할한 리스트
        - character_mention_poses: 캐릭터별로, 이름이 언급된 위치를 모두 저장한 딕셔너리 {character1_id: [[sent_idx, word_idx], ...]}
        - name_list_index: 언급된 캐릭터 이름 리스트
    """
    
    character_mention_poses = {}
    seg_sents = []
    id_pattern = ['&C{:02d}&'.format(i) for i in range(51)]

    for sent_idx, sent in enumerate(raw_sents_in_list):
        raw_sent_with_split = sent.split()

        for word_idx, word in enumerate(raw_sent_with_split):
            match =  re.search(r'&C\d{1,2}&', word)

            # &C00& 형식으로 된 이름이 있을 경우, result 변수로 지정
            if match:
                result = match.group(0)

                if alias2id[result] in character_mention_poses:
                    character_mention_poses[alias2id[result]].append([sent_idx, word_idx])
                else:
                    character_mention_poses[alias2id[result]] = [[sent_idx, word_idx]]

        seg_sents.append(raw_sent_with_split)

    name_list_index = list(character_mention_poses.keys())

    return seg_sents, character_mention_poses, name_list_index


def create_CSS(seg_sents, candidate_mention_poses, args):
    """
    각 인스턴스 내 각 발화자 후보(candidate)에 대하여 candidate-specific segments(CSS)를 만듭니다. 

    parameters:
        seg_sents: 2ws + 1 개의 문장(각 문장은 분할됨)들을 담은 리스트
        candidate_mention_poses: 발화자별로 이름이 언급된 위치를 담고 있는 딕셔너리이며, 형태는 다음과 같음.
            {character index: [[sentence index, word index in sentence] of mention 1,...]...}.
        args : 실행 인수를 담은 객체

    return:
        Returned contents are in lists, in which each element corresponds to a candidate.
        The order of candidate is consistent with that in list(candidate_mention_poses.keys()).
        many_css: 각 발화자 후보에 대한 candidate-specific segments(CSS).
        many_sent_char_len: 각 CSS의 문자 길이 정보
            [[character-level length of sentence 1,...] of the CSS of candidate 1,...].
        many_mention_pos: CSS 내에서, 인용문과 가장 가까운 이름이 언급된 위치 정보
            [(sentence-level index of nearest mention in CSS, 
             character-level index of the leftmost character of nearest mention in CSS, 
             character-level index of the rightmost character + 1) of candidate 1,...].
        many_quote_idx: CSS 내의 인용문의 문장 인덱스
        many_cut_css : 최대 길이 제한이 적용된 CSS

    """
    ws = args.ws
    max_len = args.length_limit
    model_name = args.model_name

    # assert len(seg_sents) == ws * 2 + 1

    many_css = []
    many_sent_char_lens = []
    many_mention_poses = []
    many_quote_idxes = []
    many_cut_css = []

    for candidate_idx in candidate_mention_poses.keys():
        nearest_pos = NML(seg_sents, candidate_mention_poses[candidate_idx], ws)

        if nearest_pos[0] <= ws:
            CSS = copy.deepcopy(seg_sents[nearest_pos[0]:ws + 1])
            mention_pos = [0, nearest_pos[1]]
            quote_idx = ws - nearest_pos[0]
        else:
            CSS = copy.deepcopy(seg_sents[ws:nearest_pos[0] + 1])
            mention_pos = [nearest_pos[0] - ws, nearest_pos[1]]
            quote_idx = 0

        cut_CSS, mention_pos = max_len_cut(CSS, mention_pos, max_len)
        sent_char_lens = [sum(len(word) for word in sent) for sent in cut_CSS]

        mention_pos_left = sum(sent_char_lens[:mention_pos[0]]) + sum(
            len(x) for x in cut_CSS[mention_pos[0]][:mention_pos[1]])
        mention_pos_right = mention_pos_left + len(cut_CSS[mention_pos[0]][mention_pos[1]])

        if model_name == 'CSN':
            mention_pos = (mention_pos[0], mention_pos_left, mention_pos_right)
            cat_CSS = ''.join([''.join(sent) for sent in cut_CSS])
        elif model_name == 'KCSN':
            mention_pos = (mention_pos[0], mention_pos_left, mention_pos_right, mention_pos[1])
            cat_CSS = ' '.join([' '.join(sent) for sent in cut_CSS])

        many_css.append(cat_CSS)
        many_sent_char_lens.append(sent_char_lens)
        many_mention_poses.append(mention_pos)
        many_quote_idxes.append(quote_idx)
        many_cut_css.append(cut_CSS)

    return many_css, many_sent_char_lens, many_mention_poses, many_quote_idxes, many_cut_css


class ISDataset(Dataset):
    """
    발화자 식별을 위한 데이터셋 서브클래스
    """
    def __init__(self, data_list):
        super(ISDataset, self).__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_data_loader(data_file, alias2id, args, save_name=None) -> DataLoader:
    """
    학습을 위한 데이터로더를 생성합니다.
    """
    # 사전에 이름을 추가
    for alias in alias2id:
        twitter.add_dictionary(alias, 'Noun')

    # 파일을 줄별로 불러들임
    with open(data_file, 'r', encoding='utf-8') as fin:
        data_lines = fin.readlines()

    # 전처리
    data_list = []

    for i, line in enumerate(tqdm(data_lines)):
        offset = i % 31

        if offset == 0:
            instance_index = line.strip().split()[-1]
            raw_sents_in_list = []
            continue

        if offset < 22:
            raw_sents_in_list.append(line.strip())

        if offset == 22:
            speaker_name = line.strip().split()[-1]

            # 빈 리스트는 제거
            filtered_list = [li for li in raw_sents_in_list if li]

            # 문장 분할 및 등장인물 언급 위치 추출
            seg_sents, candidate_mention_poses, name_list_index = seg_and_mention_location(
                filtered_list, alias2id)

            # CSS 생성
            css, sent_char_lens, mention_poses, quote_idxes, cut_css = create_CSS(
                seg_sents, candidate_mention_poses, args)

            # 후보자 리스트
            candidates_list = list(candidate_mention_poses.keys())

            # 원핫 레이블 생성
            one_hot_label = [0 if character_idx != alias2id[speaker_name]
                             else 1 for character_idx in candidate_mention_poses.keys()]

            true_index = one_hot_label.index(1) if 1 in one_hot_label else 0

        if offset == 24:
            category = line.strip().split()[-1]

        if offset == 25:
            name = ' '.join(line.strip().split()[1:])

        if offset == 26:
            scene = line.strip().split()[-1]

        if offset == 27:
            place = line.strip().split()[-1]

        if offset == 28:
            time = line.strip().split()[-1]

        if offset == 29:
            cut_position = line.strip().split()[-1]
            data_list.append((seg_sents, css, sent_char_lens, mention_poses, quote_idxes,
                              cut_css, one_hot_label, true_index, category, name_list_index,
                              name, scene, place, time, cut_position, candidates_list,
                              instance_index))
    # 데이터로더 생성
    data_loader = DataLoader(ISDataset(data_list), batch_size=1, collate_fn=lambda x: x[0])

    # 저장할 이름이 주어진 경우 데이터 리스트 저장
    if save_name is not None:
        torch.save(data_list, save_name)

    return data_loader


def load_data_loader(saved_filename: str) -> DataLoader:
    """
    저장된 파일에서 데이터를 로드하고 DataLoader 객체로 변환합니다.
    """
    # 저장된 데이터 리스트 로드
    data_list = load_data(saved_filename)
    return DataLoader(ISDataset(data_list), batch_size=1, collate_fn=lambda x: x[0])


def split_train_val_test(data_file, alias2id, args, save_name=None, test_size=0.2, val_size=0.1, random_state=13):
    """
    기존 검증 방식을 적용하여 데이터 로더를 빌드합니다.
    주어진 데이터 파일을 훈련, 검증, 테스트 세트로 분할하고 각각의 DataLoader를 생성합니다.

    Parameters:
        - data_file: 분할할 데이터 파일 경로
        - alias2id: 등장인물 이름과 ID를 매핑한 딕셔너리
        - args: 실행 인자를 담은 객체
        - save_name: 분할된 데이터를 저장할 파일 이름
        - test_size: 테스트 세트의 비율 (기본값: 0.2)
        - val_size: 검증 세트의 비율 (기본값: 0.1)
        - random_state: 랜덤 시드 (기본값: 13)

    Returns:
        - train_loader: 훈련 데이터로더
        - val_loader: 검증 데이터로더
        - test_loader: 테스트 데이터로더
    """

    # 사전에 이름 추가
    for alias in alias2id:
        twitter.add_dictionary(alias, 'Noun')

    # 파일에서 인스턴스 로드
    with open(data_file, 'r', encoding='utf-8') as fin:
        data_lines = fin.readlines()

    # 전처리
    data_list = []

    for i, line in enumerate(tqdm(data_lines)):
        offset = i % 31

        if offset == 0:
            instance_index = line.strip().split()[-1]
            raw_sents_in_list = []
            continue

        if offset < 22:
            raw_sents_in_list.append(line.strip())

        if offset == 22:
            speaker_name = line.strip().split()[-1]

            # 빈 리스트는 제거합니다.
            filtered_list = [li for li in raw_sents_in_list if li]

            # 문장 분할 및 등장인물 언급 위치 추출
            seg_sents, candidate_mention_poses, name_list_index = seg_and_mention_location(
                filtered_list, alias2id)

            # CSS 생성
            css, sent_char_lens, mention_poses, quote_idxes, cut_css = create_CSS(
                seg_sents, candidate_mention_poses, args)

            # 후보자 리스트
            candidates_list = list(candidate_mention_poses.keys())

            # 원핫 레이블 생성
            one_hot_label = [0 if character_idx != alias2id[speaker_name]
                             else 1 for character_idx in candidate_mention_poses.keys()]

            true_index = one_hot_label.index(1) if 1 in one_hot_label else 0

        if offset == 24:
            category = line.strip().split()[-1]

        if offset == 25:
            name = ' '.join(line.strip().split()[1:])

        if offset == 26:
            scene = line.strip().split()[-1]

        if offset == 27:
            place = line.strip().split()[-1]

        if offset == 28:
            time = line.strip().split()[-1]

        if offset == 29:
            cut_position = line.strip().split()[-1]
            data_list.append((seg_sents, css, sent_char_lens, mention_poses, quote_idxes,
                              cut_css, one_hot_label, true_index, category, name_list_index,
                              name, scene, place, time, cut_position, candidates_list,
                              instance_index))

    # train-validation-test로 데이터를 나누기
    train_data, test_data = train_test_split(
        data_list, test_size=test_size, random_state=random_state)
    train_data, val_data = train_test_split(
        train_data, test_size=val_size, random_state=random_state)

    # train DataLoader 생성
    train_loader = DataLoader(ISDataset(train_data), batch_size=1, collate_fn=lambda x: x[0])

    # validation DataLoader 생성
    val_loader = DataLoader(ISDataset(val_data), batch_size=1, collate_fn=lambda x: x[0])

    # test DataLoader 생성
    test_loader = DataLoader(ISDataset(test_data), batch_size=1, collate_fn=lambda x: x[0])

    if save_name is not None:
        # 각각의 데이터를 저장
        torch.save(train_data, save_name.replace(".pt", "_train.pt"))
        torch.save(val_data, save_name.replace(".pt", "_val.pt"))
        torch.save(test_data, save_name.replace(".pt", "_test.pt"))

    return train_loader, val_loader, test_loader
