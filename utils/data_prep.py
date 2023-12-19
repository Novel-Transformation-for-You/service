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

twitter = Twitter()


def load_data(filename) -> Any:
    """
    지정된 파일에서 데이터를 로드합니다.
    """
    return torch.load(filename)


def NML(seg_sents, mention_positions, ws):
    """
    Nearest Mention Location
    """
    def word_dist(pos):
        """
        The word level distance between quote and the mention position
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

    sorted_positions = sorted(mention_positions, key=lambda x: word_dist(x))

    return sorted_positions[0]


def max_len_cut(seg_sents, mention_pos, max_len):
    sent_char_lens = [sum(len(word) for word in sent) for sent in seg_sents]
    sum_char_len = sum(sent_char_lens)

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

        del seg_sents[max_len_sent_idx][running_cut_idx[max_len_sent_idx]]

        running_cut_idx[max_len_sent_idx] -= 1

    return seg_sents, mention_pos


def seg_and_mention_location(raw_sents_in_list, alias2id):
    """
    
    """
    character_mention_poses = {}
    seg_sents = []
    id_pattern = ['&C{:02d}&'.format(i) for i in range(51)]

    for sent_idx, sent in enumerate(raw_sents_in_list):
        raw_sent_with_split = sent.split()

        for word_idx, word in enumerate(raw_sent_with_split):
            match =  re.search(r'&C\d{1,2}&', word)

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
    Create candidate-specific segments for each candidate in an instance.

    params:
        seg_sents: 2ws + 1 segmented sentences in a list.
        candidate_mention_poses: a dict which contains the position of candiate mentions,
        with format {character index: [[sentence index, word index in sentence] of mention 1,...]...}.
        ws: single-sided context window size.
        max_len: maximum length limit.

    return:
        Returned contents are in lists, in which each element corresponds to a candidate.
        The order of candidate is consistent with that in list(candidate_mention_poses.keys()).
        many_CSS: candidate-specific segments.
        many_sent_char_len: segmentation information of candidate-specific segments.
            [[character-level length of sentence 1,...] of the CSS of candidate 1,...].
        many_mention_pos: the position of the nearest mention in CSS. 
            [(sentence-level index of nearest mention in CSS, 
             character-level index of the leftmost character of nearest mention in CSS, 
             character-level index of the rightmost character + 1) of candidate 1,...].
        many_quote_idx: the sentence-level index of quote sentence in CSS.

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
    Dataset subclass for Identifying speaker.
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
    Build the dataloader for training.
    """
    # Add dictionary
    for alias in alias2id:
        twitter.add_dictionary(alias, 'Noun')

    # load instances from file
    with open(data_file, 'r', encoding='utf-8') as fin:
        data_lines = fin.readlines()

    # pre-processing
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

            # segmentation and character mention location
            seg_sents, candidate_mention_poses, name_list_index = seg_and_mention_location(
                filtered_list, alias2id)

            css, sent_char_lens, mention_poses, quote_idxes, cut_css = create_CSS(
                seg_sents, candidate_mention_poses, args)

            candidates_list = list(candidate_mention_poses.keys())

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

    data_loader = DataLoader(ISDataset(data_list), batch_size=1, collate_fn=lambda x: x[0])

    if save_name is not None:
        torch.save(data_list, save_name)

    return data_loader


def load_data_loader(saved_filename: str) -> DataLoader:
    """
    저장된 파일에서 데이터를 로드하고 DataLoader 객체로 변환합니다.
    """
    data_list = load_data(saved_filename)
    return DataLoader(ISDataset(data_list), batch_size=1, collate_fn=lambda x: x[0])


def split_train_val_test(data_file, alias2id, args, save_name=None, test_size=0.2, val_size=0.1, random_state=13):
    # 기존의 데이터 로딩 및 전처리 부분
    for alias in alias2id:
        twitter.add_dictionary(alias, 'Noun')

    with open(data_file, 'r', encoding='utf-8') as fin:
        data_lines = fin.readlines()

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

            # segmentation and character mention location
            seg_sents, candidate_mention_poses, name_list_index = seg_and_mention_location(
                filtered_list, alias2id)

            css, sent_char_lens, mention_poses, quote_idxes, cut_css = create_CSS(
                seg_sents, candidate_mention_poses, args)

            candidates_list = list(candidate_mention_poses.keys())

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
