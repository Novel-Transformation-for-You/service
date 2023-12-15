"""
사용자 입력을 가공하는 모듈
"""
import copy
import re

from torch.utils.data import DataLoader, Dataset


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


def make_instance_list(text: str, ws=10) -> list:
    """
    입력받은 문장을 기초적인 인스턴스 리스트로 만들어줍니다.
    """
    lines = text.splitlines()
    max_line = len(lines)

    utterance = ['"', '“']
    instance_num = []

    for idx, line in enumerate(lines):
        if any(u in line for u in utterance):
            instance_num.append(idx)

    instance = [[] for _ in range(len(instance_num))]

    for i, num in enumerate(instance_num):
        if num - ws <= 0 and num + ws + 1 < max_line:
            instance[i] += ([''] * (ws - num))
            instance[i] +=(lines[:num + 1 + ws])
        elif num - ws <= 0 and num + ws + 1 >= max_line:
            instance[i] += ([''] * (ws - num))
            instance[i] +=(lines)
            instance[i] += ([''] * (ws * 2 - len(instance[i]) + 1))
        elif num + ws + 1 >= max_line:
            instance[i] +=(lines[num-ws:max_line+1])
            instance[i] += ([''] * (num + ws + 1 - max_line))
        else:
            instance[i] += (lines[num-ws:num + ws + 1])

    return instance


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


def create_css(seg_sents, candidate_mention_poses, ws=10):
    """
    Create candidate-specific segments for each candidate in an instance.
    """
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

        cut_CSS, mention_pos = max_len_cut(CSS, mention_pos, 510)
        sent_char_lens = [sum(len(word) for word in sent) for sent in cut_CSS]

        mention_pos_left = sum(sent_char_lens[:mention_pos[0]]) + sum(
            len(x) for x in cut_CSS[mention_pos[0]][:mention_pos[1]])
        mention_pos_right = mention_pos_left + len(cut_CSS[mention_pos[0]][mention_pos[1]])
        mention_pos = (mention_pos[0], mention_pos_left, mention_pos_right, mention_pos[1])
        cat_CSS = ' '.join([' '.join(sent) for sent in cut_CSS])

        many_css.append(cat_CSS)
        many_sent_char_lens.append(sent_char_lens)
        many_mention_poses.append(mention_pos)
        many_quote_idxes.append(quote_idx)
        many_cut_css.append(cut_CSS)

    return many_css, many_sent_char_lens, many_mention_poses, many_quote_idxes, many_cut_css


def input_data_loader(instances: list, alias2id) -> DataLoader:
    """
    나눠진 데이터를 맞추기 위해 가공
    """
    data_list = []

    for instance in instances:
        seg_sents, candidate_mention_poses, name_list_index = seg_and_mention_location(
            instance, alias2id)
        css, sent_char_lens, mention_poses, quote_idxes, cut_css = create_css(
            seg_sents, candidate_mention_poses)

        data_list.append((seg_sents, css, sent_char_lens, mention_poses, quote_idxes,
                          cut_css, name_list_index))

    data_loader = DataLoader(ISDataset(data_list), batch_size=1, collate_fn=lambda x: x[0])

    return data_loader


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
        first_newline_position =  max_text-500 + f_sentence.find('\n')
        split_sentences.append(text[first_newline_position:])

    return split_sentences


def making_script(text, speaker:list, instance_num:list) -> str:
    """
    스크립트를 만드는 함수
    """
    lines = text.splitlines()
    for num, people in zip(instance_num, speaker):
        lines[num] = f'{people}: {lines[num]}'
    return lines
