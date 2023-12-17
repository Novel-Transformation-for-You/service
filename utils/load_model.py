"""
모델들 불러오는 모듈
"""
import torch
# from .load_model import KCSN
# from .arguments import get_train_args


# args = get_train_args()

def load_ner(path ='model/NER.pth'):
    """
    NER 모델
    """
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint


# def load_fs(path = 'model/FS.pth'):
#     """
#     Find Speaker 모델
#     """
#     model = KCSN(args)
#     checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model, checkpoint
