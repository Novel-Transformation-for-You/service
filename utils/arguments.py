"""
A
"""
# 미리 설정된 인수들
from argparse import ArgumentParser

# 사용자 선정 변수들
ROOT_DIR = ""  # 프로젝트 루트 디렉토리 경로 
BERT_PRETRAINED_DIR = "klue/roberta-large"  # BERT 사전 훈련된 모델 디렉토리 경로
DATA_PREFIX = "data"  # 데이터 파일들의 상위 디렉토리 경로
CHECKPOINT_DIR = 'model'  # 모델 체크포인트 저장 디렉토리 경로
LOG_FATH = 'logs'  # 훈련 로그 저장 디렉토리 경로

def get_train_args():
    """
    훈련 인수 설정
    """
    parser = ArgumentParser(description='I_S', allow_abbrev=False)

    # 인수 파싱
    parser.add_argument('--model_name', type=str, default='KCSN')  

    # 모델 설정
    parser.add_argument('--pooling_type', type=str, default='max_pooling')  
    parser.add_argument('--classifier_intermediate_dim', type=int, default=100) 
    parser.add_argument('--nonlinear_type', type=str, default='tanh')  

    # BERT 설정
    parser.add_argument('--bert_pretrained_dir', type=str, default=BERT_PRETRAINED_DIR)  

    # 훈련 설정
    parser.add_argument('--margin', type=float, default=1.0)  
    parser.add_argument('--lr', type=float, default=2e-5)  
    parser.add_argument('--optimizer', type=str, default='adam')  
    parser.add_argument('--dropout', type=float, default=0.5)  
    parser.add_argument('--num_epochs', type=int, default=50)  
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--lr_decay', type=float, default=0.95)  
    parser.add_argument('--patience', type=int, default=10)  

    # 훈련, 개발 및 테스트 데이터 파일 경로
    parser.add_argument('--train_file', type=str, default=f'{DATA_PREFIX}/train_unsplit.txt')
    parser.add_argument('--dev_file', type=str, default=f'{DATA_PREFIX}/dev_unsplit.txt')
    parser.add_argument('--test_file', type=str, default=f'{DATA_PREFIX}/test_unsplit.txt')
    parser.add_argument('--name_list_path', type=str, default=f'{DATA_PREFIX}/name_list.txt')
    parser.add_argument('--ws', type=int, default=10)  # 윈도우 크기

    parser.add_argument('--length_limit', type=int, default=510)  # 시퀀스 길이 제한

    # 체크포인트 및 로그 저장 디렉토리
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR)
    parser.add_argument('--training_logs', type=str, default=LOG_FATH)

    args, _ = parser.parse_known_args()

    return args
