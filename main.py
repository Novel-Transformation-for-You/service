"""
This is main.py
"""
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List

class AppData:
    def __init__(self):
        self.file_content = ""
        self.name_list = []
        self.place = []
        self.times = []
        self.name_dic = {}
        self.end_output = []


class ItemListRequest(BaseModel):
    nameList: List[str]


app_data = AppData()

# 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    """INDEX.HTML 화면"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/put.html", response_class=HTMLResponse)
async def page_put(request: Request):
    """PUT.HTML 화면"""
    return templates.TemplateResponse("put.html", {"request": request})


@app.get("/confirm.html", response_class=HTMLResponse)
async def page_confirm(request: Request):
    """confirm.HTML 화면"""
    return templates.TemplateResponse("confirm.html",{
        "request": request, "file_content": app_data.file_content})


@app.get("/result.html", response_class=HTMLResponse)
async def page_result(request: Request):
    """result.HTML 화면"""
    return templates.TemplateResponse("result.html", {"request": request})


@app.get("/user.html", response_class=HTMLResponse)
async def page_user(request: Request):
    """user.HTML 화면"""
    return templates.TemplateResponse("user.html", {"request": request})


@app.get("/final.html", response_class=HTMLResponse)
async def page_final(request: Request):
    """final.HTML 화면"""
    return templates.TemplateResponse("final.html", {"request": request,
                                                     "output": app_data.end_output,
                                                     "place": app_data.place,
                                                     "time": app_data.times})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 및 저장"""
    with open("uploads/" + file.filename, "wb") as f:
        f.write(file.file.read())

    with open("uploads/" + file.filename, "r", encoding="utf-8") as f:
        app_data.file_content = f.read()

    return RedirectResponse(url="/put.html")


@app.post("/ners", response_class=JSONResponse)
async def ner_file():
    """저장된 파일에 NER 작업을 해서 화자랑 장소를 구분"""
    from utils.load_model import load_ner
    from utils.input_process import make_ner_input
    from utils.ner_utils import make_name_list, show_name_list, combine_similar_names

    content = app_data.file_content
    _, ner_checkpoint = load_ner()

    contents = make_ner_input(content)
    name_list, app_data.place, app_data.times = make_name_list(contents, ner_checkpoint)
    name_dic = show_name_list(name_list)
    similar_name = combine_similar_names(name_dic)
    result_list = [', '.join(names) for names, _ in similar_name.items()]

    # JSONResponse로 응답
    return JSONResponse(content={"itemList": result_list})


@app.post("/kcsn", response_class=JSONResponse)
async def kcsn_file(request_data: ItemListRequest):
    """사용자가 올려준 파일에 대해서 KCSN 모델 동작"""
    import torch
    from utils.fs_utils import get_alias2id, find_speak, making_script
    from utils.input_process import make_instance_list, input_data_loader
    from utils.train_model import KCSN
    from utils.ner_utils import convert_name2codename, convert_codename2name

    content = app_data.file_content
    name_list = request_data.nameList
    name_dic = {}

    for idx, name in enumerate(name_list):
        name_dic[f'&C{idx:02d}&'] = name.split(', ')

    content_re = convert_name2codename(name_dic, content)

    # checkpoint = torch.load('./model/final.pth')
    # model = checkpoint['model']
    # model.to('cpu')
    # tokenizer = checkpoint['tokenizer']

    from utils.arguments import get_train_args
    from transformers import AutoTokenizer

    args = get_train_args()
    path ='model/model.ckpt'
    model = KCSN(args)
    model.to('cpu')

    checkpoint = torch.load(path)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
    model.load_state_dict(checkpoint['model'])

    check_name = 'data/name.txt'
    alias2id = get_alias2id(check_name)
    instances, instance_num = make_instance_list(content_re)
    inputs = input_data_loader(instances, alias2id)
    output = find_speak(model, inputs, tokenizer, alias2id)
    outputs = convert_codename2name(name_dic, output)
    app_data.end_output = making_script(content, outputs, instance_num)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
