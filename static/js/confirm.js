// app.js

// 샘플 데이터
let itemList = [];

// 수정 모달 열기 함수
function editItem(index) {
    const item = itemList[index];

    const editItemModal = document.getElementById('editItemModal');
    editItemModal.innerHTML = `
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="editItemModalLabel">항목 수정</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <input type="text" class="form-control" id="editedItem" value="${item}">
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">취소</button>
                    <button type="button" class="btn btn-primary" onclick="saveEdit(${index})">저장</button>
                </div>
            </div>
        </div>
    `;

    $('#editItemModal').modal('show');
}

// 수정 저장 함수
function saveEdit(index) {
    const editedItem = document.getElementById('editedItem').value;
    itemList[index] = editedItem;
    updateItemList();
    $('#editItemModal').modal('hide');
}

// 삭제 함수
function deleteItem(index) {
    itemList.splice(index, 1);
    updateItemList();
}

// 추가 모달 열기 함수
function openAddItemModal() {
    const addItemModal = document.getElementById('addItemModal');
    addItemModal.innerHTML = `
        <div class="modal-dialog" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="addItemModalLabel">새 항목 추가</h5>
                    <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="modal-body">
                    <input type="text" class="form-control" id="newItem">
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">취소</button>
                    <button type="button" class="btn btn-primary" onclick="addItem()">추가</button>
                </div>
            </div>
        </div>
    `;

    $('#addItemModal').modal('show');
}

// 추가 함수
function addItem() {
    const newItem = document.getElementById('newItem').value;
    itemList.push(newItem);
    updateItemList();
    $('#addItemModal').modal('hide');
    event.preventDefault();
}

// 
async function handlePageLoad() {
    console.log('페이지 로드가 완료되었습니다!');

    try {
        const response = await fetch('/ners', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                // 필요한 데이터를 여기에 추가
            }),
        });

        if (response.ok) {
            const responseData = await response.json(); // JSON 응답을 파싱
            itemList = responseData.itemList; // itemList 업데이트
            updateItemList(); // updateItemList 함수 호출
        } else {
            console.error('NER 요청이 실패했습니다:', response.statusText);
        }
    } catch (error) {
        console.error('NER 요청 중 오류 발생:', error);
    }
}

// 초기 목록 생성
function updateItemList(event) {
    const itemListElement = document.getElementById('itemList');
    itemListElement.innerHTML = '';

    if (Array.isArray(itemList)) {
        // itemList이 배열인 경우
        itemList.forEach((item, index) => {
            const liElement = document.createElement('li');
            liElement.className = 'list-group-item';
            liElement.innerHTML = `
                ${item}
                <button class="btn btn-warning btn-sm float-right mx-2 margin" onclick="editItem(${index})">수정</button>
                <button class="btn btn-danger btn-sm float-right" onclick="deleteItem(${index})">삭제</button>
            `;
            itemListElement.appendChild(liElement);
        });
    } else {
        // itemList이 배열이 아닌 경우 처리 (원하는 형태에 따라 수정 필요)
        const liElement = document.createElement('li');
        liElement.className = 'list-group-item';
        liElement.innerHTML = itemList;
        itemListElement.appendChild(liElement);
    }

    if (event) {
        event.preventDefault();
    }

    console.log('itemList 업데이트 로직 수행');

    // 작업이 완료되었음을 나타내는 부분 수정
    const explainBox = document.querySelector('.explain_box');
    const explainText = document.querySelector('.explain');
    explainText.textContent = '아래 내용은 ai가 감지한 txt파일 속 등장인물들의 목록입니다. 목록을 확인하고 등장인물의 이름을 수정한 다음 확인 버튼을 클릭해주세요.';
}

function handleButtonClick() {
    let itemList = document.getElementById('itemList');
    let items = itemList.getElementsByTagName('li');

    let nameList = [];

    for (var i = 0; i < items.length; i++) {
        // 각 li 요소의 자식 요소 중에서 버튼을 제외한 텍스트만 추출
        var text = Array.from(items[i].childNodes)
            .filter(node => node.nodeType === Node.TEXT_NODE)
            .map(node => node.textContent.trim())
            .join('');
        nameList.push(text);
        console.log("Text from item " + i + ": " + text);
    }

    // Fetch API를 사용하여 FastAPI 서버로 데이터 전송
    fetch('/kcsn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ nameList: nameList }),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);

        // fetch 작업이 완료되면 이동
        location.href = '/final.html';
    })
    .catch(error => console.error('Error:', error));
}

