// app.js

// 샘플 데이터
const itemList = ["항목 1", "항목 2", "항목 3"];

// 초기 목록 생성
updateItemList();

// 목록 업데이트 함수
function updateItemList() {

    const itemListElement = document.getElementById('itemList');
    itemListElement.innerHTML = '';

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
    event.preventDefault();
}

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
}
