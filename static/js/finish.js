document.addEventListener("DOMContentLoaded", function () {
    loadFileContent();
});

async function loadFileContent() {
    try {
        const response = await fetch('../file/result_example.txt'); // 파일 경로 지정
        const textData = await response.text();

        // 파일 데이터를 HTML에 출력
        displayFileContent(textData);
    } catch (error) {
        console.error('Error loading file content:', error);
    }
}

function displayFileContent(data) {
    const fileContentContainer = document.getElementById('resultContainer');

    // 예제: 파일 데이터를 텍스트로 표시
    const textElement = document.createElement('p');

    // 텍스트에서 각 줄바꿈 문자를 <br> 태그로 대체
    const formattedText = data.replace(/\n/g, '<br>');

    textElement.innerHTML = formattedText;

    fileContentContainer.appendChild(textElement);
}


document.getElementById('downloadButton').addEventListener('click', function () {
    downloadTextFile();
});

async function downloadTextFile() {
    const response = await fetch('../file/result_example.txt'); // 파일 경로 지정
    const textData = await response.text(); // 여기에 실제 텍스트 데이터를 넣어주세요
    const filename = "example.txt"; // 다운로드될 파일 이름

    const blob = new Blob([textData], { type: 'text/plain' });
    const link = document.createElement('a');

    link.href = window.URL.createObjectURL(blob);
    link.download = filename;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
}
