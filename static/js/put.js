function saveAndRedirect() {
  var selectedOption = document.querySelector('input[name="displayOption"]:checked');

  if (selectedOption) {
    // 선택된 옵션을 세션 스토리지에 저장
    sessionStorage.setItem('selectedOption', selectedOption.value);

    // 결과를 보여주는 페이지로 리다이렉트
    window.location.href = 'templates/confirm.html';
  } else {
    alert('라디오 버튼을 선택하세요.');
  }
}

function showSelectedScreen() {
  // 선택된 라디오 버튼의 값을 가져옴
  var selectedOption = document.querySelector('input[name="displayOption"]:checked');

  if (selectedOption) {
    // 선택된 옵션에 따라 화면을 변경
    var resultScreen = document.getElementById('resultScreen');
    var optionResult = document.getElementById('optionResult');

    // 여기에서 선택된 옵션에 따라 다른 내용을 보여주도록 설정
    switch (selectedOption.value) {
      case 'option1':
        optionResult.textContent = '옵션 1을 선택했습니다.';
        break;
      case 'option2':
        optionResult.textContent = '옵션 2를 선택했습니다.';
        break;
      case 'option3':
        optionResult.textContent = '옵션 3을 선택했습니다.';
        break;
    }

    // 결과 화면을 보이도록 변경
    resultScreen.style.display = 'block';
  } else {
    alert('라디오 버튼을 선택하세요.');
  }
}



// 새로운 코드.....!!
async function handleButtonClick() {
  await uploadFile();  // uploadFile 함수가 비동기 함수이므로 await를 사용하여 기다립니다.
  location.href = 'confirm.html';  // 파일 업로드 후에 confirm.html로 이동합니다.
}

async function uploadFile() {
  const fileInput = document.getElementById('formFileSm');
  const file = fileInput.files[0];

  if (file) {
      const formData = new FormData();
      formData.append('file', file);

      try {
          const response = await fetch('/upload', {
              method: 'POST',
              body: formData
          });

          if (response.ok) {
              console.log('파일이 성공적으로 업로드되었습니다.');
          } else {
              console.error('파일 업로드 실패:', response.statusText);
          }
      } catch (error) {
          console.error('파일 업로드 중 오류 발생:', error);
      }
  } else {
      console.warn('파일을 선택하세요.');
  }
}