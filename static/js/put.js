async function handleButtonClick() {
  var radios = document.getElementsByName('displayOption');
  await uploadFile();  // uploadFile 함수가 비동기 함수이므로 await를 사용하여 기다립니다.

  for (var i = 0; i < radios.length; i++) {
    if (radios[i].checked) {
      selectedPage = radios[i].value;
      break;
    }
  }

  if (selectedPage === 'pro') {
    location.href = '/confirm.html'; // 1번 페이지로 이동
  } else if (selectedPage === 'reader') {
    location.href = '/user.html'; // 2번 페이지로 이동
  }
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