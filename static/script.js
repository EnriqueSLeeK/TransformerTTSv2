
document.getElementById('generate-button').onclick = function() {

  var userInput = document.getElementById('text-input').value;

  // Mostra a mensagem de carregamento
  document.getElementById('loading-message').style.display = 'block';


  fetch('/generate_audio', {
    method: 'POST', 
    headers: {
      'Content-Type': 'application/json' 
    },
    body: JSON.stringify({ text: userInput }) 
  })
  .then(response => response.json()) 
  .then(data => {
    document.getElementById('loading-message').style.display = 'none';

    // Configura o player de áudio com o arquivo gerado e mostra o player
    var audioPlayer = document.getElementById('audio-player');
    audioPlayer.src = '/audio/' + data.audio_path;
    audioPlayer.style.display = 'block';
    audioPlayer.load(); // Carrega o áudio
    audioPlayer.play(); // Começa a reprodução
  })
  .catch(err => {
    console.error(err);
    document.getElementById('loading-message').style.display = 'none';
  }); 
};
