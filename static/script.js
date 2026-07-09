const form = document.getElementById('register-form');
const nameInput = document.getElementById('name');
const message = document.getElementById('message');
const statusText = document.getElementById('status');

async function register(event) {
  event.preventDefault();

  const name = nameInput.value.trim();
  if (!name) {
    showMessage('Please enter a name.', false);
    return;
  }

  const body = new URLSearchParams({ name });
  const response = await fetch('/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body
  });
  const result = await response.json();

  showMessage(result.message, result.ok);
  if (result.ok) {
    nameInput.value = '';
  }
}

async function refreshStatus() {
  try {
    const response = await fetch('/status', { cache: 'no-store' });
    const result = await response.json();
    statusText.textContent = result.status;
  } catch (_error) {
    statusText.textContent = 'Waiting for camera stream...';
  }
}

function showMessage(text, ok) {
  message.textContent = text;
  message.className = ok ? 'message success' : 'message error';
}

form.addEventListener('submit', register);
refreshStatus();
setInterval(refreshStatus, 500);
