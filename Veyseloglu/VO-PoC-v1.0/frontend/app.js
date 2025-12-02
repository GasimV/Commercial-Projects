const API = 'http://127.0.0.1:8000';

async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

async function getJSON(url) {
  const res = await fetch(url);
  return res.json();
}

document.getElementById('trainBtn').onclick = async () => {
  const N_days = parseInt(document.getElementById('nDays').value, 10);
  const out = document.getElementById('trainOut');
  out.textContent = 'Training...';
  try {
    const resp = await postJSON(`${API}/train`, { N_days });
    out.textContent = JSON.stringify(resp, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
};

document.getElementById('likBtn').onclick = async () => {
  const cid = parseInt(document.getElementById('custId1').value, 10);
  const out = document.getElementById('likOut');
  out.textContent = 'Loading...';
  try {
    const data = await getJSON(`${API}/likelihood?customer_id=${cid}`);
    out.textContent = JSON.stringify(data.slice(0, 15), null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
};

document.getElementById('qtyBtn').onclick = async () => {
  const cid = parseInt(document.getElementById('custId2').value, 10);
  const out = document.getElementById('qtyOut');
  out.textContent = 'Loading...';
  try {
    const data = await getJSON(`${API}/quantity?customer_id=${cid}`);
    out.textContent = JSON.stringify(data.slice(0, 15), null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
};

document.getElementById('recBtn').onclick = async () => {
  const cid = parseInt(document.getElementById('custId3').value, 10);
  const k = parseInt(document.getElementById('topK').value, 10);
  const out = document.getElementById('recOut');
  out.textContent = 'Loading...';
  try {
    const data = await getJSON(`${API}/recommend?customer_id=${cid}&k=${k}`);
    out.textContent = JSON.stringify(data, null, 2);
  } catch (e) {
    out.textContent = 'Error: ' + e;
  }
};
