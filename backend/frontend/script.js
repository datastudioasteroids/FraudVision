document.addEventListener('DOMContentLoaded', () => {
  // Toggle form
  const toggleBtn = document.getElementById('toggle-form');
  const formSec   = document.getElementById('form-section');
  toggleBtn.addEventListener('click', () => formSec.classList.toggle('hidden'));

  const txForm = document.getElementById('tx-form');
  const loading = document.getElementById('loading');
  const resultDiv = document.getElementById('result');

  txForm.addEventListener('submit', async e => {
    e.preventDefault();
    resultDiv.innerHTML = '';
    // Show loader
    loading.classList.remove('hidden');

    // Build payload
    const payload = {};
    new FormData(txForm).forEach((val, key) => {
      // For numeric fields:
      if (key !== 'type') payload[key] = parseFloat(val);
      else payload[key] = val;
    });

    // Call API
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const { is_fraud, fraud_probability, new_point } = await res.json();

    // Hide loader
    loading.classList.add('hidden');

    // Show result
    resultDiv.innerHTML = `
      <p class="result-text ${is_fraud ? 'alert' : 'ok'}">
        ${is_fraud
          ? `🚨 <strong>Alerta de fraude</strong>: ${(fraud_probability*100).toFixed(1)}%`
          : `✅ Transacción limpia: ${((1-fraud_probability)*100).toFixed(1)}% de confianza`
        }
      </p>
      <canvas id="txChart" width="200" height="200"></canvas>
    `;

    new Chart(document.getElementById('txChart'), {
      type: 'doughnut',
      data: {
        labels: ['No Fraude','Fraude'],
        datasets: [{
          data: [1 - fraud_probability, fraud_probability],
        }]
      },
      options: { plugins: { legend: { position: 'bottom' }}}
    });

    // Trigger update
    window.dispatchEvent(new CustomEvent('newPrediction', { detail: new_point }));
  });

  // Dashboard init & real-time update
  let fraudRateChart, probChart, historyData = [];
  async function cargarDashboard() {
    const { current, history } = await (await fetch('/metrics')).json();
    historyData = history;
    document.getElementById('metrics-box').innerHTML = `
      <p>Tasa fraude: ${(current.fraudRate*100).toFixed(2)}%</p>
      <p>Txns/hora: ${current.txnPerHour.toFixed(1)}</p>
    `;
    fraudRateChart = new Chart(
      document.getElementById('fraudRateChart'),
      { type:'line',
        data: {
          labels: history.map(h=>h.timestamp),
          datasets:[{
            label:'Fraude %',
            data: history.map(h=>h.avgFraudProbability*100),
            tension:0.2, fill:false
          }]
        }
      }
    );
    probChart = new Chart(
      document.getElementById('probChart'),
      { type:'line',
        data: {
          labels: history.map(h=>h.timestamp),
          datasets:[{
            label:'Prob. media',
            data: history.map(h=>h.avgFraudProbability),
            tension:0.2, fill:false
          }]
        }
      }
    );
  }
  window.addEventListener('load', cargarDashboard);

  const evtSrc = new EventSource('/stream');
  evtSrc.onmessage = e => {
    const pt = JSON.parse(e.data);
    historyData.push({ timestamp: pt.timestamp, avgFraudProbability: pt.fraud_probability });
    if (historyData.length > 100) historyData.shift();
    // update charts
    fraudRateChart.data.labels.push(pt.timestamp);
    fraudRateChart.data.datasets[0].data.push(pt.fraud_probability*100);
    fraudRateChart.update();
    probChart.data.labels.push(pt.timestamp);
    probChart.data.datasets[0].data.push(pt.fraud_probability);
    probChart.update();
  };

  // Batch & Chat code remains unchanged…


  // 5) Batch scoring
  document.getElementById('batch-upload').addEventListener('click', async () => {
    const fileInput = document.getElementById('batch-file');
    if (!fileInput.files.length) return alert('Selecciona un archivo');
    const form = new FormData();
    form.append('file', fileInput.files[0]);
    const { frauds_detected } = await (await fetch('/batch',{ method:'POST', body: form })).json();
    document.getElementById('batch-result').textContent = `Fraudes detectados: ${frauds_detected}`;
  });

  // 6) Chat de tickets
  document.getElementById('send-ticket').addEventListener('click', async () => {
    const fileInput = document.getElementById('ticket-file-chat');
    if (!fileInput.files.length) return alert('Selecciona un ticket o factura');
    const chatBox = document.getElementById('chat-box');
    // Mensaje usuario
    const um = document.createElement('div');
    um.className = 'message user';
    um.textContent = `📤 Enviando ${fileInput.files[0].name}…`;
    chatBox.appendChild(um); chatBox.scrollTop = chatBox.scrollHeight;

    const form = new FormData();
    form.append('file', fileInput.files[0]);
    try {
      const { is_fraud, fraud_probability } = await (await fetch('/upload_ticket',{ method:'POST', body: form })).json();
      const bm = document.createElement('div');
      bm.className = 'message bot';
      bm.innerHTML = is_fraud
        ? `🚨 <strong>¡Fraude detectado!</strong> ${(fraud_probability*100).toFixed(1)}%`
        : `✅ Limpio (${((1-fraud_probability)*100).toFixed(1)}% de confianza)`;
      chatBox.appendChild(bm);
    } catch {
      const err = document.createElement('div');
      err.className = 'message bot';
      err.textContent = '❌ Error al procesar el ticket.';
      chatBox.appendChild(err);
    }
    chatBox.scrollTop = chatBox.scrollHeight;
  });

});
const zone = document.getElementById('upload-zone');
const input = document.getElementById('file-input');
const list  = document.getElementById('file-list');

zone.addEventListener('click', () => input.click());
zone.addEventListener('dragover', e => {
  e.preventDefault();
  zone.classList.add('dragover');
});
zone.addEventListener('dragleave', () => {
  zone.classList.remove('dragover');
});
zone.addEventListener('drop', e => {
  e.preventDefault();
  zone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});
input.addEventListener('change', () => {
  handleFiles(input.files);
});

function handleFiles(files) {
  list.innerHTML = '';
  Array.from(files).forEach(file => {
    const li = document.createElement('li');
    li.textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB)`;
    list.appendChild(li);
  });
  // Guarda los archivos en un FormData para enviarlos luego
  currentFiles = files;
}