const API_URL = document.getElementById('api-url').value;
const textsEl = document.getElementById('texts');
const topKEl = document.getElementById('top-k');
const resultsSection = document.getElementById('results');
const resultsContent = document.getElementById('results-content');
const errorSection = document.getElementById('error');
const errorContent = document.getElementById('error-content');

function showError(message) {
    errorSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    errorContent.textContent = message;
}

function hideError() {
    errorSection.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
}

function clearResults() {
    resultsContent.innerHTML = '';
}

function getTexts() {
    return textsEl.value
        .split('\n')
        .map(t => t.trim())
        .filter(t => t.length > 0);
}

function renderPredictionCard(text, prediction, explanation) {
    const card = document.createElement('div');
    const isPositive = prediction === 1;
    card.className = `result-card ${isPositive ? 'positive' : 'negative'}`;

    const textDiv = document.createElement('div');
    textDiv.className = 'text';
    textDiv.textContent = text;
    card.appendChild(textDiv);

    const predDiv = document.createElement('div');
    predDiv.className = `prediction ${isPositive ? 'positive' : 'negative'}`;
    predDiv.textContent = isPositive ? 'POSITIVO (1)' : 'NEGATIVO (0)';
    card.appendChild(predDiv);

    if (explanation && explanation.top_tokens && explanation.top_tokens.length > 0) {
        const tokensDiv = document.createElement('div');
        tokensDiv.className = 'tokens';
        explanation.top_tokens.forEach((item, idx) => {
            const span = document.createElement('span');
            span.className = 'token';
            if (idx === 0) span.classList.add('influence-high');
            else span.classList.add('influence-medium');
            span.textContent = `${item.token} (${item.influence.toFixed(3)})`;
            tokensDiv.appendChild(span);
        });
        card.appendChild(tokensDiv);
    }

    return card;
}

async function doPredict(explain = false) {
    hideError();
    clearResults();

    const texts = getTexts();
    if (texts.length === 0) {
        showError('Por favor, introduce al menos un texto.');
        return;
    }

    const apiUrl = document.getElementById('api-url').value.replace(/\/$/, '');
    const endpoint = explain ? '/predict/explain' : '/predict';
    const url = `${apiUrl}${endpoint}`;
    const payload = { texts };
    const params = explain ? `?top_k=${topKEl.value}` : '';

    try {
        const response = await fetch(url + params, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`HTTP ${response.status}: ${text}`);
        }

        const data = await response.json();
        const predictions = data.predictions;
        const explanations = data.explanations || [];

        predictions.forEach((pred, idx) => {
            const exp = explanations[idx] || null;
            const card = renderPredictionCard(texts[idx], pred, exp);
            resultsContent.appendChild(card);
        });

        showResults();
    } catch (err) {
        showError(String(err));
    }
}

document.getElementById('btn-predict').addEventListener('click', () => doPredict(false));
document.getElementById('btn-explain').addEventListener('click', () => doPredict(true));
