document.body.addEventListener('htmx:afterSwap', function(event) {
    if (event.detail.target.matches('table')) {
        addRowClickListeners();
        filterLogs(); // Apply any existing filter
    }
    if (event.detail.target.id === 'log-details') {
        const logDetails = JSON.parse(event.detail.target.textContent);
        const formattedHtml = `
            <div class="log-detail">
                <p><strong>Timestamp:</strong> ${logDetails.timestamp}</p>
                <p><strong>Model Name:</strong> ${logDetails.model_name || 'N/A'}</p>
                <p><strong>Temperature:</strong> ${logDetails.temperature || 'N/A'}</p>
                <h3>Message Log:</h3>
                <div class="message-log">
                    ${logDetails.message_log.map(message => `
                        <div class="message ${message.role}">
                            <p class="role">${message.role}</p>
                            ${message.prompt_hash ? `
                                <p class="prompt-info">
                                    <span class="prompt-name">${message.prompt_name || 'Unknown'}</span>
                                    <a href="#" class="prompt-hash" data-hash="${message.prompt_hash}" data-template="${escapeHtml(message.prompt_template || '')}">${message.prompt_hash.substring(0, 8)}...</a>
                                </p>
                            ` : ''}
                            <pre class="content">${escapeHtml(message.content)}</pre>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        event.detail.target.innerHTML = formattedHtml;
        addPromptHashClickListeners();
    }
});

function formatContent(content) {
    return content.replace(/\n/g, '<br>');
}

function escapeHtml(unsafe) {
     return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;")
         .replace(/`/g, "&#96;")
         .replace(/{/g, "&#123;")
         .replace(/}/g, "&#125;")
         .replace(/,/g, "&#44;");
}

function addPromptHashClickListeners() {
    const promptHashes = document.querySelectorAll('.prompt-hash');
    promptHashes.forEach(hash => {
        hash.addEventListener('click', function(event) {
            event.preventDefault();
            const template = this.getAttribute('data-template');
            showModal(template);
        });
    });
}

function showModal(content) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <span class="close">&times;</span>
            <pre class="modal-pre">${escapeHtml(content)}</pre>
        </div>
    `;
    document.body.appendChild(modal);

    const closeBtn = modal.querySelector('.close');
    closeBtn.onclick = function() {
        document.body.removeChild(modal);
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            document.body.removeChild(modal);
        }
    }

    // Show the modal
    modal.style.display = 'block';
}

function addRowClickListeners() {
    const rows = document.querySelectorAll('tbody tr');
    rows.forEach(row => {
        row.addEventListener('click', function(event) {
            const logId = this.cells[0].textContent;
            fetch(`/log/${logId}`)
                .then(response => response.json())
                .then(data => {
                    const logDetailsElement = document.getElementById('log-details');
                    logDetailsElement.innerHTML = JSON.stringify(data);
                    const event = new CustomEvent('htmx:afterSwap', {
                        detail: { target: logDetailsElement }
                    });
                    document.body.dispatchEvent(event);
                })
                .catch(error => console.error('Error:', error));
        });
    });
}

document.addEventListener('DOMContentLoaded', (event) => {
    const filterInput = document.getElementById('log-filter');
    filterInput.addEventListener('input', filterLogs);

    // Fetch and populate function names
    const functionNameSelect = document.getElementById('function-name-select');
    fetch('/prompt_functions')
        .then(response => response.json())
        .then(data => {
            data.function_names.forEach(item => {
                const option = document.createElement('option');
                option.value = item.name;
                option.textContent = `${item.name} (${item.count} versions)`;
                functionNameSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Error fetching function names:', error));

    // Load prompt history when a function is selected
    const promptHistoryContainer = document.getElementById('prompt-history-container');
    functionNameSelect.addEventListener('change', () => {
        const functionName = functionNameSelect.value;
        if (functionName) {
            fetch(`/prompt_history/${functionName}`)
                .then(response => response.text())
                .then(html => {
                    promptHistoryContainer.innerHTML = html;
                    hljs.highlightAll();
                })
                .catch(error => {
                    promptHistoryContainer.innerHTML = `<p>Error loading prompt history: ${error}</p>`;
                });
        } else {
            promptHistoryContainer.innerHTML = '<p>Please select a function name.</p>';
        }
    });

    // Initial load of all logs
    reloadLogs();

    // Handle prompt selection
    const promptSelect = document.getElementById('prompt-select');
    if (promptSelect) {
        promptSelect.addEventListener('change', (event) => {
            const selectedFunctionName = event.target.value;
            reloadLogs(selectedFunctionName);
        });
    }

    const exportButton = document.getElementById('export-logs');
    if (exportButton) {
        exportButton.addEventListener('click', exportLogs);
    }
});

function filterLogs() {
    const filterValue = document.getElementById('log-filter').value.toLowerCase();
    const rows = document.querySelectorAll('tbody tr');

    rows.forEach(row => {
        const visibleText = Array.from(row.cells).map(cell => cell.textContent).join(' ').toLowerCase();
        const fullContent = row.dataset.fullContent.toLowerCase();
        if (visibleText.includes(filterValue) || fullContent.includes(filterValue)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

function filterLogsByPrompt(promptHash) {
    const logRows = document.querySelectorAll('.log-row');
    logRows.forEach(row => {
        const promptNames = row.getAttribute('data-prompt-names');
        if (promptNames && promptNames.includes(promptHash)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

function showAllLogs() {
    const logRows = document.querySelectorAll('.log-row');
    logRows.forEach(row => {
        row.style.display = '';
    });
}

function reloadLogs(functionName = '') {
    const logList = document.querySelector('.log-list');
    if (logList) {
        const url = functionName ? `/logs?function_name=${encodeURIComponent(functionName)}` : '/logs';
        htmx.ajax('GET', url, {target: '.log-list', swap: 'innerHTML'});
    }
}

function exportLogs() {
    const promptSelect = document.getElementById('prompt-select');
    const functionName = promptSelect.value;
    const url = functionName ? `/export_logs?function_name=${encodeURIComponent(functionName)}` : '/export_logs';

    window.location.href = url;
}
