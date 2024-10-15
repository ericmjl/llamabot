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
                                    <span class="prompt-hash">${message.prompt_hash.substring(0, 6)}</span>
                                </p>
                            ` : ''}
                            <p class="content">${formatContent(message.content)}</p>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        event.detail.target.innerHTML = formattedHtml;
    }
});

function formatContent(content) {
    return content.replace(/\n/g, '<br>');
}

function addRowClickListeners() {
    const rows = document.querySelectorAll('tbody tr');
    rows.forEach(row => {
        row.addEventListener('click', function(event) {
            if (!event.target.closest('input[type="checkbox"]')) {
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
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const filterInput = document.getElementById('log-filter');
    filterInput.addEventListener('input', filterLogs);
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

// Initial call to add listeners when the page loads
document.addEventListener('DOMContentLoaded', addRowClickListeners);
