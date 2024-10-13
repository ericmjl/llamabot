document.body.addEventListener('htmx:afterSwap', function(event) {
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
    // Replace newline characters with HTML line breaks
    return content.replace(/\n/g, '<br>');
}
