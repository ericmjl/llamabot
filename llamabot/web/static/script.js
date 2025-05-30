document.body.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail.target.id === 'modal-content') {
        const modal = document.getElementById('prompt-modal');
        modal.style.display = 'block';

        // Add close button functionality
        const closeBtn = modal.querySelector('.close');
        closeBtn.onclick = function() {
            modal.style.display = 'none';
        }

        // Close when clicking outside the modal
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    }
});

// Initialize tooltips and popovers
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Initialize sorting functionality
    const sortableHeaders = document.querySelectorAll('.sortable');
    sortableHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const sortField = this.dataset.sort;
            const currentDirection = this.classList.contains('asc') ? 'desc' : 'asc';

            // Remove sort classes from all headers
            sortableHeaders.forEach(h => h.classList.remove('asc', 'desc'));

            // Add sort class to clicked header
            this.classList.add(currentDirection);

            // Get the table body
            const tbody = this.closest('table').querySelector('tbody');

            // Convert rows to array for sorting
            const rows = Array.from(tbody.querySelectorAll('tr'));

            // Sort rows
            rows.sort((a, b) => {
                const aValue = a.children[Array.from(this.parentElement.children).indexOf(this)].textContent;
                const bValue = b.children[Array.from(this.parentElement.children).indexOf(this)].textContent;

                // Handle numeric values
                if (!isNaN(aValue) && !isNaN(bValue)) {
                    return currentDirection === 'asc' ? aValue - bValue : bValue - aValue;
                }

                // Handle dates
                if (sortField === 'timestamp') {
                    return currentDirection === 'asc'
                        ? new Date(aValue) - new Date(bValue)
                        : new Date(bValue) - new Date(aValue);
                }

                // Handle text
                return currentDirection === 'asc'
                    ? aValue.localeCompare(bValue)
                    : bValue.localeCompare(aValue);
            });

            // Reorder rows
            rows.forEach(row => tbody.appendChild(row));
        });
    });

    // Initialize code highlighting
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });

    // Handle modal closing
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
        modal.addEventListener('hidden.bs.modal', function () {
            const content = this.querySelector('#modal-content');
            if (content) {
                content.innerHTML = '';
            }
        });
    });

    initExperimentColumnToggles();
});

// Export functionality
function exportLogs() {
    const format = document.getElementById('export-format').value;
    const positiveOnly = document.getElementById('positive-only').checked;
    const textFilter = document.querySelector('[name="text_filter"]').value;
    const functionName = document.querySelector('[name="function_name"]').value;

    const url = `/logs/export/${format}?text_filter=${encodeURIComponent(textFilter)}&function_name=${encodeURIComponent(functionName)}&positive_only=${positiveOnly}`;
    window.location.href = url;
}

// Handle HTMX after swap events
document.body.addEventListener('htmx:afterSwap', function(evt) {
    // Reinitialize code highlighting for new content
    evt.detail.target.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });

    initExperimentColumnToggles();
});

function initExperimentColumnToggles() {
    const toggles = document.querySelectorAll('.column-toggle');
    if (!toggles.length) return;
    console.log('Experiment column toggle JS loaded');
    // Initialize column visibility state from localStorage
    const savedState = JSON.parse(localStorage.getItem('experimentColumnVisibility') || '{}');

    toggles.forEach(toggle => {
        const metric = toggle.dataset.column;
        if (savedState[metric] !== undefined) {
            toggle.checked = savedState[metric];
            updateColumnVisibility(metric, savedState[metric]);
        }
    });

    // Handle individual column toggles
    toggles.forEach(toggle => {
        toggle.addEventListener('change', function() {
            const metric = this.dataset.column;
            const isVisible = this.checked;
            updateColumnVisibility(metric, isVisible);
            saveColumnState();
        });
    });

    // Handle toggle all button
    const toggleAllBtn = document.getElementById('toggle-all-columns');
    if (toggleAllBtn) {
        toggleAllBtn.addEventListener('click', function() {
            const allChecked = Array.from(toggles).every(t => t.checked);
            toggles.forEach(toggle => {
                toggle.checked = !allChecked;
                const metric = toggle.dataset.column;
                updateColumnVisibility(metric, !allChecked);
            });
            saveColumnState();
        });
    }

    function updateColumnVisibility(metric, isVisible) {
        // Hide/show both th and td for this metric
        const columns = document.querySelectorAll(`.metric-col-${metric}`);
        columns.forEach(col => {
            col.style.display = isVisible ? '' : 'none';
        });
    }

    function saveColumnState() {
        const state = {};
        toggles.forEach(toggle => {
            state[toggle.dataset.column] = toggle.checked;
        });
        localStorage.setItem('experimentColumnVisibility', JSON.stringify(state));
    }
}
