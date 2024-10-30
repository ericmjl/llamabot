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
