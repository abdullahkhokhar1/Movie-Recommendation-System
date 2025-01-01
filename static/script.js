document.addEventListener('DOMContentLoaded', () => {
    const movieInput = document.getElementById('movie_name');
    const form = document.querySelector('form');

    form.addEventListener('submit', (event) => {
        const inputValue = movieInput.value.trim();
        if (inputValue === '') {
            event.preventDefault();
            alert('Please enter a movie name.');
        }
    });
});