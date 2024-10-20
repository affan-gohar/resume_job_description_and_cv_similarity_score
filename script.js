document.addEventListener('DOMContentLoaded', function () {
    const choiceSelect = document.getElementById('choice');
    const cvInputs = document.getElementById('cvInputs');
    const jobInputs = document.getElementById('jobInputs');

    choiceSelect.addEventListener('change', function () {
        if (this.value === 'cv') {
            cvInputs.style.display = 'block';
            jobInputs.style.display = 'none';
        } else {
            cvInputs.style.display = 'none';
            jobInputs.style.display = 'block';
        }
    });
});
