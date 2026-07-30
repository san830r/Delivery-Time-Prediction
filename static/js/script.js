/* ==========================================================================
   SwiftETA · script.js
   Small, dependency-free UX helpers.
   ========================================================================== */

/**
 * Animates a number element counting up from 0 to its data-target value.
 * Used on the result page for the big predicted-minutes number.
 */
function animateCounter(el) {
  if (!el) return;
  const target = parseFloat(el.getAttribute('data-target')) || 0;
  const duration = 900; // ms
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    // ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = target * eased;
    el.textContent = target % 1 === 0
      ? Math.round(value)
      : value.toFixed(1);

    if (progress < 1) {
      requestAnimationFrame(tick);
    } else {
      el.textContent = target % 1 === 0 ? target : target.toFixed(1);
    }
  }
  requestAnimationFrame(tick);
}

/**
 * Basic front-end guard on the prediction form so users get instant
 * feedback before the request even reaches Flask.
 */
document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('predictForm');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    const distance = parseFloat(form.distance.value);
    const prep = parseFloat(form.prep_time.value);

    if (isNaN(distance) || distance <= 0) {
      e.preventDefault();
      alert('Please enter a valid distance greater than 0.');
      form.distance.focus();
      return;
    }
    if (isNaN(prep) || prep < 0) {
      e.preventDefault();
      alert('Please enter a valid preparation time.');
      form.prep_time.focus();
      return;
    }

    const submitBtn = form.querySelector('button[type="submit"]');
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';
    }
  });
});
