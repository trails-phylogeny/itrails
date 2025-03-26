window.onload = function () {
  const injectVersionSelector = () => {
    const el = document.createElement('div');
    el.innerHTML = document.getElementById('readthedocs-version')?.innerHTML;
    if (el.innerHTML) {
      el.className = 'rtd-version-selector';
      const sidebar = document.querySelector('nav[role="navigation"]');
      if (sidebar) {
        sidebar.appendChild(el);
      }
    }
  };
  injectVersionSelector();
};