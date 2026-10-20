  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide')), i = 0;
  var m = /[#&]s=?(\d+)/.exec(location.hash); if (m) i = Math.min(slides.length - 1, Math.max(0, parseInt(m[1], 10) - 1));
  function show(n) {
    i = Math.max(0, Math.min(slides.length - 1, n));
    slides.forEach(function (s, j) { s.classList.toggle('on', j === i); });
    document.getElementById('count').textContent = (i + 1) + ' / ' + slides.length;
    try { history.replaceState(null, '', '#s' + (i + 1)); } catch (e) {}
  }
  function fit() {
    var vp = document.getElementById('viewport'), st = document.getElementById('stage');
    var sc = Math.min(vp.clientWidth / 1280, vp.clientHeight / 720);
    st.style.transform = 'scale(' + sc + ')';
  }
  document.getElementById('prev').onclick = function () { show(i - 1); };
  document.getElementById('next').onclick = function () { show(i + 1); };
  document.getElementById('full').onclick = function () { var d = document.documentElement; if (!document.fullscreenElement && d.requestFullscreen) d.requestFullscreen(); else if (document.exitFullscreen) document.exitFullscreen(); };
  document.getElementById('print').onclick = function () { window.print(); };
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { show(i + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') { show(i - 1); e.preventDefault(); }
    if (e.key === 'Home') show(0); if (e.key === 'End') show(slides.length - 1);
  });
  var tx = null;
  document.addEventListener('touchstart', function (e) { tx = e.touches[0].clientX; }, { passive: true });
  document.addEventListener('touchend', function (e) { if (tx === null) return; var dx = e.changedTouches[0].clientX - tx; if (Math.abs(dx) > 50) show(i + (dx < 0 ? 1 : -1)); tx = null; });
  document.getElementById('stage').addEventListener('click', function (e) {
    var b = e.target.closest('.reveal button'); if (b) b.parentNode.classList.add('open');
  });
  window.addEventListener('resize', fit);
  fit(); show(i);
})();
</script>
</body>
</html>
