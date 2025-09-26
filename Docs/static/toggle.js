document.addEventListener('DOMContentLoaded', function () {
    var toggleButton = document.getElementById('sidebar-toggle');
    var sidebar = document.querySelector('.sidebar');
    if (toggleButton && sidebar) {
        toggleButton.addEventListener('click', function () {
            sidebar.classList.toggle('hidden');
        });
    }

    // collapse all directories and expand current path
    document.querySelectorAll('.sidebar details').forEach(function (d) {
        d.open = false;
    });
    var current = document.querySelector(
        '.sidebar a[href="' + window.location.pathname.split('/').pop() + '"]'
    );
    if (current) {
        var el = current.parentElement;
        while (el) {
            if (el.tagName && el.tagName.toLowerCase() === 'details') {
                el.open = true;
            }
            el = el.parentElement;
        }
    }
});
