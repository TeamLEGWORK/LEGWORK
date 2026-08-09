/* Light/dark theme handling.

   This file is loaded from <head>, so the theme is applied to <html> before the
   page is painted and there's no flash of the wrong theme on load. The attribute
   is mirrored onto <body> as well because sphinx-tabs styles itself off
   `body[data-theme="dark"]`, so we get its dark theme for free. */
const LEGWORK_THEME_KEY = "legwork-theme";

// how long the colours take to fade when switching, in step with .theme-switching
// in custom.css. Transitions are off until the first theme has been applied, so
// that the page paints straight into the right colours rather than animating in
const THEME_TRANSITION_MS = 300;
let legworkThemeAnimates = false;
let legworkThemeTimer = null;

function storedLegworkTheme() {
    // storage can throw when cookies are blocked
    try {
        const theme = localStorage.getItem(LEGWORK_THEME_KEY);
        return theme === "light" || theme === "dark" ? theme : null;
    } catch (err) {
        return null;
    }
}

function setLegworkTheme(theme, remember) {
    if (legworkThemeAnimates && theme !== document.documentElement.getAttribute("data-theme")) {
        document.documentElement.classList.add("theme-switching");
        clearTimeout(legworkThemeTimer);   // restart the clock if the reader toggles again mid-fade
        legworkThemeTimer = setTimeout(function() {
            document.documentElement.classList.remove("theme-switching");
        }, THEME_TRANSITION_MS);
    }

    document.documentElement.setAttribute("data-theme", theme);
    if (document.body) {
        document.body.setAttribute("data-theme", theme);
    }

    if (remember) {
        try {
            localStorage.setItem(LEGWORK_THEME_KEY, theme);
        } catch (err) {
            // nothing to do, the choice just won't survive a page load
        }
    }

    const button = document.querySelector(".theme-toggle");
    if (button) {
        const label = "Switch to " + (theme === "dark" ? "light" : "dark") + " mode";
        button.setAttribute("aria-label", label);
        button.setAttribute("title", label);
    }
}

// apply the theme immediately: an explicit choice wins, otherwise follow the OS
setLegworkTheme(storedLegworkTheme()
                || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"), false);

// every change from here on is a real switch, so it should fade
legworkThemeAnimates = true;

// keep following the OS unless the reader has picked a theme themselves
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function(event) {
    if (storedLegworkTheme() === null) {
        setLegworkTheme(event.matches ? "dark" : "light", false);
    }
});

document.addEventListener("DOMContentLoaded", function() {
    // add the theme toggle to the bottom of the sidebar header
    const sidebar = document.querySelector(".wy-side-nav-search");
    if (sidebar) {
        const button = document.createElement("button");
        button.className = "theme-toggle";
        button.type = "button";
        button.innerHTML = `
            <svg class="only-light" viewBox="0 0 24 24" width="16" height="16" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
            </svg>
            <svg class="only-dark" viewBox="0 0 24 24" width="16" height="16" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="4"/>
                <path d="M12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/>
            </svg>
            <span class="only-light">Dark mode</span>
            <span class="only-dark">Light mode</span>`;

        button.addEventListener("click", function() {
            setLegworkTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark",
                            true);
        });

        sidebar.appendChild(button);
    }

    // mirror the theme onto <body> (and label the button) now that they exist
    setLegworkTheme(document.documentElement.getAttribute("data-theme"), false);

    // add links to nav boxes
    boxes = document.querySelectorAll(".toms-nav-container .box, .toms-nav-box");
    boxes.forEach(element => {
        element.addEventListener("click", function() {
            window.location.href = this.getAttribute("data-href");
        })
    });

    // fix no-title issues
    if (document.querySelector("title").innerText == "<no title> — LEGWORK  documentation") {
        document.querySelector("title").innerText == "LEGWORK"
        document.title = "LEGWORK";

        breadcrumbs = document.querySelectorAll(".wy-breadcrumbs li");
        breadcrumbs.forEach(el => {
            if (el.innerText == "<no title>") {
                el.innerText = "Home";
            }
        });
    }
})
