document.addEventListener("DOMContentLoaded", function() {
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