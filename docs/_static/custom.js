document.addEventListener("DOMContentLoaded", function() {
    // add links to nav boxes
    boxes = document.querySelectorAll(".toms-nav-container .box");
    console.log(boxes)
    boxes.forEach(element => {
        console.log(element);
        element.addEventListener("click", function() {
            console.log(this);
            window.location.href = this.getAttribute("data-href");
        })
    });

    // fix no-title issues
    if (document.querySelector("title").innerText == "<no title> — LEGWORK  documentation") {
        document.querySelector("title").innerText == "Home — LEGWORK  documentation"
        document.title = "Home — LEGWORK  documentation";

        breadcrumbs = document.querySelectorAll(".wy-breadcrumbs li");
        breadcrumbs.forEach(el => {
            if (el.innerText == "<no title>") {
                el.innerText = "Home";
            }
        });
    }
})