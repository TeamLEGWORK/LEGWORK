document.addEventListener("DOMContentLoaded", function() {
    boxes = document.querySelectorAll(".container .box");
    console.log(boxes)
    boxes.forEach(element => {
        console.log(element);
        element.addEventListener("click", function() {
            console.log(this);
            window.location.href = this.getAttribute("data-href");
        })
    });
})