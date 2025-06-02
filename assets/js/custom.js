document.addEventListener("DOMContentLoaded", function () {
    // Get all statements
    const statements = document.querySelectorAll(".statement");
    const statementCites = document.querySelectorAll(".cite-stmt");

    statementCites.forEach((cite) => {
        const targetId = cite.getAttribute("href").substring(1); // Remove #
        const target = document.getElementById(targetId);
        if (target) {
            // Get the index in the statement sequence (matches CSS counter)
            const index = Array.from(statements).indexOf(target) + 1;

            // Determine the statement type based on its classes
            let type = "Statement"; // Default
            if (target.classList.contains("definition")) {
                type = "Definition";
            } else if (target.classList.contains("lemma")) {
                type = "Lemma";
            } else if (target.classList.contains("corollary")) {
                type = "Corollary";
            } else if (target.classList.contains("proposition")) {
                type = "Proposition";
            }

            // Update citation text
            cite.textContent = `${type} ${index}`;
        }
    });
});

document.addEventListener("DOMContentLoaded", function () {
    // Get all statements
    const statements = document.querySelectorAll(".figure");
    const statementCites = document.querySelectorAll(".cite-fig");

    statementCites.forEach((cite) => {
        const targetId = cite.getAttribute("href").substring(1); // Remove #
        const target = document.getElementById(targetId);
        if (target) {
            // Get the index in the statement sequence (matches CSS counter)
            const index = Array.from(statements).indexOf(target) + 1;

            // Determine the statement type based on its classes
            let type = "Statement"; // Default
            if (target.classList.contains("figure")) {
                type = "Figure";
            }

            // Update citation text
            cite.textContent = `${type} ${index}`;
        }
    });
});