// Populate repo cards from the GitHub API: description, language (with a
// colored dot), and star count. Only the owner/name slug is stored in
// _data/repositories.yml; everything else is fetched live here.
//
// Unauthenticated GitHub API allows 60 requests/hour per IP — plenty for a
// personal site. The repo name and link are rendered server-side, so if a
// request fails (offline, rate limited, repo renamed) the card still works;
// it just won't show the description, language, or stars.
document.addEventListener("DOMContentLoaded", function () {
  // GitHub Linguist language colors (the repos endpoint doesn't return these).
  // Add more as needed: https://github.com/github-linguist/linguist/blob/main/lib/linguist/languages.yml
  const LANGUAGE_COLORS = {
    Python: "#3572A5",
    "Jupyter Notebook": "#DA5B0B",
    JavaScript: "#F1E05A",
    TypeScript: "#3178C6",
    C: "#555555",
    "C++": "#F34B7D",
    Java: "#B07219",
    Go: "#00ADD8",
    Rust: "#DEA584",
    Julia: "#A270BA",
    R: "#198CE7",
    Shell: "#89E051",
    HTML: "#E34C26",
    CSS: "#563D7C",
    TeX: "#3D6117",
  };

  const cards = document.querySelectorAll(".repo-card[data-repo]");

  cards.forEach(function (card) {
    const repo = card.getAttribute("data-repo");
    if (!repo) return;

    fetch("https://api.github.com/repos/" + repo)
      .then(function (response) {
        if (!response.ok) throw new Error("GitHub API error: " + response.status);
        return response.json();
      })
      .then(function (data) {
        // Description
        if (data.description) {
          const desc = card.querySelector(".repo-card-description");
          desc.textContent = data.description;
          desc.hidden = false;
        }

        // Language + colored dot
        if (data.language) {
          const lang = card.querySelector(".repo-card-lang");
          card.querySelector(".repo-lang-name").textContent = data.language;
          card.querySelector(".repo-lang-dot").style.backgroundColor =
            LANGUAGE_COLORS[data.language] || "#858585";
          lang.hidden = false;
        }

        // Stars
        if (typeof data.stargazers_count === "number") {
          const stars = card.querySelector(".repo-stars");
          stars.querySelector(".repo-stars-count").textContent =
            data.stargazers_count.toLocaleString();
          stars.hidden = false;
        }
      })
      .catch(function (error) {
        console.warn("Could not load repo data for " + repo + ":", error);
      });
  });
});
