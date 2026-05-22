const selectedTitles = new Map();

const searchInput = document.getElementById("search-input");
const searchButton = document.getElementById("search-button");
const clearButton = document.getElementById("clear-button");
const recommendButton = document.getElementById("recommend-button");
const searchResults = document.getElementById("search-results");
const selectedMovies = document.getElementById("selected-movies");
const recommendations = document.getElementById("recommendations");
const statusMessage = document.getElementById("status-message");

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function setStatus(message) {
  statusMessage.textContent = message || "";
}

function renderSelected() {
  if (selectedTitles.size === 0) {
    selectedMovies.className = "chip-list empty";
    selectedMovies.innerHTML = "<span>아직 선택된 영화가 없습니다.</span>";
    return;
  }

  selectedMovies.className = "chip-list";
  selectedMovies.innerHTML = "";
  for (const [title, item] of selectedTitles.entries()) {
    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `
      <span>${item.title} (${item.year})</span>
      <button type="button" aria-label="remove">×</button>
    `;
    chip.querySelector("button").addEventListener("click", () => {
      selectedTitles.delete(title);
      renderSelected();
    });
    selectedMovies.appendChild(chip);
  }
}

function renderSearchResults(items) {
  searchResults.innerHTML = "";
  if (items.length === 0) {
    searchResults.innerHTML = "<p class=\"status-message\">검색 결과가 없습니다.</p>";
    return;
  }

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "result-item";
    row.innerHTML = `
      <div class="movie-meta">
        <strong>${item.title}</strong>
        <span>${item.year} · 평점 ${item.avg_rating} · 평가 ${item.rating_count.toLocaleString()}개</span>
      </div>
      <button class="add-button" type="button">추가</button>
    `;
    row.querySelector("button").addEventListener("click", () => {
      selectedTitles.set(item.title, item);
      renderSelected();
      setStatus(`${item.title}을(를) 선택했습니다.`);
    });
    searchResults.appendChild(row);
  });
}

function renderRecommendations(items) {
  if (items.length === 0) {
    recommendations.className = "recommendations empty-state";
    recommendations.innerHTML = "<p>추천 결과가 없습니다. 다른 영화를 조합해 보세요.</p>";
    return;
  }

  recommendations.className = "recommendations";
  recommendations.innerHTML = "";
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    card.innerHTML = `
      <strong>${item.title}</strong>
      <div class="year">${item.year}</div>
      <div class="badge-row">
        <span class="badge">평점 ${item.avg_rating}</span>
        <span class="badge">평가 ${item.rating_count.toLocaleString()}개</span>
        <span class="badge">유사취향 ${item.support.toLocaleString()}명</span>
      </div>
    `;
    recommendations.appendChild(card);
  });
}

async function runSearch() {
  const query = searchInput.value.trim();
  if (!query) {
    setStatus("검색어를 입력하세요.");
    return;
  }

  setStatus("영화를 찾는 중입니다...");
  try {
    const items = await fetchJson(`/api/search?q=${encodeURIComponent(query)}`);
    renderSearchResults(items);
    setStatus(items.length ? "검색 결과를 불러왔습니다." : "검색 결과가 없습니다.");
  } catch (error) {
    setStatus("검색 중 오류가 발생했습니다.");
  }
}

async function runRecommendation() {
  const titles = Array.from(selectedTitles.keys());
  if (titles.length === 0) {
    setStatus("좋아하는 영화를 먼저 선택하세요.");
    return;
  }

  setStatus("추천 결과를 계산하는 중입니다...");
  try {
    const result = await fetchJson("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ titles, limit: 12 }),
    });
    renderRecommendations(result.recommendations || []);
    setStatus(result.message || "추천이 완료되었습니다.");
  } catch (error) {
    setStatus("추천 중 오류가 발생했습니다. 데이터셋 설정을 확인하세요.");
  }
}

searchButton.addEventListener("click", runSearch);
recommendButton.addEventListener("click", runRecommendation);
clearButton.addEventListener("click", () => {
  selectedTitles.clear();
  renderSelected();
  setStatus("선택한 영화를 초기화했습니다.");
});

searchInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    runSearch();
  }
});

renderSelected();
