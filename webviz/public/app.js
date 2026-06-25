const GENRE_COLORS = {
  fantasy: "#8b5cf6",
  mystery: "#ef4444",
  history: "#f59e0b",
  ya: "#22d3ee",
  romance: "#ec4899",
  multi: "#a3a3a3",
  other: "#64748b",
};

fetch("graph_viz.json")
  .then((r) => r.json())
  .then(render);

function render(data) {
  const { nodes, edges, meta } = data;
  const maxPagerank = Math.max(...nodes.map((n) => n.pagerank));

  const cy = cytoscape({
    container: document.getElementById("cy"),
    elements: [
      ...nodes.map((n) => ({
        data: { id: n.id, ...n },
      })),
      ...edges.map((e) => ({
        data: { id: `${e.source}-${e.target}`, ...e },
      })),
    ],
    style: [
      {
        selector: "node",
        style: {
          "background-color": (n) => GENRE_COLORS[n.data("genre")] || GENRE_COLORS.other,
          width: (n) => 8 + 40 * (n.data("pagerank") / maxPagerank),
          height: (n) => 8 + 40 * (n.data("pagerank") / maxPagerank),
          label: (n) => (n.data("pagerank") / maxPagerank > 0.25 ? n.data("title") : ""),
          "font-size": 8,
          color: "#cbd5e1",
          "text-outline-width": 0,
          "text-valign": "bottom",
          "text-margin-y": 4,
        },
      },
      {
        selector: "edge",
        style: {
          width: (e) => Math.max(0.5, Math.min(4, e.data("pmi"))),
          "line-color": "#3a4154",
          "curve-style": "haystack",
          opacity: 0.5,
        },
      },
      { selector: "node:selected", style: { "background-color": "#fde047", "border-width": 2, "border-color": "#fff" } },
      { selector: ".dimmed", style: { opacity: 0.08 } },
    ],
    layout: { name: "cose", animate: false, nodeRepulsion: 8000, idealEdgeLength: 60 },
    minZoom: 0.1,
    maxZoom: 5,
  });

  // --- info panel ---
  const info = document.getElementById("info");
  function showInfo(n) {
    info.innerHTML = `
      <b>${n.title}</b><br/>
      genero: ${n.genre}<br/>
      rating promedio: ${n.average_rating?.toFixed(2) ?? "n/a"} (${n.ratings_count.toLocaleString()} ratings)<br/>
      grado: ${n.degree} · grado ponderado: ${n.weighted_degree.toFixed(2)}<br/>
      PageRank: ${n.pagerank.toExponential(3)}<br/>
      betweenness: ${n.betweenness_centrality.toExponential(3)}<br/>
      componente: #${n.component_id} (tamaño ${n.component_size.toLocaleString()})
    `;
  }
  cy.on("tap", "node", (evt) => showInfo(evt.target.data()));

  // --- search ---
  const searchInput = document.getElementById("search");
  const suggestions = document.getElementById("suggestions");
  searchInput.addEventListener("input", () => {
    const q = searchInput.value.trim().toLowerCase();
    suggestions.innerHTML = "";
    if (!q) return;
    const matches = nodes.filter((n) => n.title?.toLowerCase().includes(q)).slice(0, 15);
    for (const m of matches) {
      const div = document.createElement("div");
      div.className = "suggestion";
      div.textContent = m.title;
      div.onclick = () => {
        const el = cy.getElementById(m.id);
        cy.elements().unselect();
        el.select();
        cy.animate({ center: { eles: el }, zoom: 2.5 }, { duration: 300 });
        showInfo(el.data());
        searchInput.value = m.title;
        suggestions.innerHTML = "";
      };
      suggestions.appendChild(div);
    }
  });

  // --- PMI filter ---
  const pmiSlider = document.getElementById("pmiSlider");
  const pmiValue = document.getElementById("pmiValue");
  pmiSlider.addEventListener("input", () => {
    const threshold = parseFloat(pmiSlider.value);
    pmiValue.textContent = threshold.toFixed(1);
    cy.edges().forEach((e) => {
      e.style("display", e.data("pmi") >= threshold ? "element" : "none");
    });
  });

  // --- genre filter ---
  const genreFilter = document.getElementById("genreFilter");
  const genres = [...new Set(nodes.map((n) => n.genre))].sort();
  for (const g of genres) {
    const opt = document.createElement("option");
    opt.value = g;
    opt.textContent = g;
    genreFilter.appendChild(opt);
  }
  genreFilter.addEventListener("change", () => {
    const g = genreFilter.value;
    cy.nodes().forEach((n) => {
      n.style("display", g === "all" || n.data("genre") === g ? "element" : "none");
    });
    cy.edges().forEach((e) => {
      const visible = e.source().style("display") !== "none" && e.target().style("display") !== "none";
      e.style("display", visible ? "element" : "none");
    });
  });

  // --- legend ---
  const legend = document.getElementById("legend");
  legend.innerHTML =
    "<div style='color:#b6bdcc;margin-bottom:4px;'>Color = género dominante · tamaño = PageRank</div>" +
    genres
      .map((g) => `<div class="legend-item"><span class="dot" style="background:${GENRE_COLORS[g] || GENRE_COLORS.other}"></span>${g}</div>`)
      .join("");

  // --- deliverable comparison stats ---
  const metaDiv = document.getElementById("meta");
  const cmp = meta.popularity_comparison;
  const diag = meta.full_graph_diagnostics;
  metaDiv.textContent =
    `Vista: ${meta.selection}\n\n` +
    `Grafo completo: ${diag.n_nodes.toLocaleString()} nodos, ${diag.n_edges.toLocaleString()} aristas\n` +
    `Componentes: ${diag.n_components.toLocaleString()} (mayor: ${(diag.largest_component_fraction * 100).toFixed(1)}%)\n` +
    `Nodos aislados: ${diag.isolated_node_count.toLocaleString()}\n\n` +
    `PageRank vs popularidad (Spearman): ${cmp.pagerank_vs_popularity_spearman.toFixed(3)}\n` +
    `Solapamiento top-${cmp.k}: ${(cmp.pagerank_top_k_overlap * 100).toFixed(1)}%`;
}
