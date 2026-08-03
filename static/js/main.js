document.addEventListener("DOMContentLoaded", () => {
    // ----------------------------------------------------
    // 1. Navigation Tab Switching
    // ----------------------------------------------------
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-tab");
            const targetContent = document.getElementById(targetId);
            if (targetContent) targetContent.classList.add("active");
        });
    });

    // ----------------------------------------------------
    // 2. Preset Job Postings Samples
    // ----------------------------------------------------
    const fakePresetBtn = document.getElementById("load-fake-preset");
    const realPresetBtn = document.getElementById("load-real-preset");

    const sampleFakeJob = {
        title: "Data Entry Clerk - Earn $5000/Week Remote - Immediate Hire!",
        company_profile: "",
        description: "Urgent opening for remote Data Entry & Wire Transfer Coordinators! Work just 2 hours daily from home processing transactions and cashing test checks. High weekly payout guaranteed. No interview or prior experience needed. Equipment fee will be reimbursed upon start.",
        requirements: "Must have active bank account for direct deposits and wire transfers via Western Union or Bitcoin. Instant start.",
        benefits: "Earn up to $10,000 monthly, instant cash bonuses, work from couch.",
        salary_range: "5000-10000",
        telecommuting: true,
        has_company_logo: false,
        has_questions: false
    };

    const sampleRealJob = {
        title: "Senior Full Stack Software Engineer (Python / React)",
        company_profile: "Acme Cloud Systems is a enterprise infrastructure provider delivering scalable microservice architectures for Fortune 500 organizations worldwide.",
        description: "We are looking for a Senior Full Stack Engineer to lead backend microservice development and interactive dashboard features. You will collaborate with product designers, implement REST APIs in Python, and maintain high code quality standards.",
        requirements: "Bachelor's degree in Computer Science or equivalent. 5+ years experience with Python, React, PostgreSQL, Docker, and CI/CD pipelines.",
        benefits: "Competitive salary, 401(k) 5% match, comprehensive medical/dental insurance, flexible remote work allowance, $2000 annual learning budget.",
        salary_range: "120000-150000",
        telecommuting: true,
        has_company_logo: true,
        has_questions: true
    };

    function populateForm(data) {
        document.getElementById("job-title").value = data.title;
        document.getElementById("company-profile").value = data.company_profile;
        document.getElementById("job-description").value = data.description;
        document.getElementById("job-requirements").value = data.requirements;
        document.getElementById("job-benefits").value = data.benefits;
        document.getElementById("salary-range").value = data.salary_range;

        document.getElementById("telecommuting").checked = data.telecommuting;
        document.getElementById("has-company-logo").checked = data.has_company_logo;
        document.getElementById("has-questions").checked = data.has_questions;
    }

    if (fakePresetBtn) fakePresetBtn.addEventListener("click", () => populateForm(sampleFakeJob));
    if (realPresetBtn) realPresetBtn.addEventListener("click", () => populateForm(sampleRealJob));

    // ----------------------------------------------------
    // 3. Live Prediction API Call
    // ----------------------------------------------------
    const form = document.getElementById("prediction-form");
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContent = document.getElementById("result-content");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const btnPredict = document.getElementById("btn-predict");
        btnPredict.disabled = true;
        btnPredict.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Posting...`;

        const payload = {
            title: document.getElementById("job-title").value,
            company_profile: document.getElementById("company-profile").value,
            description: document.getElementById("job-description").value,
            requirements: document.getElementById("job-requirements").value,
            benefits: document.getElementById("job-benefits").value,
            salary_range: document.getElementById("salary-range").value,
            telecommuting: document.getElementById("telecommuting").checked ? 1 : 0,
            has_company_logo: document.getElementById("has-company-logo").checked ? 1 : 0,
            has_questions: document.getElementById("has-questions").checked ? 1 : 0
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            if (data.error) {
                alert("Error analyzing posting: " + data.error);
                return;
            }

            renderAssessmentResult(data);

        } catch (err) {
            console.error(err);
            alert("Failed to connect to detection backend.");
        } finally {
            btnPredict.disabled = false;
            btnPredict.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Analyze Job Posting`;
        }
    });

    function renderAssessmentResult(data) {
        if (resultPlaceholder) resultPlaceholder.classList.add("hidden");
        if (resultContent) resultContent.classList.remove("hidden");

        const riskPercent = data.risk_score_percent;
        const isFraud = data.is_fraudulent;

        // Gauge Ring Color Gradient Update
        const gaugeRing = document.getElementById("gauge-ring");
        const riskVal = document.getElementById("risk-score-val");
        riskVal.textContent = `${riskPercent}%`;

        let color = "#10b981"; // Emerald
        if (riskPercent > 60) color = "#f43f5e"; // Rose
        else if (riskPercent > 30) color = "#f59e0b"; // Amber

        gaugeRing.style.background = `conic-gradient(${color} ${riskPercent * 3.6}deg, rgba(255,255,255,0.05) 0deg)`;

        // Status Badge
        const statusBadge = document.getElementById("status-badge");
        const riskTag = document.getElementById("risk-level-tag");

        statusBadge.textContent = data.status;
        statusBadge.className = `status-badge ${isFraud ? 'fake' : 'real'}`;
        riskTag.textContent = data.risk_level;

        // Flagged Triggers
        const triggersList = document.getElementById("triggers-list");
        triggersList.innerHTML = "";

        if (data.detected_triggers && data.detected_triggers.length > 0) {
            data.detected_triggers.forEach(trig => {
                const badge = document.createElement("span");
                badge.className = "trigger-badge";
                badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${trig}`;
                triggersList.appendChild(badge);
            });
        } else {
            triggersList.innerHTML = `<span class="text-muted" style="font-size:0.8rem;">No high-risk scam triggers flagged.</span>`;
        }

        // Decision Explanation Factors
        const factorsList = document.getElementById("factors-list");
        factorsList.innerHTML = "";

        if (data.risk_factors && data.risk_factors.length > 0) {
            data.risk_factors.forEach(factor => {
                const li = document.createElement("li");
                li.textContent = factor;
                factorsList.appendChild(li);
            });
        }
    }

    // ----------------------------------------------------
    // 4. EDA Analytics & Charts Integration
    // ----------------------------------------------------
    const btnLoadEda = document.getElementById("btn-load-eda");
    let edaLoaded = false;

    if (btnLoadEda) {
        btnLoadEda.addEventListener("click", () => {
            if (!edaLoaded) {
                fetchEdaStats();
                edaLoaded = true;
            }
        });
    }

    async function fetchEdaStats() {
        try {
            const res = await fetch("/api/eda-stats");
            const data = await res.json();

            // Counters
            document.getElementById("eda-total-postings").textContent = data.total_postings.toLocaleString();
            document.getElementById("eda-fraud-percent").textContent = `${data.class_distribution.fraud_percentage}%`;
            document.getElementById("eda-real-count").textContent = data.class_distribution.legitimate.toLocaleString();

            // Chart 1: Class Distribution
            const ctx1 = document.getElementById("chart-class-dist").getContext("2d");
            new Chart(ctx1, {
                type: "doughnut",
                data: {
                    labels: ["Legitimate Jobs", "Fraudulent Jobs"],
                    datasets: [{
                        data: [data.class_distribution.legitimate, data.class_distribution.fraudulent],
                        backgroundColor: ["#10b981", "#f43f5e"],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: "#9ca3af" } } }
                }
            });

            // Chart 2: Missing Metadata Analysis
            const ctx2 = document.getElementById("chart-missing-meta").getContext("2d");
            const meta = data.missing_features_analysis;
            new Chart(ctx2, {
                type: "bar",
                data: {
                    labels: ["Company Profile Missing", "Company Logo Missing"],
                    datasets: [
                        {
                            label: "Fraudulent Jobs",
                            data: [meta.missing_company_profile.fraudulent_lacking_profile, meta.missing_company_logo.fraudulent_lacking_logo],
                            backgroundColor: "#f43f5e"
                        },
                        {
                            label: "Legitimate Jobs",
                            data: [meta.missing_company_profile.legitimate_lacking_profile, meta.missing_company_logo.legitimate_lacking_logo],
                            backgroundColor: "#10b981"
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: "#9ca3af" } } },
                    scales: {
                        x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } },
                        y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } }
                    }
                }
            });

            // Chart 3: Fraudulent Industries
            const ctx3 = document.getElementById("chart-fraud-industries").getContext("2d");
            const indData = data.top_fraudulent_industries;
            new Chart(ctx3, {
                type: "bar",
                data: {
                    labels: Object.keys(indData),
                    datasets: [{
                        label: "Fraudulent Job Count",
                        data: Object.values(indData),
                        backgroundColor: "#06b6d4",
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: "y",
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } },
                        y: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } }
                    }
                }
            });

        } catch (err) {
            console.error("Failed to load EDA stats:", err);
        }
    }

    // ----------------------------------------------------
    // 5. Model Benchmarks & Indicators
    // ----------------------------------------------------
    const btnLoadModels = document.getElementById("btn-load-models");
    let modelsLoaded = false;

    if (btnLoadModels) {
        btnLoadModels.addEventListener("click", () => {
            if (!modelsLoaded) {
                fetchModelPerformance();
                fetchFeatureIndicators();
                modelsLoaded = true;
            }
        });
    }

    async function fetchModelPerformance() {
        try {
            const res = await fetch("/api/model-performance");
            const data = await res.json();

            const tbody = document.getElementById("models-table-body");
            tbody.innerHTML = "";

            if (data.models) {
                Object.entries(data.models).forEach(([name, m]) => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><strong>${name}</strong></td>
                        <td>${m.accuracy}</td>
                        <td>${m.precision}</td>
                        <td><span style="color:#f43f5e; font-weight:700;">${m.recall}</span></td>
                        <td><span style="color:#10b981; font-weight:700;">${m.f1_score}</span></td>
                        <td>${m.roc_auc}</td>
                    `;
                    tbody.appendChild(tr);
                });
            }

            // Ablation Summary
            const expContainer = document.getElementById("ablation-experiments-container");
            if (data.experiments && expContainer) {
                expContainer.innerHTML = "";
                Object.entries(data.experiments).forEach(([expName, expVal]) => {
                    const box = document.createElement("div");
                    box.style.marginBottom = "1rem";
                    box.style.padding = "0.75rem";
                    box.style.background = "rgba(255,255,255,0.03)";
                    box.style.borderRadius = "8px";

                    let details = "";
                    Object.entries(expVal).forEach(([k, val]) => {
                        details += `<div style="font-size:0.85rem; color:#9ca3af;">• <strong>${k.replace(/_/g, ' ')}</strong>: F1 = ${val.f1_score}, Recall = ${val.recall}, Acc = ${val.accuracy}</div>`;
                    });

                    box.innerHTML = `
                        <h4 style="font-size:0.95rem; color:#06b6d4; margin-bottom:0.4rem;">${expName.replace(/_/g, ' ')}</h4>
                        ${details}
                    `;
                    expContainer.appendChild(box);
                });
            }

        } catch (err) {
            console.error("Failed to load model performance:", err);
        }
    }

    async function fetchFeatureIndicators() {
        try {
            const res = await fetch("/api/feature-indicators");
            const data = await res.json();

            const fraudList = document.getElementById("list-fraud-indicators");
            const realList = document.getElementById("list-real-indicators");

            if (fraudList && data.top_fraudulent_indicators) {
                fraudList.innerHTML = "";
                data.top_fraudulent_indicators.forEach(term => {
                    const li = document.createElement("li");
                    li.textContent = term;
                    fraudList.appendChild(li);
                });
            }

            if (realList && data.top_legitimate_indicators) {
                realList.innerHTML = "";
                data.top_legitimate_indicators.forEach(term => {
                    const li = document.createElement("li");
                    li.textContent = term;
                    realList.appendChild(li);
                });
            }

        } catch (err) {
            console.error("Failed to load feature indicators:", err);
        }
    }

    // ----------------------------------------------------
    // 6. Batch CSV Upload
    // ----------------------------------------------------
    const batchForm = document.getElementById("batch-upload-form");
    const csvInput = document.getElementById("batch-csv-input");
    const fileNameDisplay = document.getElementById("file-name-display");

    if (csvInput) {
        csvInput.addEventListener("change", (e) => {
            if (e.target.files.length > 0) {
                fileNameDisplay.textContent = `Selected: ${e.target.files[0].name}`;
            }
        });
    }

    if (batchForm) {
        batchForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (!csvInput.files[0]) return;

            const btn = document.getElementById("btn-process-batch");
            btn.disabled = true;
            btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing CSV File...`;

            const formData = new FormData();
            formData.append("file", csvInput.files[0]);

            try {
                const res = await fetch("/api/batch-predict", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();

                if (data.error) {
                    alert("Error: " + data.error);
                    return;
                }

                // Render Batch Card
                const batchCard = document.getElementById("batch-results-card");
                batchCard.classList.remove("hidden");

                document.getElementById("batch-total-scanned").textContent = data.total_scanned;
                document.getElementById("batch-flagged-count").textContent = data.flagged_fraudulent;
                document.getElementById("batch-fraud-rate").textContent = `${data.fraud_rate_percent}%`;

                const tbody = document.getElementById("batch-table-body");
                tbody.innerHTML = "";

                data.results.forEach(row => {
                    const tr = document.createElement("tr");
                    const isFraud = row.prediction === "FRAUDULENT";
                    tr.innerHTML = `
                        <td>${row.index}</td>
                        <td>${row.title}</td>
                        <td><span class="status-badge ${isFraud ? 'fake' : 'real'}" style="font-size:0.75rem; padding:0.2rem 0.6rem;">${row.prediction}</span></td>
                        <td>${row.fraud_probability}</td>
                        <td><strong style="color:${isFraud ? '#f43f5e' : '#10b981'}">${row.risk_score}%</strong></td>
                    `;
                    tbody.appendChild(tr);
                });

            } catch (err) {
                console.error("Batch processing failed:", err);
                alert("Batch processing failed. Make sure the CSV file has valid column names.");
            } finally {
                btn.disabled = false;
                btn.innerHTML = `<i class="fa-solid fa-gears"></i> Process Batch CSV`;
            }
        });
    }
});
