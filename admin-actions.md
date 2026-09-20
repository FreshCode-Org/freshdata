# FreshData GitHub Repository Administrative Actions

The following administrative actions require GitHub repository administrator privileges on [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata). These settings cannot be altered directly via Git commits.

---

## 1. Repository Details & Description

Navigate to the repository homepage `https://github.com/FreshCode-Org/freshdata` and click the **⚙️ (Edit repository details)** button on the top-right of the "About" box:

* **Description**:
  ```text
  Automated, explainable data cleaning for pandas and Polars — repair messy tabular data safely and see exactly what changed.
  ```
* **Website**:
  ```text
  https://freshcode-org.github.io/freshdata/
  ```

---

## 2. GitHub Repository Topics (Discoverability & Search Intent)

In the same "About" settings dialog, set the following **15 developer-search-intent topics** (replace any stale or generic tags):

```text
data-cleaning
data-quality
data-preprocessing
dataframe
pandas
polars
python
tabular-data
data-validation
etl
data-engineering
data-science
machine-learning
missing-values
outlier-detection
```

> **Why these tags**: Developers actively search GitHub for `data-cleaning`, `missing-values`, and `pandas` when looking for libraries to replace fragile manual preprocessing scripts.

---

## 3. GitHub Discussions Setup

Navigate to **Settings** $\rightarrow$ **General** $\rightarrow$ **Features** $\rightarrow$ Check **Discussions**.

Once enabled, navigate to the **Discussions** tab $\rightarrow$ **Categories** (click edit pencil) and configure these 6 official categories:

| Category | Format | Description |
|---|---|---|
| 📢 **Announcements** | Announcement (maintainers only) | Project releases, benchmarks, and community roadmap updates |
| 💬 **Q&A** | Q&A (mark answers) | Ask how to clean specific messy datasets or configure rules |
| 💡 **Ideas** | Open discussion | Share ideas for new cleaning heuristics, formats, or algorithms |
| 🙌 **Show and Tell** | Open discussion | Share datasets, pipelines, and projects you cleaned using FreshData |
| ⚡ **Benchmarks** | Open discussion | Share benchmark runs across different machines and dataset sizes |
| 🔌 **Integrations** | Open discussion | Discuss integrations with Polars, DuckDB, Airflow, dbt, and GX |

---

## 4. Public Issues & Pull Requests Settings

Ensure the following are enabled under **Settings** $\rightarrow$ **General**:

* **Issues**: Checked (Enabled)
* **Automatically delete head branches**: Checked (keeps repository clean after PR merges)
* **Squash and merge**: Allowed (recommended with PR title and commit description)
