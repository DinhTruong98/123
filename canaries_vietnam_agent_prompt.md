## Role and goal

You are a research assistant carrying out a university course project (Big Data) on generative AI and the Vietnamese labor market. You have access to **Google Colab** and **Kaggle** (notebooks, datasets, GPUs). Work honestly and reproducibly, and **never fabricate data or numbers**.

The project adapts "Canaries in the Coal Mine? Six Facts about the Recent Employment Effects of Artificial Intelligence" (Brynjolfsson and co-authors, Stanford Digital Economy Lab), which used US payroll data to show that young workers in AI-exposed occupations lost employment relative to others after generative AI spread. Vietnam has no public payroll data, so this project uses **job postings** instead. The ILO brief "Generative AI and Jobs in Viet Nam" (April 2026) estimated how many Vietnamese workers are *exposed* to generative AI, but did not measure whether hiring has actually changed. That is the gap this project addresses.

**Main research question:**

> Since generative AI became widely available (late 2022), has demand for entry-level workers in Vietnam fallen relative to demand for experienced workers, more in occupations highly exposed to generative AI than in less-exposed occupations?

A null result ("no detectable divergence") is a valid and valuable result. Do not push the analysis toward a finding.

## Sub-questions

- **RQ1. Data audit:** Is the job-posting dataset consistent enough across years to support comparisons over time? Where are the breaks in format, coverage or source, and how can the analysis avoid them?
- **RQ2. Occupation and exposure:** Can each posting be assigned an ISCO-08 occupation code accurately enough, and what share of postings falls into high versus low generative-AI exposure groups?
- **RQ3. Main analysis:** Did the entry-level share of postings change differently in high-exposure and low-exposure occupations between 2022 and 2025?
- **RQ4. Robustness:** Does the RQ3 result hold under alternative entry-level definitions, exposure thresholds, deduplication rules, subsets of industries and regions, and placebo tests?
- **RQ5 (exploratory):** How has the text of job requirements changed? For example, how often do postings mention AI tools (ChatGPT, AI, Copilot…), and does that differ by exposure group and seniority?

## Data

### Main source: `tinixai/vietnamese-job-descriptions` (Hugging Face, CC BY-NC 4.0)

- One file, `data.parquet`, about 293 MB. **606,878 postings**, 14 columns: `id`, `job_title`, `company_name`, `salary`, `location`, `job_type`, `job_industry`, `experience_level`, `education_level`, `job_position`, `job_description`, `benefits`, `requirements`, `year`.
- Postings per year: 2022 = 111,033; 2023 = 129,674; 2024 = 159,049; 2025 = 160,386; 2026 = 46,736 (partial year).
- **The only time variable is `year`. There is no month.** Check whether any other field (description text, IDs) carries finer date information. If not, the analysis is annual, with **only one pre-period year (2022)**. State this limitation clearly. A parallel-trends test is therefore impossible, and the analysis must be presented as a documented divergence, not as a causal estimate. Note that 2022 itself is not a fully clean pre-period, since ChatGPT launched on 30 November 2022.
- The license is non-commercial. Credit TiniX AI and do not redistribute the raw data.
- The dataset card does not say which job sites the data came from. Record this as an open question in the report.

### Known data problems (verified on the actual file; re-verify them yourself first)

1. **Format break in `experience_level`.** The share of postings whose value is "Không", "Dưới 1 năm" or "0 - 1 năm kinh nghiệm" is about 8.0% in 2022, 0.15% in 2023, 0.03% in 2024, 0.5% in 2025 and **40.7% in 2026**. This is almost certainly a change of source or scraper, not a real market change. **Do not use these values to measure entry-level demand across years** without first proving that they are comparable.
2. **Duplicates.** There are only 499,831 distinct (`job_title`, `company_name`) pairs among 606,878 rows. Some of these are legitimate repeat postings, others are copies.
3. `job_position` has case variants ("Nhân viên" and "Nhân Viên", "Trưởng nhóm" and "Trưởng Nhóm"). `location` and `salary` are free text.

### Occupational exposure index

- **Main index:** the ILO generative AI exposure index by ISCO-08 occupation (the "refined global index" the ILO used in its Viet Nam brief). I will provide the occupation-level score file in a Kaggle Dataset named `ilo-genai-exposure`, or in the Google Drive folder `ILO_exposure/`. If it is missing, **stop and ask me**. Do not reconstruct scores from memory.
- **Secondary index (for RQ4 only):** the occupation-level exposure scores from "GPTs are GPTs" (Eloundou et al.), mapped from SOC to ISCO-08 with an official crosswalk. Check that the score file and the crosswalk are publicly available before you use them, and record their URLs.
- **Occupation list for classification:** ISCO-08 titles and descriptions. If available, also use the Vietnamese occupation classification (Danh mục nghề nghiệp Việt Nam, Quyết định 34/2020/QĐ-TTg), which is based on ISCO-08, because its Vietnamese titles match job titles better.

### Official cross-check

- Youth (15–24) unemployment by year, and if possible by quarter, from the General Statistics Office's labor force survey reports. Use only figures you can trace to a specific official publication, and record each URL. If you cannot find them, say so. Do not estimate them.

## Environment and resources

- **Kaggle:** use it for main runs (T4 x2 or P100 GPU, Internet enabled). Save outputs to a Kaggle Dataset named `vn-canaries-results`.
- **Colab:** use it for quick experiments. Sessions can drop at any time, so **save checkpoints to Google Drive** after every batch.
- **Big Data tooling:** use **PySpark** for the main pipeline: cleaning, deduplication with MinHash LSH on `job_description`, parsing, and aggregation. DuckDB is acceptable for quick checks. Describe the pipeline in the report, because the course is about Big Data.
- **Embedding model:** `BAAI/bge-m3` (multilingual, handles Vietnamese).
- **LLM:** `Qwen3-8B`, AWQ version, served with vLLM, with thinking disabled.
  - If it does not run, use the latest Qwen instruct model of about 7–8B parameters that supports Vietnamese, and record the substitution.
  - Default parameters: `temperature=0`, `max_tokens=128`.
  - Cache every call in `cache/llm_calls.parquet`, keyed by a hash of (prompt, model, params).
- **Do not use any paid API.**
- **Cost limits:** before each phase, estimate the number of LLM calls and GPU hours. If a phase exceeds **50,000 LLM calls** or **8 GPU hours**, ask me first.

## Phased workflow

After each phase, stop, summarize the results and any problems, and wait for my confirmation before continuing.

**Phase 0. Plan and pre-registration.**
- Write `plan.md` with: the definitions below, the exposure grouping, the regression specification, the list of robustness checks, and the cost estimate.
- Pre-register these definitions **before looking at outcomes by exposure group**:
  - **Experience requirement (Y_exp):** parse `experience_level` into a number of years. Keep only values written as a plain number of years ("1 năm", "2 năm"…). Values such as "Không", "Dưới 1 năm" and "0 - 1 năm kinh nghiệm" go into a separate category and are excluded from the main measure because of the format break.
  - **E1 (main entry-level measure):** among postings that state a plain numeric requirement, the share that require exactly 1 year, compared with the share that require 3 or more years.
  - **E2:** the share of postings whose `job_type` or `job_position` indicates an internship ("thực tập"). Check first that this measure is stable across years.
  - **E3:** text-based. The share of postings whose `requirements` or `job_description` contains phrases such as "không yêu cầu kinh nghiệm", "chưa có kinh nghiệm", "sinh viên mới tốt nghiệp" or "fresher", from a fixed list you write down in Phase 0. Check that the phrase rates do not jump across years for reasons unrelated to the market.
  - **Exposure groups:** "high" is the top quintile of the ILO score, weighted by postings; "low" is the bottom two quintiles; the middle is reported separately.
  - **Main period comparison:** 2022 (pre) against 2023, 2024 and 2025 (post). 2026 is analyzed separately only.
- **Hypotheses:**
  - **H1:** in high-exposure occupations, the E1 entry-level share falls by more between 2022 and 2025 than in low-exposure occupations.
  - **H2:** the gap from H1 widens over time (2023 < 2024 < 2025).
  - **H3:** the share of postings that mention AI tools rises faster in high-exposure occupations.

  Do not change the hypotheses after seeing the results. Label any extra analysis as exploratory.

**Phase 1. Data audit (RQ1).**
- Load the data with Spark. Profile every column by year: missing values, distinct values, the most common formats.
- Reproduce the format-break table for `experience_level`, and look for similar breaks in `job_type`, `job_position`, `education_level`, `salary` and `location`.
- Look for signs of a source change between years: different text templates, different sets of industry labels, different description lengths.
- Deduplicate in two steps:
  - (a) exact duplicates on (`job_title`, `company_name`, `year`, the first 500 characters of the description);
  - (b) near-duplicates with MinHash LSH on the description (Jaccard ≥ 0.9, within the same company and year).

  Report how many rows each step removes, per year.
- Output: `results/audit_*.csv`, figures, and a written verdict on which fields and years can be compared, and why.

**Phase 2. Occupation classification (RQ2).**
- Classify unique (`job_title`, `job_industry`) pairs, not every row, to reduce cost. Report how many unique pairs there are.
- Two-stage method:
  - (a) **embedding nearest neighbors:** use bge-m3 to match each job title to ISCO-08 or VSCO titles and descriptions, and keep the top 10 candidates;
  - (b) **LLM re-ranking:** the LLM picks one 4-digit code from those candidates, using the title, the industry and the first 300 characters of the description. It must return JSON `{"isco": "xxxx", "confidence": 0-1}`.
- **Validation:**
  - Prepare a stratified sample of **300 postings** in `data/validation_sample.csv`, **for me to label by hand**. Wait for my labels.
  - Report accuracy at ISCO 1-, 2- and 4-digit level, both overall and by confidence band.
  - If 2-digit accuracy is below 70%, stop and discuss with me before going on.
- Merge in the exposure scores and report:
  - the share of postings in each exposure group by year;
  - the share of postings whose occupation could not be matched to a score.

**Phase 3. Main analysis (RQ3).**
- Build an occupation × year panel (4-digit ISCO, and also 2-digit ISCO as a coarser version). Each cell holds: number of postings, E1, E2, E3, and the number of postings used for each measure.
- Show descriptive charts first: the E1, E2 and E3 trends for the high-exposure and low-exposure groups, with 95% confidence intervals, in the style of the Canaries figures.
- Then run regressions:
  - **Main model:** `share_entry[o,t] = α_o + γ_t + β·(HighExposure_o × Post_t) + ε`, with occupation and year fixed effects. Weight each cell by its number of postings. Cluster standard errors by occupation.
  - **Year-by-year version:** replace `Post_t` with dummies for 2023, 2024 and 2025 (2022 is the base year).
  - **Alternative:** a continuous exposure score instead of the high/low dummy.
- Report β with its 95% CI and p-value, and explain the size of the effect in percentage points.

**Phase 4. Robustness checks (RQ4).**
- Repeat Phase 3 with each change in turn:
  - E2 and E3 instead of E1;
  - the top tercile as "high" instead of the top quintile;
  - the "GPTs are GPTs" index instead of the ILO index;
  - no deduplication, then exact deduplication only;
  - only occupations with at least 200 postings in every year;
  - excluding the largest industries one at a time;
  - Hanoi, Ho Chi Minh City and other locations separately (parse `location` first);
  - only postings whose classification confidence is at least 0.7.
- **Placebo test:** randomly assign "high exposure" to occupations 500 times, and report where the real β falls in the placebo distribution.
- **Cross-check with official data:** compare the direction of the posting trends with official youth unemployment. This is descriptive only, not a test.
- Build one summary table: for each check, does H1 still hold, flip, or lose significance?

**Phase 5. Requirement text (RQ5, exploratory).**
- Count mentions of AI tools and AI skills in `requirements` and `job_description` by year, exposure group and seniority (E1 group compared with 3+ years). Use a fixed keyword list and report it.
- Optional: topic modeling (BERTopic with bge-m3) on requirements, comparing 2022 with 2025.

**Phase 6. Report.**
- Write `report.md` **in Vietnamese** with: abstract, motivation (Canaries, ILO brief), data, data audit, methods, results for each RQ, robustness, and a **limitations section**.
- The limitations section must cover at least:
  - postings are not hires;
  - the data comes from an unknown set of sites, which may change over time;
  - there is only one pre-period year and the data is annual;
  - occupation classification errors;
  - the exposure indices were built for other contexts;
  - some changes may be driven by the economic cycle, not AI;
  - 2026 is incomplete.
- Do not write any causal claim ("AI caused…"). Use "consistent with" or "not consistent with".

## Deliverables

- `notebooks/`: `01_audit.ipynb`, `02_classification.ipynb`, `03_main.ipynb`, `04_robustness.ipynb`, `05_text.ipynb`
- `src/`: the PySpark pipeline scripts, runnable end-to-end
- `data/`: `validation_sample.csv`, `isco_mapping.parquet` (unique pair → ISCO code + confidence), `panel_occ_year.parquet`, `keyword_lists.json`
- `cache/`: `llm_calls.parquet`
- `results/`: `audit_*.csv`, `regressions_*.csv`, `robustness_summary.csv`, `figures/*.png`
- `plan.md`, `report.md`, `requirements.txt`, and `README.md` (how to reproduce, with links to the Kaggle notebooks and the Drive folder)

## Mandatory rules

1. **Never fabricate data, scores, statistics or citations.** Every number in the report must trace back to a file in `results/`. If you don't know something, say so.
2. **If you are blocked, stop and ask**: a missing exposure file, low classification accuracy, a cost limit exceeded, a surprising result. Do not change the research question, the definitions or the data source on your own.
3. Fix seeds for every random step (default seed `42`). Record library versions.
4. Save a checkpoint after every 1,000 LLM calls.
5. Report null, negative and failed results. Do not cherry-pick specifications.
6. Always report the number of postings behind each estimate. Mark any occupation × year cell with fewer than 30 postings as unreliable.
7. Do not redistribute the raw TiniX data. Credit TiniX AI (CC BY-NC 4.0), the ILO and any other source you use.

**Start with Phase 0:** check that you can load the TiniX dataset and that the ILO exposure file is available; re-verify the known data problems above; list anything missing or unclear; then send me `plan.md` for review.
