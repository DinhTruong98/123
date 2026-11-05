## Role and goal

You are a research assistant carrying out a university course project on multi-agent social simulation with LLMs. You have access to **Google Colab** and **Kaggle** (notebooks, datasets, GPUs). Work honestly and reproducibly, and **never fabricate data or numbers**.

**Main research question:**

> How reliable is a multi-agent LLM simulation that uses Vietnamese personas (Nemotron-Personas-Vietnam), and where does it deviate from reality?

This study does **not** try to predict how Vietnamese society would react to anything. Its goal is to **measure the reliability of the simulation method itself**. A negative result (the simulation deviates a lot) is a valid and valuable result.

## Sub-questions

- **RQ1. Individual level, against a real survey:** When agents answer survey questions, does the distribution of their answers by age group, sex, education, region and urban/rural match real survey data from Vietnamese respondents? Which groups deviate the most?
- **RQ2. Persona fidelity:** From an agent's answers, can a separate model infer the persona's attributes (age group, education, region, urban/rural)? Do personas actually change agent behavior, or do all agents answer alike?
- **RQ3. Interaction level:** When agents discuss topics on a simple social network, do the opinion dynamics reproduce known sociological patterns (homophily, convergence or polarization)? How do they compare with a classical opinion-dynamics model?
- **RQ4. Robustness:** Do the conclusions of RQ1–RQ3 hold when the random seed, the prompt wording or the LLM is changed?

## Data

### Personas: `nvidia/Nemotron-Personas-Vietnam` (Hugging Face, CC BY 4.0)

- 1,000,000 rows, 21 columns, about 2.25 GB of parquet. Read the parquet files and **load only the columns you need** to save RAM. For sampling, first load `uuid` and the demographic columns, then fetch the text columns only for the sampled rows.
- Demographic columns: `sex`, `age` (18+), `marital_status`, `education_level` (7 levels), `occupation` (144 occupations), `zone` (Đô Thị/Nông Thôn, i.e. urban/rural), `region` (6 socio-economic regions), `country`.
- Text columns (all in Vietnamese): `persona`, `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, `cultural_background`, `skills_and_expertise(_list)`, `hobbies_and_interests(_list)`, `career_goals_and_ambitions`.
- **There is no province column and no name column.** Names and provinces appear only inside the text. This study analyzes at the region level (`region`) and does not need a province variable. Only if it becomes truly necessary, extract the province from the text using the list of the 34 provinces/municipalities in force after the 2025 merger; if you do, report the extraction rate and manually check 50 samples.
- This is **synthetic data**. The demographic attributes were sampled from a probabilistic graphical model fitted to official statistics from Vietnam's General Statistics Office. The text was written by the LLM `SaoLa-Small-32B`. Known limitations:
  - Elementary occupations make up only 5.1% of employed personas, versus 24.6% in reality.
  - When occupations were assigned, sex, age and education were assumed to act independently.
  - The text is fairly formulaic.

  **Never use this dataset as ground truth.**

### Ground truth: World Values Survey Wave 7, Vietnam sample

- I will upload the microdata file and the codebook to a Kaggle Dataset named `wvs-wave7-vietnam`, or to the Google Drive folder `WVS7_Vietnam/`. If you cannot find them, **stop and ask me**. Do not look for other sources, and do not use remembered or guessed figures.
- Do not use PAPI or any other source within this project.
- **Select 10 questions.** Each question must:
  - (a) be about attitudes or values;
  - (b) use a rating scale or closed answer options;
  - (c) not be politically sensitive.

  Prefer the topics below. The question codes in parentheses are approximate; **verify each one against the codebook**:
  - how important family, friends, leisure time, work and religion are in life (around Q1–Q6);
  - feeling of happiness (around Q46) and self-rated health (around Q47);
  - satisfaction with life (around Q49);
  - trust in other people (around Q57);
  - one or two questions on gender roles, e.g. "men make better political leaders than women do" (around Q29).

  If a question is missing from the Vietnam sample, or more than 20% of its answers are missing, replace it with another question on the same topic and record why.
- **Variable harmonization:** Recode the WVS demographic variables to match the persona dataset:
  - sex;
  - age, grouped into 18–29, 30–44, 45–59 and 60+;
  - education, collapsed into 3–4 levels shared by both sources;
  - urban/rural;
  - region, if WVS has a region variable; if not, drop this dimension and say so.

  Save the mapping table to `data/variable_mapping.csv`. Use the WVS weight variable when computing the real distributions.

## Environment and resources

- **Kaggle:** use it for the main runs, with a T4 x2 or P100 GPU and Internet enabled in Settings. Save results as notebook output and in a separate Kaggle Dataset named `persona-sim-results`. Store the HF token in Kaggle Secrets. Check the remaining weekly GPU quota before every large run.
- **Colab:** use it for quick experiments and debugging. Sessions can be cut off at any time, so **save checkpoints to Google Drive** after every batch.
- **Main LLM:** `Qwen3-8B`, the AWQ-quantized version so it fits on a T4 GPU, served with vLLM.
  - Disable thinking mode (`enable_thinking=False`).
  - If this version does not run, use the latest Qwen instruct model of about 7–8B parameters that supports Vietnamese, and record the substitution.
- **Second LLM (only for RQ4 and the classifier in RQ2):** `meta-llama/Llama-3.1-8B-Instruct`, AWQ or 4-bit version.
  - This model requires an HF token with its license accepted. If you cannot access it, choose an open model of about 8B parameters from a family other than Qwen that supports Vietnamese, and tell me.
- **Do not use any paid API.** The API budget is 0 USD.
- **Default generation parameters:** `temperature=0.7`, `top_p=0.9`, `max_tokens=256`. Record the exact model name, revision and all parameters.
- **Cache every LLM call** in `cache/llm_calls.parquet`, keyed by a hash of (prompt, model, params, seed), so reruns cost nothing.
- **Cost limits:** before each phase, estimate the number of LLM calls and GPU hours and report them to me. If a phase exceeds **20,000 calls** or **6 GPU hours**, ask me before running it.

## Phased workflow

After each phase, stop, summarize the results and any issues, and wait for my confirmation before continuing.

**Phase 0. Plan and pre-register hypotheses.**
- Write `plan.md` with: the 10 selected questions and their codebook codes, the variable mapping table, sample sizes, metrics, baselines, and a cost estimate for each phase.
- Write specific hypotheses **before running anything**. At minimum:
  - **H1:** the persona condition (b) has a lower Jensen–Shannon divergence (JSD) than the no-persona condition (c).
  - **H2:** the spread of answers across agents is lower than the real spread in WVS, i.e. agents answer more alike than real people do.
  - **H3:** the deviation (JSD) for the 60+ age group and the low-education group is larger than for the 18–29 age group and university graduates.
  - **H4:** from agents' answers, a classifier predicts age group better than a majority-class baseline, but does not predict region better than that baseline.

  Do not change the hypotheses after seeing the results. Any additional analysis must be labeled as exploratory.

**Phase 1. Data loading and exploratory data analysis (EDA).**
- Load the persona dataset and WVS.
- Compare the distribution of each demographic variable between the two sources (tables and charts).
- Draw a stratified sample of **400 personas**. The strata are age group × sex × urban/rural, in the weighted proportions of WVS. Use seed `42`.
- Save `data/sampled_uuids.csv`.

**Phase 2. RQ1: survey replication.**
- Each agent receives its persona and answers each of the 10 questions. The agent must return JSON of the form `{"answer": <valid answer code>, "reason": "<1–2 sentences>"}`.
  - Parse strictly. Re-ask an invalid answer at most 2 times.
  - Record the rate of invalid answers.
- Ask the questions in Vietnamese. If the codebook has an official Vietnamese version, use it; otherwise translate faithfully and save the translations to `data/survey_questions.csv`.
- Three prompt conditions, each with all 400 agents × 10 questions (about 12,000 calls in total):
  - (a) demographic attributes only;
  - (b) demographics plus `persona` and `cultural_background`;
  - (c) no persona: ask with no persona 400 times using different seeds.
- Statistical baselines:
  - uniform random answers;
  - the national WVS distribution, ignoring groups.
- Metrics:
  - JSD between the simulated and real distributions, per question and per group; for rating-scale questions, also report the Wasserstein distance;
  - Spearman correlation between simulated and real group means, to see whether the simulation keeps the right order of which groups score higher or lower;
  - the ratio of the simulated standard deviation to the real standard deviation (to test H2).
- Every metric must come with a **95% bootstrap confidence interval (1,000 resamples)**.

**Phase 3. RQ2: persona fidelity.**
- Use the results of condition (b): each agent's 10 answers and reasons, **with all persona-identifying details removed** (names, ages, place names, occupations).
- Classifier 1: the second LLM predicts age group, education, region and urban/rural.
- Classifier 2: logistic regression on the answer vector only (no text), evaluated with 5-fold cross-validation.
- The labels are the original persona attributes. Report accuracy and macro-F1 for each attribute, compared with a majority-class baseline.

**Phase 4. RQ3: interaction level.**
- **Network:** 60 agents taken from the Phase 1 sample. Build a Watts–Strogatz graph with k=6 and p=0.1, then add homophily by preferring links between agents in the same region or age group. Save the graph to `data/network.graphml`.
- **Topics:** 2 rating-scale questions from Phase 2, namely trust in other people and one gender-role question.
  - Each agent's initial state is its own answer from Phase 2, condition (b).
- **Dynamics:** 8 rounds. In each round, an agent reads its neighbors' opinions and reasons, then answers again in JSON as `{"answer": <scale value>, "reason": "..."}`. That is about 60 × 8 × 2 = 960 calls per run.
- **Comparison model:** Friedkin–Johnsen on the same graph with the same initial states. Also report DeGroot as a special case of Friedkin–Johnsen.
- **Measure in every round:**
  - mean and variance of opinions;
  - a polarization index (bimodality coefficient);
  - agreement across regional clusters;
  - the share of agents who changed their opinion.
- **State clearly in the report that this level has no ground truth.** Results here assess plausibility, not accuracy.

**Phase 5. RQ4: robustness.**
- Rerun Phase 2 condition (b) on 200 agents, and all of Phase 4, varying in turn:
  - 3 seeds: `42`, `7`, `2026`;
  - 2 prompt wordings (the original and a paraphrase with the same content);
  - 2 models: the main LLM and the second LLM.
- Change one factor at a time; do not run the full factorial design.
- Build a table showing, for each hypothesis H1–H4 and each RQ3 conclusion, whether it holds or flips under each change.

**Phase 6. Report.**
- Write `report.md` (in Vietnamese) with: an abstract, methods, results for each RQ (with figures and tables), which hypotheses were supported or rejected, a **limitations section**, and future work.
- The limitations section must cover at least:
  - the persona data is synthetic;
  - WVS has its own time lag and sampling error;
  - the variable harmonization is imperfect;
  - the interaction level has no ground truth;
  - biases of the LLMs used;
  - small sample sizes in some groups.

## Deliverables

- `notebooks/`: `01_eda.ipynb`, `02_survey_sim.ipynb`, `03_fidelity.ipynb`, `04_network_sim.ipynb`, `05_robustness.ipynb`
- `data/`: `sampled_uuids.csv`, `variable_mapping.csv`, `survey_questions.csv`, `network.graphml`
- `cache/`: `llm_calls.parquet`
- `results/`: `metrics_*.csv`, `figures/*.png`
- `prompts/`: every prompt template used, with version numbers
- `plan.md`, `report.md`, `requirements.txt` (from `pip freeze`)
- `README.md`: how to reproduce everything from scratch, with links to the Kaggle notebooks and the Drive folder

## Mandatory rules

1. **Never fabricate data, sources or citations.** Every number in the report must trace back to a file in `results/`. If you don't know something, say so.
2. **If you are blocked, stop and ask**: data you cannot download, a cost limit exceeded, unexpected results. Do not change the research question or the ground-truth source on your own.
3. Fix the seed for every random step. Record library versions.
4. Save a checkpoint after every batch of 500 LLM calls so an interrupted session can resume.
5. Report bad results, non-significant results and failed runs. Do not cherry-pick.
6. When analyzing by group, always report the group's sample size. Mark any group with fewer than 30 samples as unreliable and do not draw conclusions from it.
7. Credit the persona dataset under its CC BY 4.0 license (NVIDIA, FPT Smart Cloud, Quantum AI & Cyber Security Institute) and WVS under its terms of use.

**Start with Phase 0:** check that you can access the persona dataset, the WVS files and both models; list anything missing or unclear; then send me `plan.md` for review.
