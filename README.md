# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

<!-- What topic or category of knowledge does your system cover?
     Why is this knowledge valuable, and why is it hard to find through official channels?
     Example: "Student reviews of CS professors at [university] — useful because official
     course descriptions don't reflect teaching style, exam difficulty, or workload." -->

Off-campus housing near Georgia Tech. This knowledge is valuable because there have been a lot of new student housing complexes that have been built recently. A lot of them have several issues which is not immediately obvious so this could help students make a more informed decision.


---

## Document Sources

<!-- List every source you collected documents from.
     Be specific: include URLs, subreddit names, forum thread titles, or file names.
     Aim for variety — sources that together cover different subtopics or perspectives. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | r/gatech subreddit| megathreads and search results for     "off-campus housing" containing student reviews | reddit.com/r/gatech |
| 2 | Rambler website | official website for one of the newest student housing complexes | rambleratlanta.com |
| 3 | Inspire Atlanta website | official website for Inspire housing complex | liveatinspireatl.com |
| 4 | Hub Atlanta website| official website for Hub Atlanta | huboncampus.com/atlanta |
| 5 | Square on Fifth | official website for Square on Fifth complex | squareonfifth.com |
| 6 | The Standard Atlanta | official website for the Standard | landmarkproperties.com/property/the-standard-at-atlanta |
| 7 | r/ATLHousing subreddit | safety reviews, info on rental rates, and neighborhood comparisons | reddit.com/r/ATLHousing |
| 8 | GT Off-Campus Housing Resources | official university site on tenant rights and neighborhood safety | housing.gatech.edu |
| 9 | Google Maps Reviews | scraped reviews for specific complexes to cross-reference with Reddit complaints | Google Maps API / Scraped Data |
| 10 | Standard lease agreements | examples of student housing leases to go over guarantor requirements/fine print | Local property management PDFs |

---

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:**
250 tokens
**Overlap:**
40 tokens
**Why these choices fit your documents:**
A 250-token chunk is large enough to capture an entire Reddit comment without cutting them off. The 40-token overlap was important for the HTML pricing tables as it ensured if an apartment building's name was at the top, the context would bleed into the next chunk.
**Final chunk count:**
86
---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**
sentence-transformers/all-MiniLM-L6-v2 via native ChromaDB
**Production tradeoff reflection:**
If this were to be deployed to thousands of students, the model I used would help keep cloud compute costs at zero since it runs locally with minimal latency.
However, its small context window and semantic simplicity lead to a lot of entity confusion. So if cost wasn't a constraint, I might upgrade to a commercial API to help resolve these issues.
---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**
"You are an expert, highly objective assistant helping Georgia Tech students find off-campus housing. Your task is to answer the user's question using ONLY the provided document chunks below. CRITICAL RULES: 1. Base your answer strictly on the text provided. Do NOT use outside knowledge, general training assumptions, or extrapolate. 2. If the provided document chunks do not contain the answer, you must state exactly: 'I am sorry, but the provided documents do not contain enough information to answer this question.'"
**How source attribution is surfaced in the response:**
Source citatioins are programtically extracted from the vector database's metadata output. The pipeline aggregates the source names and adds them to a text box below the generated answer.
---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What's the cheapest 4-bedroom at The Rambler right now? | Should pull the starting price from the floor plan data | The cheapest 4-bedroom options are The Carlyle and The Roosevelt, both starting at $1,369* and $1,479 respectively, and The Ivy and The Pearl, starting at $1,489 and $1,399* respectively. The Pearl is the cheapest option at $1,399*. | Relevant | Accurate
| 2 | Which apartment complexes are actually less than a 5-minute walk to campus? | Should list Inspire Atlanta and SQ5 based on location specs | I am sorry, but the provided documents do not contain enough information to answer this question. | Off-target | Inaccurate
| 3 | What is the worst thing about living at The Standard based on Reddit? | Should synthesize Reddit data mentioning management or maintenance issues | Listed complaints from Reddit in response | Relevant | Accurate
| 4 | Does Hub Atlanta do individual leases? | Should confirm they do individual leasing based on FAQ |
| 5 | Is Home Park better than Midtown high-rises? | Should compare cheaper rent in Home Park with the luxury amenities/higher costs of Midtown | I am sorry, but the provided documents do not contain enough information to answer this question. | Off-target | Inaccurate

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**

**What the system returned:**

**Root cause (tied to a specific pipeline stage):**

**What you would change to fix it:**

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

**One way your implementation diverged from the spec, and why:**

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*

**Instance 2**

- *What I gave the AI:*
- *What it produced:*
- *What I changed or overrode:*
