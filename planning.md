# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
Off-campus housing near Georgia Tech. This knowledge is valuable because there have been a lot of new student housing complexes that have been built recently. A lot of them have several issues which is not immediately obvious so this could help students make a more informed decision.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

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

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
250 tokens
**Overlap:**
40 tokens
**Reasoning:**
The dataset is a mix of structured stuff (like rental rate tables) and unstructured Reddit comments. A 250-token chunk is big enough to be able to grab a long Reddit comment without cutting it off in the middle. The 40-token overlap is to make sure the building name doesn't get cut off (if the name of the complex is at the beginning, want to make sure any reviews are captured in the same chunk).
---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:**
text-embedding-3-small (OpenAI)
**Top-k:**
7
**Production tradeoff reflection:**
If cost wasn't a thing, I might have used a heavier model with a massive context window. But, I also think a heavy model would lag the UI and need to keep latency in mind as well. Also, domain=specific accuray is important as the model needs to understand "Tech Square" and "West Campus" are different places, and that "Home Park" is a neighborhood (and not a park).
---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What's the cheapest 4-bedroom at The Rambler right now? | Should pull the starting price from the floor plan data |
| 2 | Which apartment complexes are actually less than a 5-minute walk to campus? | Should list Inspire Atlanta and SQ5 based on location specs |
| 3 | What is the worst thing about living at The Standard based on Reddit? | Should synthesize Reddit data mentioning management or maintenance issues |
| 4 | Does Hub Atlanta do individual leases? | Should confirm they do individual leasing based on FAQ |
| 5 | Is Home Park better than Midtown high-rises? | Should compare cheaper rent in Home Park with the luxury amenities/higher costs of Midtown |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. A lot of the luxury student apartments have the same amenities (rooftop pool, 4x4 layout, gym). I could see entity confusion being a problem as the retrieval pipeline might mix them up and pull up a review for one complex and attribute it to another.

2. Conflicting data may also be a challenge. Official websites will mention "quiet luxury studying environments" but Reddit reviews will say "the walls are paper thin". The model may struggle with deciding which source to trust unless I give it guidance.

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

[ Document Ingestion ] -> (Web Scraper / Reddit API)
        |
        v
[    Chunking      ] -> (Python / LangChain RecursiveCharacterTextSplitter)
        |
        v
[ Embedding & Vector Store ] -> (OpenAI Embeddings / ChromaDB)
        |
        v
[    Retrieval     ] -> (Similarity Search with MMR so it's not just 7 duplicate comments)
        |
        v
[ Generation & UI  ] -> (React Frontend / Node.js Backend / LLM API)

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
tool: Claude
input: The Documents and Chunking Strategy sections. I'll aslk it to write the Python ingestion scripts.
expectation: a Python script using LangChain's RecursiveCharacterTextSplitter that outputs clean JSON objects with the text and source URLs
verification: I'll run the script on a sample Reddit dump and one apartment pricing page, then manually check the JSON to make sure sentences aren't chopped in half and metadata isn't lost.
**Milestone 4 — Embedding and retrieval:**
tool: Gemini
input: the Retrieval Approach section and chunked JSONs from M3
expectation: Python code to generate embeddings, load them into ChromaDB, and a retrieval function that takes a query, embeds it, and returns the top 7 chunks using MMR
verification: I'll pass in evaluation question 3 and manually read the terminal output to make sure the 7 chunks are actually Reddit reviews about the Standard and not the same comment repeated 7 times.
**Milestone 5 — Generation and interface:**
tool: Claude for React/node boilerplate, Gemini to tweak final prompts
input: the Evaluation Plan and retrieval function from M4
expectation: an Express server in Node.js to handle the API calls and prompt formatting, plus a clean React frontend for the chat UI
verification: I'll ask the 5 Evaluation questions through the React interface and see if the answers match my expected outcomes.
