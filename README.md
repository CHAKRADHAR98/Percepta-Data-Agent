![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

# 🤖 Percepta Data Agent — FPV Annotation Assistant

**Agent Address:** `agent1qwdfq7pjv573kay7fnumpkw4768m5lauq5gzlv398dafauxrs8j3z3v85nj`

---

## 🧠 Overview

The **Percepta Data Agent** is an autonomous AI agent for the **ASI Alliance** ecosystem that automates the entire pipeline for processing **first-person (FPV) egocentric datasets** for bimanual (two-handed) manipulation tasks.

Built with **Fetch.ai's uAgents**, it provides a simple chat interface (compatible with **ASI:One**) for researchers to submit raw videos from IPFS. The agent then perceives, reasons, and acts: it runs a complex computer vision pipeline to generate annotations, validates the data quality, and dynamically adds the new, structured dataset to a decentralized **SingularityNET MeTTa Knowledge Graph**.

This project solves a major bottleneck in robotics research: the creation and sharing of high-quality, labeled training data.

---

## 📺 Demo Video

[**Watch our 3-minute demo video here**](https://www.loom.com/share/be462918fe64414b85f58557ab3a4de5)

---

## ⚙️ Technology Stack

This agent is built entirely on the ASI Alliance stack:

* **Fetch.ai uAgents:** The core framework for the autonomous agent, managing all logic and communication.
* **ASI:One Chat Protocol:** The agent implements the `chat_protocol_spec`, making it fully discoverable and interactive via the ASI:One interface.
* **Agentverse:** The agent is registered and discoverable on the Agentverse, allowing anyone to find and use it.
* **SingularityNET MeTTa:** The agent's persistent memory. All processed datasets, tasks, and quality metadata are stored in a MeTTa knowledge graph, allowing for powerful semantic queries.
* **IPFS (via Pinata):** Used for decentralized storage. The agent downloads raw videos from IPFS and uploads the final annotated dataset folders back to IPFS.

---

## 💡 What You Can Do

-   **Process a new video:**
    ```text
    Process ipfs://Qm... (task: opening bottle)
    ```
    The agent will download the video, run the full CV pipeline, upload the results to IPFS, and update the MeTTa knowledge graph.

-   **Retrieve existing datasets:**
    ```text
    I need fpv data for folding clothes
    ```
    The agent queries the MeTTa graph to find all known datasets for this task.

-   **Explore the knowledge graph:**
    ```text
    /stats
    /tasks
    /recent
    /help
    ```
    The agent provides statistics and summaries directly from its MeTTa memory.

---

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/CHAKRADHAR98/Percepta-Data-Agent.git
    cd Percepta-Data-Agent
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    * Create a `.env` file (you can copy `.env.example`).
    * Add your `ASI_ONE_API_KEY`, `PINATA_API_KEY`, and `PINATA_SECRET_KEY`.

4.  **Run the agent:**
    ```bash
    python agent.py
    ```
    (Note: Your agent address will be printed to the console when you first run it.)

---

## 🚫 Limitations
* Does not train or fine-tune models — only manages and processes data.
* Currently supports bimanual manipulation tasks.
* Requires videos with a stable FPV perspective for accurate annotation.

---

## 🔍 Keywords
`ASI Alliance` · `Fetch.ai` · `SingularityNET` · `uAgent` · `MeTTa` · `FPV datasets` · `Egocentric AI` · `Bimanual manipulation` · `Embodied AI` · `Dataset annotation` · `MediaPipe`

---

## 🌐 Official Links
-   [Website](https://www.percepta.world/)
-   [X (Twitter)](https://x.com/Perceptaworld)