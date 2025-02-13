# **Auto Query AI**

**Auto Query AI** is a proof-of-concept project that explores the integration of large language models (LLMs) into vehicles to provide real-time guidance to drivers. By combining car manuals with state-of-the-art AI technologies, this project aims to develop a context-aware chatbot capable of assisting drivers with understanding warning messages, recommended actions, and other vehicle-related queries.

---

## **Project Overview**

Modern vehicles come with a wide range of features and warning systems that can be challenging for drivers to interpret. Auto Query AI uses Retrieval Augmented Generation (RAG) to create an intelligent chatbot that:

- **Understands Car Warnings**: Provides explanations for warning messages found in the car manual.
- **Recommends Actions**: Suggests appropriate steps to address issues.


---

## **Key Features**

### **1. Context-Aware Responses**
The chatbot retrieves relevant information from the car manual to answer driver queries accurately.

### **2. HTML File Integration**
The project uses an HTML file (`mg-zs-warning-messages.html`) containing warning messages from the MG ZS manual.

### **3. RAG Framework**
Combines LLMs with retrieval-based techniques to ensure factual, up-to-date, and relevant responses.

---

## **Project Workflow**

1. **Data Source**
   - The project uses an HTML file, `mg-zs-warning-messages.html`, which contains several pages of car warning messages, their meanings, and recommended actions.

2. **Preprocessing**
   - Extract warning messages and associated content from the HTML file.
   - Clean and structure the data for integration with the chatbot.

3. **Model Development**
   - Implement the **Retrieval Augmented Generation (RAG)** framework to allow the LLM to retrieve specific, relevant sections from the car manual during interactions.
   - Use LangChain to build the pipeline for document retrieval and generation.

4. **User Interaction**
   - Design the chatbot to accept driver queries (e.g., "What does the red oil light mean?").
   - Retrieve the corresponding warning message and recommended action from the manual.
   - Generate a response using the LLM, which can optionally be read aloud using text-to-speech software.

---

## **Tech Stack**

- **Language Model**: OpenAI GPT (or any other LLM compatible with LangChain).
- **Framework**: LangChain for RAG implementation.
- **File Format**: HTML for manual integration.
- **Programming Language**: Python for building the chatbot.

---

## **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/auto-query-ai.git
   cd auto-query-ai
2. Install dependencies:

bash
Copy code
pip install -r requirements.txt

3. Place the car manual file (mg-zs-warning-messages.html) in the data directory
4. Run th3 code file version 2

---

## **Challenges and future opportunities**

**Challenges:**
   - Data Quality & Availability – Not all car manuals are available in digital format; some might be incomplete or poorly structured.
   - Accuracy & Reliability – Ensuring the AI provides correct interpretations without misguiding the driver is critical.
   - Real-Time Processing – The chatbot must quickly retrieve and generate responses, especially in urgent situations.
   - Integration with In-Car Systems – Some vehicles might not support AI integrations, making deployment challenging.
   - Multimodal Capabilities – Adding image recognition (e.g., detecting dashboard icons using a camera) would enhance usability but adds complexity.
**Future Opportunities:**
   - Expansion to More Vehicles – The system can be trained with manuals from multiple car manufacturers.
   - Voice-Activated Assistance – Full voice integration with in-car AI assistants (e.g., Siri, Google Assistant, Alexa).
   - Computer Vision Integration – Drivers could take a photo of a warning light, and the AI could recognize and explain it.
   - Predictive Maintenance – The chatbot could analyze past queries and warn about potential future issues (e.g., “Your tire pressure has been low multiple times this month; consider a check-up”).
   - Multi-Language Support – Expanding to multiple languages would make it globally accessible.
   - Offline Mode – Enabling offline functionality using on-device models for areas with poor internet connectivity.

