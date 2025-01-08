from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.vectorstores import Chroma
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
# Hugging Face token for gated models
hf_token = "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"

# Load the Mistral model and tokenizer explicitly using tokenizer.json
def load_falcon_model():
    model_name = "tiiuae/falcon-7b"
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
        print("Falcon model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading Falcon model: {e}")
        raise
# Generate response using Mistral
def generate_response(model, tokenizer, prompt, max_length=150, temperature=0.7):
    """
    Generate a response from the Mistral model.
    :param model: The Mistral model.
    :param tokenizer: The tokenizer for the model.
    :param prompt: The input prompt (str).
    :param max_length: The maximum length of the output (int).
    :param temperature: Sampling temperature for randomness (float).
    :return: Generated response (str).
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating response: {e}")
        raise

# Load and process documents
def load_and_process_documents():
    try:
        loader = UnstructuredHTMLLoader(file_path="mg-zs-warning-messages.html")
        car_docs = loader.load()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(car_docs)
        
        # Display chunks
        for i, chunk in enumerate(splits):
            print(f"Chunk {i + 1}:\n{chunk.page_content}\n")
        return splits
    except Exception as e:
        print(f"Error loading and processing documents: {e}")
        raise

# Initialize Chroma vectorstore
def setup_vectorstore(splits):
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore.as_retriever()
    except Exception as e:
        print(f"Error setting up vectorstore: {e}")
        raise

# Main Execution

   
# Load the Mistral model and tokenizer
tokenizer, model = load_falcon_model()

# Load and process documents
splits = load_and_process_documents()

# Setup retriever
retriever = setup_vectorstore(splits)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_template(
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\nContext: {context}\nAnswer:"
)

# Query the retriever
question = "The Gasoline Particulate Filter Full warning has appeared. What does this mean and what should I do about it?"
docs = retriever.get_relevant_documents(question)
context = "\n".join([doc.page_content for doc in docs])

# Create the full prompt
full_prompt = prompt_template.format(question=question, context=context)

# Generate response using Mistral
response = generate_response(model, tokenizer, full_prompt)
print("\nAnswer:")
print(response)
