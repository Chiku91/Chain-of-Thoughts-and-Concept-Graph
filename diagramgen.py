import subprocess
import re
from graphviz import Source
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt

def query_ollama(question):
    """
    Sends a question to the Ollama model and expects a valid Graphviz diagram in return.
    """
    prompt = f"""
    You are an expert educational assistant who creates diagrams to explain complex topics.
    Your task is to answer the user's question with a clear, step-by-step explanation.
    After the explanation, you MUST provide a complete and valid Graphviz DOT diagram that visually represents the concept.

    **CRITICAL INSTRUCTIONS:**
    1. First, write the textual explanation.
    2. After the explanation, create the Graphviz diagram.
    3. The diagram code MUST be enclosed in a single markdown block starting with ```dot and ending with ```.

    Example:
    ```dot
    digraph example {{
        A -> B;
        B -> C;
    }}
    ```

    Now, please answer: "{question}"
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=120
        )
        return result.stdout
    except FileNotFoundError:
        print("❌ Error: 'ollama' command not found. Ensure Ollama is installed and in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Ollama command failed:\n{e.stderr}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("❌ Error: Ollama model took too long to respond.")
        sys.exit(1)

def extract_dot_code(response):
    """Extracts DOT code block from model response."""
    match = re.search(r"```dot(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_chain_of_thought(response):
    """Extracts explanation by removing DOT code block."""
    return re.sub(r"```dot.*?```", "", response, flags=re.DOTALL).strip()

def generate_diagram_from_query(question):
    """
    Generates a DOT diagram and explanation for a given student query.
    Returns a tuple: (explanation, dot_code) or None if generation fails.
    """
    if not question.strip():
        print("⚠️ No question provided.")
        return None

    print("\n⏳ Querying Ollama model...")
    model_response = query_ollama(question)

    if not model_response:
        print("❌ Empty response from model.")
        return None

    # Chain of thought (explanation)
    chain_of_thought = extract_chain_of_thought(model_response)
    print("\n💭 Chain of Thought:\n", chain_of_thought)

    # DOT code
    dot_code = extract_dot_code(model_response)
    if not dot_code:
        print("❌ Could not find valid DOT diagram in response.")
        return None

    print("\n✅ Extracted DOT Code:\n", dot_code)

    try:
        # Create and render diagram
        filename = "concept_graph"
        s = Source(dot_code, filename=filename, format="png")
        output_path = s.render(cleanup=True)
        print(f"\n📌 Diagram saved to: {output_path}")

        # Display with matplotlib
        try:
            img = Image.open(output_path)
            plt.imshow(img)
            plt.axis("off")
            plt.title("Generated Diagram")
            plt.show()
        except Exception as e:
            print("⚠️ Diagram generated but could not display image:", e)

    except Exception as e:
        print("❌ Error generating diagram:", e)

    return chain_of_thought, dot_code

# Optional CLI test
def main():
    try:
        question = input("📚 Enter a student question: ")
        generate_diagram_from_query(question)
    except EOFError:
        print("\n⚠️ No input received. Exiting.")

if __name__ == "__main__":
    main()
