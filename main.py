import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import spacy
import faiss
import numpy as np
import time
import random
from typing import List, Dict, Tuple
from diagramgen import generate_diagram_from_query
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

# ✅ Add Graphviz to PATH
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ----------------------------- AGENT 1: INPUT AGENT -----------------------------
def input_agent(text: str) -> str:
    return text.strip()

# ----------------------------- AGENT 2: NLP AGENT -----------------------------
nlp = spacy.load("en_core_web_sm")

def extract_key_terms(doc) -> List[str]:
    noun_chunks = {chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1}
    named_entities = {ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 1}
    return list(noun_chunks.union(named_entities))[:5]

def extract_svo_triples(doc) -> List[tuple]:
    triples = []
    for sent in doc.sents:
        root = [token for token in sent if token.dep_ == "ROOT"]
        if root:
            verb = root[0]
            subj = [w for w in verb.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w for w in verb.rights if w.dep_ in ("dobj", "attr", "pobj")]
            if subj and obj:
                triples.append((subj[0].text, verb.text, obj[0].text))
    return triples

def nlp_agent(text: str) -> Dict:
    doc = nlp(text)
    key_terms = extract_key_terms(doc)
    triples = extract_svo_triples(doc)
    topic_type = "process" if "how" in text.lower() else "theory"
    return {
        "key_terms": key_terms,
        "triples": triples,
        "topic_type": topic_type
    }

# ----------------------------- AGENT 3: INTERACTIVE CHARTS -----------------------------
def show_key_terms_chart(key_terms: List[str]):
    freq = {term: random.randint(1, 10) for term in key_terms}
    fig = px.bar(x=list(freq.keys()), y=list(freq.values()), labels={'x': 'Key Terms', 'y': 'Frequency'},
                 title="Key Term Importance (Simulated)")
    fig.show()

def show_line_chart():
    x = list(range(10))
    y = [random.uniform(1, 10) for _ in range(10)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Simulated Trend'))
    fig.update_layout(title='Trend Over Time (Simulated)', xaxis_title='Time', yaxis_title='Value')
    fig.show()

def show_pie_chart(key_terms: List[str]):
    values = [random.randint(1, 10) for _ in key_terms]
    fig = px.pie(names=key_terms, values=values, title='Distribution of Key Concepts')
    fig.show()

def show_svo_network(triples: List[tuple]):
    G = nx.DiGraph()
    for s, v, o in triples:
        G.add_edge(s, v)
        G.add_edge(v, o)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    node_text = list(G.nodes)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                            hoverinfo='none', mode='lines')

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text',
                            text=node_text, textposition="bottom center",
                            hoverinfo='text',
                            marker=dict(showscale=False, size=20, color='skyblue'))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='Subject-Verb-Object Graph',
                                     showlegend=False, hovermode='closest'))
    fig.show()

# ----------------------------- AGENT 4: DIALOGUE MEMORY AGENT -----------------------------
dialogue_memory = []

def store_dialogue(text: str, embedding: np.ndarray):
    dialogue_memory.append((text, embedding))

def retrieve_similar_query(embedding: np.ndarray):
    if not dialogue_memory:
        return None
    dim = len(embedding)
    index = faiss.IndexFlatL2(dim)
    data = np.array([e for _, e in dialogue_memory]).astype('float32')
    index.add(data)
    D, I = index.search(np.array([embedding]).astype('float32'), 1)
    return dialogue_memory[I[0][0]][0] if D[0][0] < 0.1 else None

# ----------------------------- AGENT 5: ENGAGEMENT MONITOR AGENT -----------------------------
def monitor_engagement() -> float:
    simulated_emotion = random.choice(['happy', 'surprise', 'neutral', 'sad', 'angry'])
    print(f"\n(Simulated Emotion Detected: {simulated_emotion})")

    if simulated_emotion in ['happy', 'surprise']:
        return 0.9
    elif simulated_emotion == 'neutral':
        return 0.7
    else:
        return 0.5

# ----------------------------- AGENT 6: ADAPTIVE TEACHING AGENT -----------------------------
def adaptive_teaching(response: str, engagement_score: float):
    if engagement_score < 0.4:
        print("\nYou seem disengaged. Here's a simplified explanation with a diagram:")
        return response + "\n(Simplified with visual aid)"
    elif engagement_score > 0.7:
        return response + "\n(Going deeper with more detail...)"
    else:
        return response

# ----------------------------- MAIN SYSTEM FLOW -----------------------------
def teaching_assistant_pipeline(text: str):
    input_text = input_agent(text)

    parsed = nlp_agent(input_text)
    print("\n🔍 NLP Agent Output:", parsed)

    result = generate_diagram_from_query(input_text)
    if result is None:
        return
    explanation, _ = result
    print("\n📘 Explanation from model:\n", explanation)

    # Interactive Charts
    if parsed['key_terms']:
        show_key_terms_chart(parsed['key_terms'])
        show_pie_chart(parsed['key_terms'])

    if parsed['triples']:
        show_svo_network(parsed['triples'])

    # Always show a simulated line chart
    show_line_chart()

    # Dialogue Memory
    embed_vector = np.random.rand(300).astype('float32')
    store_dialogue(input_text, embed_vector)
    similar = retrieve_similar_query(embed_vector)
    if similar:
        print(f"\n🧠 Previously you asked something similar: '{similar}'")

    engagement = monitor_engagement()
    print(f"\n📊 Engagement Score: {engagement}")

    final_response = adaptive_teaching("Here’s your explanation based on input.", engagement)
    print("\n🤖 Final Teaching Response:\n", final_response)

# ----------------------------- TEST -----------------------------
if __name__ == "__main__":
    user_query = input("Please enter your query: ")
    teaching_assistant_pipeline(user_query)
