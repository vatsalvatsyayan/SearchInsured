import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from enum import Enum
from collections import defaultdict
import re

from service import (
    CitySearchRequest,
    ProximityCityRequest,
    SearchRequest,
    clean_results_compact,
    clean_results_compact_with_distance,
    query_by_city,
    query_by_zip,
    save_clean_compact,
)

# Configuration
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8

load_dotenv()

app = FastAPI(title="Agentic Insurance RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
index = None
chunks = None
model = None

# Conversation memory storage
conversation_history = defaultdict(list)

def get_session_id():
    """Generate a session ID"""
    return "default_session"

class ToolName(str, Enum):
    SEARCH_DOCUMENT = "search_document"
    SEARCH_PROVIDER = "search_provider"
    CALCULATE_COST = "calculate_cost"
    COMPARE_OPTIONS = "compare_options"
    FINAL_ANSWER = "final_answer"

class AgentStep(BaseModel):
    thought: str
    tool: ToolName
    tool_input: Dict
    observation: Optional[str] = None

class AgentResponse(BaseModel):
    question: str
    steps: List[AgentStep]
    final_answer: str
    confidence: str

class Question(BaseModel):
    question: str
    max_iterations: Optional[int] = 5
    session_id: Optional[str] = None

class Answer(BaseModel):
    answer: str
    retrieved_chunks: list

class ClearHistoryRequest(BaseModel):
    session_id: Optional[str] = None

@app.on_event("startup")
def load_resources():
    """Load index, metadata, and model on startup."""
    global index, chunks, model
    
    print("Loading index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    
    with open(META_PATH, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {index.ntotal} vectors and {len(chunks)} chunks")
    
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully")

def expand_query(query):
    """Expand the user query with related terms for better retrieval."""
    expansions = {
        'pediatrician': 'pediatrician paediatrician child doctor specialist primary care',
        'paediatrician': 'pediatrician paediatrician child doctor specialist primary care',
        'cardiologist': 'cardiologist heart doctor specialist cardiovascular',
        'dermatologist': 'dermatologist skin doctor specialist',
        'orthopedist': 'orthopedist orthopedic bone doctor specialist surgeon',
        'neurologist': 'neurologist brain doctor specialist neurology',
        'psychiatrist': 'psychiatrist mental health doctor specialist therapy',
        'dentist': 'dentist dental orthodontist oral care',
        'therapist': 'therapist therapy mental health counseling psychologist',
        'physical therapy': 'physical therapy PT rehabilitation therapy',
        'prescription': 'prescription medication drug pharmacy',
        'surgery': 'surgery surgical procedure operation hospital',
        'emergency': 'emergency urgent care ER hospital',
        'maternity': 'maternity pregnancy prenatal childbirth delivery',
        'preventive': 'preventive screening wellness checkup annual exam',
    }
    
    query_lower = query.lower()
    expanded = query
    
    for key, expansion in expansions.items():
        if key in query_lower:
            expanded = f"{query} {expansion}"
            break
    
    return expanded

def retrieve_similar_chunks(query, k=TOP_K):
    """Retrieve top-k similar chunks for a query."""
    expanded_query = expand_query(query)
    query_embedding = model.encode([expanded_query])
    distances, indices = index.search(query_embedding.astype('float32'), k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "text": chunks[idx]["text"],
            "page": chunks[idx]["page"],
            "distance": float(dist)
        })
    
    return results

# ============ AGENT TOOLS ============

def tool_search_document(query: str) -> Dict:
    """Search the insurance document for relevant information."""
    results = retrieve_similar_chunks(query, k=5)
    return {
        "chunks": results,
        "summary": f"Found {len(results)} relevant sections"
    }

async def tool_search_provider(city: str, state: str, specialty: str) -> Dict:
    """Search for healthcare providers."""
    try:
        raw = await query_by_city(city, state, specialty)
        cleaned = clean_results_compact(
            raw,
            specialty,
            scope_filter={"city": city, "state": state},
        )
        
        if not cleaned or len(cleaned) == 0:
            return {
                "status": "no_results",
                "message": f"No {specialty} providers found in {city}, {state}"
            }
        
        top_results = cleaned[:5]
        formatted_results = []
        for provider in top_results:
            formatted_results.append({
                "name": provider.get("name", "Unknown"),
                "address": provider.get("address", ""),
                "phone": provider.get("phone", ""),
                "accepting_new_patients": provider.get("accepting_new_patients", "Unknown")
            })
        
        return {
            "status": "success",
            "count": len(cleaned),
            "providers": formatted_results,
            "message": f"Found {len(cleaned)} {specialty} providers in {city}, {state}"
        }
    except Exception as e:
        print(f"Error in tool_search_provider: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error searching providers: {str(e)}"
        }

def tool_calculate_cost(service: str, context: Dict) -> Dict:
    """Calculate estimated costs based on coverage info."""
    return {
        "calculation": "Cost calculation based on retrieved coverage info",
        "context_used": context
    }

def tool_compare_options(options: List[str], criteria: str) -> Dict:
    """Compare different coverage or provider options."""
    return {
        "comparison": f"Comparing {len(options)} options based on {criteria}"
    }

# ============ LOCATION PARSING ============

def normalize_state(state_text: str) -> Optional[str]:
    """Convert state name or abbreviation to standard 2-letter code."""
    if not state_text:
        return None
        
    state_text = state_text.strip().upper()
    
    state_map = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
        'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
        'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
        'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
        'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI', 'WYOMING': 'WY'
    }
    
    if len(state_text) == 2 and state_text.isalpha():
        return state_text
    
    return state_map.get(state_text.upper())

def extract_location_and_specialty(question: str) -> Optional[Dict]:
    """Extract city, state, and specialty from question."""
    question_lower = question.lower()
    
    specialties = [
        'cardiologist', 'cardiologists',
        'pediatrician', 'pediatricians', 
        'dermatologist', 'dermatologists',
        'neurologist', 'neurologists',
        'psychiatrist', 'psychiatrists',
        'dentist', 'dentists',
        'therapist', 'therapists',
        'orthopedist', 'orthopedists',
        'oncologist', 'oncologists',
        'ophthalmologist', 'ophthalmologists',
        'primary care', 'family doctor', 'general practitioner'
    ]
    
    specialty = None
    for spec in specialties:
        if spec in question_lower:
            specialty = spec.rstrip('s') if spec.endswith('s') and spec not in ['primary care'] else spec
            break
    
    if not specialty:
        return None
    
    # More flexible patterns that work with case-insensitive matching
    patterns = [
        # Pattern 1: "in City, State" - case insensitive
        r'(?:in|near|around)\s+([a-zA-Z][\w\s]+?),\s*([a-zA-Z][\w\s]+?)(?:\s|$|\.|\?|!)',
        # Pattern 2: "in City State" (no comma) - with 2-letter state code
        r'(?:in|near|around)\s+([a-zA-Z][\w\s]+?)\s+([A-Z]{2})(?:\s|$|\.|\?|!)',
        # Pattern 3: Just "City, State" anywhere
        r'([a-zA-Z][\w\s]{2,}?),\s*([a-zA-Z][\w\s]+?)(?:\s|$|\.|\?|!)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            state_text = match.group(2).strip()
            
            # Capitalize city name properly
            city = ' '.join(word.capitalize() for word in city.split())
            
            state = normalize_state(state_text)
            
            # Validate that city looks reasonable (not too short, not just "me" or "a")
            if state and city and len(city) > 2 and city.lower() not in ['me', 'and', 'the', 'for', 'you']:
                print(f"[EXTRACTED] City: {city}, State: {state}, Specialty: {specialty}")
                return {
                    "city": city,
                    "state": state,
                    "specialty": specialty
                }
    
    return None

# ============ AGENT REASONING ============

def parse_agent_action(llm_response: str) -> Dict:
    """Parse LLM response into structured action."""
    lines = llm_response.strip().split('\n')
    action = {
        "thought": "",
        "tool": None,
        "tool_input": {}
    }
    
    current_section = None
    input_buffer = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("Thought:"):
            action["thought"] = line.replace("Thought:", "").strip()
            current_section = "thought"
        elif line.startswith("Action:"):
            tool_name = line.replace("Action:", "").strip()
            action["tool"] = tool_name
            current_section = "action"
        elif line.startswith("Action Input:"):
            input_str = line.replace("Action Input:", "").strip()
            input_buffer.append(input_str)
            current_section = "input"
        elif current_section == "input" and line:
            input_buffer.append(line)
    
    if input_buffer:
        try:
            input_text = ' '.join(input_buffer)
            action["tool_input"] = json.loads(input_text)
        except:
            action["tool_input"] = {"raw": ' '.join(input_buffer)}
    
    return action

def execute_tool(tool_name: str, tool_input: Dict) -> str:
    """Execute the specified tool with given input."""
    
    if tool_name == "search_document":
        result = tool_search_document(tool_input.get("query", ""))
        return json.dumps(result, indent=2)
    elif tool_name == "search_provider":
        return "ASYNC_PROVIDER_SEARCH"
    elif tool_name == "calculate_cost":
        result = tool_calculate_cost(
            tool_input.get("service", ""),
            tool_input.get("context", {})
        )
        return json.dumps(result, indent=2)
    elif tool_name == "compare_options":
        result = tool_compare_options(
            tool_input.get("options", []),
            tool_input.get("criteria", "")
        )
        return json.dumps(result, indent=2)
    else:
        return f"Unknown tool: {tool_name}"

async def run_agent(question: str, session_id: str, max_iterations: int = 5) -> AgentResponse:
    """Run the agent to answer a question through multi-step reasoning."""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    history = conversation_history[session_id]
    
    conversation_context = ""
    if history:
        conversation_context = "\n\nPREVIOUS CONVERSATION:\n"
        for msg in history[-6:]:
            conversation_context += f"{msg['role']}: {msg['content']}\n"
    
    steps = []
    agent_scratchpad = ""
    
    system_prompt = """You are an intelligent insurance assistant agent.

Available tools:
1. search_document - Search the insurance policy document
   Input: {"query": "search terms"}

2. search_provider - Find healthcare providers by location
   Input: {"city": "City Name", "state": "CA", "specialty": "cardiologist"}

3. final_answer - Provide the final answer to the user
   Input: {"answer": "your complete answer"}

Response format:
Thought: [what I need to do]
Action: [tool name]
Action Input: {"key": "value"}

After Observation, either do next action OR use final_answer."""

    for i in range(max_iterations):
        prompt = f"""{system_prompt}
{conversation_context}

Current Question: {question}

{agent_scratchpad}

What should you do next? (Step {i+1}/{max_iterations})"""

        try:
            response = gemini_model.generate_content(prompt)
            llm_output = response.text
            
            print(f"\n=== Step {i+1} ===")
            print(f"LLM Output: {llm_output[:300]}")
            
            if "final_answer" in llm_output.lower() or i == max_iterations - 1:
                final_answer = llm_output
                
                if "Action Input:" in llm_output:
                    try:
                        input_start = llm_output.index("Action Input:") + len("Action Input:")
                        final_text = llm_output[input_start:].strip()
                        
                        try:
                            if "```json" in final_text:
                                final_text = final_text.split("```json")[1].split("```")[0].strip()
                            elif "```" in final_text:
                                final_text = final_text.split("```")[1].split("```")[0].strip()
                            
                            final_json = json.loads(final_text)
                            final_answer = final_json.get("answer", final_text)
                        except:
                            if '"answer"' in final_text:
                                try:
                                    start = final_text.index('"answer":') + len('"answer":')
                                    rest = final_text[start:].strip()
                                    if rest.startswith('"'):
                                        rest = rest[1:]
                                        end = rest.find('"')
                                        if end > 0:
                                            final_answer = rest[:end]
                                except:
                                    pass
                    except:
                        pass
                
                if final_answer == llm_output:
                    if "Thought:" in final_answer:
                        parts = final_answer.split("Action Input:")
                        if len(parts) > 1:
                            final_answer = parts[1].strip()
                
                conversation_history[session_id].append({
                    "role": "user",
                    "content": question
                })
                conversation_history[session_id].append({
                    "role": "assistant",
                    "content": final_answer
                })
                
                return AgentResponse(
                    question=question,
                    steps=steps,
                    final_answer=final_answer,
                    confidence="high" if len(steps) >= 1 else "medium"
                )
            
            action = parse_agent_action(llm_output)
            
            if not action["tool"] or action["tool"] == "final_answer":
                return AgentResponse(
                    question=question,
                    steps=steps,
                    final_answer=llm_output,
                    confidence="medium"
                )
            
            print(f"Executing tool: {action['tool']} with input: {action['tool_input']}")
            
            if action["tool"] == "search_provider":
                result = await tool_search_provider(
                    action['tool_input'].get("city", ""),
                    action['tool_input'].get("state", ""),
                    action['tool_input'].get("specialty", "")
                )
                observation = json.dumps(result, indent=2)
            else:
                observation = execute_tool(action["tool"], action["tool_input"])
            
            print(f"Observation: {observation[:200]}")
            
            step = AgentStep(
                thought=action["thought"],
                tool=action["tool"],
                tool_input=action["tool_input"],
                observation=observation
            )
            steps.append(step)
            
            agent_scratchpad += f"\nThought: {action['thought']}\n"
            agent_scratchpad += f"Action: {action['tool']}\n"
            agent_scratchpad += f"Action Input: {json.dumps(action['tool_input'])}\n"
            agent_scratchpad += f"Observation: {observation}\n"
            
        except Exception as e:
            print(f"Error in agent loop: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return AgentResponse(
                question=question,
                steps=steps,
                final_answer=f"I encountered an error: {str(e)}",
                confidence="low"
            )
    
    return AgentResponse(
        question=question,
        steps=steps,
        final_answer="I need more steps to fully answer this question. Please try rephrasing.",
        confidence="low"
    )

async def execute_provider_search(question: str, city: str, state: str, specialty: str, session_id: str) -> AgentResponse:
    """Execute a direct provider search and format the response."""
    try:
        print(f"[SEARCH] Looking for {specialty} in {city}, {state}")
        
        raw = await query_by_city(city, state, specialty)
        cleaned = clean_results_compact(
            raw,
            specialty,
            scope_filter={"city": city, "state": state},
        )
        print(cleaned)
        
        if cleaned and len(cleaned) > 0:
            providers_text = f"I found {len(cleaned)} {specialty}(s) in {city}, {state}:\n\n"
            
            for i, p in enumerate(cleaned[:5], 1):
                name = p.get('name', 'Unknown Provider')
                address = p.get('address', 'Address not available')
                phone = p.get('phone', 'Phone not available')
                accepting = p.get('accepting_new_patients', False)
                
                providers_text += f"{i}. {name}\n"
                providers_text += f"   Location: {address}\n"
                providers_text += f"   Phone: {phone}\n"
                if accepting:
                    providers_text += f"   Status: Accepting new patients\n"
                providers_text += "\n"
            
            if len(cleaned) > 5:
                providers_text += f"... and {len(cleaned) - 5} more providers available.\n\n"
            
            providers_text += "Would you like more information about any of these providers?"
            
            conversation_history[session_id].append({
                "role": "user",
                "content": question
            })
            conversation_history[session_id].append({
                "role": "assistant",
                "content": providers_text
            })
            
            return AgentResponse(
                question=question,
                steps=[AgentStep(
                    thought=f"User wants to find {specialty}s in {city}, {state}",
                    tool="search_provider",
                    tool_input={"city": city, "state": state, "specialty": specialty},
                    observation=f"Found {len(cleaned)} providers"
                )],
                final_answer=providers_text,
                confidence="high"
            )
        else:
            no_results = f"I couldn't find any {specialty}s in {city}, {state}.\n\n"
            no_results += "This could mean:\n"
            no_results += "1. There are no providers of this type in the area\n"
            no_results += "2. The specialty name might need adjustment\n"
            no_results += "3. Try searching in a nearby city\n\n"
            no_results += "Would you like to search in a different location?"
            
            conversation_history[session_id].append({
                "role": "user",
                "content": question
            })
            conversation_history[session_id].append({
                "role": "assistant",
                "content": no_results
            })
            
            return AgentResponse(
                question=question,
                steps=[AgentStep(
                    thought=f"Searching for {specialty}s in {city}, {state}",
                    tool="search_provider",
                    tool_input={"city": city, "state": state, "specialty": specialty},
                    observation="No results found"
                )],
                final_answer=no_results,
                confidence="medium"
            )
            
    except Exception as e:
        print(f"[ERROR] Direct search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"I encountered an error while searching for providers: {str(e)}\n\n"
        error_msg += "Please try again or rephrase your question."
        
        return AgentResponse(
            question=question,
            steps=[],
            final_answer=error_msg,
            confidence="low"
        )

# ============ API ENDPOINTS ============

@app.post("/ask/agent", response_model=AgentResponse)
async def ask_agent(question: Question):
    """Ask a question using the agentic approach."""
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        session_id = question.session_id or get_session_id()
        
        location_info = extract_location_and_specialty(question.question)
        
        if location_info:
            print(f"[DETECTED] Location search: {location_info}")
            return await execute_provider_search(
                question.question,
                location_info["city"],
                location_info["state"],
                location_info["specialty"],
                session_id
            )
        
        return await run_agent(question.question, session_id, question.max_iterations or 5)
        
    except Exception as e:
        print(f"[ERROR] in ask_agent: {str(e)}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.post("/ask", response_model=Answer)
def ask_question(question: Question):
    """Original non-agentic endpoint - single-step RAG."""
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    retrieved = retrieve_similar_chunks(question.question)
    
    context_parts = []
    for i, chunk in enumerate(retrieved, 1):
        context_parts.append(f"[Chunk {i}, Page {chunk['page']}]:\n{chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful insurance assistant. Answer based on the context provided.

CONTEXT:
{context}

QUESTION: {question.question}

ANSWER:"""
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    try:
        response = gemini_model.generate_content(prompt)
        return Answer(
            answer=response.text,
            retrieved_chunks=[
                {
                    "page": chunk["page"],
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "distance": chunk["distance"]
                }
                for chunk in retrieved
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

@app.post("/clear-history")
def clear_history(request: ClearHistoryRequest):
    """Clear conversation history for a session."""
    session_id = request.session_id or get_session_id()
    if session_id in conversation_history:
        conversation_history[session_id].clear()
    return {"status": "success", "message": "Conversation history cleared"}

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "index_size": index.ntotal if index else 0,
        "chunks_loaded": len(chunks) if chunks else 0,
        "agent_enabled": True
    }

@app.post("/search/city/clean/compact")
async def search_clean_compact_city(req: CitySearchRequest):
    try:
        raw = await query_by_city(req.city, req.state, req.specialty)
        cleaned = clean_results_compact(
            raw,
            req.specialty,
            scope_filter={"city": req.city, "state": req.state},
        )
        save_clean_compact(cleaned, f"{req.city.lower().replace(' ', '_')}_{req.state}_{req.specialty}")
        return cleaned
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/city/near")
async def search_city_near(req: ProximityCityRequest):
    """City-wide search, sorted by distance from origin_zip."""
    try:
        raw = await query_by_city(req.city, req.state, req.specialty)
        cleaned = clean_results_compact_with_distance(
            raw,
            req.specialty,
            origin_zip=req.origin_zip,
            scope_filter={"city": req.city, "state": req.state},
        )
        save_clean_compact(cleaned, f"{req.city.lower().replace(' ', '_')}_{req.state}_{req.specialty}_near_{req.origin_zip}")
        return cleaned
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)