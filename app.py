"""
Flask Backend for Loop AI Hospital Network Assistant
Integrates Gemini API with RAG and Function Calling
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
import os
import json
import pickle
import tempfile
import io
import time
import traceback
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import speech_recognition as sr
import edge_tts
import asyncio
from pydub import AudioSegment
from dotenv import load_dotenv
import numpy as np

# Set ffmpeg path for pydub - add to PATH
ffmpeg_path = r"C:\Users\oditi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_path
AudioSegment.converter = os.path.join(ffmpeg_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_path, "ffprobe.exe")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='.')
CORS(app)

# Store conversation history per session (in-memory)
# Format: {session_id: [{'role': 'user', 'text': '...'}, {'role': 'assistant', 'text': '...'}]}
conversation_sessions = {}

# Store conversation history per session (in-memory)
conversation_history = {}

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

genai.configure(api_key=GEMINI_API_KEY)

# Load RAG artifacts
print("🔄 Loading RAG artifacts...")
faiss_index = faiss.read_index("hospital_index.faiss")
with open("hospital_data.pkl", 'rb') as f:
    hospital_df = pickle.load(f)
with open("network_status.json", 'r', encoding='utf-8') as f:
    network_status_db = json.load(f)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ RAG artifacts loaded successfully")

# Initialize Gemini model with function calling
def get_network_status(hospital_name: str, city: str = None) -> dict:
    """
    Check if a hospital is in the network by searching the actual hospital database.
    
    Args:
        hospital_name: Name of the hospital (can include location like "Manipal Sarjapur")
        city: City where the hospital is located (optional)
    
    Returns:
        dict with network status information
    """
    print(f"🔍 Function called: get_network_status('{hospital_name}', '{city}')")
    
    # Normalize inputs
    hospital_lower = hospital_name.lower().strip()
    search_terms = hospital_lower.split()
    
    print(f"🔍 Search terms: {search_terms}")
    
    # Search in the main hospital dataframe
    matches = []
    
    # For each hospital, check if it matches the search
    for _, row in hospital_df.iterrows():
        hospital_name_db = row['HOSPITAL NAME'].lower()
        address_db = row['Address'].lower()
        city_db = row['CITY'].lower()
        
        # Combine name and address for searching
        full_text = f"{hospital_name_db} {address_db}"
        
        # Check if all search terms are present in name+address
        all_terms_match = all(term in full_text for term in search_terms if len(term) > 2)
        
        # Debug Manipal hospitals
        if 'manipal' in hospital_name_db:
            print(f"🏥 Checking: {row['HOSPITAL NAME'][:70]}")
            print(f"   Address: {address_db[:70]}")
            print(f"   Full text: {full_text[:100]}")
            print(f"   All terms match: {all_terms_match}")
            for term in search_terms:
                if len(term) > 2:
                    print(f"   - '{term}' found: {term in full_text}")
        
        # Apply city filter if provided
        city_match = True
        if city:
            city_lower = city.lower()
            # Handle Bangalore/Bengaluru variations
            if 'bangalor' in city_lower or 'bengalur' in city_lower:
                city_match = 'bangalor' in city_db or 'bengalur' in city_db
            else:
                city_match = city_lower in city_db or city_db in city_lower
        
        if all_terms_match and city_match:
            matches.append({
                "hospital_name": row['HOSPITAL NAME'],
                "city": row['CITY'],
                "address": row['Address'],
                "in_network": True,
                "copay": 20,
                "coverage": "100%"
            })
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return {
            "status": "multiple_found",
            "message": f"I found {len(matches)} hospitals matching your search.",
            "matches": matches[:5]
        }
    else:
        return {
            "status": "not_found",
            "message": f"I couldn't find a hospital matching '{hospital_name}' in our database."
        }

# Define function schema for Gemini
get_network_status_declaration = {
    "name": "get_network_status",
    "description": "Check if a specific hospital is in the network. Use this when user asks about network status or confirmation of a specific hospital.",
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "hospital_name": {
                "type": "STRING",
                "description": "The name of the hospital to check"
            },
            "city": {
                "type": "STRING",
                "description": "The city where the hospital is located (optional but helps with accuracy)"
            }
        },
        "required": ["hospital_name"]
    }
}

# System prompt
SYSTEM_PROMPT = """You are Loop AI, a helpful voice assistant for a hospital network.

YOUR CAPABILITIES:
- Search for hospitals in specific cities
- Confirm if a hospital is in the network
- Answer follow-up questions about hospitals

YOUR BEHAVIOR:
1. Only introduce yourself as "Loop AI" at the VERY START of a brand new conversation (first user query only)
2. For follow-up questions in the same conversation, do NOT re-introduce yourself - just answer directly
3. Be conversational, friendly, and concise in your responses
4. Format responses professionally:
   - Start with a short introductory sentence
   - List each hospital using bullet points with this EXACT structure:
   * Hospital Name, located at full address.
5. Keep consistent spacing, punctuation, and capitalization
6. If user query is unclear, ask clarifying questions (e.g., "Which city?")

OUT-OF-SCOPE HANDLING:
If the user asks about anything OTHER than hospitals (weather, news, math, general knowledge, etc.), respond EXACTLY:
"I'm sorry, I can't help with that. I am forwarding this to a human agent."

IMPORTANT: Use the get_network_status function when user asks "is X hospital in network" or similar confirmation queries.
"""

def search_hospitals_rag(query: str, city: str = None, top_k: int = 5) -> dict:
    """
    Search hospitals using RAG (vector similarity search)
    
    Args:
        query: User's search query
        city: Optional city filter
        top_k: Number of results to return
    
    Returns:
        Dictionary with results and metadata
    """
    print(f"🔍 RAG search: '{query}', city: '{city}'")
    
    # Generate query embedding
    query_embedding = embedding_model.encode([query])[0].astype('float32')
    
    # Search FAISS index
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k * 3)
    
    # Get results
    results = []
    unique_cities = set()
    for idx in indices[0]:
        if idx < len(hospital_df):
            hospital = hospital_df.iloc[idx]
            
            # Apply city filter if provided
            if city and city.lower() not in hospital['CITY'].lower():
                continue
            
            results.append({
                "hospital_name": hospital['HOSPITAL NAME'],
                "city": hospital['CITY'],
                "address": hospital['Address']
            })
            unique_cities.add(hospital['CITY'])
            
            if len(results) >= top_k:
                break
    
    print(f"✅ Found {len(results)} hospitals in {len(unique_cities)} cities")
    return {
        "results": results,
        "total_found": len(results),
        "cities": list(unique_cities),
        "needs_clarification": len(unique_cities) > 1 and city is None
    }

def process_with_gemini(user_message: str, conversation_history: list = None, system_instruction: str = None, is_first_message: bool = False) -> str:
    """
    Process user message with Gemini, handling function calls and RAG
    
    Args:
        user_message: User's text input
        conversation_history: List of previous messages
        system_instruction: Custom system instruction (overrides default if provided)
        is_first_message: Whether this is the first message in a conversation
    
    Returns:
        AI response text
    """
    if conversation_history is None:
        conversation_history = []
    
    if system_instruction is None:
        system_instruction = SYSTEM_PROMPT
    
    # Create model with tools
    model = genai.GenerativeModel(
        model_name='models/gemini-2.5-flash',
        tools=[get_network_status_declaration],
        system_instruction=system_instruction
    )
    
    # Convert conversation history to Gemini format
    gemini_history = []
    for msg in conversation_history:
        # Gemini uses "model" instead of "assistant"
        role = "model" if msg["role"] == "assistant" else msg["role"]
        gemini_history.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    
    # Build chat history
    chat = model.start_chat(history=gemini_history)
    
    # Send message
    response = chat.send_message(user_message)
    
    # Handle function calls
    if response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                # Execute function call
                function_name = part.function_call.name
                function_args = dict(part.function_call.args)
                
                print(f"🔧 Executing function: {function_name}({function_args})")
                
                if function_name == "get_network_status":
                    result = get_network_status(**function_args)
                    
                    # Send function response back to model
                    response = chat.send_message(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=function_name,
                                    response={"result": result}
                                )
                            )]
                        )
                    )
    
    # Check if RAG search is needed (heuristic: query contains "hospital", "near", "around", city names)
    search_keywords = ["hospital", "around", "near", "in", "list", "tell me", "show me", "know", "any"]
    if any(keyword in user_message.lower() for keyword in search_keywords):
        # Extract city from message (simple approach)
        city = None
        for _, row in hospital_df.iterrows():
            if row['CITY'].lower() in user_message.lower():
                city = row['CITY']
                break
        
        # Perform RAG search
        if city or any(word in user_message.lower() for word in ["hospital", "hospitals"]):
            search_result = search_hospitals_rag(user_message, city=city, top_k=5)
            hospitals = search_result["results"]
            
            if hospitals:
                # Check if clarification is needed
                if search_result["needs_clarification"]:
                    # Multiple cities found, ask for clarification
                    cities_list = ", ".join(search_result["cities"][:5])
                    clarification_message = f"I found several hospitals matching your search in multiple cities: {cities_list}. In which city are you looking for the hospital?"
                    
                    # Add introduction for first message
                    if is_first_message:
                        clarification_message = f"Hello! I'm Loop AI. {clarification_message}"
                    
                    return clarification_message
                else:
                    # Single city or city specified, show results
                    hospital_info = "\n\n".join([
                        f"- {h['hospital_name']} in {h['city']}, {h['address']}"
                        for h in hospitals
                    ])
                    
                    # Get final response with context
                    enhanced_prompt = f"Based on the search results:\n{hospital_info}\n\nProvide a natural response to: {user_message}"
                    response = chat.send_message(enhanced_prompt)
    
    # Handle function call responses safely
    if response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text:
                return part.text
    
    return response.text

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_file(filename):
    """Serve static files like images"""
    try:
        return send_from_directory('.', filename)
    except:
        return "File not found", 404

@app.route('/test_audio', methods=['POST'])
def test_audio():
    """Test endpoint to check if audio is being received"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file"}), 400
        
        audio_file = request.files['audio']
        file_size = len(audio_file.read())
        audio_file.seek(0)
        
        return jsonify({
            "status": "Audio received",
            "filename": audio_file.filename,
            "content_type": audio_file.content_type,
            "size": file_size
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_text', methods=['POST'])
def process_text():
    """Process text query without voice (for example prompts)"""
    try:
        data = request.get_json()
        text_query = data.get('text', '').strip()
        
        if not text_query:
            return jsonify({'error': 'No text provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"TEXT QUERY PROCESSING")
        print(f"{'='*60}")
        print(f"Query: {text_query}")
        
        # Process with Gemini and get response
        response_text = process_with_gemini(text_query)
        print(f"\nGemini Response: {response_text}")
        
        # Generate speech using edge-tts
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb') as temp_file:
            temp_path = temp_file.name
        
        async def generate_speech():
            communicate = edge_tts.Communicate(response_text, "en-US-AriaNeural")
            await communicate.save(temp_path)
        
        asyncio.run(generate_speech())
        
        # Read the audio file
        with open(temp_path, 'rb') as f:
            audio_data = f.read()
        
        # Cleanup
        os.unlink(temp_path)
        
        return jsonify({
            'transcript': text_query,
            'response': response_text,
            'audio': audio_data.hex()
        })
        
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_voice', methods=['POST'])
def process_voice():
    """
    Main endpoint: receives audio, processes with STT -> Gemini -> TTS
    """
    temp_audio_path = None
    temp_response_path = None
    
    try:
        print("\n" + "="*60)
        print("🎤 New voice request received")
        
        # Get session ID from request (or create new one)
        session_id = request.form.get('session_id', 'default')
        
        # Initialize session if new
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = []
            print(f"📝 New conversation session: {session_id}")
        else:
            print(f"📝 Continuing session: {session_id} ({len(conversation_sessions[session_id])} messages)")
        
        # Get audio file from request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        print(f"📁 Received file: {audio_file.filename}, Content-Type: {audio_file.content_type}")
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm', mode='wb') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
            print(f"💾 Saved to: {temp_audio_path}")
        
        # Get file size for debugging
        file_size = os.path.getsize(temp_audio_path)
        print(f"📏 File size: {file_size} bytes")
        
        if file_size < 100:
            return jsonify({"error": "Audio file too small. Please record for at least 1 second."}), 400
        
        # Speech-to-Text using Google's API
        print("🔄 Converting speech to text...")
        recognizer = sr.Recognizer()
        
        user_text = None
        wav_path = None
        
        try:
            # Convert WebM to WAV using pydub
            print("🔄 Converting WebM to WAV...")
            audio = AudioSegment.from_file(temp_audio_path, format="webm")
            
            # Export as WAV
            wav_path = temp_audio_path.replace('.webm', '.wav')
            audio.export(wav_path, format="wav")
            print(f"✅ Converted to WAV: {wav_path}")
            
            # Now use SpeechRecognition with the WAV file
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                
                # Retry logic for speech recognition
                max_retries = 3
                retry_count = 0
                last_error = None
                
                while retry_count < max_retries:
                    try:
                        # Use recognize_google with longer timeout
                        user_text = recognizer.recognize_google(
                            audio_data, 
                            show_all=False,
                            with_confidence=False
                        )
                        print(f"✅ Transcription successful: {user_text}")
                        break
                    except sr.RequestError as e:
                        retry_count += 1
                        last_error = e
                        print(f"⚠️ Speech recognition attempt {retry_count} failed: {e}")
                        if retry_count < max_retries:
                            time.sleep(1)  # Wait 1 second before retry
                        continue
                    except sr.UnknownValueError:
                        print("⚠️ Could not understand audio")
                        break
                
                if retry_count >= max_retries and not user_text:
                    raise sr.RequestError(f"Failed after {max_retries} attempts: {last_error}")
                
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return jsonify({"error": "Could not connect to speech recognition service. Please check your internet connection and try again."}), 503
        except Exception as e:
            print(f"❌ Conversion/transcription error: {e}")
            return jsonify({"error": f"Audio processing failed: {str(e)}"}), 400
        finally:
            # Clean up WAV file
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
        
        if not user_text:
            return jsonify({"error": "Could not understand audio. Please speak clearly for at least 2 seconds and try again."}), 400
        
        print(f"📝 User said: {user_text}")
        
        # Get conversation history for this session
        session_history = conversation_sessions[session_id]
        is_first_message = len(session_history) == 0
        
        # Add system instruction for first message
        system_instruction = SYSTEM_PROMPT
        if is_first_message:
            system_instruction += "\n\nIMPORTANT: This is the FIRST message in a new conversation. You MUST start your response with 'Hello! I'm Loop AI.' or 'Hi! I'm Loop AI.' to introduce yourself."
        
        # Check for out-of-scope
        out_of_scope_keywords = ["weather", "news", "joke", "calculate", "math", "cook", "recipe"]
        is_out_of_scope = False
        
        if any(keyword in user_text.lower() for keyword in out_of_scope_keywords):
            if not any(word in user_text.lower() for word in ["hospital", "network", "doctor"]):
                response_text = "I'm sorry, I can't help with that. I am forwarding this to a human agent."
                print(f"⚠️ Out-of-scope query detected - ending session")
                is_out_of_scope = True
            else:
                response_text = process_with_gemini(user_text, session_history, system_instruction, is_first_message)
        else:
            # Process with Gemini
            print("🤖 Processing with Gemini...")
            response_text = process_with_gemini(user_text, session_history, system_instruction, is_first_message)
        
        # Add introduction on first message if not already present
        if is_first_message and not response_text.lower().startswith(("hello", "hi")):
            response_text = f"Hello! I'm Loop AI. {response_text}"
        
        # Store in conversation history
        session_history.append({"role": "user", "content": user_text})
        session_history.append({"role": "assistant", "content": response_text})
        
        print(f"💬 AI response: {response_text}")
        
        # If out-of-scope, clear the session to end interaction
        if is_out_of_scope:
            # Keep the message for this response, but mark session for termination
            conversation_sessions[session_id].append({"terminated": True})
        
        # Text-to-Speech using edge-tts
        print("🔊 Converting text to speech...")
        
        # Create temp file for audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3', mode='wb') as temp_response:
            temp_response_path = temp_response.name
        
        # Generate speech using edge-tts
        async def generate_speech():
            communicate = edge_tts.Communicate(response_text, "en-US-AriaNeural")
            await communicate.save(temp_response_path)
        
        asyncio.run(generate_speech())
        
        # Read audio file
        with open(temp_response_path, 'rb') as f:
            audio_data = f.read()
        
        print("✅ Request processed successfully")
        print("="*60 + "\n")
        
        result = jsonify({
            "transcript": user_text,
            "response": response_text,
            "audio": audio_data.hex(),
            "session_ended": is_out_of_scope  # Signal frontend to end session
        })
        
        # Cleanup
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if temp_response_path and os.path.exists(temp_response_path):
            os.unlink(temp_response_path)
        
        return result
        
    except sr.UnknownValueError:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return jsonify({"error": "Could not understand audio. Please speak clearly and try again."}), 400
    except sr.RequestError as e:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        return jsonify({"error": f"Speech recognition service error: {str(e)}"}), 500
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        if temp_response_path and os.path.exists(temp_response_path):
            os.unlink(temp_response_path)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "hospitals_loaded": len(hospital_df),
        "faiss_index_size": faiss_index.ntotal
    })

# ============================================================
# TWILIO VOICE INTEGRATION
# ============================================================

@app.route('/twilio/voice', methods=['POST'])
def twilio_voice():
    """Handle incoming Twilio voice calls"""
    print("\n" + "="*60)
    print("📞 Incoming Twilio call")
    print("="*60)
    
    response = VoiceResponse()
    
    # Get or create session for this call
    call_sid = request.form.get('CallSid')
    
    if call_sid not in conversation_sessions:
        conversation_sessions[call_sid] = []
        # Welcome message
        response.say("Hello! I'm Loop AI. How can I assist you today?", voice='Polly.Joanna')
    
    # Gather user's speech
    gather = Gather(
        input='speech',
        action='/twilio/process',
        method='POST',
        speech_timeout='auto',
        language='en-US'
    )
    
    response.append(gather)
    response.say("I didn't hear anything. Please call back.", voice='Polly.Joanna')
    
    return str(response), 200, {'Content-Type': 'text/xml'}

@app.route('/twilio/process', methods=['POST'])
def twilio_process():
    """Process transcribed speech from Twilio"""
    print("\n" + "="*60)
    print("🎤 Processing Twilio speech")
    print("="*60)
    
    call_sid = request.form.get('CallSid')
    user_text = request.form.get('SpeechResult', '')
    
    print(f"📝 User said: {user_text}")
    
    # Get or create session
    if call_sid not in conversation_sessions:
        conversation_sessions[call_sid] = []
    
    session_history = conversation_sessions[call_sid]
    is_first_message = len(session_history) == 0
    
    # Process with Gemini
    system_instruction = SYSTEM_PROMPT
    if is_first_message:
        system_instruction += "\n\nIMPORTANT: Keep responses concise for phone calls (under 30 seconds)."
    
    try:
        # Check for out-of-scope
        out_of_scope_keywords = ["weather", "news", "joke", "calculate", "math", "cook", "recipe"]
        if any(keyword in user_text.lower() for keyword in out_of_scope_keywords):
            if not any(word in user_text.lower() for word in ["hospital", "network", "doctor"]):
                response_text = "I'm sorry, I can't help with that. I am forwarding this to a human agent. Goodbye."
                
                # Store and end call
                session_history.append({"role": "user", "content": user_text})
                session_history.append({"role": "assistant", "content": response_text})
                
                response = VoiceResponse()
                response.say(response_text, voice='Polly.Joanna')
                response.hangup()
                return str(response), 200, {'Content-Type': 'text/xml'}
        
        # Process with Gemini
        response_text = process_with_gemini(user_text, session_history, system_instruction, is_first_message)
        
        # Store conversation
        session_history.append({"role": "user", "content": user_text})
        session_history.append({"role": "assistant", "content": response_text})
        
        print(f"💬 AI response: {response_text}")
        
        # Create TwiML response
        response = VoiceResponse()
        response.say(response_text, voice='Polly.Joanna')
        
        # Continue conversation
        gather = Gather(
            input='speech',
            action='/twilio/process',
            method='POST',
            speech_timeout='auto',
            language='en-US'
        )
        response.append(gather)
        
        # Timeout message
        response.say("Thank you for calling. Goodbye!", voice='Polly.Joanna')
        
        return str(response), 200, {'Content-Type': 'text/xml'}
        
    except Exception as e:
        print(f"❌ Error processing Twilio request: {e}")
        response = VoiceResponse()
        response.say("I'm sorry, there was an error. Please try again later.", voice='Polly.Joanna')
        response.hangup()
        return str(response), 200, {'Content-Type': 'text/xml'}

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 Loop AI Hospital Network Assistant")
    print("="*60)
    print("✅ Server starting on http://localhost:5000")
    print("📱 Open the URL in your browser to use the voice interface")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
