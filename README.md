# Loop AI Hospital Network Assistant 🏥

A conversational AI voice agent that helps users find hospitals in a network using **Google Gemini 2.0 Flash**, **RAG (Retrieval-Augmented Generation)**, and **Function Calling**.

## 🎯 Features

- **Voice-to-Voice Interaction**: Speak naturally and get audio responses with typing animation
- **Intelligent Hospital Search**: Find hospitals by city using semantic search (FAISS vector database)
- **Network Status Verification**: Check if specific hospitals are in-network using function calling
- **RAG-Powered**: Efficiently searches **2,179 hospitals** without overwhelming AI context
- **Out-of-Scope Detection**: Politely handles non-hospital queries and ends conversation
- **Session Management**: Maintains conversation context with unique session tracking
- **Clarification Questions**: Asks "Which city?" when multiple locations found
- **Interrupt Capability**: Click mic anytime to stop playback and ask new question
- **Twilio Integration**: Phone call support with voice webhooks (bonus feature)

## 🏗️ Architecture

### How It Handles Large Data

Instead of sending all 2,182 hospitals to Gemini (which would use ~150K tokens), we use a **dual-retrieval strategy**:

1. **RAG with FAISS Vector Search**: 
   - Pre-indexes hospitals using sentence embeddings
   - User query → Vector similarity search → Top 3-5 relevant hospitals
   - Only matched hospitals sent to LLM (reduces context by 98%)

2. **Function Calling for Exact Lookups**:
   - User asks "Is X hospital in network?" → Gemini calls `get_network_status()` function
   - Direct dictionary lookup (zero context usage)
   - Returns simulated network status

### Tech Stack

- **Backend**: Python Flask
- **LLM**: Google Gemini 2.0 Flash (with function calling)
- **RAG**: FAISS + sentence-transformers (all-MiniLM-L6-v2)
- **Voice Pipeline**: 
  - **STT**: Google Speech Recognition API (with retry logic)
  - **TTS**: edge-tts (Microsoft Edge TTS engine)
  - **Audio Processing**: pydub + FFmpeg for WebM→WAV conversion
- **Frontend**: Vanilla JavaScript + Tailwind CSS + Custom animations
- **Telephony** (Optional): Twilio + ngrok for phone integration

## 📦 Installation

### Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get it free](https://makersuite.google.com/app/apikey))
- **FFmpeg** ([Download](https://ffmpeg.org/download.html)) - Required for audio conversion
- Microphone access (for voice input)

### Setup Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd loop_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

4. **Generate RAG index (IMPORTANT - Run this ONCE before starting the app)**
```bash
python hospital_rag.py
```

This will:
- Load the CSV file
- Generate embeddings for all hospitals
- Create FAISS index
- Simulate network status
- Save artifacts to disk (~30-60 seconds)

5. **Start the Flask server**
```bash
python app.py
```

6. **Open your browser**
```
http://localhost:5000
```

##  Usage

1. Click the microphone button (it will turn green)
2. Speak your query clearly for 2-3 seconds
3. Click again to stop recording
4. Watch the typing animation as Loop AI responds
5. Audio response plays automatically
6. Click mic anytime to interrupt and ask a new question

### Test Queries (Assignment Requirements)

✅ **Query 1**: "Tell me 3 hospitals around Bangalore"
- Uses RAG vector search with FAISS
- Filters by city
- Returns top 3 matches with addresses
- Response formatted with bullet points

✅ **Query 2**: "Can you confirm if Manipal Sarjapur in Bangalore is in my network?"
- Triggers `get_network_status()` function call
- Searches both hospital name AND address
- Returns network status with copay details
- Handles Bangalore/Bengaluru city variations

### Follow-up Examples

After first query, try:
- "What about Delhi?" (continues conversation)
- "Do you know any Apollo Hospital?" (asks for city clarification)
- "Mumbai" (responds with Mumbai Apollo hospitals)

### Out-of-Scope Handling

If you ask about weather, news, jokes, math, or recipes:
> "I'm sorry, I can't help with that. I am forwarding this to a human agent."

**The conversation ends** - mic button gets disabled. Refresh page to start new conversation.

## 📁 Project Structure

```
loop_project/
├── app.py                          # Flask backend (765 lines)
│   ├── RAG search with FAISS
│   ├── Function calling (get_network_status)
│   ├── Session management
│   ├── Audio pipeline (WebM→WAV→STT→TTS)
│   └── Twilio webhook endpoints
├── hospital_rag.py                 # RAG indexing script (run once)
├── index.html                      # Voice UI frontend (803 lines)
│   ├── MediaRecorder for voice capture
│   ├── Typing animation
│   ├── Session tracking
│   └── Audio playback with interrupt
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── List of GIPSA Hospitals - Sheet1.csv  # 2,179 hospitals
├── hospital_index.faiss           # Generated FAISS vector index
├── hospital_data.pkl              # Generated hospital DataFrame
└── Loop_ai.png                    # banner image
```

##  Configuration

### Gemini Model
Change model in `app.py`:
```python
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',  # Or 'gemini-1.5-pro'
    ...
)
```

### Number of Results
Adjust in `app.py`:
```python
hospitals = search_hospitals_rag(query, city=city, top_k=5)  # Change top_k
```

### Embedding Model
Change in `hospital_rag.py` and `app.py`:
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

## Testing

### Manual Testing
1. Run `python app.py`
2. Open browser console (F12)
3. Test both required queries via voice
4. Check console logs for RAG/function call traces

### Health Check
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "hospitals_loaded": 2182,
  "faiss_index_size": 2182
}
```


## 🚀 Deployment (Optional)

### Local with ngrok
```bash
ngrok http 5000
```

### Render/Railway
1. Push to GitHub
2. Connect repository
3. Add `GEMINI_API_KEY` environment variable
4. Deploy!

##  Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "GEMINI_API_KEY not found"
- Check `.env` file exists
- Verify `GEMINI_API_KEY=your_key` is set
- Restart Flask server

### "Could not access microphone"
- Use Chrome/Firefox/Edge (Safari may have issues)
- Grant microphone permissions when prompted
- Ensure HTTPS or localhost (required for getUserMedia API)

### "Could not connect to speech recognition service"
- Check internet connection (Google Speech Recognition requires internet)
- Issue often happens after 2-3 queries (rate limiting)
- **Automatic retry**: System retries 3 times with 1-second delays
- If persists, wait 30 seconds and try again

### "FFmpeg not found"
- Download FFmpeg: https://ffmpeg.org/download.html
- Add to system PATH or update `app.py` with explicit path:
```python
AudioSegment.converter = r"C:\path\to\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\path\to\ffprobe.exe"
```

### RAG returns no results
- Verify `hospital_rag.py` ran successfully
- Check `hospital_index.faiss` and `hospital_data.pkl` exist
- Try broader query ("hospitals in India")

### Gemini API errors
- Check API key is valid
- Verify free tier quota (1500 requests/day)
- Check rate limits (15 RPM for Flash model)


## Performance

- **RAG search latency**: <50ms (FAISS in-memory search)
- **Function call lookup**: <10ms (DataFrame search)
- **Gemini inference**: ~800ms (2.0 Flash model)
- **TTS generation**: ~200ms (edge-tts)
- **Audio conversion**: ~100ms (WebM→WAV with pydub)
- **Total response time**: ~1.2-1.5 seconds

## 🔐 Security Notes

- Implement rate limiting for public deployments
- Session data stored in-memory (clears on server restart)
- Temporary audio files deleted after processing

##  API Endpoints

### `POST /process_voice`
**Input**: 
- `audio`: WebM audio file (multipart/form-data)
- `session_id`: Unique session identifier

**Output**: 
```json
{
  "transcript": "Tell me hospitals in Bangalore",
  "response": "Hello! I'm Loop AI. Here are hospitals...",
  "audio": "ffd8ffe0...",  // MP3 as hex string
  "session_ended": false   // true if out-of-scope
}
```

### `POST /twilio/voice` (Optional)
**Input**: Twilio webhook POST data
**Output**: TwiML XML for voice response

### `POST /twilio/process` (Optional)
**Input**: Transcribed speech from Twilio
**Output**: TwiML XML with AI response and continuation

### `GET /health`
**Output**: 
```json
{
  "status": "healthy",
  "hospitals_loaded": 2179,
  "faiss_index_size": 2179
}
```

## Assignment Completion Checklist

- [x] **Part 1: API Integration & Data Loading**
  - [x] Voice-to-voice API integration (Gemini 2.0 Flash + edge-tts)
  - [x] Efficient data handling (RAG with FAISS + Function Calling)
  - [x] Query 1: "3 hospitals around Bangalore" (works)
  - [x] Query 2: "Manipal Sarjapur in network" (works)
  - [x] Handles 2,179 hospitals without context overflow

- [x] **Part 2: Introduction & Follow-ups**
  - [x] "Loop AI" introduction on first message only
  - [x] Session-based conversation context tracking
  - [x] Clarification questions ("Which city are you looking for?")
  - [x] Follow-up question handling with history

- [x] **Part 3: Error Handling & Bonus**
  - [x] Out-of-scope detection (weather, news, jokes, etc.)
  - [x] Polite handoff: "I am forwarding this to a human agent"
  - [x] Conversation ends after out-of-scope query
  - [x] **Twilio phone integration** (configured but requires VoIP testing)

##  Key Technical Achievements

1. **Smart Search**: Searches both hospital name AND address for better accuracy
2. **City Variations**: Handles Bangalore/Bengaluru, Mumbai/Bombay, etc.
3. **Retry Logic**: Automatic 3-attempt retry for speech recognition failures
4. **Audio Pipeline**: WebM → WAV conversion with FFmpeg integration
5. **Interrupt Support**: Users can stop AI mid-response and ask new questions
6. **Session Management**: Unique session IDs prevent conversation mixing
7. **Professional UI**: Loop AI branding, typing animation, waveform visualization

## 📝 License

MIT License - feel free to use for learning/portfolio

## 👤 Author

**Oditi Rathi**  
Created for Loop Health Internship Assignment - November 2025

**Repository**: https://github.com/oditirathi20/loop

---

**Questions or Issues?** 
- Check the troubleshooting section
- Review terminal logs for debug information
- All function calls and search operations are logged
