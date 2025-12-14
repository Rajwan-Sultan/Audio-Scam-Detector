import asyncio
import json
import os
from typing import Optional, List, Literal, Annotated
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Real-time Scam Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# from google import genai
# from google.genai import types

from google import genai
from google.genai import types



API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = None

try:
    gemini_client = genai.Client(api_key=API_KEY)
    print("✓ Gemini client initialized successfully")
except Exception as e:
    print(f"✗ Warning: Gemini client initialization failed: {e}")

class ScamTag(BaseModel):
    tag: Literal[
        "Robocall",
        "Phishing",
        "Telemarketing",
        "Debt Collection",
        "Survey Scam",
        "Charity Scam",
        "Tech Support Scam",
        "Lottery Scam",
        "Investment Scam",
        "Other"
    ] = Field(..., description="Type of scam detected")

class RiskFactors(BaseModel):
    urgency_pressure: Annotated[Optional[int], Field(ge=0, le=10, description="Risk factor score for urgency and pressure tactics")] = None
    suspicious_payment: Annotated[Optional[int], Field(ge=0, le=10, description="Risk factor score for suspicious payment requests")] = None
    impersonation: Annotated[Optional[int], Field(ge=0, le=10, description="Risk factor score for impersonation attempts")] = None
    information_request: Annotated[Optional[int], Field(ge=0, le=10, description="Risk factor score for personal information requests")] = None
    emotional_manipulation: Annotated[Optional[int], Field(ge=0, le=10, description="Risk factor score for emotional manipulation tactics")] = None

class OutputSchema(BaseModel):
    scam_detected: Annotated[Literal['scam','legitimate'], Field(description="Indicates if the call is a scam or not")] = None
    confidence_score: Annotated[Optional[int], Field(ge=0, le=100, description="Confidence score in percentage from 0 to 100")] = None
    risk_factors: Annotated[Optional[RiskFactors], Field(description="Detailed risk factor scores")] = None
    explanation: Annotated[Optional[str], Field(description="Summary of the call in two lines")] = None
    explanation_arabic: Annotated[Optional[str], Field(description="Summary of the call in Arabic in two lines")] = None
    scam_tags: Annotated[Optional[List[ScamTag]], Field(description="List of scam tags detected in the call")] = None

def check_mime_type(file: UploadFile):
    ext = file.filename.split('.')[-1].lower()
    allowed_extensions = {
         'flac':'audio/flac', 
         'mp3':'audio/mp3',
         'm4a':'audio/m4a', 
         'mpeg':'audio/mpeg',
         'mpga':'audio/mpga', 
         'mp4':'audio/mp4',
         'ogg': 'audio/ogg', 
         'pcm': 'audio/pcm', 
         'wav': 'audio/wav', 
         'webm': 'audio/webm' 
    }
    if ext not in allowed_extensions:
        raise Exception(f"Unsupported file extension: {ext}")
    return allowed_extensions[ext]

def prompt_text():
    prompt = (
    """You are an expert fraud detection analyst specializing in telecommunications scams.Strictly remove any markdown formatting from your response.
    # CLASSIFICATION CRITERIA
    ## CLASSIFY AS "SCAM" IF ANY OF THESE RED FLAGS ARE PRESENT:
    - **Urgency & Pressure**: Immediate threats, limited-time offers, demands for quick action
    - **Suspicious Payment Requests**: Gift cards, wire transfers, cryptocurrency, unconventional payment methods
    - **Impersonation**: Claiming to be from government agencies (IRS, SSA), tech support, utilities, or banks without verification
    - **Personal Information Requests**: Asking for SSN, passwords, bank details, or remote computer access
    - **Too-Good-To-Be-True Offers**: Prizes, grants, or deals that seem unrealistic
    - **Threats & Intimidation**: Legal threats, account suspension warnings, or arrest warrants
    - **Secrecy Demands**: Instructions not to tell family, friends, or financial institutions
    ## Only give the Arabic translation in the explanation_arabic field without any other text. Ony for this part the translation should be in Arabic and every other part of the response should be in English. Follow this strictly.
    ## CLASSIFY AS "AMBIGUOUS" IF:
    - Some suspicious elements exist but no clear scam indicators
    - Caller identity cannot be verified but no direct requests for money/information
    - Mixed signals - some legitimate aspects with minor red flags
    - Insufficient information to make definitive classification  
    - Potential telemarketing or survey calls with borderline practices
    ## CLASSIFY AS "LEGITIMATE" IF:
    - Caller identity is verifiable and matches claimed organization
    - No pressure tactics or urgency
    - Standard business practices (appointment reminders, customer service follow-ups)
    - Clear, transparent purpose with no hidden requests
    - Professional tone without emotional manipulation
    # ANALYSIS GUIDELINES
    1. Be conservative - when in doubt, classify as AMBIGUOUS
    2. Consider the context and caller's stated purpose
    3. Look for patterns rather than isolated phrases
    4. Weight multiple weak indicators higher than single strong ones
    5. Consider cultural and regional communication norms\n\n
    Extract the call data and return ONLY valid JSON. No explanations or formatting.\nFocus on relevant sections like if call is a possible scam, reason for contacting the person and the tags for possible scam\n
    If any information is missing, then just leave it empty or use empty arrays.\nIgnore irrelevant sections like references, hobbies, or template artifacts.\n\n""")
    return prompt

async def parse_with_llm_audio(audio_bytes: bytes, mime_type: str, segment_number: int = 1) -> OutputSchema:
    if not gemini_client:
        raise Exception("Gemini client not available")
    
    prompt = prompt_text()
    prompt += f"\n\nThis is segment #{segment_number} of an ongoing conversation (approximately 10 seconds of audio). Analyze this segment for scam indicators."
    
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_bytes(
            data=audio_bytes,
            mime_type=mime_type,
        )
    ]

    generation_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=1000, 
        response_mime_type="application/json",
        response_schema=OutputSchema.model_json_schema()
    )

    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash-lite",
        # model="gemini-2.0-flash-exp",
        contents=parts,
        config=generation_config
    )
    
    if not getattr(response, "text", None):
        raise Exception("Empty LLM response")
    return OutputSchema.model_validate_json(response.text)

async def parse_with_llm_text(file_content: str) -> OutputSchema:
    if not gemini_client:
        raise Exception("Gemini client not available")
    
    prompt = prompt_text()
    
    parts = [
        types.Part.from_text(text=prompt),
        types.Part.from_text(text=file_content),
    ]

    generation_config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=1000, 
        response_mime_type="application/json",
        response_schema=OutputSchema.model_json_schema()
    )

    response = await gemini_client.aio.models.generate_content(
        # model="gemini-2.0-flash-exp",
        model="gemini-2.0-flash-lite",
        contents=parts,
        config=generation_config
    )
     
    if not getattr(response, "text", None):
        raise Exception("Empty LLM response")
    return OutputSchema.model_validate_json(response.text)

class ConversationState:
    def __init__(self):
        self.segments_analyzed = 0
        self.scam_detections = []
        self.highest_confidence = 0
        self.cumulative_risk = {
            "urgency_pressure": 0,
            "suspicious_payment": 0,
            "impersonation": 0,
            "information_request": 0,
            "emotional_manipulation": 0
        }
        self.all_tags = set()
    
    def add_segment_result(self, result: OutputSchema):
        self.segments_analyzed += 1
        self.scam_detections.append(result.scam_detected)
        
        if result.confidence_score:
            self.highest_confidence = max(self.highest_confidence, result.confidence_score)
        
        if result.risk_factors:
            factors = result.risk_factors.model_dump()
            for key, value in factors.items():
                if value is not None:
                    self.cumulative_risk[key] = max(self.cumulative_risk[key], value)
        
        if result.scam_tags:
            for tag in result.scam_tags:
                self.all_tags.add(tag.tag)
    
    def get_overall_assessment(self) -> dict:
        scam_count = self.scam_detections.count('scam')
        legitimate_count = self.scam_detections.count('legitimate')
        ambiguous_count = self.scam_detections.count('ambiguous')
        
        if scam_count > 0:
            overall = "scam"
        elif legitimate_count > ambiguous_count:
            overall = "legitimate"
        else:
            overall = "ambiguous"
        
        return {
            "overall_verdict": overall,
            "segments_analyzed": self.segments_analyzed,
            "highest_confidence": self.highest_confidence,
            "scam_segments": scam_count,
            "legitimate_segments": legitimate_count,
            "ambiguous_segments": ambiguous_count,
            "cumulative_risk_factors": self.cumulative_risk,
            "all_detected_tags": list(self.all_tags)
        }

@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    await websocket.accept()
    conversation_state = ConversationState()
    mime_type = "audio/mp3"
    connection_established = False
    INACTIVITY_TIMEOUT = 30  # seconds
    
    print(f"✓ WebSocket client connected - Waiting for first audio file")
    
    try:
        while True:
            try:
                # Wait for audio data with 30-second timeout
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=INACTIVITY_TIMEOUT
                )
                
                # Handle different message types
                if "bytes" in message:
                    audio_segment = message["bytes"]
                    
                    # Establish connection on first file
                    if not connection_established:
                        connection_established = True
                        await websocket.send_json({
                            "type": "connected",
                            "message": "Connection established. Ready to analyze audio segments.",
                            "config": {
                                "mime_type": mime_type,
                                "inactivity_timeout": f"{INACTIVITY_TIMEOUT} seconds"
                            }
                        })
                        print(f"✓ Connection established with first file")
                    
                    segment_number = conversation_state.segments_analyzed + 1
                    print(f"→ Received segment #{segment_number} ({len(audio_segment)} bytes)")
                    
                    await websocket.send_json({
                        "type": "segment_received",
                        "segment_number": segment_number,
                        "segment_size_bytes": len(audio_segment)
                    })
                    
                    try:
                        await websocket.send_json({
                            "type": "analyzing",
                            "message": f"Analyzing segment #{segment_number}...",
                            "segment_number": segment_number
                        })
                        
                        result = await parse_with_llm_audio(audio_segment, mime_type, segment_number)
                        conversation_state.add_segment_result(result)
                        overall = conversation_state.get_overall_assessment()
                        
                        await websocket.send_json({
                            "type": "analysis",
                            "segment_number": segment_number,
                            "segment_analysis": result.model_dump(),
                            "cumulative_assessment": overall
                        })
                        
                        print(f"✓ Segment #{segment_number} analyzed - Scam: {result.scam_detected}, Overall: {overall['overall_verdict']}")
                        
                    except Exception as e:
                        print(f"✗ Analysis error for segment #{segment_number}: {str(e)}")
                        await websocket.send_json({
                            "type": "error",
                            "segment_number": segment_number,
                            "message": f"Analysis failed: {str(e)}"
                        })
                
                elif "text" in message:
                    # Handle text messages (like config or close command)
                    text_data = message["text"]
                    try:
                        data = json.loads(text_data)
                        if data.get("action") == "close":
                            await websocket.send_json({
                                "type": "closing",
                                "message": "Connection closing as requested",
                                "summary": conversation_state.get_overall_assessment()
                            })
                            break
                        elif "mime_type" in data:
                            mime_type = data.get("mime_type", "audio/mp3")
                    except json.JSONDecodeError:
                        pass
                        
            except asyncio.TimeoutError:
                # No activity for 30 seconds
                print(f"⏱ Connection timeout after {INACTIVITY_TIMEOUT}s of inactivity")
                await websocket.send_json({
                    "type": "timeout",
                    "message": f"Connection closed due to {INACTIVITY_TIMEOUT}s inactivity",
                    "summary": conversation_state.get_overall_assessment()
                })
                break
                    
    except WebSocketDisconnect:
        print(f"✗ Client disconnected - Analyzed {conversation_state.segments_analyzed} segments")
    except Exception as e:
        print(f"✗ WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except:
            pass
    finally:
        print(f"✓ Connection closed - Total segments analyzed: {conversation_state.segments_analyzed}")

@app.post("/api/v1/analyzer_audio", response_model=OutputSchema)
async def analyze_audio(file: UploadFile = File(...)):
    try:
        mime_type = check_mime_type(file)
        content = await file.read()
        result = await parse_with_llm_audio(content, mime_type)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing audio file: {str(e)}"
        )

@app.post("/analyzer_text", response_model=OutputSchema)
async def analyze_text(file: str):
    try:
        result = await parse_with_llm_text(file)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text file: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "service": "Real-time Scam Detection API",
        "endpoints": {
            "websocket": "/ws/analyze",
            "rest_audio": "/api/v1/analyzer_audio",
            "rest_text": "/analyzer_text",
            "health": "/health"
        },
        "status": "running"
    }

@app.get("/health")
async def health_check():
    gemini_status = "connected" if gemini_client else "disconnected"
    return {
        "status": "healthy",
        "gemini_client": gemini_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001,reload=True)
    












