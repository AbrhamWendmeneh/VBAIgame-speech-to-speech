# Adventure Game
import os
os.environ['SDL_VIDEODRIVER'] = 'x11'  # Use Linux system
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
import sys
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import time
import websocket
import pyaudio
import wave
import threading
import json
import base64
from pydub import AudioSegment
from pydub.playback import play
import queue

# Load environment variables
load_dotenv()
# Ensure OpenAI API Key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("[OpenAI] API key not found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)
client = OpenAI(api_key=api_key)
print("[OpenAI] API key loaded successfully.")

os.environ['SDL_VIDEODRIVER'] = 'windib'  # Set the video driver to windib

# Initialize Pygame with macOS specific settings
pygame.init()
pygame.display.init()
display = (800, 600)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
screen = pygame.display.get_surface()

glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Set up basic lighting
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 5, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

# Enable blending for transparency
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Initial camera position
glTranslatef(0.0, 0.0, -5)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
TILE_SIZE = 32
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# Game map
GAME_MAP = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W..................W",
    "W........N.........W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W....P.............W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWWWW"
]

# Add these constants near the other constants
TITLE = "Venture Builder AI"
SUBTITLE = "Our Digital Employees"
MENU_BG_COLOR = (0, 0, 0)  # Black background
MENU_TEXT_COLOR = (0, 255, 0)  # Matrix-style green
MENU_HIGHLIGHT_COLOR = (0, 200, 0)  # Slightly darker green for effects

# Add these constants after the other constants
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

def draw_cube():
    vertices = [
        # Front face
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
        # Back face
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
    ]
    
    surfaces = [
        [0, 1, 2, 3],  # Front
        [3, 2, 6, 5],  # Top
        [0, 3, 5, 4],  # Left
        [1, 7, 6, 2],  # Right
        [4, 5, 6, 7],  # Back
        [0, 4, 7, 1],  # Bottom
    ]
    
    glBegin(GL_QUADS)
    for surface in surfaces:
        glNormal3f(0, 0, 1)  # Simple normal for lighting
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)
        
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)
        
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()

class DialogueSystem:
    def __init__(self):
        self.active = False
        self.user_input = ""
        try:
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            print("[DialogueSystem] Font loaded successfully")
        except Exception as e:
            print("[DialogueSystem] Font loading failed:", e)
        self.npc_message = ""
        self.input_active = False
        self.last_npc_text = ""     # Track NPC text separately
        self.last_input_text = ""   # Track input text separately
        self.conversation_history = []  # Maintain conversation history

        # Create a surface for the UI
        self.ui_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.ui_texture = glGenTextures(1)
        self.current_npc = None  # Track which NPC we're talking to
        self.initial_player_pos = None  # Store initial position when dialogue starts

        # Add these new attributes
        self.audio_manager = AudioManager()
        self.voice_mode = False
        self.ws = None
        self.setup_websocket()

        # Add new status indicators
        self.is_listening = False
        self.is_speaking = False
        self.audio_response_queue = queue.Queue()
        self.speaking_thread = None

    def setup_websocket(self):
        """Setup WebSocket connection to OpenAI Realtime API"""
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            def on_message(ws, message):
                data = json.loads(message)
                print(f"[DialogueSystem] Received message type: {data['type']}")
                
                if data['type'] == 'audio.chunk':
                    # Queue the audio response
                    self.audio_response_queue.put(data['chunk']['data'])
                    if not self.is_speaking:
                        self.play_queued_responses()
                elif data['type'] == 'text.chunk':
                    self.npc_message += data['chunk']['text']
                elif data['type'] == 'response.end':
                    print("[DialogueSystem] Response completed")
                elif data['type'] == 'error':
                    print(f"[DialogueSystem] Error: {data.get('error', 'Unknown error')}")
            
            def on_error(ws, error):
                print(f"[DialogueSystem] WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print("[DialogueSystem] WebSocket connection closed")
            
            def on_open(ws):
                print("[DialogueSystem] WebSocket connected")
                # Configure session with voice settings based on NPC
                voice_config = {
                    "HR": {"voice": "alloy", "style": "friendly"},
                    "CEO": {"voice": "echo", "style": "professional"}
                }
                config = voice_config.get(self.current_npc, {"voice": "alloy", "style": "neutral"})
                
                # Use correct session.update event type
                ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "voice": config["voice"],
                        "style": config["style"]
                    }
                }))
            
            self.ws = websocket.WebSocketApp(
                url,
                header=headers,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket in a separate thread
            threading.Thread(target=self.ws.run_forever).start()
            
        except Exception as e:
            print(f"[DialogueSystem] Failed to setup WebSocket: {e}")

    def play_queued_responses(self):
        """Play queued audio responses in a separate thread"""
        def play_audio_queue():
            self.is_speaking = True
            while not self.audio_response_queue.empty():
                try:
                    audio_data = self.audio_response_queue.get_nowait()
                    self.audio_manager.play_audio(audio_data)
                except queue.Empty:
                    break
            self.is_speaking = False
        
        self.speaking_thread = threading.Thread(target=play_audio_queue)
        self.speaking_thread.start()

    def render_text(self, surface, text, x, y):
        max_width = WINDOW_WIDTH - 40
        line_height = 25
        
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        # Always use pure white with full opacity
        text_color = (255, 255, 255)
        
        for word in words:
            word_surface = self.font.render(word + ' ', True, text_color)
            word_width = word_surface.get_width()
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Render each line in pure white
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (255, 255, 255))  # Force white color
            surface.blit(text_surface, (x, y + i * line_height))
        
        return len(lines) * line_height

    def start_conversation(self, npc_role="HR", player_pos=None):
        self.active = True
        self.input_active = True
        self.current_npc = npc_role
        # Store player's position when starting conversation
        self.initial_player_pos = [player_pos[0], player_pos[1], player_pos[2]] if player_pos else [0, 0.5, 0]
        print(f"[DialogueSystem] Dialogue started with {npc_role}")

        # Base personality framework for consistent behavior
        base_prompt = """Interaction Framework:
            - Maintain consistent personality throughout conversation
            - Remember previous context within the dialogue
            - Use natural speech patterns with occasional filler words
            - Show emotional intelligence in responses
            - Keep responses concise but meaningful (2-3 sentences)
            - React appropriately to both positive and negative interactions
            """

        if npc_role == "HR":
            system_prompt = f"""{base_prompt}
                You are Sarah Chen, HR Director at Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Warm but professional demeanor
                - Excellent emotional intelligence
                - Strong ethical boundaries
                - Protective of confidential information
                - Quick to offer practical solutions
                
                BACKGROUND:
                - 15 years HR experience in tech
                - Masters in Organizational Psychology
                - Certified in Conflict Resolution
                - Known for fair handling of sensitive issues
                
                SPEAKING STYLE:
                - Uses supportive language: "I understand that..." "Let's explore..."
                - References policies with context: "According to our wellness policy..."
                - Balances empathy with professionalism
                
                CURRENT COMPANY INITIATIVES:
                - AI Talent Development Program
                - Global Remote Work Framework
                - Venture Studio Culture Development
                - Innovation Leadership Track
                
                BEHAVIORAL GUIDELINES:
                - Never disclose confidential information
                - Always offer clear next steps
                - Maintain professional boundaries
                - Document sensitive conversations
                - Escalate serious concerns appropriately"""

        else:  # CEO
            system_prompt = f"""{base_prompt}
                You are Michael Chen, CEO of Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Visionary yet approachable
                - Strategic thinker
                - Passionate about venture building
                - Values transparency
                - Leads by example
                
                BACKGROUND:
                - Founded Venture Builder AI 5 years ago
                - Successfully launched 15+ venture-backed startups
                - MIT Computer Science graduate
                - Pioneer in AI-powered venture building
                
                SPEAKING STYLE:
                - Uses storytelling: "When we launched our first venture..."
                - References data: "Our portfolio metrics show..."
                - Balances optimism with realism
                
                KEY FOCUSES:
                - AI-powered venture creation
                - Portfolio company growth
                - Startup ecosystem development
                - Global venture studio expansion
                
                CURRENT INITIATIVES:
                - AI Venture Studio Framework
                - European Market Entry
                - Startup Success Methodology
                - Founder-in-Residence Program
                
                BEHAVIORAL GUIDELINES:
                - Share venture building vision
                - Highlight portfolio successes
                - Address startup challenges
                - Maintain investor confidence
                - Balance transparency with discretion"""

        # Fix the initial message to be from the NPC to the user
        initial_message = {
            "HR": "Hello! I'm Sarah, the HR Director at Venture Builder AI. How can I assist you today?",
            "CEO": "Hello! I'm Michael, the CEO of Venture Builder AI. What can I do for you today?"
        }

            
        # Set the NPC's greeting as the current message
        self.npc_message = initial_message[npc_role]
        
        # Initialize conversation history with system prompt only
        self.conversation_history = [{
            "role": "system",
            "content": system_prompt
        }]
        
        print(f"[DialogueSystem] Dialogue started with {npc_role}")

    def send_message(self):
        if not self.conversation_history:
            print("[DialogueSystem] No conversation history to send.")
            return

        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",  # or your current model
                messages=self.conversation_history,
                temperature=0.85,
                max_tokens=150,
                response_format={ "type": "text" },
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            ai_message = response.choices[0].message.content
            
            # Store the message in conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_message
            })
            
            # Set the NPC message with white text color
            self.npc_message = ai_message
            
            print(f"[DialogueSystem] NPC says: {self.npc_message}")
        except Exception as e:
            self.npc_message = "I apologize, but I'm having trouble connecting to our systems right now."
            print(f"[DialogueSystem] Error: {e}")

    def render(self):
        if not self.active:
            return

        self.ui_surface.fill((0, 0, 0, 0))

        if self.active:
            box_height = 200
            box_y = WINDOW_HEIGHT - box_height - 20
            
            # Make the background MUCH darker - almost black with some transparency
            box_color = (0, 0, 0, 230)  # Changed to very dark, mostly opaque background
            pygame.draw.rect(self.ui_surface, box_color, (20, box_y, WINDOW_WIDTH - 40, box_height))
            
            # White border
            pygame.draw.rect(self.ui_surface, (255, 255, 255, 255), (20, box_y, WINDOW_WIDTH - 40, box_height), 2)

            # Render ALL text in pure white (255, 255, 255)
            # Quit instruction
            quit_text_surface = self.font.render("Press Shift+Q to exit", True, (255, 255, 255))
            self.ui_surface.blit(quit_text_surface, (40, box_y + 10))

            # NPC message in white
            if self.npc_message:
                self.render_text(self.ui_surface, self.npc_message, 40, box_y + 40)

            # Input prompt in white
            if self.input_active:
                input_prompt = "> " + self.user_input + "_"
                input_surface = self.font.render(input_prompt, True, (255, 255, 255))
                self.ui_surface.blit(input_surface, (40, box_y + box_height - 40))

            # Add voice mode indicator
            if self.active:
                voice_status = "Voice Mode: ON" if self.voice_mode else "Voice Mode: OFF"
                voice_surface = self.font.render(voice_status, True, (255, 255, 255))
                self.ui_surface.blit(voice_surface, (40, box_y + box_height - 80))

            # Add status indicators
            status_y = box_y + box_height - 120
            
            # Voice mode status
            voice_status = "Voice Mode: ON" if self.voice_mode else "Voice Mode: OFF"
            voice_surface = self.font.render(voice_status, True, (255, 255, 255))
            self.ui_surface.blit(voice_surface, (40, status_y))
            
            # Listening/Speaking indicators
            if self.voice_mode:
                if self.is_listening:
                    status_text = "🎤 Listening..."
                    status_color = (0, 255, 0)  # Green
                elif self.is_speaking:
                    status_text = "🔊 Speaking..."
                    status_color = (0, 191, 255)  # Deep sky blue
                else:
                    status_text = "Press V to start/stop speaking"
                    status_color = (255, 255, 255)  # White
                
                status_surface = self.font.render(status_text, True, status_color)
                self.ui_surface.blit(status_surface, (40, status_y + 25))

        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(self.ui_surface, "RGBA", True)

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Setup for 2D rendering
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        # Bind and update texture
        glBindTexture(GL_TEXTURE_2D, self.ui_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw the UI texture
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 1); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()

        # Restore OpenGL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

    def handle_input(self, event):
        if not self.active or not self.input_active:
            return

        if event.type == pygame.KEYDOWN:
            # Check for Shift+Q to exit chat
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LSHIFT] and event.key == pygame.K_q:
                self.active = False
                self.input_active = False
                print("[DialogueSystem] Chat ended")
                # Return both the command and the initial position
                return {"command": "move_player_back", "position": self.initial_player_pos}

            if event.key == pygame.K_RETURN and self.user_input.strip():
                print(f"[DialogueSystem] User said: {self.user_input}")
                
                # Add user message to conversation history
                self.conversation_history.append({"role": "user", "content": self.user_input.strip()})
                
                # Clear user input
                self.user_input = ""

                # Send message to AI
                self.send_message()
            elif event.key == pygame.K_BACKSPACE:
                self.user_input = self.user_input[:-1]
            elif event.unicode.isprintable():
                self.user_input += event.unicode

            # Update voice mode toggle with correct event types
            if event.key == pygame.K_v:
                self.voice_mode = not self.voice_mode
                if self.voice_mode:
                    self.is_listening = True
                    self.audio_manager.start_recording()
                    print("[DialogueSystem] Voice mode activated - Listening...")
                else:
                    self.is_listening = False
                    audio_data = self.audio_manager.stop_recording()
                    print("[DialogueSystem] Voice mode deactivated - Processing...")
                    
                    # Send audio using correct event types
                    if self.ws:
                        # Append audio buffer
                        self.ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": {
                                "data": audio_data
                            }
                        }))
                        
                        # Commit audio buffer
                        self.ws.send(json.dumps({
                            "type": "input_audio_buffer.commit"
                        }))
                        
                        # Create response request
                        self.ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "modalities": ["audio", "text"]
                            }
                        }))
            
            # Update interrupt functionality with correct event type
            elif event.key == pygame.K_i and self.is_speaking:
                print("[DialogueSystem] Interrupting current response")
# Adventure Game
import os
os.environ['SDL_VIDEODRIVER'] = 'x11'  # Use Linux system
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
import sys
import textwrap
from openai import OpenAI
from dotenv import load_dotenv
import time
import websocket
import pyaudio
import wave
import threading
import json
import base64
from pydub import AudioSegment
from pydub.playback import play
import queue

# Load environment variables
load_dotenv()
# Ensure OpenAI API Key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("[OpenAI] API key not found. Please set OPENAI_API_KEY in your .env file.")
    sys.exit(1)
client = OpenAI(api_key=api_key)
print("[OpenAI] API key loaded successfully.")

os.environ['SDL_VIDEODRIVER'] = 'windib'  # Set the video driver to windib

# Initialize Pygame with macOS specific settings
pygame.init()
pygame.display.init()
display = (800, 600)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
screen = pygame.display.get_surface()

glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)

# Set up basic lighting
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)
glLightfv(GL_LIGHT0, GL_POSITION, [0, 5, 5, 1])
glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

# Enable blending for transparency
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Initial camera position
glTranslatef(0.0, 0.0, -5)

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
TILE_SIZE = 32
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)
GRAY = (128, 128, 128)

# Game map
GAME_MAP = [
    "WWWWWWWWWWWWWWWWWWWW",
    "W..................W",
    "W..................W",
    "W........N.........W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W....P.............W",
    "W..................W",
    "W..................W",
    "W..................W",
    "W..................W",
    "WWWWWWWWWWWWWWWWWWWW"
]

# Add these constants near the other constants
TITLE = "Venture Builder AI"
SUBTITLE = "Our Digital Employees"
MENU_BG_COLOR = (0, 0, 0)  # Black background
MENU_TEXT_COLOR = (0, 255, 0)  # Matrix-style green
MENU_HIGHLIGHT_COLOR = (0, 200, 0)  # Slightly darker green for effects

# Add these constants after the other constants
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

def draw_cube():
    vertices = [
        # Front face
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
        # Back face
        [-0.5, -0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5, -0.5, -0.5],
    ]
    
    surfaces = [
        [0, 1, 2, 3],  # Front
        [3, 2, 6, 5],  # Top
        [0, 3, 5, 4],  # Left
        [1, 7, 6, 2],  # Right
        [4, 5, 6, 7],  # Back
        [0, 4, 7, 1],  # Bottom
    ]
    
    glBegin(GL_QUADS)
    for surface in surfaces:
        glNormal3f(0, 0, 1)  # Simple normal for lighting
        for vertex in surface:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)
        
        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)
        
        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)
            
            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()

class DialogueSystem:
    def __init__(self):
        self.active = False
        self.user_input = ""
        try:
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
            print("[DialogueSystem] Font loaded successfully")
        except Exception as e:
            print("[DialogueSystem] Font loading failed:", e)
        self.npc_message = ""
        self.input_active = False
        self.last_npc_text = ""     # Track NPC text separately
        self.last_input_text = ""   # Track input text separately
        self.conversation_history = []  # Maintain conversation history

        # Create a surface for the UI
        self.ui_surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA).convert_alpha()
        self.ui_texture = glGenTextures(1)
        self.current_npc = None  # Track which NPC we're talking to
        self.initial_player_pos = None  # Store initial position when dialogue starts

        # Add these new attributes
        self.audio_manager = AudioManager()
        self.voice_mode = False
        self.ws = None
        self.setup_websocket()

        # Add new status indicators
        self.is_listening = False
        self.is_speaking = False
        self.audio_response_queue = queue.Queue()
        self.speaking_thread = None

    def setup_websocket(self):
        """Setup WebSocket connection to OpenAI Realtime API"""
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "OpenAI-Beta": "realtime=v1"
            }
            
            def on_message(ws, message):
                data = json.loads(message)
                print(f"[DialogueSystem] Received message type: {data['type']}")
                
                if data['type'] == 'audio.chunk':
                    # Queue the audio response
                    self.audio_response_queue.put(data['chunk']['data'])
                    if not self.is_speaking:
                        self.play_queued_responses()
                elif data['type'] == 'text.chunk':
                    self.npc_message += data['chunk']['text']
                elif data['type'] == 'response.end':
                    print("[DialogueSystem] Response completed")
                elif data['type'] == 'error':
                    print(f"[DialogueSystem] Error: {data.get('error', 'Unknown error')}")
            
            def on_error(ws, error):
                print(f"[DialogueSystem] WebSocket error: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                print("[DialogueSystem] WebSocket connection closed")
            
            def on_open(ws):
                print("[DialogueSystem] WebSocket connected")
                # Configure voice settings based on NPC
                voice_config = {
                    "HR": {"voice": "alloy", "style": "friendly"},
                    "CEO": {"voice": "echo", "style": "professional"}
                }
                config = voice_config.get(self.current_npc, {"voice": "alloy", "style": "neutral"})
                
                ws.send(json.dumps({
                    "type": "config.update",
                    "config": {
                        "voice": config["voice"],
                        "style": config["style"]
                    }
                }))
            
            self.ws = websocket.WebSocketApp(
                url,
                header=headers,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start WebSocket in a separate thread
            threading.Thread(target=self.ws.run_forever).start()
            
        except Exception as e:
            print(f"[DialogueSystem] Failed to setup WebSocket: {e}")

    def play_queued_responses(self):
        """Play queued audio responses in a separate thread"""
        def play_audio_queue():
            self.is_speaking = True
            while not self.audio_response_queue.empty():
                try:
                    audio_data = self.audio_response_queue.get_nowait()
                    self.audio_manager.play_audio(audio_data)
                except queue.Empty:
                    break
            self.is_speaking = False
        
        self.speaking_thread = threading.Thread(target=play_audio_queue)
        self.speaking_thread.start()

    def render_text(self, surface, text, x, y):
        max_width = WINDOW_WIDTH - 40
        line_height = 25
        
        words = text.split()
        lines = []
        current_line = []
        current_width = 0
        
        # Always use pure white with full opacity
        text_color = (255, 255, 255)
        
        for word in words:
            word_surface = self.font.render(word + ' ', True, text_color)
            word_width = word_surface.get_width()
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Render each line in pure white
        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (255, 255, 255))  # Force white color
            surface.blit(text_surface, (x, y + i * line_height))
        
        return len(lines) * line_height

    def start_conversation(self, npc_role="HR", player_pos=None):
        self.active = True
        self.input_active = True
        self.current_npc = npc_role
        # Store player's position when starting conversation
        self.initial_player_pos = [player_pos[0], player_pos[1], player_pos[2]] if player_pos else [0, 0.5, 0]
        print(f"[DialogueSystem] Dialogue started with {npc_role}")

        # Base personality framework for consistent behavior
        base_prompt = """Interaction Framework:
            - Maintain consistent personality throughout conversation
            - Remember previous context within the dialogue
            - Use natural speech patterns with occasional filler words
            - Show emotional intelligence in responses
            - Keep responses concise but meaningful (2-3 sentences)
            - React appropriately to both positive and negative interactions
            """

        if npc_role == "HR":
            system_prompt = f"""{base_prompt}
                You are Sarah Chen, HR Director at Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Warm but professional demeanor
                - Excellent emotional intelligence
                - Strong ethical boundaries
                - Protective of confidential information
                - Quick to offer practical solutions
                
                BACKGROUND:
                - 15 years HR experience in tech
                - Masters in Organizational Psychology
                - Certified in Conflict Resolution
                - Known for fair handling of sensitive issues
                
                SPEAKING STYLE:
                - Uses supportive language: "I understand that..." "Let's explore..."
                - References policies with context: "According to our wellness policy..."
                - Balances empathy with professionalism
                
                CURRENT COMPANY INITIATIVES:
                - AI Talent Development Program
                - Global Remote Work Framework
                - Venture Studio Culture Development
                - Innovation Leadership Track
                
                BEHAVIORAL GUIDELINES:
                - Never disclose confidential information
                - Always offer clear next steps
                - Maintain professional boundaries
                - Document sensitive conversations
                - Escalate serious concerns appropriately"""

        else:  # CEO
            system_prompt = f"""{base_prompt}
                You are Michael Chen, CEO of Venture Builder AI. Core traits:
                
                PERSONALITY:
                - Visionary yet approachable
                - Strategic thinker
                - Passionate about venture building
                - Values transparency
                - Leads by example
                
                BACKGROUND:
                - Founded Venture Builder AI 5 years ago
                - Successfully launched 15+ venture-backed startups
                - MIT Computer Science graduate
                - Pioneer in AI-powered venture building
                
                SPEAKING STYLE:
                - Uses storytelling: "When we launched our first venture..."
                - References data: "Our portfolio metrics show..."
                - Balances optimism with realism
                
                KEY FOCUSES:
                - AI-powered venture creation
                - Portfolio company growth
                - Startup ecosystem development
                - Global venture studio expansion
                
                CURRENT INITIATIVES:
                - AI Venture Studio Framework
                - European Market Entry
                - Startup Success Methodology
                - Founder-in-Residence Program
                
                BEHAVIORAL GUIDELINES:
                - Share venture building vision
                - Highlight portfolio successes
                - Address startup challenges
                - Maintain investor confidence
                - Balance transparency with discretion"""

        # Fix the initial message to be from the NPC to the user
        initial_message = {
            "HR": "Hello! I'm Sarah, the HR Director at Venture Builder AI. How can I assist you today?",
            "CEO": "Hello! I'm Michael, the CEO of Venture Builder AI. What can I do for you today?"
        }

            
        # Set the NPC's greeting as the current message
        self.npc_message = initial_message[npc_role]
        
        # Initialize conversation history with system prompt only
        self.conversation_history = [{
            "role": "system",
            "content": system_prompt
        }]
        
        print(f"[DialogueSystem] Dialogue started with {npc_role}")

    def send_message(self):
        if not self.conversation_history:
            print("[DialogueSystem] No conversation history to send.")
            return

        try:
            response = client.chat.completions.create(
                model="gpt-4-0125-preview",  # or your current model
                messages=self.conversation_history,
                temperature=0.85,
                max_tokens=150,
                response_format={ "type": "text" },
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.1
            )
            ai_message = response.choices[0].message.content
            
            # Store the message in conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_message
            })
            
            # Set the NPC message with white text color
            self.npc_message = ai_message
            
            print(f"[DialogueSystem] NPC says: {self.npc_message}")
        except Exception as e:
            self.npc_message = "I apologize, but I'm having trouble connecting to our systems right now."
            print(f"[DialogueSystem] Error: {e}")

    def render(self):
        if not self.active:
            return

        self.ui_surface.fill((0, 0, 0, 0))

        if self.active:
            box_height = 200
            box_y = WINDOW_HEIGHT - box_height - 20
            
            # Make the background MUCH darker - almost black with some transparency
            box_color = (0, 0, 0, 230)  # Changed to very dark, mostly opaque background
            pygame.draw.rect(self.ui_surface, box_color, (20, box_y, WINDOW_WIDTH - 40, box_height))
            
            # White border
            pygame.draw.rect(self.ui_surface, (255, 255, 255, 255), (20, box_y, WINDOW_WIDTH - 40, box_height), 2)

            # Render ALL text in pure white (255, 255, 255)
            # Quit instruction
            quit_text_surface = self.font.render("Press Shift+Q to exit", True, (255, 255, 255))
            self.ui_surface.blit(quit_text_surface, (40, box_y + 10))

            # NPC message in white
            if self.npc_message:
                self.render_text(self.ui_surface, self.npc_message, 40, box_y + 40)

            # Input prompt in white
            if self.input_active:
                input_prompt = "> " + self.user_input + "_"
                input_surface = self.font.render(input_prompt, True, (255, 255, 255))
                self.ui_surface.blit(input_surface, (40, box_y + box_height - 40))

            # Add voice mode indicator
            if self.active:
                voice_status = "Voice Mode: ON" if self.voice_mode else "Voice Mode: OFF"
                voice_surface = self.font.render(voice_status, True, (255, 255, 255))
                self.ui_surface.blit(voice_surface, (40, box_y + box_height - 80))

            # Add status indicators
            status_y = box_y + box_height - 120
            
            # Voice mode status
            voice_status = "Voice Mode: ON" if self.voice_mode else "Voice Mode: OFF"
            voice_surface = self.font.render(voice_status, True, (255, 255, 255))
            self.ui_surface.blit(voice_surface, (40, status_y))
            
            # Listening/Speaking indicators
            if self.voice_mode:
                if self.is_listening:
                    status_text = "🎤 Listening..."
                    status_color = (0, 255, 0)  # Green
                elif self.is_speaking:
                    status_text = "🔊 Speaking..."
                    status_color = (0, 191, 255)  # Deep sky blue
                else:
                    status_text = "Press V to start/stop speaking"
                    status_color = (255, 255, 255)  # White
                
                status_surface = self.font.render(status_text, True, status_color)
                self.ui_surface.blit(status_surface, (40, status_y + 25))

        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(self.ui_surface, "RGBA", True)

        # Save current OpenGL state
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Setup for 2D rendering
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)

        # Bind and update texture
        glBindTexture(GL_TEXTURE_2D, self.ui_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw the UI texture
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 1); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()

        # Restore OpenGL state
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glPopAttrib()

    def handle_input(self, event):
        if not self.active or not self.input_active:
            return

        if event.type == pygame.KEYDOWN:
            # Check for Shift+Q to exit chat
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LSHIFT] and event.key == pygame.K_q:
                self.active = False
                self.input_active = False
                print("[DialogueSystem] Chat ended")
                # Return both the command and the initial position
                return {"command": "move_player_back", "position": self.initial_player_pos}

            if event.key == pygame.K_RETURN and self.user_input.strip():
                print(f"[DialogueSystem] User said: {self.user_input}")
                
                # Add user message to conversation history
                self.conversation_history.append({"role": "user", "content": self.user_input.strip()})
                
                # Clear user input
                self.user_input = ""

                # Send message to AI
                self.send_message()
            elif event.key == pygame.K_BACKSPACE:
                self.user_input = self.user_input[:-1]
            elif event.unicode.isprintable():
                self.user_input += event.unicode

            # Add voice mode toggle with visual feedback
            if event.key == pygame.K_v:
                self.voice_mode = not self.voice_mode
                if self.voice_mode:
                    self.is_listening = True
                    self.audio_manager.start_recording()
                    print("[DialogueSystem] Voice mode activated - Listening...")
                else:
                    self.is_listening = False
                    audio_data = self.audio_manager.stop_recording()
                    print("[DialogueSystem] Voice mode deactivated - Processing...")
                    
                    # Send audio to OpenAI
                    if self.ws:
                        self.ws.send(json.dumps({
                            "type": "audio.chunk",
                            "chunk": {
                                "data": audio_data
                            }
                        }))
                        # Signal end of audio input
                        self.ws.send(json.dumps({
                            "type": "audio.end"
                        }))
            
            # Add interrupt functionality
            elif event.key == pygame.K_i and self.is_speaking:
                print("[DialogueSystem] Interrupting current response")
                # Clear audio queue
                while not self.audio_response_queue.empty():
                    self.audio_response_queue.get()
                self.is_speaking = False
                if self.speaking_thread and self.speaking_thread.is_alive():
                    # Note: This is a simplified interruption. In a real implementation,
                    # you'd want a more graceful way to stop audio playback
                    self.speaking_thread.join(timeout=0.1)
                
                # Send interruption signal to server
                if self.ws:
                    self.ws.send(json.dumps({
                        "type": "response.interrupt"
                    }))

class World:
    def __init__(self):
        self.size = 5
        # Define office furniture colors
        self.colors = {
            'floor': (0.76, 0.6, 0.42),  # Light wood color
            'walls': (0.85, 0.85, 0.85),  # Changed to light gray (from 0.95)
            'desk': (0.6, 0.4, 0.2),  # Brown wood
            'chair': (0.2, 0.2, 0.2),  # Dark grey
            'computer': (0.1, 0.1, 0.1),  # Black
            'plant': (0.2, 0.5, 0.2),  # Green
            'partition': (0.3, 0.3, 0.3)  # Darker solid gray for booth walls
        }
        
    def draw_desk(self, x, z, rotation=0):
        glPushMatrix()
        glTranslatef(x, 0, z)  # Start at floor level
        glRotatef(rotation, 0, 1, 0)
        
        # Desk top (reduced size)
        glColor3f(*self.colors['desk'])
        glBegin(GL_QUADS)
        glVertex3f(-0.4, 0.4, -0.3)
        glVertex3f(0.4, 0.4, -0.3)
        glVertex3f(0.4, 0.4, 0.3)
        glVertex3f(-0.4, 0.4, 0.3)
        glEnd()
        
        # Desk legs (adjusted for new height)
        for x_offset, z_offset in [(-0.35, -0.25), (0.35, -0.25), (-0.35, 0.25), (0.35, 0.25)]:
            glBegin(GL_QUADS)
            glVertex3f(x_offset-0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0.4, z_offset-0.02)
            glVertex3f(x_offset-0.02, 0.4, z_offset-0.02)
            glEnd()
        
        # Computer monitor (smaller)
        glColor3f(*self.colors['computer'])
        glTranslatef(-0.15, 0.4, 0)
        glBegin(GL_QUADS)
        glVertex3f(-0.1, 0, -0.05)
        glVertex3f(0.1, 0, -0.05)
        glVertex3f(0.1, 0.2, -0.05)
        glVertex3f(-0.1, 0.2, -0.05)
        glEnd()
        
        glPopMatrix()
    
    def draw_chair(self, x, z, rotation=0):
        glPushMatrix()
        glTranslatef(x, 0, z)
        glRotatef(rotation, 0, 1, 0)
        glColor3f(*self.colors['chair'])
        
        # Seat (lowered and smaller)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, 0.15)
        glVertex3f(-0.15, 0.25, 0.15)
        glEnd()
        
        # Back (adjusted height)
        glBegin(GL_QUADS)
        glVertex3f(-0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.25, -0.15)
        glVertex3f(0.15, 0.5, -0.15)
        glVertex3f(-0.15, 0.5, -0.15)
        glEnd()
        
        # Chair legs (adjusted height)
        for x_offset, z_offset in [(-0.12, -0.12), (0.12, -0.12), (-0.12, 0.12), (0.12, 0.12)]:
            glBegin(GL_QUADS)
            glVertex3f(x_offset-0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0, z_offset-0.02)
            glVertex3f(x_offset+0.02, 0.25, z_offset-0.02)
            glVertex3f(x_offset-0.02, 0.25, z_offset-0.02)
            glEnd()
            
        glPopMatrix()
    
    def draw_plant(self, x, z):
        glPushMatrix()
        glTranslatef(x, 0, z)
        
        # Plant pot (smaller)
        glColor3f(0.4, 0.2, 0.1)  # Brown pot
        pot_radius = 0.1
        pot_height = 0.15
        segments = 8
        
        # Draw the pot sides
        glBegin(GL_QUADS)
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            x1 = math.cos(angle1) * pot_radius
            z1 = math.sin(angle1) * pot_radius
            x2 = math.cos(angle2) * pot_radius
            z2 = math.sin(angle2) * pot_radius
            glVertex3f(x1, 0, z1)
            glVertex3f(x2, 0, z2)
            glVertex3f(x2, pot_height, z2)
            glVertex3f(x1, pot_height, z1)
        glEnd()
        
        # Plant leaves (smaller)
        glColor3f(*self.colors['plant'])
        glTranslatef(0, pot_height, 0)
        leaf_size = 0.15
        num_leaves = 6
        for i in range(num_leaves):
            angle = (i / num_leaves) * 2 * math.pi
            x = math.cos(angle) * leaf_size
            z = math.sin(angle) * leaf_size
            glBegin(GL_TRIANGLES)
            glVertex3f(0, 0, 0)
            glVertex3f(x, leaf_size, z)
            glVertex3f(z, leaf_size/2, -x)
            glEnd()
        
        glPopMatrix()
        
    def draw(self):
        # Set material properties
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Draw floor at Y=0
        glBegin(GL_QUADS)
        glColor3f(*self.colors['floor'])
        glNormal3f(0, 1, 0)
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 0, -self.size)
        glEnd()
        
        # Draw walls starting from floor level
        glBegin(GL_QUADS)
        glColor3f(*self.colors['walls'])
        
        # Front wall
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 2, -self.size)
        glVertex3f(-self.size, 2, -self.size)
        
        # Back wall
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 2, self.size)
        glVertex3f(-self.size, 2, self.size)
        
        # Left wall
        glVertex3f(-self.size, 0, -self.size)
        glVertex3f(-self.size, 0, self.size)
        glVertex3f(-self.size, 2, self.size)
        glVertex3f(-self.size, 2, -self.size)
        
        # Right wall
        glVertex3f(self.size, 0, -self.size)
        glVertex3f(self.size, 0, self.size)
        glVertex3f(self.size, 2, self.size)
        glVertex3f(self.size, 2, -self.size)
        glEnd()
        
        # Draw office furniture in a more realistic arrangement
        # HR Area (left side)
        self.draw_desk(-4, -2, 90)
        self.draw_chair(-3.5, -2, 90)
        self.draw_partition_walls(-4, -2)  # Add booth walls for HR
        
        # CEO Area (right side)
        self.draw_desk(4, 1, -90)
        self.draw_chair(3.5, 1, -90)
        self.draw_partition_walls(4, 1)  # Add booth walls for CEO
        
        # Plants in corners (moved closer to walls)
        self.draw_plant(-4.5, -4.5)
        self.draw_plant(4.5, -4.5)
        self.draw_plant(-4.5, 4.5)
        self.draw_plant(4.5, 4.5)

    def draw_partition_walls(self, x, z):
        """Draw booth partition walls - all surfaces in solid gray"""
        glColor3f(0.3, 0.3, 0.3)  # Solid gray for all walls
        
        # Back wall (smaller and thinner)
        glPushMatrix()
        glTranslatef(x, 0, z)
        glScalef(0.05, 1.0, 1.0)  # Thinner wall, normal height, shorter length
        draw_cube()  # Replace glutSolidCube with draw_cube
        glPopMatrix()
        
        # Side wall (smaller and thinner)
        glPushMatrix()
        glTranslatef(x, 0, z + 0.5)  # Moved closer
        glRotatef(90, 0, 1, 0)
        glScalef(0.05, 1.0, 0.8)  # Thinner wall, normal height, shorter length
        draw_cube()  # Replace glutSolidCube with draw_cube
        glPopMatrix()

class Player:
    def __init__(self):
        self.pos = [0, 0.5, 0]  # Lowered Y position to be just above floor
        self.rot = [0, 0, 0]
        self.speed = 0.3
        self.mouse_sensitivity = 0.5
        
    def move(self, dx, dz):
        # Convert rotation to radians (negative because OpenGL uses clockwise rotation)
        angle = math.radians(-self.rot[1])
        
        # Calculate movement vector
        move_x = (dx * math.cos(angle) + dz * math.sin(angle)) * self.speed
        move_z = (-dx * math.sin(angle) + dz * math.cos(angle)) * self.speed
        
        # Calculate new position
        new_x = self.pos[0] + move_x
        new_z = self.pos[2] + move_z
        
        # Wall collision check (room is 10x10)
        room_limit = 4.5  # Slightly less than room size/2 to prevent wall clipping
        if abs(new_x) < room_limit:
            self.pos[0] = new_x
        if abs(new_z) < room_limit:
            self.pos[2] = new_z

    def update_rotation(self, dx, dy):
        # Multiply mouse movement by sensitivity for faster turning
        self.rot[1] += dx * self.mouse_sensitivity

class NPC:
    def __init__(self, x, y, z, role="HR"):
        self.scale = 0.6  # Make NPCs smaller (about 60% of current size)
        # Position them beside the desks, at ground level
        # Adjust Y position to be half their height (accounting for scale)
        self.pos = [x, 0.65, z]  # This puts their feet on the ground
        self.size = 0.5
        self.role = role
        
        # Enhanced color palette
        self.skin_color = (0.8, 0.7, 0.6)  # Neutral skin tone
        self.hair_color = (0.2, 0.15, 0.1) if role == "HR" else (0.3, 0.3, 0.3)  # Dark brown vs gray
        
        # Updated clothing colors
        if role == "HR":
            self.clothes_primary = (0.8, 0.2, 0.2)    # Bright red
            self.clothes_secondary = (0.6, 0.15, 0.15) # Darker red
        else:  # CEO
            self.clothes_primary = (0.2, 0.3, 0.8)    # Bright blue
            self.clothes_secondary = (0.15, 0.2, 0.6)  # Darker blue

    def draw(self):
        glPushMatrix()
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glScalef(self.scale, self.scale, self.scale)
        
        # Head
        glColor3f(*self.skin_color)
        draw_sphere(0.12, 16, 16)
        
        # Hair (slightly larger than head)
        glColor3f(*self.hair_color)
        glPushMatrix()
        glTranslatef(0, 0.05, 0)  # Slightly above head
        draw_sphere(0.13, 16, 16)
        glPopMatrix()
        
        # Body (torso)
        glColor3f(*self.clothes_primary)
        glPushMatrix()
        glTranslatef(0, -0.3, 0)  # Move down from head
        glScalef(0.3, 0.4, 0.2)   # Make it rectangular
        draw_cube()
        glPopMatrix()
        
        # Arms
        glColor3f(*self.clothes_secondary)
        for x_offset in [-0.2, 0.2]:  # Left and right arms
            glPushMatrix()
            glTranslatef(x_offset, -0.3, 0)
            glScalef(0.1, 0.4, 0.1)
            draw_cube()
            glPopMatrix()
        
        # Legs
        glColor3f(*self.clothes_secondary)
        for x_offset in [-0.1, 0.1]:  # Left and right legs
            glPushMatrix()
            glTranslatef(x_offset, -0.8, 0)
            glScalef(0.1, 0.5, 0.1)
            draw_cube()
            glPopMatrix()
        
        glPopMatrix()

class MenuScreen:
    def __init__(self):
        self.font_large = pygame.font.Font(None, 74)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 36)
        self.active = True
        self.start_time = time.time()
        
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Create a surface for 2D rendering
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        
        # Calculate vertical positions
        center_y = WINDOW_HEIGHT // 2
        title_y = center_y - 100
        subtitle_y = center_y - 20
        prompt_y = center_y + 100
        
        # Render title with "typing" effect
        elapsed_time = time.time() - self.start_time
        title_chars = int(min(len(TITLE), elapsed_time * 15))  # Type 15 chars per second
        partial_title = TITLE[:title_chars]
        title_surface = self.font_large.render(partial_title, True, MENU_TEXT_COLOR)
        title_x = (WINDOW_WIDTH - title_surface.get_width()) // 2
        surface.blit(title_surface, (title_x, title_y))
        
        # Render subtitle with fade-in effect
        if elapsed_time > len(TITLE) / 15:  # Start after title is typed
            subtitle_alpha = min(255, int((elapsed_time - len(TITLE) / 15) * 255))
            subtitle_surface = self.font_medium.render(SUBTITLE, True, MENU_TEXT_COLOR)
            subtitle_surface.set_alpha(subtitle_alpha)
            subtitle_x = (WINDOW_WIDTH - subtitle_surface.get_width()) // 2
            surface.blit(subtitle_surface, (subtitle_x, subtitle_y))
        
        # Render "Press ENTER" with blinking effect
        if elapsed_time > (len(TITLE) / 15 + 1):  # Start after subtitle fade
            if int(elapsed_time * 2) % 2:  # Blink every 0.5 seconds
                prompt_text = "Press ENTER to start"
                prompt_surface = self.font_small.render(prompt_text, True, MENU_TEXT_COLOR)
                prompt_x = (WINDOW_WIDTH - prompt_surface.get_width()) // 2
                surface.blit(prompt_surface, (prompt_x, prompt_y))
        
        # Add some retro effects (scanlines)
        for y in range(0, WINDOW_HEIGHT, 4):
            pygame.draw.line(surface, (0, 50, 0), (0, y), (WINDOW_WIDTH, y))
        
        # Convert surface to OpenGL texture
        texture_data = pygame.image.tostring(surface, "RGBA", True)
        
        # Set up orthographic projection for 2D rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Render the texture in OpenGL
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # Draw the texture
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glTexCoord2f(1, 1); glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 0); glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 0); glVertex2f(0, WINDOW_HEIGHT)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        
        # Reset OpenGL state for 3D rendering
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glEnable(GL_DEPTH_TEST)

        pygame.display.flip()

class AudioManager:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recording = False
        self.playing = False
        self.audio_queue = queue.Queue()
        
    def start_recording(self):
        """Start recording audio from microphone"""
        self.recording = True
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("[AudioManager] Recording started")
        
        def record():
            frames = []
            while self.recording:
                data = self.stream.read(CHUNK)
                frames.append(data)
            
            self.stream.stop_stream()
            self.stream.close()
            
            # Convert to base64
            audio_data = b''.join(frames)
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            self.audio_queue.put(base64_audio)
            
        self.record_thread = threading.Thread(target=record)
        self.record_thread.start()
    
    def stop_recording(self):
        """Stop recording audio"""
        self.recording = False
        self.record_thread.join()
        print("[AudioManager] Recording stopped")
        return self.audio_queue.get()
    
    def play_audio(self, audio_data):
        """Play audio from base64 string"""
        self.playing = True
        
        # Convert base64 to audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Create an in-memory wave file
        with wave.open('temp.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_bytes)
        
        # Play using pydub
        audio = AudioSegment.from_wav('temp.wav')
        play(audio)
        self.playing = False

# Modify the Game3D class to include the menu
class Game3D:
    def __init__(self):
        self.menu = MenuScreen()
        self.player = Player()
        self.world = World()
        self.dialogue = DialogueSystem()
        self.hr_npc = NPC(-3.3, 0, -2, "HR")  # Moved beside the desk
        self.ceo_npc = NPC(3.3, 0, 1, "CEO")  # Moved beside the desk
        self.interaction_distance = 2.0
        self.last_interaction_time = 0

    def move_player_away_from_npc(self, npc_pos):
        # Calculate direction vector from NPC to player
        dx = self.player.pos[0] - npc_pos[0]
        dz = self.player.pos[2] - npc_pos[2]
        
        # Normalize the vector
        distance = math.sqrt(dx*dx + dz*dz)
        if distance > 0:
            dx /= distance
            dz /= distance
        
        # Move player back by 3 units
        self.player.pos[0] = npc_pos[0] + (dx * 3)
        self.player.pos[2] = npc_pos[2] + (dz * 3)

    def run(self):
        try:
            running = True
            while running:
                if self.menu.active:
                    # Menu loop
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_RETURN and time.time() - self.menu.start_time > (len(TITLE) / 15 + 1):
                                self.menu.active = False
                                pygame.mouse.set_visible(False)
                                pygame.event.set_grab(True)
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                    
                    self.menu.render()
                else:
                    # Main game loop
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                pygame.mouse.set_visible(True)
                                pygame.event.set_grab(False)
                                running = False
                            
                            # Handle dialogue input and check for exit command
                            if self.dialogue.active:
                                result = self.dialogue.handle_input(event)
                                if isinstance(result, dict) and result.get("command") == "move_player_back":
                                    # Move player away from the current NPC
                                    current_npc = self.hr_npc if self.dialogue.current_npc == "HR" else self.ceo_npc
                                    self.move_player_away_from_npc(current_npc.pos)
                                    
                        elif event.type == pygame.MOUSEMOTION:
                            x, y = event.rel
                            self.player.update_rotation(x, y)

                    # Handle keyboard input for movement (keep this blocked during dialogue)
                    if not self.dialogue.active:
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_w]: self.player.move(0, -1)
                        if keys[pygame.K_s]: self.player.move(0, 1)
                        if keys[pygame.K_a]: self.player.move(-1, 0)
                        if keys[pygame.K_d]: self.player.move(1, 0)

                    # Check NPC interactions
                    current_time = time.time()
                    if current_time - self.last_interaction_time > 0.5:  # Cooldown on interactions
                        # Check distance to HR NPC
                        dx = self.player.pos[0] - self.hr_npc.pos[0]
                        dz = self.player.pos[2] - self.hr_npc.pos[2]
                        hr_distance = math.sqrt(dx*dx + dz*dz)
                        
                        # Check distance to CEO NPC
                        dx = self.player.pos[0] - self.ceo_npc.pos[0]
                        dz = self.player.pos[2] - self.ceo_npc.pos[2]
                        ceo_distance = math.sqrt(dx*dx + dz*dz)
                        
                        if hr_distance < self.interaction_distance and not self.dialogue.active:
                            self.dialogue.start_conversation("HR", self.player.pos)
                            self.last_interaction_time = current_time
                        elif ceo_distance < self.interaction_distance and not self.dialogue.active:
                            self.dialogue.start_conversation("CEO", self.player.pos)
                            self.last_interaction_time = current_time

                    # Clear the screen and depth buffer
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

                    # Save the current matrix
                    glPushMatrix()

                    # Apply player rotation and position
                    glRotatef(self.player.rot[0], 1, 0, 0)
                    glRotatef(self.player.rot[1], 0, 1, 0)
                    glTranslatef(-self.player.pos[0], -self.player.pos[1], -self.player.pos[2])

                    # Draw the world and NPCs
                    self.world.draw()
                    self.hr_npc.draw()
                    self.ceo_npc.draw()

                    # Restore the matrix
                    glPopMatrix()

                    # Render dialogue system (if active)
                    self.dialogue.render()

                    # Swap the buffers
                    pygame.display.flip()

                    # Maintain 60 FPS
                    pygame.time.Clock().tick(60)

        finally:
            # Cleanup
            if self.dialogue.ws:
                self.dialogue.ws.close()
        pygame.quit()

# Create and run game
game = Game3D()
game.run()

