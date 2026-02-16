import logging
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import google, sarvam

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("voice-agent")
logger.setLevel(logging.INFO)


class VoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            # Your agent's personality and instructions
            instructions=(
                "You are the Bizom Cafe voice feedback agent. You handle feedback from "
                "Bizom employees about the cafe: food quality, menu items, and suggestions. "
                "Be warm and solution-oriented. Acknowledge what they said before responding. "
                "Keep responses under 3 sentences. "
                "Focus only on the cafe menu. Today's menu:\n"
                "BREAKFAST: Set dosa and sambar.\n"
                "LUNCH: Roti, Rice, Dal fry, Aloo gobhi, Baigan bharta, Salad.\n"
                "SNACKS: Maggi.\n"
                "If they talk about something unrelated, briefly redirect: e.g. 'I'm here for "
                "cafe feedback only—how was the food today?' "
                "When they give feedback, tie it to the menu when possible and say you'll pass it to "
                "the kitchen. If their feedback is vague, ask which item or meal they mean. "
                "If a colleague is upset or frustrated, apologize sincerely and offer a concrete "
                "next step (e.g. 'Sorry about that—we'll share this with the kitchen right away.'). "
                "Never argue or make excuses. "
                "If they seem done—e.g. that's it, nothing else, no more—say a brief thank you "
                "and close (e.g. 'Thanks, that's really helpful. Take care!'). "
                "Respond in Indian English: natural, warm tone; use 'kindly', 'sure' where it fits."
            ),
            
            # Saaras v3 STT - Converts speech to text
            stt=sarvam.STT(
                language="en-IN",  # or "hi-IN", etc.
                model="saaras:v3",
                flush_signal=True 
            ),
            
            # OpenAI LLM - The "brain" that processes and generates responses
            llm=google.LLM(model="gemini-2.0-flash"),
            
            # Bulbul TTS - Converts text to speech
            tts=sarvam.TTS(
                target_language_code="en-IN",
                model="bulbul:v2",
                speaker="anushka"  # Female: priya, simran, ishita, kavya | Male: aditya, anand, rohan
            ),
        )
    
    async def on_enter(self):
        """Called when user joins - agent starts the conversation"""
        self.session.generate_reply()


async def entrypoint(ctx: JobContext):
    """Main entry point - LiveKit calls this when a user connects"""
    logger.info(f"User connected to room: {ctx.room.name}")
    
    # Create and start the agent session
    session = AgentSession(turn_detection="stt")
    await session.start(
        agent=VoiceAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    # Run the agent
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
