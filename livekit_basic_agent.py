"""
LiveKit Voice Agent - Quick Start
==================================
The simplest possible LiveKit voice agent to get you started.
Requires only OpenAI and Deepgram API keys.
"""

import asyncio
import os
import random
from datetime import datetime

from ddgs import DDGS
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import openai, deepgram, silero

# Short, varied hold messages used to fill the dead air during a slow tool call.
# Production rule of thumb: any tool that takes >500ms on the happy path should
# play one of these so the user doesn't sit in silence wondering if the agent
# crashed. Picked at random each call so the agent doesn't sound robotic.
_HOLD_MESSAGES_SEARCH = [
    "Let me look that up for you.",
    "One moment, checking that.",
    "Hold on, let me search for that.",
    "Give me a second.",
    "Let me pull that up.",
]

# Load environment variables
load_dotenv(".env")

class Assistant(Agent):
    """Basic voice assistant with Airbnb booking capabilities."""

    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful and friendly voice assistant. You can: "
                "(1) search for and book Airbnbs in supported cities — "
                "San Francisco, New York, and Los Angeles; "
                "(2) tell the user the current date and time; "
                "(3) search the live web for current events, recent news, "
                "or any factual question you don't already know the answer to. "
                "When the user asks anything that requires fresh information "
                "from the internet, call `search_the_web` rather than guessing. "
                "Keep responses concise and natural — you are speaking, not writing. "
                "Before invoking a tool that may take a moment to return — anything "
                "that hits the network, books a reservation, or looks something up — "
                "first say a brief natural acknowledgment so the user doesn't sit in "
                "silence. Examples: 'let me check that', 'one sec', 'hold on a moment', "
                "'looking that up'. Vary the phrasing across turns, keep it to a few "
                "words, and never name the tool. Skip the acknowledgment for instant "
                "operations like reading the current time."
            )
        )

        # Mock Airbnb database
        self.airbnbs = {
            "san francisco": [
                {
                    "id": "sf001",
                    "name": "Cozy Downtown Loft",
                    "address": "123 Market Street, San Francisco, CA",
                    "price": 150,
                    "amenities": ["WiFi", "Kitchen", "Workspace"],
                },
                {
                    "id": "sf002",
                    "name": "Victorian House with Bay Views",
                    "address": "456 Castro Street, San Francisco, CA",
                    "price": 220,
                    "amenities": ["WiFi", "Parking", "Washer/Dryer", "Bay Views"],
                },
                {
                    "id": "sf003",
                    "name": "Modern Studio near Golden Gate",
                    "address": "789 Presidio Avenue, San Francisco, CA",
                    "price": 180,
                    "amenities": ["WiFi", "Kitchen", "Pet Friendly"],
                },
            ],
            "new york": [
                {
                    "id": "ny001",
                    "name": "Brooklyn Brownstone Apartment",
                    "address": "321 Bedford Avenue, Brooklyn, NY",
                    "price": 175,
                    "amenities": ["WiFi", "Kitchen", "Backyard Access"],
                },
                {
                    "id": "ny002",
                    "name": "Manhattan Skyline Penthouse",
                    "address": "555 Fifth Avenue, Manhattan, NY",
                    "price": 350,
                    "amenities": ["WiFi", "Gym", "Doorman", "City Views"],
                },
                {
                    "id": "ny003",
                    "name": "Artsy East Village Loft",
                    "address": "88 Avenue A, Manhattan, NY",
                    "price": 195,
                    "amenities": ["WiFi", "Washer/Dryer", "Exposed Brick"],
                },
            ],
            "los angeles": [
                {
                    "id": "la001",
                    "name": "Venice Beach Bungalow",
                    "address": "234 Ocean Front Walk, Venice, CA",
                    "price": 200,
                    "amenities": ["WiFi", "Beach Access", "Patio"],
                },
                {
                    "id": "la002",
                    "name": "Hollywood Hills Villa",
                    "address": "777 Mulholland Drive, Los Angeles, CA",
                    "price": 400,
                    "amenities": ["WiFi", "Pool", "City Views", "Hot Tub"],
                },
            ],
        }

        # Track bookings
        self.bookings = []

    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"

    @function_tool
    async def search_the_web(self, context: RunContext, query: str) -> str:
        """Search the live web for current information.

        Use this for current events, recent news, prices, weather, sports
        scores, or any factual question that requires up-to-date information
        from the internet.

        Args:
            query: A natural-language search query. Examples: "weather in Tokyo
                today", "latest AI news", "current price of Bitcoin".
        """
        # Fire a hold message so the user doesn't sit in silence while the
        # search runs. ``session.say`` returns immediately (TTS plays in the
        # background) so the acknowledgment is heard CONCURRENTLY with the
        # search — perceived latency drops from ~3s of dead air to ~0s.
        #
        # `add_to_chat_ctx=False` keeps the acknowledgment out of the
        # conversation history, so the LLM and the judges don't see filler
        # phrases as part of the substantive conversation. This is a UX cue,
        # not conversation content.
        #
        # `allow_interruptions=True` lets the user cancel mid-acknowledgment
        # (e.g. "actually never mind") — the framework will then cancel the
        # in-flight tool and discard the result.
        context.session.say(
            random.choice(_HOLD_MESSAGES_SEARCH),
            allow_interruptions=True,
            add_to_chat_ctx=False,
        )

        def _ddg_search() -> list[dict]:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=3))

        try:
            results = await asyncio.to_thread(_ddg_search)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the LLM
            return f"Web search failed: {exc}. The user should try a different query."

        if not results:
            return f"No web results were found for '{query}'."

        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"• {title}\n  {body}\n  Source: {href}")
        return "Top web results:\n\n" + "\n\n".join(lines)

    @function_tool
    async def search_airbnbs(self, context: RunContext, city: str) -> str:
        """Search for available Airbnbs in a city.

        Args:
            city: The city name to search for Airbnbs (e.g., 'San Francisco', 'New York', 'Los Angeles')
        """
        city_lower = city.lower()

        if city_lower not in self.airbnbs:
            return f"Sorry, I don't have any Airbnb listings for {city} at the moment. Available cities are: San Francisco, New York, and Los Angeles."

        listings = self.airbnbs[city_lower]
        result = f"Found {len(listings)} Airbnbs in {city}:\n\n"

        for listing in listings:
            result += f"• {listing['name']}\n"
            result += f"  Address: {listing['address']}\n"
            result += f"  Price: ${listing['price']} per night\n"
            result += f"  Amenities: {', '.join(listing['amenities'])}\n"
            result += f"  ID: {listing['id']}\n\n"

        return result

    @function_tool
    async def book_airbnb(self, context: RunContext, airbnb_id: str, guest_name: str, check_in_date: str, check_out_date: str) -> str:
        """Book an Airbnb.

        Args:
            airbnb_id: The ID of the Airbnb to book (e.g., 'sf001')
            guest_name: Name of the guest making the booking
            check_in_date: Check-in date (e.g., 'January 15, 2025')
            check_out_date: Check-out date (e.g., 'January 20, 2025')
        """
        # Find the Airbnb
        airbnb = None
        for city_listings in self.airbnbs.values():
            for listing in city_listings:
                if listing['id'] == airbnb_id:
                    airbnb = listing
                    break
            if airbnb:
                break

        if not airbnb:
            return f"Sorry, I couldn't find an Airbnb with ID {airbnb_id}. Please search for available listings first."

        # Create booking
        booking = {
            "confirmation_number": f"BK{len(self.bookings) + 1001}",
            "airbnb_name": airbnb['name'],
            "address": airbnb['address'],
            "guest_name": guest_name,
            "check_in": check_in_date,
            "check_out": check_out_date,
            "total_price": airbnb['price'],
        }

        self.bookings.append(booking)

        result = f"✓ Booking confirmed!\n\n"
        result += f"Confirmation Number: {booking['confirmation_number']}\n"
        result += f"Property: {booking['airbnb_name']}\n"
        result += f"Address: {booking['address']}\n"
        result += f"Guest: {booking['guest_name']}\n"
        result += f"Check-in: {booking['check_in']}\n"
        result += f"Check-out: {booking['check_out']}\n"
        result += f"Nightly Rate: ${booking['total_price']}\n\n"
        result += f"You'll receive a confirmation email shortly. Have a great stay!"

        return result        

async def entrypoint(ctx: agents.JobContext):
    """Entry point for the agent."""

    # Configure the voice pipeline with the essentials
    session = AgentSession(
        stt=deepgram.STT(model="nova-2"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4.1-mini")),
        tts=openai.TTS(voice="echo"),
        vad=silero.VAD.load(),
    )

    # Start the session
    await session.start(
        room=ctx.room,
        agent=Assistant()
    )

    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly and ask how you can help."
    )

if __name__ == "__main__":
    # Run the agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))