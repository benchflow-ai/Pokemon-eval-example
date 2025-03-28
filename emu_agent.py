# /// script
# dependencies = [
# "morphcloud",
# "anthropic",
# "sqlalchemy",
# "psycopg2-binary"
# ]
# ///

import os
import time
import json
import base64
import anthropic
import logging
import re
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import MorphComputer from local file
from morph_computer import MorphComputer

# Import database models
from models import init_db, GameSession, GameStep, ActionType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmuAgent")

class EmuAgent:
    """
    A fully autonomous agent that uses Claude to play games through
    the MorphComputer interface, automatically taking screenshots after each action.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20240620",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        computer: Optional[MorphComputer] = None,
        snapshot_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        setup_computer: bool = True,
        verbose: bool = True,
        save_base64: bool = True,  # Parameter to save base64 strings
    ):
        """Initialize the EmuAgent."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required")
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.save_base64 = save_base64  # Store the preference
        
        db_url = os.environ.get("DATABASE_URL")
        # Initialize database
        self.engine, self.SessionLocal = init_db(db_url)
        self.db = self.SessionLocal()
        
        # Create a unique session ID
        self.session_id = str(uuid.uuid4())
        
        # Create a unique folder for this run (for backward compatibility)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.screenshot_dir = f"screenshots_{timestamp}"
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        # Create a subfolder for base64 strings if needed (for backward compatibility)
        if self.save_base64:
            self.base64_dir = f"base64_{timestamp}"
            os.makedirs(self.base64_dir, exist_ok=True)
            # Create an index file for easy sequential access
            self.index_file = os.path.join(self.base64_dir, "index.txt")
            with open(self.index_file, 'w') as f:
                f.write("# List of base64 screenshot files\n")
        
        self.step_counter = 0
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize computer if needed
        self.computer = computer
        if self.computer is None and setup_computer:
            if instance_id:
                self.log(f"Connecting to existing instance: {instance_id}")
                self.computer = MorphComputer(instance_id=instance_id)
            elif snapshot_id:
                self.log(f"Creating new computer from snapshot: {snapshot_id}")
                self.computer = MorphComputer(snapshot_id=snapshot_id)
            else:
                self.log("Creating new computer from default snapshot")
                self.computer = MorphComputer()
        
        # Conversation history
        self.messages = []
        self.system_prompt = """
You are an AI game-playing assistant that can see and interact with a game through screenshots.
You'll receive screenshots of the game state and can take actions by pressing keys.

CAPABILITIES:
- Observe the game through screenshots
- Press specific keys to control the game

AVAILABLE KEYS (based on what you see in the interface):
- "UP" (arrow up)
- "DOWN" (arrow down)
- "LEFT" (arrow left)
- "RIGHT" (arrow right)
- "ENTER" (start)
- "SPACE" (select)
- "Z" (A)
- "X" (B)

HOW THE SYSTEM WORKS:
1. You'll receive a screenshot of the game
2. Analyze the game state and decide on the best action
3. Specify the key to press using the action format below
4. The system will press the key and automatically take a new screenshot
5. The new screenshot will be sent to you to decide on your next action
6. This loop continues until the game session ends

To specify a key press, use this format:
```action
{
  "action_type": "keypress",
  "keys": ["Z"]
}
```

You can also wait if needed:
```action
{
  "action_type": "wait",
  "ms": 1000
}
```

As you play, explain your reasoning and strategy. Describe what you observe in the game and why you're making specific moves.
"""
        self.init_conversation()
        
        # Create a database session entry
        self.game_session = GameSession(
            session_id=self.session_id,
            model=self.model,
            snapshot_id=snapshot_id,
            instance_id=instance_id
        )
        self.db.add(self.game_session)
        self.db.commit()
        
    def init_conversation(self):
        """Initialize or reset the conversation history."""
        # Initialize with empty message list (system prompt is passed separately)
        self.messages = []
    
    def log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            logger.info(message)
    
    def __enter__(self):
        """Context manager entry."""
        if self.computer:
            self.computer.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.computer:
            self.computer.__exit__(exc_type, exc_val, exc_tb)
        
        # Update session end time
        if hasattr(self, 'game_session'):
            self.game_session.ended_at = datetime.utcnow()
            self.db.commit()
            
        # Close database connection
        if hasattr(self, 'db'):
            self.db.close()
            
    def take_screenshot(self) -> str:
        """Take a screenshot and return the encoded image data."""
        self.log("Taking screenshot...")
        try:
            return self.computer.screenshot()
        except Exception as e:
            self.log(f"Error taking screenshot: {e}")
            return None
            
    def take_save_state(self) -> str:
        """Take a save state and return the encoded Core.bin data."""
        self.log("Taking save state...")
        try:
            return self.computer.take_save_state()
        except Exception as e:
            self.log(f"Error taking save state: {e}")
            return None
            
    def add_screenshot_to_conversation(self) -> None:
        """Take a screenshot and add it to the conversation as an image."""
        try:
            screenshot_data = self.take_screenshot()
            if screenshot_data:
                # Add screenshot as a user message with image
                if len(self.messages) > 0 and self.messages[-1]["role"] == "assistant":
                    # If the last message was from the assistant, add the screenshot as a user message
                    self.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "Screenshot result from your last action:"
                        }, {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_data
                            }
                        }]
                    })
                else:
                    # For the initial screenshot or if conversation flow needs correction
                    self.messages.append({
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "Here's the current game state. What action will you take next?"
                        }, {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": screenshot_data
                            }
                        }]
                    })
                self.log("Added screenshot to conversation")
                
                # Add screenshot to database
                self.step_counter += 1
                game_step = GameStep(
                    session_id=self.game_session.id,
                    step_number=self.step_counter,
                    screenshot_b64=screenshot_data,
                    timestamp=datetime.utcnow()
                )
                self.db.add(game_step)
                self.db.commit()
                
            else:
                self.log("Failed to add screenshot - no data")
        except Exception as e:
            self.log(f"Error adding screenshot: {e}")
            
    def add_save_state_to_conversation(self) -> None:
        """Take a save state and add the Core.bin data to the conversation."""
        try:
            save_state_data = self.take_save_state()
            if save_state_data:
                message = {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Here's the emulator save state data (Core.bin in base64 format): " + save_state_data
                    }]
                }
                self.messages.append(message)
                self.log("Added save state data to conversation")
            else:
                self.log("Failed to add save state - no data")
        except Exception as e:
            self.log(f"Error adding save state: {e}")
            
    def save_screenshot(self, action_type: str, action: str, screenshot_data: str = None) -> str:
        """Save a screenshot with the specified naming format (backward compatibility)."""
        try:
            if screenshot_data is None:
                screenshot_data = self.take_screenshot()
                
            if screenshot_data:
                # Decode base64 image data
                image_data = base64.b64decode(screenshot_data)
                
                # Create filename with the specified format
                filename = f"step_{self.step_counter}_{action_type}_{action}.png"
                filepath = os.path.join(self.screenshot_dir, filename)
                
                # Save the image
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                # Save base64 string if enabled
                if self.save_base64:
                    base64_filename = f"step_{self.step_counter}_{action_type}_{action}.b64"
                    base64_filepath = os.path.join(self.base64_dir, base64_filename)
                    
                    # Save base64 string to a file
                    with open(base64_filepath, 'w') as f:
                        f.write(screenshot_data)
                    
                    # Append to index file
                    with open(self.index_file, 'a') as f:
                        f.write(f"{base64_filename}\n")
                    
                    self.log(f"Saved base64 screenshot to {base64_filepath}")
                
                self.log(f"Saved screenshot to {filepath}")
                return filepath
            return None
        except Exception as e:
            self.log(f"Error saving screenshot: {e}")
            return None

    def execute_action(self, action_type: str, **params) -> bool:
        """Execute an action on the desktop."""
        self.log(f"Executing action: {action_type} with params: {params}")
        try:
            # Take screenshot before the action for the DB record
            screenshot_data = self.take_screenshot()
            
            success = False
            action_enum = None
            action_data = json.dumps(params)
            action_str = ""
            
            if action_type == "keypress":
                keys = params["keys"]
                self.computer.keypress(keys, params.get("press_ms", 500))
                action_enum = ActionType.KEYPRESS
                action_str = "_".join(keys)
                success = True
            elif action_type == "wait":
                self.computer.wait(params.get("ms", 1000))
                action_enum = ActionType.WAIT
                action_str = f"wait_{params.get('ms', 1000)}"
                success = True
            else:
                self.log(f"Unsupported action type: {action_type}")
                return False
            
            # For backward compatibility - save file
            self.save_screenshot(action_type, action_str, screenshot_data)
            
            # Update the database step with action information
            game_step = self.db.query(GameStep).filter_by(
                session_id=self.game_session.id,
                step_number=self.step_counter
            ).first()
            
            if game_step:
                game_step.action_type = action_enum
                game_step.action_data = action_data
                self.db.commit()
                
            return success
        except Exception as e:
            self.log(f"Error executing action {action_type}: {e}")
            return False
            
    def play(self, initial_prompt: str = "Analyze this game and start playing", 
             max_turns: int = 100, 
             max_no_action_turns: int = 3,
             include_save_states: bool = False) -> str:
        """
        Start a fully autonomous gameplay session.
        
        Args:
            initial_prompt: Initial instruction to Claude
            max_turns: Maximum number of turns to play
            max_no_action_turns: Maximum consecutive turns without actions before stopping
            include_save_states: Whether to include save state data with each turn
            
        Returns:
            Final response from Claude
        """
        self.log(f"Starting autonomous gameplay with prompt: {initial_prompt}")
        
        # Add initial instruction to the conversation
        self.messages.append({
            "role": "user", 
            "content": initial_prompt
        })
        
        # Add initial screenshot to conversation
        self.add_screenshot_to_conversation()
        
        # Optionally add initial save state
        if include_save_states:
            self.add_save_state_to_conversation()
        
        # Get Claude's first response
        response = self.get_next_action()
        last_response = response
        
        # Process action loop
        no_action_count = 0
        for turn in range(max_turns):
            self.log(f"Turn {turn+1}/{max_turns}")
            
            # Check if Claude wants to take an action
            action = self.extract_action(response)
            
            if not action:
                # No action requested, count it and potentially break
                no_action_count += 1
                self.log(f"No action requested ({no_action_count}/{max_no_action_turns})")
                
                if no_action_count >= max_no_action_turns:
                    self.log("Maximum no-action turns reached, ending gameplay")
                    break
                    
                # Prompt Claude again for an action
                self.messages.append({
                    "role": "user", 
                    "content": "Please specify an action to take in the game using the ```action{...}``` format."
                })
                response = self.get_next_action()
                last_response = response
                continue
                
            # Reset no-action counter when an action is found
            no_action_count = 0
                
            # Execute the action
            self.execute_action(**action)
            
            # IMPORTANT: Always take a new screenshot after action
            self.add_screenshot_to_conversation()
            
            # Optionally add save state after each action
            if include_save_states:
                self.add_save_state_to_conversation()
            
            # Get Claude's next step
            response = self.get_next_action()
            last_response = response
        
        # Update session end time
        self.game_session.ended_at = datetime.utcnow()
        self.db.commit()
        
        return last_response
    
    def get_next_action(self) -> str:
        """Get Claude's next action based on the conversation so far."""
        try:
            self.log("Getting next action from Claude...")
        
            # Call Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=self.messages
            )
            
            response_text = response.content[0].text
            self.log(f"Claude response: {response_text[:100]}...")
            
            # Add to conversation history
            self.messages.append({"role": "assistant", "content": response_text})
            
            return response_text

        except Exception as e:
            self.log(f"Error getting response from Claude: {e}")
            return f"Error: {str(e)}"
    
    def extract_action(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract an action from Claude's response."""
        # Look for action blocks
        action_match = re.search(r'```action\n(.*?)\n```', response, re.DOTALL)
        
        if not action_match:
            return None
            
        try:
            action_json = action_match.group(1).strip()
            action = json.loads(action_json)
            return action
        except json.JSONDecodeError:
            self.log(f"Failed to parse action JSON: {action_match.group(1)}")
            return None
    
    def close(self):
        """Clean up resources used by the agent."""
        if hasattr(self, 'computer') and self.computer:
            try:
                self.computer.cleanup()
                self.log("Cleaned up computer resources")
            except Exception as e:
                self.log(f"Error cleaning up computer: {e}")
        
        # Close database connection
        if hasattr(self, 'db'):
            self.db.close()

# Simple command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the EmuAgent')
    parser.add_argument('--snapshot', '-s', help='Snapshot ID to use')
    parser.add_argument('--instance', '-i', help='Instance ID to use')
    parser.add_argument('--turns', '-t', type=int, default=100, help='Max turns to play')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--save-states', action='store_true', help='Include save state data with each turn')
    parser.add_argument('--no-save-base64', action='store_true', help='Disable saving base64 string images')
    parser.add_argument('--model', '-m', default='claude-3-5-sonnet-20240620', help='Claude model to use (default: claude-3-5-sonnet-20240620)')
    
    args = parser.parse_args()
    
    with EmuAgent(
        snapshot_id=args.snapshot,
        instance_id=args.instance,
        verbose=args.verbose,
        save_base64=not args.no_save_base64,
        model=args.model,
    ) as agent:
        agent.play(
            max_turns=args.turns,
            include_save_states=args.save_states
        )
