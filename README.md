# Agent Plays Pokémon Eval Harness

This project demonstrates how to use Morph Cloud to create a fully autonomous agent that can play Pokémon games in a Game Boy emulator. Using Claude 3.7 Sonnet, the agent can see the game through screenshots, interpret the game state, and take actions by controlling the emulator.

**IMPORTANT: This project does not include or distribute any ROM files. You need to provide your own legally obtained ROM files.**

## Getting Started

1. Install dependencies:
   ```
   pip install anthropic morphcloud
   ```
   
   You can also use `uv` to automatically download dependencies:
   ```
   uv pip install anthropic morphcloud
   ```

2. Set your API keys:
   ```
   export ANTHROPIC_API_KEY="your_api_key"
   export MORPH_API_KEY="your_morph_api_key"
   ```

3. Run the emulator setup with your ROM file:
   ```
   python emulator/emulator_setup_rom.py --rom path/to/your/rom.gb
   ```

4. Run the agent using the snapshot ID from the setup:
   ```
   python emu_agent.py --snapshot snapshot_id --turns 100 --model gpt-4o-mini
   ```

# Pokémon Game Agent

This project uses an AI agent to automatically play Pokémon through an emulator in a cloud environment.

## Setup

### Prerequisites
- Python 3.8+
- An OpenAI API key (for the GPT-4o model)
- A MorphCloud API key (for the cloud emulator)
- A PostgreSQL database (e.g. can use SQLite or Neon)

### Installation

1. Install the required dependencies:
```bash
pip install morphcloud openai sqlalchemy psycopg2-binary
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export MORPH_API_KEY="your_morph_cloud_api_key"
export DATABASE_URL="postgres://username:password@endpoint-name.neon.tech/dbname"
```

## Using NeonDB

### Setting up NeonDB

1. Create a NeonDB account at https://neon.tech/
2. Create a new project and database
3. Get your connection string from the NeonDB dashboard
4. Set the `DATABASE_URL` environment variable with your connection string

### Connection String Format

The NeonDB connection string should be in the following format:
```
postgres://username:password@endpoint-name.neon.tech/dbname
```

The system will automatically add the necessary SSL parameters.

## Running the Agent

To run the agent with the NeonDB database:

```bash
python emu_agent.py --model gpt-4o --verbose
```

## Evaluating Game Progress

To evaluate a gameplay session:

```bash
python evaluate_instance.py --session-id SESSION_ID
```

Replace `SESSION_ID` with the UUID of the game session you want to evaluate.

## Database Schema

The system uses the following tables:

- `game_sessions`: Stores information about each gameplay session
- `game_steps`: Stores individual steps in the gameplay, including screenshots
- `milestone_achievements`: Records when each gameplay milestone was achieved
- `game_evaluations`: Stores evaluation results for each session

## Milestones

The system tracks the following gameplay milestones:

- Game Start
- Character Naming
- First Pokemon
- First Battle
- First Gym
- Pokedex 10
- Second Gym
- Evolution
- Third Gym
- Pokedex 30
- Elite Four

Each milestone has a base score, and the final score includes time efficiency bonuses.
