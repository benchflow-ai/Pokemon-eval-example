"""
Evaluation module for Pokemon game progress.
This module can read from the database or from saved base64 images directory.
"""

import os
import json
import glob
import argparse
import openai
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

# Import database models
from models import init_db, GameSession, GameStep, GameEvaluation, MilestoneAchievement, Milestone, ActionType, MILESTONE_BENCHMARKS

# Milestone scoring weights
MILESTONE_SCORES = {
    Milestone.START: 5,
    Milestone.NAMING: 5,
    Milestone.FIRST_POKEMON: 10,
    Milestone.FIRST_BATTLE: 10,
    Milestone.FIRST_GYM: 15,
    Milestone.POKEDEX_10: 10,
    Milestone.SECOND_GYM: 15,
    Milestone.EVOLUTION: 10,
    Milestone.THIRD_GYM: 15,
    Milestone.POKEDEX_30: 10,
    Milestone.ELITE_FOUR: 20
}

def read_action_from_filename(filename: str) -> Dict[str, Any]:
    """Extract action data from the filename format used by EmuAgent."""
    try:
        # Expected format: step_1_keypress_UP.b64 or step_2_wait_wait_1000.b64
        parts = os.path.basename(filename).split('_')
        step_num = int(parts[1])
        action_type = parts[2]
        
        if action_type == "keypress":
            keys = parts[3:]
            # Remove the .b64 extension from the last key
            if keys and keys[-1].endswith('.b64'):
                keys[-1] = keys[-1].split('.')[0]
            return {"type": "keypress", "data": keys, "step": step_num}
        elif action_type == "wait":
            # Expected format: step_2_wait_wait_1000.b64
            # Parts[4] would contain the wait time with .b64 extension
            wait_time = parts[4].split('.')[0]
            return {"type": "wait", "data": int(wait_time), "step": step_num}
        else:
            return {"type": "unknown", "data": None, "step": step_num}
    except Exception as e:
        print(f"Error parsing action from filename {filename}: {e}")
        return {"type": "error", "data": None, "step": 0}

def load_base64_images(base64_dir: str) -> Dict[str, Any]:
    """Load all base64 images and action data from a directory (legacy support)."""
    result = {
        "screenshots": [],
        "actions": [],
        "timestamps": []
    }
    
    # Check if there's an index file
    index_file = os.path.join(base64_dir, "index.txt")
    if os.path.exists(index_file):
        print(f"Reading index file: {index_file}")
        with open(index_file, 'r') as f:
            # Skip the header line
            next(f, None)
            filenames = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    else:
        # Otherwise, glob all .b64 files
        print(f"No index file found, globbing all .b64 files in {base64_dir}")
        pattern = os.path.join(base64_dir, "*.b64")
        filenames = [os.path.basename(path) for path in glob.glob(pattern)]
        filenames.sort()  # Sort to ensure chronological order
    
    # Process each file
    for filename in filenames:
        file_path = os.path.join(base64_dir, filename)
        try:
            # Get file stats for timestamp
            stats = os.stat(file_path)
            file_timestamp = datetime.fromtimestamp(stats.st_mtime)
            
            # Read the base64 data
            with open(file_path, 'r') as f:
                image_data = f.read().strip()
            
            # Extract action from filename
            action = read_action_from_filename(filename)
            
            # Add to the result
            result["screenshots"].append(image_data)
            result["actions"].append(action)
            result["timestamps"].append(file_timestamp)
            
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
    
    # Calculate time taken
    if result["timestamps"]:
        start_time = min(result["timestamps"])
        end_time = max(result["timestamps"])
        time_taken = (end_time - start_time).total_seconds()
        result["time_taken"] = time_taken
    else:
        result["time_taken"] = 0
    
    print(f"Loaded {len(result['screenshots'])} screenshots and {len(result['actions'])} actions")
    print(f"Time taken: {result['time_taken']} seconds")
    
    return result

def load_data_from_db(session_id: str, db_session: Session) -> Dict[str, Any]:
    """Load gameplay data from the database."""
    print(f"Loading data for session: {session_id}")
    
    # Query the game session
    game_session = db_session.query(GameSession).filter_by(session_id=session_id).first()
    if not game_session:
        raise ValueError(f"No game session found with ID: {session_id}")
    
    # Query all steps for this session, ordered by step_number
    steps = db_session.query(GameStep).filter_by(session_id=game_session.id).order_by(GameStep.step_number).all()
    
    result = {
        "screenshots": [],
        "actions": [],
        "timestamps": [],
        "step_numbers": [],  # Track step numbers for milestone tracking
        "session": game_session
    }
    
    for step in steps:
        # Add screenshot
        if step.screenshot_b64:
            result["screenshots"].append(step.screenshot_b64)
        
        # Add timestamp
        result["timestamps"].append(step.timestamp)
        
        # Add step number
        result["step_numbers"].append(step.step_number)
        
        # Add action if available
        if step.action_type:
            action_data = json.loads(step.action_data) if step.action_data else {}
            
            # Convert to the expected format
            if step.action_type == ActionType.KEYPRESS:
                keys = action_data.get("keys", [])
                action = {"type": "keypress", "data": keys, "step": step.step_number}
            elif step.action_type == ActionType.WAIT:
                ms = action_data.get("ms", 1000)
                action = {"type": "wait", "data": ms, "step": step.step_number}
            else:
                action = {"type": "unknown", "data": None, "step": step.step_number}
                
            result["actions"].append(action)
    
    # Calculate time taken
    if game_session.started_at and game_session.ended_at:
        time_taken = (game_session.ended_at - game_session.started_at).total_seconds()
    elif result["timestamps"]:
        start_time = min(result["timestamps"])
        end_time = max(result["timestamps"])
        time_taken = (end_time - start_time).total_seconds()
    else:
        time_taken = 0
        
    result["time_taken"] = time_taken
    result["start_time"] = game_session.started_at if game_session.started_at else (min(result["timestamps"]) if result["timestamps"] else None)
    
    print(f"Loaded {len(result['screenshots'])} screenshots and {len(result['actions'])} actions")
    print(f"Time taken: {result['time_taken']} seconds")
    print(f"Model used: {game_session.model}")
    
    return result

def detect_milestones(game_data: Dict[str, Any], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Detect when each milestone was reached using OpenAI's GPT-4o.
    
    Args:
        game_data: Dict with screenshots, actions, timestamps, etc.
        api_key: OpenAI API key (optional, will use env var if not provided)
        
    Returns:
        List of milestone achievements with step numbers
    """
    # Initialize OpenAI client
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    client = openai.OpenAI(api_key=api_key)
    
    # Create milestone details string for prompt
    milestone_details = "\n".join([
        f"- {milestone.value}: {MILESTONE_BENCHMARKS.get(milestone, 300)}" 
        for milestone in Milestone
    ])
    
    # Number of screenshots to analyze
    num_screenshots = len(game_data["screenshots"])
    
    # Define checkpoint indices to sample the gameplay at reasonable intervals
    # Create more checkpoints at the beginning (where most milestones are likely to occur)
    if num_screenshots <= 10:
        # For very short gameplay, check every screenshot
        checkpoints = list(range(num_screenshots))
    elif num_screenshots <= 30:
        # For shorter gameplay, check every 3 screenshots
        checkpoints = list(range(0, num_screenshots, 3))
    else:
        # For longer gameplay, create a balanced distribution of checkpoints
        # More frequent at the beginning, less frequent later
        checkpoints = list(range(0, 20, 2))  # Every 2nd screenshot in first 20
        checkpoints.extend(list(range(20, 50, 5)))  # Every 5th screenshot from 20-50
        checkpoints.extend(list(range(50, num_screenshots, 10)))  # Every 10th screenshot after 50
    
    # Ensure we always check the last screenshot
    if num_screenshots - 1 not in checkpoints and num_screenshots > 0:
        checkpoints.append(num_screenshots - 1)
    
    # Sort checkpoints to ensure they're in order
    checkpoints.sort()
    
    print(f"Analyzing {len(checkpoints)} checkpoints out of {num_screenshots} total screenshots")
    
    # Dictionary to track the first occurrence of each milestone
    milestone_occurrences = {}
    milestones_detected = []
    
    for i, checkpoint in enumerate(checkpoints):
        if checkpoint >= num_screenshots:
            print(f"Skipping checkpoint {checkpoint+1} as it exceeds available screenshots ({num_screenshots})")
            continue
            
        print(f"Analyzing checkpoint {i+1}/{len(checkpoints)} (screenshot {checkpoint+1})")
        
        # Get the screenshot at this checkpoint
        screenshot_data = game_data["screenshots"][checkpoint]
        step_number = game_data["step_numbers"][checkpoint] if "step_numbers" in game_data else checkpoint + 1
        timestamp = game_data["timestamps"][checkpoint]
        
        # Create start time delta
        start_time = game_data.get("start_time")
        if start_time and timestamp:
            time_delta = (timestamp - start_time).total_seconds()
        else:
            time_delta = 0
        
        # Prepare the prompt
        prompt = f"""Analyze this Pokémon screenshot (#screenshot {checkpoint+1}) and determine which of the following game milestones have been reached:

{milestone_details}

Respond ONLY with the names of milestones that have been definitively reached based on this screenshot.
If multiple milestones have been reached, list all of them separated by commas.
If no milestones can be definitively confirmed, respond with "NONE".

Format your response as a simple comma-separated list, for example: "Game Start, Character Naming"
DO NOT include any explanation, just the milestone names.
"""
        
        # Prepare the message with the image
        messages = [
            {"role": "system", "content": "You are an expert at analyzing Pokémon gameplay screenshots and identifying progress milestones."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_data}"}}
            ]}
        ]
        
        # Call GPT-4o Vision
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=100,  # Short response
                messages=messages
            )
            
            milestone_text = response.choices[0].message.content.strip()
            print(f"Milestone detection result: {milestone_text}")
            
            if milestone_text.upper() != "NONE":
                # Parse the milestone names
                milestone_names = [name.strip() for name in milestone_text.split(',')]
                
                # Add each detected milestone
                for name in milestone_names:
                    # Find the matching milestone enum
                    detected_milestone = None
                    for m in Milestone:
                        if m.value.lower() == name.lower():
                            detected_milestone = m
                            break
                    
                    if detected_milestone:
                        # Only record the first occurrence of each milestone
                        if detected_milestone not in milestone_occurrences:
                            milestone_occurrences[detected_milestone] = {
                                "milestone": detected_milestone,
                                "step_number": step_number,
                                "screenshot_index": checkpoint,
                                "timestamp": timestamp,
                                "time_taken": time_delta
                            }
                            milestones_detected.append(milestone_occurrences[detected_milestone])
                            print(f"New milestone detected: {detected_milestone.value} at step {step_number}, time: {time_delta:.1f} seconds")
        
        except Exception as e:
            print(f"Error in milestone detection: {e}")
    
    # Sort milestones by step number to ensure chronological order
    milestones_detected.sort(key=lambda x: x["step_number"])
    
    return milestones_detected

def calculate_score(milestones: List[Dict[str, Any]]) -> Tuple[int, Dict[str, float]]:
    """
    Calculate the score based on milestones achieved and time taken.
    
    Args:
        milestones: List of milestone achievements
        
    Returns:
        Tuple of (total_score, milestone_scores)
    """
    if not milestones:
        return 0, {}
    
    milestone_scores = {}
    total_score = 0
    
    for milestone_data in milestones:
        milestone = milestone_data["milestone"]
        time_taken = milestone_data["time_taken"]
        benchmark_time = MILESTONE_BENCHMARKS.get(milestone, 300)  # Default 5 minutes if not specified
        
        # Base score for reaching this milestone
        base_score = MILESTONE_SCORES.get(milestone, 5)  # Default 5 points if not specified
        
        # Time factor: faster completion gets higher score
        # If completed in exactly the benchmark time, factor is 1.0
        # If completed in half the benchmark time, factor is 2.0
        # If completed in twice the benchmark time, factor is 0.5
        time_factor = benchmark_time / time_taken if (time_taken > 0 and benchmark_time > 0) else 1.0
        
        # Cap the time factor between 0.1 and 3.0
        time_factor = max(0.1, min(3.0, time_factor))
        
        # Calculate the milestone score
        milestone_score = base_score * time_factor
        milestone_scores[milestone.name] = milestone_score
        total_score += milestone_score
    
    return int(total_score), milestone_scores

def evaluate_game_data(game_data: Dict[str, Any], api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate the game progress using OpenAI's GPT-4o.
    
    Args:
        game_data: Dict with screenshots, actions, time_taken, and session data
        api_key: OpenAI API key (optional, will use env var if not provided)
        
    Returns:
        Dict with evaluation results including milestone achievements
    """
    # Detect milestones
    milestones = detect_milestones(game_data, api_key)
    
    # Calculate score
    total_score, milestone_scores = calculate_score(milestones)
    
    # Find the final milestone (the one with the highest progression value)
    # Safely get the final milestone to avoid list index out of range error
    final_milestone = None
    if milestones:
        final_milestone = milestones[-1]["milestone"]
    
    # Count actions by type for summary
    action_counts = {}
    for action in game_data.get("actions", []):
        action_type = action["type"]
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    # Calculate total steps
    total_steps = len(game_data.get("screenshots", []))
    
    # Create the evaluation result
    result = {
        "final_milestone": final_milestone.value if final_milestone else "None",
        "milestone_enum": final_milestone,
        "milestone_achievements": milestones,
        "milestone_scores": milestone_scores,
        "total_steps": total_steps,
        "total_score": total_score,
        "time_taken": game_data["time_taken"],
        "action_counts": action_counts
    }
    
    # Generate a human-readable evaluation summary
    milestone_summary = "\n".join([
        f"- {m['milestone'].value}: Step {m['step_number']}, Time: {m['time_taken']:.1f}s, Score: {milestone_scores.get(m['milestone'].name, 0):.1f}"
        for m in milestones
    ])
    
    summary = f"""
===== GAME EVALUATION SUMMARY =====
Final milestone: {result['final_milestone']}
Total steps: {total_steps}
Time taken: {game_data['time_taken']:.1f} seconds
Total score: {total_score}

Milestone achievements:
{milestone_summary}

Action breakdown:
{', '.join([f"{k}: {v}" for k, v in action_counts.items()])}
"""
    result["evaluation_text"] = summary
    
    print(summary)
    
    return result

def evaluate_folder(base64_dir: str, output_file: Optional[str] = None, api_key: Optional[str] = None) -> Dict:
    """Evaluate game progress from a folder of base64 screenshots (legacy support)."""
    print(f"Evaluating folder: {base64_dir}")
    
    # Load the game data
    game_data = load_base64_images(base64_dir)
    
    if not game_data["screenshots"]:
        print("No screenshots found. Cannot evaluate empty game data.")
        return {"error": "No screenshots found"}
    
    # Evaluate the game data
    evaluation = evaluate_game_data(game_data, api_key)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation, f, indent=2)
        print(f"Evaluation results saved to {output_file}")
    
    # Print summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Final milestone: {evaluation.get('final_milestone', 'Unknown')}")
    print(f"Total steps: {evaluation.get('total_steps', 0)}")
    print(f"Score: {evaluation.get('total_score', 0)}")
    print(f"Time taken: {game_data['time_taken']:.1f} seconds")
    
    return evaluation

def evaluate_session(session_id: str, output_file: Optional[str] = None, api_key: Optional[str] = None, db_url: Optional[str] = None) -> Dict:
    """Evaluate game progress for a session from the database."""
    print(f"Evaluating session: {session_id}")
    
    # Initialize database connection
    engine, SessionLocal = init_db(db_url)
    db_session = SessionLocal()
    
    try:
        # Load the game data from database
        game_data = load_data_from_db(session_id, db_session)
        
        if not game_data["screenshots"]:
            print("No screenshots found. Cannot evaluate empty game data.")
            return {"error": "No screenshots found"}
        
        # Evaluate the game data
        evaluation = evaluate_game_data(game_data, api_key)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(evaluation, f, indent=2)
            print(f"Evaluation results saved to {output_file}")
        
        # Save to database
        game_session = game_data["session"]
        
        # First, delete any existing evaluations or milestone achievements
        db_session.query(GameEvaluation).filter_by(session_id=game_session.id).delete()
        db_session.query(MilestoneAchievement).filter_by(session_id=game_session.id).delete()
        
        # Create new evaluation
        db_eval = GameEvaluation(
            session_id=game_session.id,
            final_milestone=evaluation.get("milestone_enum"),
            total_steps=evaluation.get("total_steps", 0),
            raw_score=evaluation.get("total_score", 0),
            final_score=evaluation.get("total_score", 0),
            strategy_score=0,  # This could be calculated in the future
            time_taken=game_data["time_taken"],
            evaluation_text=evaluation.get("evaluation_text", "")
        )
        db_session.add(db_eval)
        
        # Save milestone achievements to database
        for milestone_data in evaluation.get("milestone_achievements", []):
            milestone = milestone_data["milestone"]
            step_number = milestone_data["step_number"]
            timestamp = milestone_data["timestamp"]
            time_taken = milestone_data["time_taken"]
            
            milestone_achievement = MilestoneAchievement(
                session_id=game_session.id,
                milestone=milestone,
                step_number=step_number,
                timestamp=timestamp,
                time_taken=time_taken
            )
            db_session.add(milestone_achievement)
        
        db_session.commit()
        print(f"Evaluation and {len(evaluation.get('milestone_achievements', []))} milestone achievements saved to database for session: {session_id}")
        
        # Print summary
        print("\n===== EVALUATION SUMMARY =====")
        print(f"Final milestone: {evaluation.get('final_milestone', 'Unknown')}")
        print(f"Total steps: {evaluation.get('total_steps', 0)}")
        print(f"Score: {evaluation.get('total_score', 0)}")
        print(f"Time taken: {game_data['time_taken']:.1f} seconds")
        
        # Print milestone summary
        if "milestone_achievements" in evaluation and evaluation["milestone_achievements"]:
            print("\nMilestone Achievements:")
            for milestone in evaluation["milestone_achievements"]:
                m_name = milestone["milestone"].value
                m_step = milestone["step_number"]
                m_time = milestone["time_taken"]
                m_score = evaluation["milestone_scores"].get(milestone["milestone"].name, 0)
                print(f"- {m_name}: Step {m_step}, Time: {m_time:.1f}s, Score: {m_score:.1f}")
        
        return evaluation
    finally:
        db_session.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Pokemon game progress')
    parser.add_argument('--dir', '-d', help='Directory containing base64 screenshot files (legacy mode)')
    parser.add_argument('--session-id', '-s', help='Session ID to evaluate from database')
    parser.add_argument('--output', '-o', help='Output file for evaluation results (JSON)')
    parser.add_argument('--api-key', help='OpenAI API key (optional, uses env var if not provided)')
    parser.add_argument('--db-url', help='Database URL (optional, uses default if not provided)')
    
    args = parser.parse_args()
    
    if args.dir and args.session_id:
        print("Error: Cannot specify both --dir and --session-id. Choose one mode.")
        return
    
    if args.dir:
        evaluate_folder(args.dir, args.output, args.api_key)
    elif args.session_id:
        evaluate_session(args.session_id, args.output, args.api_key, args.db_url)
    else:
        print("Error: Must specify either --dir or --session-id")

if __name__ == "__main__":
    main() 