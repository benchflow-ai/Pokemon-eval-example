from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, ForeignKey, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import enum
import os
from datetime import datetime
import urllib.parse

# Create base class for SQLAlchemy models
Base = declarative_base()

# Define action types as an Enum
class ActionType(enum.Enum):
    KEYPRESS = "keypress"
    WAIT = "wait"
    
class GameSession(Base):
    """Represents a single gameplay session."""
    __tablename__ = 'game_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(100), unique=True, nullable=False)
    model = Column(String(50), nullable=False)  # Model used (e.g., gpt-4o)
    snapshot_id = Column(String(100), nullable=True)  # Snapshot ID used
    instance_id = Column(String(100), nullable=True)  # Instance ID
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    
    # Relationships
    steps = relationship("GameStep", back_populates="session", cascade="all, delete-orphan")
    evaluation = relationship("GameEvaluation", back_populates="session", uselist=False, cascade="all, delete-orphan")
    milestone_achievements = relationship("MilestoneAchievement", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<GameSession(id={self.id}, session_id='{self.session_id}')>"
    
class GameStep(Base):
    """Represents a single step in a gameplay session."""
    __tablename__ = 'game_steps'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('game_sessions.id'), nullable=False)
    step_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action_type = Column(Enum(ActionType), nullable=True)
    action_data = Column(Text, nullable=True)  # JSON string with action details
    screenshot_b64 = Column(Text, nullable=True)  # Base64-encoded screenshot
    
    # Relationships
    session = relationship("GameSession", back_populates="steps")
    
    def __repr__(self):
        return f"<GameStep(id={self.id}, session_id={self.session_id}, step_number={self.step_number})>"

class Milestone(enum.Enum):
    START = "Game Start"
    NAMING = "Character Naming"
    FIRST_POKEMON = "First Pokemon"
    FIRST_BATTLE = "First Battle"
    FIRST_GYM = "First Gym"
    POKEDEX_10 = "Pokedex 10"
    SECOND_GYM = "Second Gym"
    EVOLUTION = "First Evolution"
    THIRD_GYM = "Third Gym"
    POKEDEX_30 = "Pokedex 30"
    ELITE_FOUR = "Elite Four"

class MilestoneAchievement(Base):
    """Represents when a specific milestone was achieved in a game session."""
    __tablename__ = 'milestone_achievements'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('game_sessions.id'), nullable=False)
    milestone = Column(Enum(Milestone), nullable=False)
    step_number = Column(Integer, nullable=False)  # Step at which milestone was achieved
    timestamp = Column(DateTime, nullable=False)
    time_taken = Column(Float, nullable=False)  # Time in seconds since session start
    
    # Relationships
    session = relationship("GameSession", back_populates="milestone_achievements")
    
    def __repr__(self):
        return f"<MilestoneAchievement(milestone={self.milestone}, step={self.step_number}, time_taken={self.time_taken})>"
    
class GameEvaluation(Base):
    """Represents an evaluation of a gameplay session."""
    __tablename__ = 'game_evaluations'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('game_sessions.id'), nullable=False, unique=True)
    evaluation_timestamp = Column(DateTime, default=datetime.utcnow)
    final_milestone = Column(Enum(Milestone), nullable=True)
    total_steps = Column(Integer, nullable=True)
    raw_score = Column(Integer, nullable=True)
    final_score = Column(Integer, nullable=True)
    strategy_score = Column(Float, nullable=True)
    time_taken = Column(Float, nullable=True)  # In seconds
    evaluation_text = Column(Text, nullable=True)
    
    # Relationships
    session = relationship("GameSession", back_populates="evaluation")
    
    def __repr__(self):
        return f"<GameEvaluation(id={self.id}, session_id={self.session_id}, final_score={self.final_score})>"

# Define milestone benchmark times (in seconds) for scoring
# These are the target times for achieving each milestone
MILESTONE_BENCHMARKS = {
    Milestone.START: 60,          # 1 minute to start game
    Milestone.NAMING: 180,        # 3 minutes to name character
    Milestone.FIRST_POKEMON: 300, # 5 minutes to get first pokemon
    Milestone.FIRST_BATTLE: 480,  # 8 minutes to complete first battle
    Milestone.FIRST_GYM: 1200,    # 20 minutes to first gym
    Milestone.POKEDEX_10: 1500,   # 25 minutes for 10 pokemon
    Milestone.SECOND_GYM: 1800,   # 30 minutes to second gym
    Milestone.EVOLUTION: 1500,    # 25 minutes to evolve pokemon
    Milestone.THIRD_GYM: 2400,    # 40 minutes to third gym
    Milestone.POKEDEX_30: 3000,   # 50 minutes for 30 pokemon
    Milestone.ELITE_FOUR: 3600    # 60 minutes to reach elite four
}

def init_db(db_url=None):
    """Initialize the database connection and create tables if they don't exist."""
    if db_url is None:
        # Default to NeonDB URL if available, otherwise use SQLite
        db_url = os.environ.get('DATABASE_URL', os.environ.get('DATABASE_URL', 'sqlite:///pokemon_agent.db'))
    
    # If using NeonDB, ensure SSL parameters are set properly
    if 'neon.tech' in db_url:
        # Parse the URL to add SSL requirements
        url_parts = urllib.parse.urlparse(db_url)
        query_params = dict(urllib.parse.parse_qsl(url_parts.query))
        
        # Set SSL parameters for NeonDB
        query_params['sslmode'] = 'require'
        
        # Reconstruct the URL with SSL parameters
        url_parts = url_parts._replace(query=urllib.parse.urlencode(query_params))
        db_url = urllib.parse.urlunparse(url_parts)
    
    # Create engine with appropriate connection pooling settings
    engine_args = {}
    
    # For NeonDB (serverless), configure connection pooling for efficient connection handling
    if 'neon.tech' in db_url:
        # Serverless databases like NeonDB work better with these settings
        engine_args = {
            'pool_size': 5,               # Smaller pool size for serverless
            'max_overflow': 10,           # Allow some overflow connections
            'pool_timeout': 30,           # Wait longer for available connections
            'pool_recycle': 300,          # Recycle connections after 5 minutes
            'pool_pre_ping': True,        # Check connection validity before using
            'connect_args': {
                'connect_timeout': 10     # Timeout when establishing a connection
            }
        }
    
    # Create the engine with appropriate args
    engine = create_engine(db_url, **engine_args)
    
    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    
    # Create a session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    return engine, SessionLocal

def get_db_session(session_local):
    """Get a database session."""
    db = session_local()
    try:
        return db
    except:
        db.close()
        raise 