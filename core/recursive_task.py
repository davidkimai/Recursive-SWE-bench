# recursive_swe_bench/core/recursive_task.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import datetime
import uuid
import json
import copy

class TaskStatus(Enum):
    """Status of a recursive task."""
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress" 
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    PERFECT_SOLUTION = "perfect_solution"
    ABANDONED = "abandoned"


@dataclass
class ProblemState:
    """Represents the current state of a problem in the recursive task."""
    problem_id: str
    description: str
    code_context: Dict[str, Any]
    requirements: List[Dict[str, Any]]
    difficulty: float  # 0.0 to 1.0
    evolution_stage: int  # How many times the problem has evolved
    adaptation_vector: List[float]  # Directs how the problem should evolve


@dataclass
class EvaluationResult:
    """Results from evaluating a solution."""
    success: bool
    score: float  # 0.0 to 1.0 
    execution_results: Dict[str, Any]
    error_details: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class Feedback:
    """Structured feedback on a solution."""
    summary: str
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    focus_areas: List[str]
    adaptation_hints: List[Dict[str, Any]]


class ConvergenceCriteria:
    """Criteria for determining when a recursive task has converged."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.score_threshold = self.config.get("score_threshold", 0.95)
        self.min_iterations = self.config.get("min_iterations", 1)
        self.max_iterations = self.config.get("max_iterations", 10)
        self.score_delta_threshold = self.config.get("score_delta_threshold", 0.01)
        self.consecutive_plateau_limit = self.config.get("consecutive_plateau_limit", 3)
    
    def has_converged(self, trajectory: "Trajectory") -> bool:
        """Determine if the task has converged based on the trajectory."""
        if len(trajectory.steps) < self.min_iterations:
            return False
            
        if len(trajectory.steps) >= self.max_iterations:
            return True
            
        # Check if we've reached the score threshold
        latest_score = trajectory.steps[-1].result.score
        if latest_score >= self.score_threshold:
            return True
            
        # Check for plateau (little improvement over consecutive iterations)
        if len(trajectory.steps) >= self.consecutive_plateau_limit + 1:
            recent_scores = [step.result.score for step in 
                            trajectory.steps[-self.consecutive_plateau_limit-1:]]
            deltas = [abs(recent_scores[i+1] - recent_scores[i]) 
                     for i in range(len(recent_scores)-1)]
            
            if all(delta < self.score_delta_threshold for delta in deltas):
                return True
                
        return False


@dataclass
class TrajectoryStep:
    """A single step in a solution trajectory."""
    step_id: str
    timestamp: datetime.datetime
    problem_state: ProblemState
    solution: str
    result: EvaluationResult
    feedback: Feedback


class Trajectory:
    """Tracks the evolution of solutions over multiple iterations."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.steps: List[TrajectoryStep] = []
        self.metadata: Dict[str, Any] = {
            "start_time": datetime.datetime.now(),
            "task_id": task_id
        }
        
    def add_step(self, problem_state: ProblemState, solution: str, 
                result: EvaluationResult, feedback: Feedback) -> None:
        """Add a step to the trajectory."""
        step = TrajectoryStep(
            step_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(),
            problem_state=problem_state,
            solution=solution,
            result=result,
            feedback=feedback
        )
        self.steps.append(step)
        
    def get_solution_series(self) -> List[str]:
        """Return the series of solutions."""
        return [step.solution for step in self.steps]
    
    def get_score_series(self) -> List[float]:
        """Return the series of scores."""
        return [step.result.score for step in self.steps]
    
    def get_latest_step(self) -> Optional[TrajectoryStep]:
        """Get the most recent step in the trajectory."""
        if not self.steps:
            return None
        return self.steps[-1]
    
    def calculate_improvement_rate(self) -> float:
        """Calculate the rate of improvement across iterations."""
        scores = self.get_score_series()
        if len(scores) < 2:
            return 0.0
            
        return (scores[-1] - scores[0]) / len(scores)
    
    def calculate_volatility(self) -> float:
        """Calculate the volatility of scores across iterations."""
        scores = self.get_score_series()
        if len(scores) < 2:
            return 0.0
            
        deltas = [abs(scores[i+1] - scores[i]) for i in range(len(scores)-1)]
        return sum(deltas) / len(deltas)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the trajectory to a dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "metadata": self.metadata,
            "steps": [
                {
                    "step_id": step.step_id,
                    "timestamp": step.timestamp.isoformat(),
                    "problem_state": {
                        "problem_id": step.problem_state.problem_id,
                        "description": step.problem_state.description,
                        "code_context": step.problem_state.code_context,
                        "requirements": step.problem_state.requirements,
                        "difficulty": step.problem_state.difficulty,
                        "evolution_stage": step.problem_state.evolution_stage,
                        "adaptation_vector": step.problem_state.adaptation_vector
                    },
                    "solution": step.solution,
                    "result": {
                        "success": step.result.success,
                        "score": step.result.score,
                        "execution_results": step.result.execution_results,
                        "error_details": step.result.error_details,
                        "test_results": step.result.test_results,
                        "metrics": step.result.metrics
                    },
                    "feedback": {
                        "summary": step.feedback.summary,
                        "issues": step.feedback.issues,
                        "suggestions": step.feedback.suggestions,
                        "focus_areas": step.feedback.focus_areas,
                        "adaptation_hints": step.feedback.adaptation_hints
                    }
                }
                for step in self.steps
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Create a trajectory from a dictionary."""
        trajectory = cls(data["task_id"])
        trajectory.metadata = data["metadata"]
        
        for step_data in data["steps"]:
            problem_state = ProblemState(
                problem_id=step_data["problem_state"]["problem_id"],
                description=step_data["problem_state"]["description"],
                code_context=step_data["problem_state"]["code_context"],
                requirements=step_data["problem_state"]["requirements"],
                difficulty=step_data["problem_state"]["difficulty"],
                evolution_stage=step_data["problem_state"]["evolution_stage"],
                adaptation_vector=step_data["problem_state"]["adaptation_vector"]
            )
            
            result = EvaluationResult(
                success=step_data["result"]["success"],
                score=step_data["result"]["score"],
                execution_results=step_data["result"]["execution_results"],
                error_details=step_data["result"]["error_details"],
                test_results=step_data["result"]["test_results"],
                metrics=step_data["result"]["metrics"]
            )
            
            feedback = Feedback(
                summary=step_data["feedback"]["summary"],
                issues=step_data["feedback"]["issues"],
                suggestions=step_data["feedback"]["suggestions"],
                focus_areas=step_data["feedback"]["focus_areas"],
                adaptation_hints=step_data["feedback"]["adaptation_hints"]
            )
            
            trajectory.add_step(
                problem_state=problem_state,
                solution=step_data["solution"],
                result=result,
                feedback=feedback
            )
            
        return trajectory
    
    def save(self, filepath: str) -> None:
        """Save the trajectory to a file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Trajectory":
        """Load a trajectory from a file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class RecursiveTask:
    """
    Base class for recursive tasks that evolve based on model solutions.
    
    A recursive task provides a dynamic problem that adapts based on the 
    model's attempted solutions, creating a feedback loop that more accurately
    reflects real-world software engineering challenges.
    """
    
    def __init__(self, 
                 initial_state: ProblemState, 
                 config: Dict[str, Any] = None):
        """
        Initialize the recursive task with an initial problem state.
        
        Args:
            initial_state: The initial state of the problem
            config: Configuration options for the task
        """
        self.task_id = str(uuid.uuid4())
        self.state = initial_state
        self.config = config or {}
        self.trajectory = Trajectory(self.task_id)
        self.status = TaskStatus.INITIALIZED
        self.convergence_criteria = ConvergenceCriteria(
            config.get("convergence_criteria", {}))
        
    def get_current_problem(self) -> Dict[str, Any]:
        """
        Return the current problem description and context.
        
        Returns:
            A dictionary containing the current problem description and context
        """
        return {
            "description": self.state.description,
            "code_context": self.state.code_context,
            "requirements": self.state.requirements,
            "evolution_stage": self.state.evolution_stage
        }
    
    def evaluate_solution(self, solution: str) -> Tuple[EvaluationResult, Feedback]:
        """
        Evaluate a solution and generate feedback.
        
        Args:
            solution: The solution to evaluate
            
        Returns:
            A tuple containing the evaluation result and feedback
        """
        # Run the evaluation logic
        result = self._run_evaluation(solution)
        
        # Generate feedback based on the evaluation
        feedback = self._generate_feedback(solution, result)
        
        return result, feedback
    
    def update_state(self, 
                    solution: str, 
                    result: EvaluationResult, 
                    feedback: Feedback) -> ProblemState:
        """
        Update the problem state based on the solution and feedback.
        
        This method implements the recursive nature of the benchmark by
        evolving the problem based on the model's solution attempt.
        
        Args:
            solution: The attempted solution
            result: The evaluation result
            feedback: The feedback provided
            
        Returns:
            The updated problem state
        """
        # Add the current step to the trajectory
        self.trajectory.add_step(
            problem_state=self.state,
            solution=solution,
            result=result,
            feedback=feedback
        )
        
        # Check if we've converged
        if self.convergence_criteria.has_converged(self.trajectory):
            if self.trajectory.steps[-1].result.score >= self.convergence_criteria.score_threshold:
                self.status = TaskStatus.PERFECT_SOLUTION
            elif len(self.trajectory.steps) >= self.convergence_criteria.max_iterations:
                self.status = TaskStatus.MAX_ITERATIONS
            else:
                self.status = TaskStatus.CONVERGED
            return self.state
        
        # Evolve the problem state based on the solution
        self.state = self._evolve_state(solution, result, feedback)
        
        # Update the status
        self.status = TaskStatus.IN_PROGRESS
        
        return self.state
    
    def _run_evaluation(self, solution: str) -> EvaluationResult:
        """
        Run evaluation logic specific to this task.
        
        Args:
            solution: The solution to evaluate
            
        Returns:
            The evaluation result
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _generate_feedback(self, 
                         solution: str, 
                         result: EvaluationResult) -> Feedback:
        """
        Generate structured feedback based on evaluation results.
        
        Args:
            solution: The solution that was evaluated
            result: The evaluation result
            
        Returns:
            Structured feedback
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _evolve_state(self, 
                    solution: str, 
                    result: EvaluationResult, 
                    feedback: Feedback) -> ProblemState:
        """
        Evolve the problem state based on the solution and feedback.
        
        This method implements the recursive nature of the benchmark by
        defining how the problem changes in response to solution attempts.
        
        Args:
            solution: The attempted solution
            result: The evaluation result
            feedback: The feedback provided
            
        Returns:
            The evolved problem state
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_trajectory(self) -> Trajectory:
        """
        Get the complete solution trajectory for this task.
        
        Returns:
            The solution trajectory
        """
        return self.trajectory
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary for serialization.
        
        Returns:
            A dictionary representation of the task
        """
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "state": {
                "problem_id": self.state.problem_id,
                "description": self.state.description,
                "code_context": self.state.code_context,
                "requirements": self.state.requirements,
                "difficulty": self.state.difficulty,
                "evolution_stage": self.state.evolution_stage,
                "adaptation_vector": self.state.adaptation_vector
            },
            "config": self.config,
            "trajectory": self.trajectory.to_dict()
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the task to a file.
        
        Args:
            filepath: Path to save the task
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "RecursiveTask":
        """
        Load a task from a file.
        
        Args:
            filepath: Path to load the task from
            
        Returns:
            The loaded task
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # This method needs to be implemented by subclasses
        # as they need to implement the abstract methods
        raise NotImplementedError("Subclasses must implement this method")
