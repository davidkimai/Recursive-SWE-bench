# recursive_swe_bench/evaluation/harness.py

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import datetime
import uuid
import json
import os
import logging
from dataclasses import dataclass, field

from recursive_swe_bench.core.recursive_task import (
    RecursiveTask, Trajectory, TrajectoryStep, ProblemState, 
    EvaluationResult, Feedback, TaskStatus
)

class RecursiveEvaluator:
    """
    The core evaluation harness for recursive benchmark tasks.
    
    This class orchestrates the recursive evaluation process, managing the interactions
    between models and tasks, tracking trajectories, and calculating metrics.
    """
    
    def __init__(
        self,
        model: Any,  # Model interface
        metrics: Dict[str, Any],  # Metric calculators
        config: Dict[str, Any] = None
    ):
        """
        Initialize the recursive evaluator.
        
        Args:
            model: The model to evaluate
            metrics: Dictionary of metric calculators
            config: Configuration options
        """
        self.model = model
        self.metrics = metrics
        self.config = config or {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the evaluator."""
        logger = logging.getLogger("RecursiveEvaluator")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.config.get("log_level", logging.INFO))
        return logger
    
    def evaluate_task(
        self, 
        task: RecursiveTask,
        max_iterations: int = 5
    ) -> Tuple[Trajectory, Dict[str, float]]:
        """
        Run a full recursive evaluation on a single task.
        
        Args:
            task: The task to evaluate
            max_iterations: Maximum number of iterations
            
        Returns:
            The trajectory and calculated metrics
        """
        self.logger.info(f"Starting evaluation of task {task.task_id}")
        
        for i in range(max_iterations):
            self.logger.info(f"Starting iteration {i+1}/{max_iterations}")
            
            # Get the current problem
            problem = task.get_current_problem()
            self.logger.debug(f"Problem state: evolution_stage={problem['evolution_stage']}")
            
            # Format the problem for the model
            formatted_problem = self._format_problem_for_model(problem, task.trajectory)
            
            # Get model solution
            self.logger.debug("Requesting solution from model")
            solution = self.model.solve(formatted_problem)
            
            # Evaluate the solution
            self.logger.debug("Evaluating solution")
            result, feedback = task.evaluate_solution(solution)
            
            # Log the results
            self.logger.info(f"Solution score: {result.score:.4f}, Success: {result.success}")
            
            # Update the task state based on the solution
            new_state = task.update_state(solution, result, feedback)
            
            # Check if we've reached a terminal state
            if task.status != TaskStatus.IN_PROGRESS:
                self.logger.info(f"Task complete with status: {task.status.value}")
                break
                
        # Calculate metrics across the trajectory
        self.logger.info("Calculating metrics")
        metrics_result = self._calculate_metrics(task.trajectory)
        
        return task.trajectory, metrics_result
    
    def evaluate_task_set(
        self,
        tasks: List[RecursiveTask],
        max_iterations: int = 5,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a set of tasks and aggregate the results.
        
        Args:
            tasks: List of tasks to evaluate
            max_iterations: Maximum iterations per task
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary of aggregated results
        """
        self.logger.info(f"Evaluating {len(tasks)} tasks")
        
        results = {}
        trajectories = {}
        all_metrics = {}
        
        for i, task in enumerate(tasks):
            self.logger.info(f"Evaluating task {i+1}/{len(tasks)}: {task.task_id}")
            
            # Evaluate the task
            trajectory, metrics = self.evaluate_task(task, max_iterations)
            
            # Store the results
            trajectories[task.task_id] = trajectory
            all_metrics[task.task_id] = metrics
            
            # Save the trajectory if output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                task_output_path = os.path.join(output_dir, f"task_{task.task_id}.json")
                task.save(task_output_path)
                self.logger.info(f"Saved task to {task_output_path}")
        
        # Aggregate metrics across all tasks
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Compile results
        results = {
            "aggregated_metrics": aggregated_metrics,
            "task_metrics": all_metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_info": self.model.get_meta_information(),
            "total_tasks": len(tasks),
            "config": self.config
        }
        
        # Save aggregated results if output_dir is provided
        if output_dir:
            results_path = os.path.join(output_dir, "aggregated_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved aggregated results to {results_path}")
        
        return results
    
    def _format_problem_for_model(
        self, 
        problem: Dict[str, Any],
        trajectory: Trajectory
    ) -> Dict[str, Any]:
        """
        Format the problem in a way the model can understand.
        
        Args:
            problem: The problem state
            trajectory: The trajectory so far
            
        Returns:
            Formatted problem for the model
        """
        # Extract the previous steps if they exist
        previous_steps = []
        for step in trajectory.steps:
            previous_steps.append({
                "problem": {
                    "description": step.problem_state.description,
                    "requirements": step.problem_state.requirements,
                    "evolution_stage": step.problem_state.evolution_stage
                },
                "solution": step.solution,
                "feedback": {
                    "summary": step.feedback.summary,
                    "issues": step.feedback.issues,
                    "suggestions": step.feedback.suggestions,
                    "focus_areas": step.feedback.focus_areas
                }
            })
        
        # Format the problem with the trajectory context
        formatted_problem = {
            "description": problem["description"],
            "code_context": problem["code_context"],
            "requirements": problem["requirements"],
            "iteration": problem["evolution_stage"] + 1,
            "previous_attempts": previous_steps
        }
        
        return formatted_problem
    
    def _calculate_metrics(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        Calculate metrics across the trajectory.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            Dictionary of metric values
        """
        return {name: metric.calculate(trajectory) 
                for name, metric in self.metrics.items()}
    
    def _aggregate_metrics(
        self, 
        all_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple tasks.
        
        Args:
            all_metrics: Dictionary of metrics per task
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Initialize aggregated metrics
        if not all_metrics:
            return {}
            
        sample_metrics = next(iter(all_metrics.values()))
        aggregated = {name: 0.0 for name in sample_metrics.keys()}
        
        # Sum up metrics
        for task_metrics in all_metrics.values():
            for name, value in task_metrics.items():
                aggregated[name] += value
        
        # Calculate averages
        for name in aggregated:
            aggregated[name] /= len(all_metrics)
            
        return aggregated


# recursive_swe_bench/evaluation/metrics/recursive.py

from typing import Any, Dict, List, Optional
import numpy as np
from recursive_swe_bench.core.recursive_task import Trajectory


class RecursiveMetric:
    """Base class for recursive metrics."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the metric value for a trajectory.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The metric value
        """
        raise NotImplementedError("Subclasses must implement this method")


class ConvergenceRate(RecursiveMetric):
    """
    Measures how quickly the model reaches a stable solution.
    
    A lower value indicates faster convergence.
    """
    
    def calculate(self, trajectory: Trajectory) -> float:
        scores = trajectory.get_score_series()
        if len(scores) < 2:
            return 0.0
            
        # Calculate changes between consecutive scores
        deltas = [abs(scores[i+1] - scores[i]) 
                 for i in range(len(scores)-1)]
        
        # A lower sum indicates faster convergence
        # Normalize by the number of iterations
        return sum(deltas) / len(deltas)


class AdaptationEfficiency(RecursiveMetric):
    """
    Measures improvement per feedback iteration.
    
    A higher value indicates more efficient adaptation.
    """
    
    def calculate(self, trajectory: Trajectory) -> float:
        scores = trajectory.get_score_series()
        if len(scores) < 2:
            return 0.0
            
        # Calculate the improvement from first to last iteration
        total_improvement = max(0.0, scores[-1] - scores[0])
        
        # Normalize by the number of iterations
        return total_improvement / (len(scores) - 1)


class LearningCurveArea(RecursiveMetric):
    """
    Measures the area under the learning curve.
    
    A higher value indicates better overall performance across iterations.
    """
    
    def calculate(self, trajectory: Trajectory) -> float:
        scores = trajectory.get_score_series()
        if not scores:
            return 0.0
        
        # Calculate the area under the curve
        # Normalize by the maximum possible area (perfect score from the start)
        max_score = self.config.get("max_score", 1.0)
        max_area = max_score * len(scores)
        
        return sum(scores) / max_area


class ProbabilisticSolutionQuality(RecursiveMetric):
    """
    Measures the distribution of solution quality using non-deterministic assessment.
    
    This metric captures the robustness of solutions by measuring the variability in quality
    across multiple probabilistic evaluations.
    """
    
    def calculate(self, trajectory: Trajectory) -> float:
        # For each step, we expect the result.metrics to contain probabilistic assessments
        steps = trajectory.steps
        if not steps:
            return 0.0
        
        # Extract probabilistic quality distributions if available
        distributions = []
        for step in steps:
            if (step.result.metrics and 
                "probabilistic_quality_distribution" in step.result.metrics):
                distributions.append(
                    step.result.metrics["probabilistic_quality_distribution"])
        
        if not distributions:
            # Fall back to deterministic scores if no distributions are available
            return trajectory.get_score_series()[-1]
        
        # Calculate the expected value of the final distribution
        final_distribution = distributions[-1]
        return sum(prob * val for val, prob in final_distribution.items())


class TransferLearningFactor(RecursiveMetric):
    """
    Measures how well learning transfers across related problems.
    
    This requires multiple trajectories from related tasks.
    """
    
    def __init__(self, config: Dict[str, Any] = None, related_trajectories: List[Trajectory] = None):
        super().__init__(config)
        self.related_trajectories = related_trajectories or []
    
    def calculate(self, trajectory: Trajectory) -> float:
        # This metric requires related trajectories
        if not self.related_trajectories:
            return 0.0
        
        # Get learning rates for the current trajectory and related ones
        current_learning_rate = self._calculate_learning_rate(trajectory)
        
        related_learning_rates = [
            self._calculate_learning_rate(rel_traj)
            for rel_traj in self.related_trajectories
        ]
        
        # Filter out invalid learning rates
        valid_related_rates = [rate for rate in related_learning_rates if rate is not None]
        
        if not valid_related_rates:
            return 0.0
        
        # Calculate the transfer factor as the ratio of the current learning rate
        # to the average of related learning rates
        avg_related_rate = sum(valid_related_rates) / len(valid_related_rates)
        
        if avg_related_rate == 0:
            return 0.0
            
        return current_learning_rate / avg_related_rate
    
    def _calculate_learning_rate(self, trajectory: Trajectory) -> Optional[float]:
        """Calculate the learning rate for a trajectory."""
        scores = trajectory.get_score_series()
        if len(scores) < 2:
            return None
            
        # Calculate improvement per iteration
        return (scores[-1] - scores[0]) / (len(scores) - 1)


class DynamicComplexityHandling(RecursiveMetric):
    """
    Measures how well the model handles varying problem complexity.
    
    This metric evaluates performance while accounting for changes in problem difficulty.
    """
    
    def calculate(self, trajectory: Trajectory) -> float:
        if not trajectory.steps:
            return 0.0
        
        # Extract scores and difficulties
        scores = trajectory.get_score_series()
        difficulties = [step.problem_state.difficulty for step in trajectory.steps]
        
        if len(scores) < 2:
            return scores[0]  # Return the single score if only one step
        
        # Calculate normalized scores (adjusted by difficulty)
        normalized_scores = [scores[i] * (1 + difficulties[i]) 
                           for i in range(len(scores))]
        
        # Return the average normalized score
        return sum(normalized_scores) / len(normalized_scores)
