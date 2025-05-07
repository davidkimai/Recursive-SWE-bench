# recursive_swe_bench/models/anthropic.py

import json
import backoff
import time
import anthropic
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import logging

from recursive_swe_bench.models.base_model import ModelInterface

class AnthropicModel(ModelInterface):
    """
    Integration with Anthropic models (Claude).
    
    This class provides integration with Anthropic's API for evaluating
    Claude models with Recursive-SWE-bench through recursive evaluation loops.
    The implementation features dynamic adaptation to feedback through a 
    self-reflective mechanism that traces attribution paths through recursive iterations.
    """
    
    def __init__(
        self, 
        model_identifier: str, 
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Anthropic model interface.
        
        Args:
            model_identifier: Anthropic model identifier (e.g., "claude-3-opus-20240229")
            api_key: Anthropic API key (optional if set in environment)
            config: Additional configuration options
        """
        super().__init__(model_identifier, config)
        
        # Initialize Anthropic client
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            self.client = anthropic.Anthropic()
        
        # Set up system prompt and templates
        self.prompts = self.config.get("prompts", {
            "system": "You are an expert software engineer who specializes in debugging and fixing complex code. Your task is to fix bugs in code based on the description and test requirements provided.",
            "user_template": "# Bug Fixing Task\n\n{description}\n\n# Code\n```python\n{code}\n```\n\n{tests_description}\n\n# Your task\nFix the bugs in the code above. Focus on making the code pass all tests while maintaining good practices. Provide only the corrected code without additional explanations.",
            "reflection_template": "# Feedback on Previous Solution\n\nYour previous solution had the following issues:\n{issues}\n\n# Suggested Improvements\n{suggestions}\n\n# Test Results\n{test_results}\n\n# Reflection Prompt\nBefore providing a new solution, analyze what went wrong in your previous attempt and how you'll approach fixing it differently this time."
        })
        
        # Configure API parameters
        self.api_params = self.config.get("api_params", {
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 0.95,
            "top_k": 50
        })
        
        # Set up recursive adaptation configuration
        self.recursive_config = self.config.get("recursive_config", {
            "enable_self_reflection": True,
            "adaptation_threshold": 0.5,  # Minimum score to trigger adaptation
            "max_reflection_depth": 3,    # Maximum depth of recursive reflection
            "attribution_tracking": True,  # Track attribution patterns across iterations
            "dynamic_prompting": True,    # Adjust prompts based on failure patterns
        })
        
        # Initialize recursive state
        self.recursive_state = {
            "reflection_depth": 0,
            "adaptation_vector": [0.0] * 5,  # Tracks adaptation across dimensions
            "attribution_map": {},           # Maps error types to attribution patterns
            "error_frequency": {},           # Tracks frequency of error types
            "solution_quality_trend": [],    # Tracks solution quality over iterations
        }
        
        self.logger.info(f"Initialized Anthropic model: {model_identifier} with recursive capability")
    
    @backoff.on_exception(
        backoff.expo,
        (anthropic.APIError, anthropic.APITimeoutError, anthropic.RateLimitError),
        max_tries=5
    )
    def solve(
        self, 
        problem: Dict[str, Any], 
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a solution using the Anthropic model with recursive adaptation.
        
        Args:
            problem: The problem to solve
            history: Optional history of previous solution attempts
            
        Returns:
            The generated solution
        """
        self.logger.info(f"Solving problem with Anthropic model: {self.model_identifier}")
        start_time = time.time()
        
        # Reset recursive state for new problems if no history
        if not history:
            self._reset_recursive_state()
        elif history:
            # Update recursive state based on history
            self._update_recursive_state(history)
        
        # Format messages for the model
        system_prompt, user_message = self._format_messages(problem, history)
        
        # Make API call
        response = self.client.messages.create(
            model=self.model_identifier,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ],
            max_tokens=self.api_params.get("max_tokens", 2000),
            temperature=self.api_params.get("temperature", 0.2),
            top_p=self.api_params.get("top_p", 0.95),
            top_k=self.api_params.get("top_k", 50)
        )
        
        # Extract the solution from the response
        solution = response.content[0].text
        
        end_time = time.time()
        self.logger.info(f"Solution generated in {end_time - start_time:.2f} seconds")
        
        # Track solution in recursive state
        if solution:
            self.recursive_state["reflection_depth"] += 1
            
        return self._extract_code(solution)
    
    def _format_messages(
        self, 
        problem: Dict[str, Any], 
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, str]:
        """
        Format the problem and history into messages for the Anthropic API.
        
        Args:
            problem: The problem to solve
            history: Optional history of previous solution attempts
            
        Returns:
            Tuple of (system_prompt, user_message)
        """
        # Start with base system prompt
        system_prompt = self.prompts["system"]
        
        # Enhance system prompt with recursive adaptation if enabled
        if self.recursive_config.get("enable_self_reflection", True) and history:
            # Add adaptation guidance based on error patterns
            if self.recursive_state["error_frequency"]:
                top_errors = sorted(
                    self.recursive_state["error_frequency"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                error_guidance = "Focus particularly on addressing these recurring issues:\n"
                for error_type, count in top_errors:
                    error_guidance += f"- {error_type} (appeared {count} times)\n"
                
                system_prompt += f"\n\n{error_guidance}"
            
            # Add reflection guidance based on solution quality trend
            if len(self.recursive_state["solution_quality_trend"]) > 1:
                trend = self.recursive_state["solution_quality_trend"]
                if trend[-1] > trend[-2]:
                    system_prompt += "\n\nYour solutions are improving. Continue this trajectory."
                elif trend[-1] < trend[-2]:
                    system_prompt += "\n\nYour solutions are declining in quality. Carefully reconsider your approach."
                else:
                    system_prompt += "\n\nYour solutions maintain the same quality. Try a different approach."
        
        # Format code and tests
        code = problem["code_context"]["code"]
        
        # Prepare tests description
        tests_description = "# Tests\n"
        if "tests" in problem["code_context"]:
            tests_description += "The code must pass the following tests:\n\n"
            for i, test in enumerate(problem["code_context"]["tests"]):
                tests_description += f"## Test {i+1}: {test['name']}\n```python\n{test['content']}\n```\n\n"
        else:
            tests_description += "The code must work correctly according to its intended functionality."
        
        # Base user message
        user_message = self.prompts["user_template"].format(
            description=problem["description"],
            code=code,
            tests_description=tests_description
        )
        
        # Add history if available - with recursive reflection
        if history and self.recursive_config.get("enable_self_reflection", True):
            # Get the most recent entry for reflection
            latest_entry = history[-1]
            
            # Format issues
            issues_text = "- " + "\n- ".join([issue["message"] for issue in latest_entry["feedback"]["issues"]])
            
            # Format suggestions
            suggestions_text = "- " + "\n- ".join([suggestion["message"] for suggestion in latest_entry["feedback"]["suggestions"]])
            
            # Format test results
            test_results = latest_entry.get("result", {})
            passed_tests = test_results.get("passed_tests", 0)
            total_tests = test_results.get("total_tests", 0)
            
            test_results_text = f"Passed {passed_tests}/{total_tests} tests."
            if "tests" in test_results:
                test_results_text += "\n\nIndividual test results:"
                for test_name, test_result in test_results["tests"].items():
                    status = "✅ Passed" if test_result.get("passed", False) else "❌ Failed"
                    test_results_text += f"\n- {test_name}: {status}"
                    if not test_result.get("passed", False) and "message" in test_result:
                        test_results_text += f"\n  Error: {test_result['message']}"
            
            # Add reflection prompt
            reflection_prompt = self.prompts["reflection_template"].format(
                issues=issues_text,
                suggestions=suggestions_text,
                test_results=test_results_text
            )
            
            # Prepend reflection to user message
            user_message = f"{reflection_prompt}\n\n{user_message}"
            
            # Add dynamic adaptation based on error patterns if enabled
            if self.recursive_config.get("dynamic_prompting", True):
                # Look for specific error patterns and add targeted guidance
                error_types = [issue.get("type", "") for issue in latest_entry["feedback"]["issues"]]
                
                if "syntax" in " ".join(error_types).lower():
                    user_message += "\n\nPay careful attention to syntax correctness. Double-check all parentheses, indentation, and function definitions."
                
                if "test_failure" in " ".join(error_types).lower():
                    user_message += "\n\nFocus on making the code pass the failing tests. Carefully trace through the code execution for each test case."
                
                if "edge_case" in " ".join(error_types).lower() or "boundary" in " ".join(error_types).lower():
                    user_message += "\n\nBe sure to handle edge cases such as empty inputs, boundary values, and special cases."
                
                if "performance" in " ".join(error_types).lower():
                    user_message += "\n\nOptimize your solution for better performance. Avoid unnecessary operations and inefficient data structures."
        
        return system_prompt, user_message
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code from the model's response.
        
        Args:
            text: The model's response
            
        Returns:
            Extracted code
        """
        # Try to extract code from markdown code blocks
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the full text (it might be just code)
        return text.strip()
    
    def _reset_recursive_state(self):
        """Reset the recursive state for a new problem."""
        self.recursive_state = {
            "reflection_depth": 0,
            "adaptation_vector": [0.0] * 5,
            "attribution_map": {},
            "error_frequency": {},
            "solution_quality_trend": [],
        }
    
    def _update_recursive_state(self, history: List[Dict[str, Any]]):
        """
        Update recursive state based on solution history.
        
        Args:
            history: History of previous solution attempts
        """
        # Extract scores from history
        scores = [entry.get("result", {}).get("score", 0.0) for entry in history]
        self.recursive_state["solution_quality_trend"] = scores
        
        # Calculate adaptation vector
        if len(scores) >= 2:
            # Dimension 0: Overall improvement trajectory
            improvement = scores[-1] - scores[0]
            self.recursive_state["adaptation_vector"][0] = max(-1.0, min(1.0, improvement))
            
            # Dimension 1: Recent improvement
            recent_improvement = scores[-1] - scores[-2]
            self.recursive_state["adaptation_vector"][1] = max(-1.0, min(1.0, recent_improvement))
        
        # Update error frequency from latest feedback
        if history:
            latest_feedback = history[-1].get("feedback", {})
            issues = latest_feedback.get("issues", [])
            
            for issue in issues:
                issue_type = issue.get("type", "unknown")
                self.recursive_state["error_frequency"][issue_type] = self.recursive_state["error_frequency"].get(issue_type, 0) + 1
        
        # Update reflection depth
        self.recursive_state["reflection_depth"] = len(history)
    
    def get_meta_information(self) -> Dict[str, Any]:
        """
        Get meta information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_identifier,
            "provider": "Anthropic",
            "type": "API",
            "parameters": self.api_params,
            "system_prompt": self.prompts["system"],
            "recursive_capability": self.recursive_config.get("enable_self_reflection", True),
            "reflection_depth": self.recursive_state["reflection_depth"],
            "adaptation_vector": self.recursive_state["adaptation_vector"]
        }


# recursive_swe_bench/evaluation/recursive_metrics.py

import numpy as np
import scipy.stats
from typing import Any, Dict, List, Optional, Union
import dataclasses
import math

from recursive_swe_bench.core.recursive_task import Trajectory


class RecursiveLearningCurveArea:
    """
    Measures the area under the learning curve across iterations.
    
    This metric captures the overall performance of a model throughout its
    learning trajectory, rewarding both high scores and quick improvement.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the recursive learning curve area metric.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.max_score = self.config.get("max_score", 1.0)
        self.normalize = self.config.get("normalize", True)
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the area under the learning curve.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The normalized area under the learning curve
        """
        scores = trajectory.get_score_series()
        if not scores:
            return 0.0
        
        # Calculate the area under the curve using trapezoidal rule
        area = np.trapz(scores, dx=1.0)
        
        # Normalize by the maximum possible area if requested
        if self.normalize:
            max_area = self.max_score * len(scores)
            return area / max_area
        
        return area


class AdaptationRate:
    """
    Measures the rate at which the model improves its solutions.
    
    This metric captures how quickly a model adapts to feedback and
    improves its solutions across iterations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptation rate metric.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.min_iterations = self.config.get("min_iterations", 2)
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the adaptation rate.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The adaptation rate
        """
        scores = trajectory.get_score_series()
        if len(scores) < self.min_iterations:
            return 0.0
        
        # Calculate the average improvement per iteration
        total_improvement = scores[-1] - scores[0]
        iterations = len(scores) - 1
        
        return total_improvement / iterations


class RecursiveVolatility:
    """
    Measures the volatility of solution quality across iterations.
    
    This metric captures how stable or erratic a model's performance
    is across iterations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the recursive volatility metric.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.min_iterations = self.config.get("min_iterations", 3)
        self.normalize = self.config.get("normalize", True)
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the recursive volatility.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The normalized volatility
        """
        scores = trajectory.get_score_series()
        if len(scores) < self.min_iterations:
            return 0.0
        
        # Calculate the standard deviation of score changes
        changes = [abs(scores[i] - scores[i-1]) for i in range(1, len(scores))]
        volatility = np.std(changes)
        
        # Normalize by the average score if requested
        if self.normalize and np.mean(scores) > 0:
            return volatility / np.mean(scores)
        
        return volatility


class ConvergenceIndex:
    """
    Measures how quickly the model converges to a stable solution.
    
    This metric captures how efficiently a model reaches a stable solution
    across iterations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the convergence index metric.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.stability_threshold = self.config.get("stability_threshold", 0.05)
        self.max_score_threshold = self.config.get("max_score_threshold", 0.95)
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the convergence index.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The convergence index (lower is better)
        """
        scores = trajectory.get_score_series()
        if not scores:
            return 0.0
        
        # Find the first iteration where the score stabilizes
        # (subsequent changes are below the stability threshold)
        convergence_point = len(scores) - 1
        for i in range(1, len(scores)):
            remaining_changes = [abs(scores[j] - scores[j-1]) for j in range(i, len(scores))]
            if all(change <= self.stability_threshold for change in remaining_changes):
                convergence_point = i
                break
        
        # Find the first iteration where the score exceeds the max score threshold
        max_score_point = len(scores)
        for i, score in enumerate(scores):
            if score >= self.max_score_threshold:
                max_score_point = i
                break
        
        # Return a combined index
        # Lower is better - converging quickly to a high score is ideal
        return (convergence_point / len(scores)) * (1.0 - max(0.0, min(1.0, scores[-1])))


class ErrorRecoveryEfficiency:
    """
    Measures how efficiently the model recovers from errors.
    
    This metric captures how well a model addresses and fixes specific
    errors across iterations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the error recovery efficiency metric.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the error recovery efficiency.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The error recovery efficiency
        """
        if not trajectory.steps or len(trajectory.steps) < 2:
            return 0.0
        
        # Extract error counts from each step
        error_counts = []
        for step in trajectory.steps:
            if hasattr(step, "result") and hasattr(step.result, "error_details"):
                error_counts.append(len(step.result.error_details or {}))
            else:
                # If no error details available, use issue count from feedback
                error_counts.append(len(step.feedback.issues))
        
        if not error_counts or error_counts[0] == 0:
            return 1.0  # Perfect if no initial errors
        
        # Calculate the rate at which errors are fixed
        initial_errors = error_counts[0]
        final_errors = error_counts[-1]
        
        # Return the proportion of errors fixed
        return (initial_errors - final_errors) / initial_errors


class DynamicComplexityHandling:
    """
    Measures how well the model handles varying problem complexity.
    
    This metric evaluates performance while accounting for changes in
    problem difficulty across iterations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dynamic complexity handling metric.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
    
    def calculate(self, trajectory: Trajectory) -> float:
        """
        Calculate the dynamic complexity handling score.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            The dynamic complexity handling score
        """
        if not trajectory.steps:
            return 0.0
        
        # Extract scores and difficulties from each step
        scores = []
        difficulties = []
        
        for step in trajectory.steps:
            scores.append(step.result.score)
            difficulties.append(step.problem_state.difficulty)
        
        # Calculate difficulty-weighted scores
        weighted_scores = [scores[i] / max(0.1, difficulties[i]) for i in range(len(scores))]
        
        # Return the average weighted score
        return sum(weighted_scores) / len(weighted_scores)


class RecursiveFrameworkMetrics:
    """
    Comprehensive collection of metrics for recursive evaluation.
    
    This class provides easy access to all recursive metrics and
    standardized calculation across trajectories.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the recursive framework metrics.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        
        # Initialize all metrics
        self.metrics = {
            "learning_curve_area": RecursiveLearningCurveArea(self.config.get("learning_curve_area")),
            "adaptation_rate": AdaptationRate(self.config.get("adaptation_rate")),
            "volatility": RecursiveVolatility(self.config.get("volatility")),
            "convergence_index": ConvergenceIndex(self.config.get("convergence_index")),
            "error_recovery": ErrorRecoveryEfficiency(self.config.get("error_recovery")),
            "complexity_handling": DynamicComplexityHandling(self.config.get("complexity_handling"))
        }
        
        # Add custom metrics from config if provided
        if "custom_metrics" in self.config:
            for name, metric in self.config["custom_metrics"].items():
                self.metrics[name] = metric
    
    def calculate_all(self, trajectory: Trajectory) -> Dict[str, float]:
        """
        Calculate all metrics for a trajectory.
        
        Args:
            trajectory: The solution trajectory
            
        Returns:
            Dictionary of metric names and values
        """
        return {name: metric.calculate(trajectory) 
                for name, metric in self.metrics.items()}
    
    def calculate(self, trajectory: Trajectory, metric_name: str) -> float:
        """
        Calculate a specific metric for a trajectory.
        
        Args:
            trajectory: The solution trajectory
            metric_name: The name of the metric to calculate
            
        Returns:
            The calculated metric value
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        return self.metrics[metric_name].calculate(trajectory)
    
    def aggregate_metrics(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """
        Calculate aggregate metrics across multiple trajectories.
        
        Args:
            trajectories: List of solution trajectories
            
        Returns:
            Dictionary of aggregated metric values
        """
        if not trajectories:
            return {}
        
        all_metrics = [self.calculate_all(trajectory) for trajectory in trajectories]
        
        # Aggregate by averaging each metric
        aggregated = {}
        for metric_name in self.metrics:
            values = [metrics[metric_name] for metrics in all_metrics]
            aggregated[metric_name] = sum(values) / len(values)
        
        return aggregated


# recursive_swe_bench/evaluation/visualizer.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
import os
import json
import seaborn as sns
from pathlib import Path

from recursive_swe_bench.core.recursive_task import Trajectory


class RecursiveVisualizer:
    """
    Visualization tools for recursive evaluation results.
    
    This class provides methods for visualizing recursive trajectories,
    metrics, and comparative analysis across models.
    """
    
    def __init__(self, output_dir: Optional[str] = None, config: Dict[str, Any] = None):
        """
        Initialize the recursive visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            config: Configuration options
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        self.config = config or {}
        self.theme = self.config.get("theme", "default")
        
        # Set up the visualization style
        if self.theme == "dark":
            plt.style.use("dark_background")
            self.colors = sns.color_palette("viridis", 10)
        else:
            plt.style.use("seaborn-v0_8-whitegrid")
            self.colors = sns.color_palette("muted", 10)
        
        sns.set_context("talk")
    
    def plot_trajectory(
        self,
        trajectory: Trajectory,
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot a solution trajectory showing score evolution.
        
        Args:
            trajectory: The solution trajectory
            title: Optional title for the plot
            show: Whether to display the plot
            save_path: Optional path to save the plot
        """
        scores = trajectory.get_score_series()
        if not scores:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot scores
        plt.plot(range(1, len(scores) + 1), scores, marker='o', 
                 linewidth=2, markersize=8, color=self.colors[0])
        
        # Add difficulty if available
        difficulties = [step.problem_state.difficulty for step in trajectory.steps]
        if difficulties:
            plt.plot(range(1, len(difficulties) + 1), difficulties, marker='s',
                     linewidth=2, markersize=8, color=self.colors[1], linestyle='--',
                     label='Problem Difficulty')
        
        # Set plot properties
        plt.title(title or f"Solution Trajectory for Task {trajectory.task_id}")
        plt.xlabel("Iteration")
        plt.ylabel("Score / Difficulty")
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.xticks(range(1, len(scores) + 1))
        plt.legend(["Solution Score", "Problem Difficulty"])
        
        # Save if requested
        if save_path:
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_metrics_comparison(
        self,
        metrics_by_model: Dict[str, Dict[str, float]],
        title: Optional[str] = None,
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot a comparison of metrics across models.
        
        Args:
            metrics_by_model: Dictionary mapping model names to metric values
            title: Optional title for the plot
            show: Whether to display the plot
            save_path: Optional path to save the plot
        """
        if not metrics_by_model:
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(metrics_by_model).T
        
        # Create a radar chart
        categories = list(df.columns)
        N = len(categories)
        
        # Create angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Add lines for each model
        for i, (model, metrics) in enumerate(df.iterrows()):
            values = metrics.values.flatten().tolist()
            values += values[:1]  # Close the loop
            
            # Plot the line
            ax.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=model, color=self.colors[i % len(self.colors)])
            ax.fill(angles, values, alpha=0.1, color=self.colors[i % len(self.colors)])
        
        # Set category labels
        plt.xticks(angles[:-1], categories)
        
        # Set y-axis limits
        plt.ylim(0, 1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        plt.title(title or "Metrics Comparison Across Models")
        
        # Save if requested
        if save_path:
            full_path = os.path.join(self.output_dir, save_path) if self.output_dir else save_path
            plt.savefig(full_path, bbox_inches='tight',
