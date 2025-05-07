# recursive_swe_bench/task_generators/bug_fixing.py

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import uuid
import json
import re
import random
import ast
import copy
from pathlib import Path
import tempfile
import subprocess
import shutil
import os

from recursive_swe_bench.core.recursive_task import (
    RecursiveTask, ProblemState, EvaluationResult, Feedback, TaskStatus
)

class BugCategory:
    """Categories of bugs for classification and evolution."""
    SYNTAX = "syntax"
    LOGICAL = "logical"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONCURRENCY = "concurrency"
    EXCEPTION_HANDLING = "exception_handling"
    API_USAGE = "api_usage"
    MEMORY_MANAGEMENT = "memory_management"
    TYPE_ERROR = "type_error"
    EDGE_CASE = "edge_case"
    DATA_HANDLING = "data_handling"
    DEPENDENCY = "dependency"


class BugFixingTask(RecursiveTask):
    """
    A recursive task for evaluating how models fix bugs in code.
    
    The task presents a piece of code with one or more bugs, and evolves
    based on the model's fix attempts. As the model addresses issues,
    the task may introduce more subtle bugs, change requirements, or
    increase complexity to test adaptive problem-solving.
    """
    
    def __init__(
        self,
        initial_state: ProblemState,
        config: Dict[str, Any] = None,
        test_runner: Any = None
    ):
        """
        Initialize the bug fixing task.
        
        Args:
            initial_state: The initial problem state
            config: Configuration options
            test_runner: Custom test runner (optional)
        """
        super().__init__(initial_state, config)
        self.test_runner = test_runner or DefaultTestRunner()
        self.bug_categories: Set[str] = set(
            self.config.get("bug_categories", [BugCategory.LOGICAL, BugCategory.SYNTAX])
        )
        self.difficulty_progression = self.config.get(
            "difficulty_progression", [0.0, 0.15, 0.3, 0.5, 0.7]
        )
        self.evolution_strategies = self.config.get(
            "evolution_strategies", ["add_subtle_bug", "change_requirements", "increase_complexity"]
        )
        
    def _run_evaluation(self, solution: str) -> EvaluationResult:
        """
        Run tests to evaluate the solution.
        
        Args:
            solution: The solution code
            
        Returns:
            Evaluation results
        """
        # Create a temporary directory to run tests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write solution code to file
            solution_file = temp_path / "solution.py"
            with open(solution_file, "w") as f:
                f.write(solution)
            
            # Create test files
            test_files = self._create_test_files(temp_path)
            
            # Run tests
            results = self.test_runner.run_tests(
                solution_file=solution_file,
                test_files=test_files,
                code_context=self.state.code_context
            )
            
            # Calculate score based on test results
            score = self._calculate_score(results)
            
            return EvaluationResult(
                success=results["all_passed"],
                score=score,
                execution_results=results["execution"],
                error_details=results.get("errors"),
                test_results=results["tests"],
                metrics={
                    "passed_tests": results["passed_tests"],
                    "total_tests": results["total_tests"],
                    "execution_time": results["execution_time"],
                    "memory_usage": results.get("memory_usage", 0),
                    "code_complexity": self._calculate_complexity(solution)
                }
            )
    
    def _generate_feedback(self, solution: str, result: EvaluationResult) -> Feedback:
        """
        Generate structured feedback based on evaluation results.
        
        Args:
            solution: The solution code
            result: The evaluation results
            
        Returns:
            Structured feedback
        """
        issues = []
        suggestions = []
        focus_areas = []
        
        # Add issues for failing tests
        if result.test_results:
            for test_name, test_result in result.test_results.items():
                if not test_result["passed"]:
                    issues.append({
                        "type": "test_failure",
                        "test": test_name,
                        "message": test_result.get("message", "Test failed"),
                        "expected": test_result.get("expected"),
                        "actual": test_result.get("actual")
                    })
        
        # Add issues for errors
        if result.error_details:
            for error_type, error_info in result.error_details.items():
                issues.append({
                    "type": "error",
                    "error_type": error_type,
                    "message": error_info.get("message", "An error occurred"),
                    "location": error_info.get("location")
                })
        
        # Generate suggestions based on issues
        for issue in issues:
            if issue["type"] == "test_failure":
                suggestion = self._generate_suggestion_for_test_failure(
                    issue, solution, result.test_results
                )
                if suggestion:
                    suggestions.append(suggestion)
            elif issue["type"] == "error":
                suggestion = self._generate_suggestion_for_error(
                    issue, solution
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        # Determine focus areas based on issues and task state
        focus_areas = self._determine_focus_areas(issues, solution, result)
        
        # Generate adaptation hints based on the current state and results
        adaptation_hints = self._generate_adaptation_hints(solution, result)
        
        # Create summary
        if result.success:
            summary = (
                f"Your solution passes all tests with a score of {result.score:.2f}. "
                f"The code successfully addresses the bugs in the original implementation."
            )
        else:
            passed = result.metrics.get("passed_tests", 0)
            total = result.metrics.get("total_tests", 0)
            summary = (
                f"Your solution passes {passed}/{total} tests with a score of {result.score:.2f}. "
                f"There are still issues that need to be addressed."
            )
        
        return Feedback(
            summary=summary,
            issues=issues,
            suggestions=suggestions,
            focus_areas=focus_areas,
            adaptation_hints=adaptation_hints
        )
    
    def _evolve_state(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Evolve the problem state based on the solution and feedback.
        
        This method implements the recursive nature of the benchmark by
        adapting the problem to challenge the model's understanding.
        
        Args:
            solution: The attempted solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state
        """
        # If the solution perfectly solved the problem, make it more challenging
        if result.success and result.score > 0.95:
            return self._increase_difficulty(solution, result, feedback)
        
        # If the solution was close but not perfect, focus on the remaining issues
        elif result.score > 0.7:
            return self._focus_remaining_issues(solution, result, feedback)
            
        # If the solution was not very good, provide more guidance
        else:
            return self._provide_more_guidance(solution, result, feedback)
    
    def _increase_difficulty(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Increase the difficulty of the problem for models that solved it well.
        
        Args:
            solution: The successful solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state with increased difficulty
        """
        # Create a new state based on the current state
        new_state = copy.deepcopy(self.state)
        
        # Increment evolution stage
        new_state.evolution_stage += 1
        
        # Increase difficulty based on progression schedule
        current_difficulty_idx = min(new_state.evolution_stage, 
                                    len(self.difficulty_progression) - 1)
        new_state.difficulty = self.difficulty_progression[current_difficulty_idx]
        
        # Select an evolution strategy based on the current state
        strategy = self._select_evolution_strategy(solution, result, feedback)
        
        # Apply the selected strategy
        if strategy == "add_subtle_bug":
            self._add_subtle_bug(new_state, solution)
        elif strategy == "change_requirements":
            self._change_requirements(new_state, solution)
        elif strategy == "increase_complexity":
            self._increase_complexity(new_state, solution)
        
        # Update the description to reflect the changes
        new_state.description = self._generate_description(new_state)
        
        # Update adaptation vector to guide future evolution
        new_state.adaptation_vector = self._calculate_adaptation_vector(
            solution, result, feedback
        )
        
        return new_state
    
    def _focus_remaining_issues(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Evolve the state to focus on remaining issues when the solution is close but not perfect.
        
        Args:
            solution: The nearly-successful solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state focusing on remaining issues
        """
        # Create a new state based on the current state
        new_state = copy.deepcopy(self.state)
        
        # Increment evolution stage
        new_state.evolution_stage += 1
        
        # Maintain the same difficulty level
        current_difficulty_idx = min(new_state.evolution_stage - 1, 
                                    len(self.difficulty_progression) - 1)
        new_state.difficulty = self.difficulty_progression[current_difficulty_idx]
        
        # Update the code context to focus on remaining issues
        new_state.code_context["focus_areas"] = feedback.focus_areas
        
        # Highlight failing tests in the code context
        if result.test_results:
            failing_tests = [
                test_name for test_name, test_result in result.test_results.items()
                if not test_result["passed"]
            ]
            new_state.code_context["failing_tests"] = failing_tests
        
        # Update the description to be more specific about remaining issues
        new_state.description = self._generate_focused_description(
            new_state, feedback.issues
        )
        
        # Update adaptation vector to guide future evolution
        new_state.adaptation_vector = self._calculate_adaptation_vector(
            solution, result, feedback
        )
        
        return new_state
    
    def _provide_more_guidance(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Evolve the state to provide more guidance when the solution was not very good.
        
        Args:
            solution: The unsuccessful solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state with more guidance
        """
        # Create a new state based on the current state
        new_state = copy.deepcopy(self.state)
        
        # Increment evolution stage
        new_state.evolution_stage += 1
        
        # Maintain or slightly decrease difficulty
        current_difficulty_idx = max(0, min(new_state.evolution_stage - 1, 
                                          len(self.difficulty_progression) - 1) - 1)
        new_state.difficulty = self.difficulty_progression[current_difficulty_idx]
        
        # Add more hints to the code context
        new_state.code_context["hints"] = self._generate_hints(
            solution, result, feedback
        )
        
        # Add more detailed information about failing tests
        if result.test_results:
            detailed_test_results = {}
            for test_name, test_result in result.test_results.items():
                if not test_result["passed"]:
                    detailed_test_results[test_name] = {
                        "message": test_result.get("message", "Test failed"),
                        "expected": test_result.get("expected"),
                        "actual": test_result.get("actual"),
                        "hint": self._generate_test_hint(test_name, test_result)
                    }
            new_state.code_context["detailed_test_results"] = detailed_test_results
        
        # Update the description to include more guidance
        new_state.description = self._generate_guided_description(
            new_state, feedback.issues, feedback.suggestions
        )
        
        # Update adaptation vector to guide future evolution
        new_state.adaptation_vector = self._calculate_adaptation_vector(
            solution, result, feedback
        )
        
        return new_state
    
    def _select_evolution_strategy(self, solution: str, result: EvaluationResult, feedback: Feedback) -> str:
        """
        Select an evolution strategy based on the current state and solution.
        
        Args:
            solution: The current solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The selected evolution strategy
        """
        available_strategies = self.evolution_strategies.copy()
        
        # Weight the strategies based on the current state
        weights = {}
        
        # Prefer adding subtle bugs if the solution is very good
        if result.score > 0.95:
            weights["add_subtle_bug"] = 0.6
            weights["change_requirements"] = 0.3
            weights["increase_complexity"] = 0.1
        
        # Prefer changing requirements if we've already added several bugs
        elif self.state.evolution_stage >= 2 and "bug_count" in self.state.code_context and self.state.code_context["bug_count"] >= 3:
            weights["add_subtle_bug"] = 0.1
            weights["change_requirements"] = 0.7
            weights["increase_complexity"] = 0.2
            
        # Prefer increasing complexity if the solution is good but not perfect
        elif result.score > 0.85:
            weights["add_subtle_bug"] = 0.2
            weights["change_requirements"] = 0.2
            weights["increase_complexity"] = 0.6
            
        # Default to equal weights
        else:
            weights = {strategy: 1.0 / len(available_strategies) 
                      for strategy in available_strategies}
        
        # Normalize weights for available strategies
        total_weight = sum(weights.get(strategy, 0) for strategy in available_strategies)
        normalized_weights = [weights.get(strategy, 0) / total_weight 
                             for strategy in available_strategies]
        
        # Select a strategy based on weights
        return random.choices(available_strategies, weights=normalized_weights)[0]
    
    def _add_subtle_bug(self, state: ProblemState, solution: str) -> None:
        """
        Add a subtle bug to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Parse the solution to find potential bug insertion points
        try:
            parsed_solution = ast.parse(solution)
        except SyntaxError:
            # If we can't parse the solution, just add a syntax error
            self._add_syntax_error(state, solution)
            return
        
        # Choose a bug category based on available categories
        available_categories = list(self.bug_categories)
        if available_categories:
            bug_category = random.choice(available_categories)
        else:
            bug_category = BugCategory.LOGICAL
        
        # Add a bug based on the selected category
        if bug_category == BugCategory.SYNTAX:
            self._add_syntax_error(state, solution)
        elif bug_category == BugCategory.LOGICAL:
            self._add_logical_error(state, solution, parsed_solution)
        elif bug_category == BugCategory.PERFORMANCE:
            self._add_performance_issue(state, solution, parsed_solution)
        elif bug_category == BugCategory.EDGE_CASE:
            self._add_edge_case_issue(state, solution, parsed_solution)
        else:
            # Default to logical error
            self._add_logical_error(state, solution, parsed_solution)
        
        # Update bug count in code context
        if "bug_count" not in state.code_context:
            state.code_context["bug_count"] = 0
        state.code_context["bug_count"] += 1
        
        # Add the bug category to the context
        if "bug_categories" not in state.code_context:
            state.code_context["bug_categories"] = []
        state.code_context["bug_categories"].append(bug_category)
    
    def _change_requirements(self, state: ProblemState, solution: str) -> None:
        """
        Change the requirements to challenge the current solution.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Get the current requirements
        requirements = state.requirements
        
        # Add a new requirement
        new_requirement = self._generate_new_requirement(state, solution)
        if new_requirement:
            requirements.append(new_requirement)
        
        # Modify an existing requirement if possible
        if requirements and random.random() < 0.5:
            idx = random.randint(0, len(requirements) - 1)
            requirements[idx] = self._modify_requirement(requirements[idx], state, solution)
    
    def _increase_complexity(self, state: ProblemState, solution: str) -> None:
        """
        Increase the complexity of the task.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Parse the solution if possible
        try:
            parsed_solution = ast.parse(solution)
        except SyntaxError:
            # If we can't parse the solution, make a simpler change
            self._add_edge_case_requirement(state)
            return
        
        # Choose a complexity increase strategy
        strategies = [
            "add_edge_cases",
            "increase_data_volume",
            "add_performance_constraint",
            "expand_functionality"
        ]
        
        strategy = random.choice(strategies)
        
        if strategy == "add_edge_cases":
            self._add_edge_case_requirement(state)
        elif strategy == "increase_data_volume":
            self._increase_data_volume(state, solution)
        elif strategy == "add_performance_constraint":
            self._add_performance_constraint(state, solution)
        elif strategy == "expand_functionality":
            self._expand_functionality(state, solution)
    
    def _create_test_files(self, temp_path: Path) -> List[Path]:
        """
        Create test files based on the current problem state.
        
        Args:
            temp_path: The temporary directory path
            
        Returns:
            List of test file paths
        """
        test_files = []
        
        # Create test files from the code context
        if "tests" in self.state.code_context:
            for i, test in enumerate(self.state.code_context["tests"]):
                test_file = temp_path / f"test_{i}.py"
                with open(test_file, "w") as f:
                    f.write(test["content"])
                test_files.append(test_file)
        
        # Create a default test file if no tests are specified
        if not test_files:
            test_file = temp_path / "test_default.py"
            with open(test_file, "w") as f:
                f.write(self._generate_default_test())
            test_files.append(test_file)
        
        return test_files
    
    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate a score based on test results.
        
        Args:
            results: The test results
            
        Returns:
            A score between 0 and 1
        """
        # Base score on test results
        if results["total_tests"] == 0:
            test_score = 0.0
        else:
            test_score = results["passed_tests"] / results["total_tests"]
        
        # Adjust for execution success
        execution_score = 1.0 if results["execution"]["success"] else 0.0
        
        # Combine scores with weights
        weights = self.config.get("score_weights", {"test": 0.7, "execution": 0.3})
        score = (test_score * weights["test"] + execution_score * weights["execution"])
        
        # Apply difficulty modifier
        difficulty_modifier = 1.0 + (self.state.difficulty * 0.2)
        score = score / difficulty_modifier
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity(self, code: str) -> float:
        """
        Calculate the complexity of code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A complexity score
        """
        # Simple cyclomatic complexity estimation
        complexity = 1
        
        # Count control flow statements
        for pattern in ["if", "for", "while", "and", "or"]:
            complexity += code.count(f" {pattern} ")
        
        # Count function definitions
        complexity += code.count("def ")
        
        # Normalize to 0-1 range
        normalized = min(1.0, complexity / 50.0)
        
        return normalized
    
    def _generate_suggestion_for_test_failure(
        self,
        issue: Dict[str, Any],
        solution: str,
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a suggestion for a test failure.
        
        Args:
            issue: The issue data
            solution: The solution code
            test_results: The test results
            
        Returns:
            A suggestion dictionary
        """
        test_name = issue["test"]
        test_result = test_results[test_name]
        
        # Extract relevant parts of the test
        test_content = None
        for test in self.state.code_context.get("tests", []):
            if test.get("name") == test_name:
                test_content = test.get("content")
                break
        
        if test_content:
            # Try to extract the assertion that failed
            assertion_match = re.search(r"assert.*", test_content)
            assertion = assertion_match.group(0) if assertion_match else None
            
            # Look for function names in both test and solution
            test_funcs = re.findall(r"def\s+(\w+)", test_content)
            solution_funcs = re.findall(r"def\s+(\w+)", solution)
            
            # Find functions in test that aren't in solution
            missing_funcs = [f for f in test_funcs if f not in solution_funcs]
            
            if missing_funcs:
                return {
                    "type": "missing_function",
                    "message": f"Implement the missing function(s): {', '.join(missing_funcs)}",
                    "functions": missing_funcs
                }
            elif assertion:
                return {
                    "type": "fix_assertion_failure",
                    "message": f"Fix the code to pass the assertion: {assertion}",
                    "assertion": assertion,
                    "expected": test_result.get("expected"),
                    "actual": test_result.get("actual")
                }
            else:
                return {
                    "type": "fix_test_failure",
                    "message": f"Fix the code to pass the test: {test_name}",
                    "test_name": test_name
                }
        else:
            return {
                "type": "general_fix",
                "message": f"Fix the code to pass the failing test: {test_name}"
            }
    
    def _generate_suggestion_for_error(
        self,
        issue: Dict[str, Any],
        solution: str
    ) -> Dict[str, Any]:
        """
        Generate a suggestion for an error.
        
        Args:
            issue: The issue data
            solution: The solution code
            
        Returns:
            A suggestion dictionary
        """
        error_type = issue["error_type"]
        message = issue["message"]
        location = issue.get("location")
        
        if error_type == "syntax":
            return {
                "type": "fix_syntax",
                "message": f"Fix the syntax error: {message}",
                "location": location
            }
        elif error_type == "runtime":
            return {
                "type": "fix_runtime_error",
                "message": f"Fix the runtime error: {message}",
                "location": location
            }
        else:
            return {
                "type": "fix_error",
                "message": f"Fix the error: {message}",
                "error_type": error_type,
                "location": location
            }
    
    def _determine_focus_areas(
        self,
        issues: List[Dict[str, Any]],
        solution: str,
        result: EvaluationResult
    ) -> List[str]:
        """
        Determine focus areas based on issues and results.
        
        Args:
            issues: The identified issues
            solution: The solution code
            result: The evaluation results
            
        Returns:
            List of focus areas
        """
        focus_areas = []
        
        # Check for syntax issues
        syntax_issues = [i for i in issues if i.get("error_type") == "syntax"]
        if syntax_issues:
            focus_areas.append("syntax")
        
        # Check for failing tests
        test_issues = [i for i in issues if i["type"] == "test_failure"]
        if test_issues:
            if any("expected" in i and "actual" in i for i in test_issues):
                focus_areas.append("logic")
            else:
                focus_areas.append("functionality")
        
        # Check for performance issues
        if result.metrics and "execution_time" in result.metrics:
            if result.metrics["execution_time"] > self.config.get("performance_threshold", 1.0):
                focus_areas.append("performance")
        
        # Check for complexity issues
        if result.metrics and "code_complexity" in result.metrics:
            if result.metrics["code_complexity"] > self.config.get("complexity_threshold", 0.7):
                focus_areas.append("complexity")
        
        # Default focus area if none were identified
        if not focus_areas:
            focus_areas.append("general")
        
        return focus_areas
    
    def _generate_adaptation_hints(
        self,
        solution: str,
        result: EvaluationResult
    ) -> List[Dict[str, Any]]:
        """
        Generate hints about how the problem might adapt in the next iteration.
        
        Args:
            solution: The solution code
            result: The evaluation results
            
        Returns:
            List of adaptation hints
        """
        hints = []
        
        # Hint about potential complexity increases
        if result.score > 0.8:
            hints.append({
                "type": "complexity_increase",
                "message": "The problem may become more complex in the next iteration."
            })
        
        # Hint about potential requirement changes
        if result.score > 0.9 and self.state.evolution_stage >= 1:
            hints.append({
                "type": "requirement_change",
                "message": "The requirements may change in the next iteration."
            })
        
        # Hint about potential bug additions
        if result.score > 0.95:
            hints.append({
                "type": "new_bugs",
                "message": "New, more subtle bugs may be introduced in the next iteration."
            })
        
        # Hint about focus on specific areas
        if result.score > 0.7 and result.score < 0.95:
            focus_areas = result.metrics.get("focus_areas", [])
            if focus_areas:
                hints.append({
                    "type": "focus_shift",
                    "message": f"The next iteration may focus more on: {', '.join(focus_areas)}",
                    "areas": focus_areas
                })
        
        return hints
    
    def _generate_description(self, state: ProblemState) -> str:
        """
        Generate a description for the current problem state.
        
        Args:
            state: The problem state
            
        Returns:
            A descriptive prompt for the problem
        """
        # Base description
        base_desc = (
            f"Fix the bug(s) in the following code. "
            f"This is iteration {state.evolution_stage + 1} of the task."
        )
        
        # Add information about known bug categories
        if "bug_categories" in state.code_context:
            categories = state.code_context["bug_categories"]
            if categories:
                base_desc += f"\n\nThe code contains the following types of issues: {', '.join(categories)}."
        
        # Add requirements
        if state.requirements:
            base_desc += "\n\nRequirements:"
            for i, req in enumerate(state.requirements):
                base_desc += f"\n{i+1}. {req['description']}"
                
        # Add information about difficulty
        difficulty_desc = "easy"
        if state.difficulty > 0.3 and state.difficulty <= 0.6:
            difficulty_desc = "moderate"
        elif state.difficulty > 0.6 and state.difficulty <= 0.8:
            difficulty_desc = "challenging"
        elif state.difficulty > 0.8:
            difficulty_desc = "very challenging"
        
        base_desc += f"\n\nThis is a {difficulty_desc} bug fixing task."
        
        return base_desc
    
    def _generate_focused_description(self, state: ProblemState, issues: List[Dict[str, Any]]) -> str:
        """
        Generate a description focused on remaining issues.
        
        Args:
            state: The problem state
            issues: The identified issues
            
        Returns:
            A descriptive prompt focused on remaining issues
        """
        base_desc = self._generate_description(state)
        
        # Add focus on remaining issues
        if issues:
            base_desc += "\n\nFocus on the following issues:"
            for i, issue in enumerate(issues):
                if issue["type"] == "test_failure":
                    base_desc += f"\n{i+1}. Test failure in '{issue['test']}': {issue['message']}"
                else:
                    base_desc += f"\n{i+1}. {issue['error_type']} error: {issue['message']}"
        
        # Add focus areas if present
        if "focus_areas" in state.code_context:
            areas = state.code_context["focus_areas"]
            if areas:
                base_desc += f"\n\nPay particular attention to: {', '.join(areas)}."
        
        return base_desc
    
    def _generate_guided_description(
        self,
        state: ProblemState,
        issues: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a description with added guidance.
        
        Args:
            state: The problem state
            issues: The identified issues
            suggestions: The suggested fixes
            
        Returns:
            A descriptive prompt with added guidance
        """
        base_desc = self._generate_description(state)
        
        # Add detailed information about issues
        if issues:
            base_desc += "\n\nThe following issues were identified in your previous solution:"
            for i, issue in enumerate(issues):
                if issue["type"] == "test_failure":
                    base_desc += f"\n{i+1}. Test failure in '{issue['test']}': {issue['message']}"
                    if "expected" in issue and "actual" in issue:
                        base_desc += f"\n   Expected: {issue['expected']}"
                        base_desc += f"\n   Actual: {issue['actual']}"
                else:
                    base_desc += f"\n{i+1}. {issue['error_type']} error: {issue['message']}"
                    if "location" in issue:
                        base_desc += f"\n   Location: {issue['location']}"
        
        # Add suggestions
        if suggestions:
            base_desc += "\n\nConsider the following suggestions:"
            for i, suggestion in enumerate(suggestions):
                base_desc += f"\n{i+1}. {suggestion['message']}"
        
        # Add hints if present
        if "hints" in state.code_context:
            hints = state.code_context["hints"]
            if hints:
                base_desc += "\n\nHints:"
                for i, hint in enumerate(hints):
                    base_desc += f"\n{i+1}. {hint}"
        
        return base_desc
    
    def _generate_hints(
        self,
        solution: str,
        result: EvaluationResult,
        feedback: Feedback
    ) -> List[str]:
        """
        Generate hints based on the solution and feedback.
        
        Args:
            solution: The solution code
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            List of hints
        """
        hints = []
        
        # Add hints based on failing tests
        if result.test_results:
            failing_tests = [
                test_name for test_name, test_result in result.test_results.items()
                if not test_result["passed"]
            ]
            
            if failing_tests:
                test_hint = "Focus on fixing the failing tests"
                
                # Add specific information about test expectations if available
                for test_name in failing_tests[:2]:  # Limit to first two tests
                    test_result = result.test_results[test_name]
                    if "expected" in test_result and "actual" in test_result:
                        test_hint += f". For test '{test_name}', expected '{test_result['expected']}' but got '{test_result['actual']}'"
                
                hints.append(test_hint + ".")
        
        # Add hints based on errors
        if result.error_details:
            for error_type, error_info in result.error_details.items():
                hints.append(f"Fix the {error_type} error: {error_info.get('message', 'Unknown error')}.")
        
        # Add hints based on focus areas
        for area in feedback.focus_areas:
            if area == "syntax":
                hints.append("Check your syntax carefully, especially parentheses, indentation, and function definitions.")
            elif area == "logic":
                hints.append("Review the logic of your solution, especially conditional statements and loop conditions.")
            elif area == "functionality":
                hints.append("Ensure your solution implements all required functionality specified in the tests.")
            elif area == "performance":
                hints.append("Consider optimizing your solution for better performance, avoid unnecessary operations.")
            elif area == "complexity":
                hints.append("Try to simplify your solution, it may be more complex than necessary.")
        
        return hints
    
    def _generate_test_hint(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """
        Generate a hint for a specific failing test.
        
        Args:
            test_name: The name of the test
            test_result: The test result
            
        Returns:
            A hint for the test
        """
        if "expected" in test_result and "actual" in test_result:
            return f"The test expected '{test_result['expected']}' but got '{test_result['actual']}'"
        elif "message" in test_result:
            return test_result["message"]
        else:
            return "The test failed, but no detailed information is available."
    
    def _add_syntax_error(self, state: ProblemState, solution: str) -> None:
        """
        Add a syntax error to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        lines = solution.split('\n')
        if not lines:
            return
        
        # Choose a line to modify
        idx = random.randint(0, len(lines) - 1)
        line = lines[idx]
        
        # Skip empty lines or comment lines
        while not line.strip() or line.strip().startswith('#'):
            idx = random.randint(0, len(lines) - 1)
            line = lines[idx]
        
        # Choose a modification type
        mod_type = random.choice([
            "remove_character",
            "add_character",
            "swap_characters",
            "change_indent"
        ])
        
        if mod_type == "remove_character" and line:
            char_idx = random.randint(0, len(line) - 1)
            lines[idx] = line[:char_idx] + line[char_idx+1:]
        
        elif mod_type == "add_character":
            char_idx = random.randint(0, len(line))
            char = random.choice(["(", ")", "{", "}", "[", "]", ":", ";", ",", "."])
            lines[idx] = line[:char_idx] + char + line[char_idx:]
        
        elif mod_type == "swap_characters" and len(line) >= 2:
            char_idx = random.randint(0, len(line) - 2)
            lines[idx] = (line[:char_idx] + line[char_idx+1] + 
                         line[char_idx] + line[char_idx+2:])
        
        elif mod_type == "change_indent":
            # Either add or remove indentation
            if line.startswith("    "):
                lines[idx] = line[2:]  # Remove some indent
            else:
                lines[idx] = "  " + line  # Add inconsistent indent
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "syntax",
            "line": idx + 1,
            "description": f"Syntax error introduced in line {idx + 1}"
        })
    
    def _add_logical_error(self, state: ProblemState, solution: str, parsed_solution: ast.Module) -> None:
        """
        Add a logical error to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
            parsed_solution: The parsed AST of the solution
        """
        modification_types = [
            "change_comparison",
            "invert_condition",
            "off_by_one",
            "change_operator",
            "reverse_logic"
        ]
        
        mod_type = random.choice(modification_types)
        lines = solution.split('\n')
        
        # Find all if statements and loops
        if_statements = []
        for i, line in enumerate(lines):
            if re.search(r'\bif\b|\bwhile\b|\bfor\b', line):
                if_statements.append((i, line))
        
        if if_statements:
            # Choose an if statement to modify
            idx, line = random.choice(if_
            # Choose an if statement to modify
            idx, line = random.choice(if_statements)
            
            if mod_type == "change_comparison":
                # Change a comparison operator
                new_line = re.sub(r'==', '!=', line)
                if new_line == line:  # No change made
                    new_line = re.sub(r'!=', '==', line)
                if new_line == line:  # Still no change
                    new_line = re.sub(r'>', '>=', line)
                if new_line == line:  # Still no change
                    new_line = re.sub(r'<', '<=', line)
                
                lines[idx] = new_line
            
            elif mod_type == "invert_condition":
                # Add or remove a 'not'
                if "not" in line:
                    new_line = re.sub(r'not\s*', '', line)
                else:
                    match = re.search(r'(if|while)\s*\(?(.*?)\)?:', line)
                    if match:
                        condition = match.group(2)
                        replacement = f"{match.group(1)} not ({condition}):"
                        new_line = line.replace(match.group(0), replacement)
                    else:
                        # If we can't parse it properly, just add 'not' before the condition
                        new_line = re.sub(r'(if|while)\s*', r'\1 not ', line)
                
                lines[idx] = new_line
            
            elif mod_type == "off_by_one":
                # Introduce an off-by-one error in a range or comparison
                if "range(" in line:
                    # Modify a range function
                    new_line = re.sub(r'range\(([^,]+),\s*([^)]+)\)', 
                                     r'range(\1, \2 - 1)', line)
                    if new_line == line:  # No change made
                        new_line = re.sub(r'range\(([^)]+)\)', r'range(\1 - 1)', line)
                else:
                    # Modify a comparison
                    new_line = re.sub(r'(>|>=)\s*(\d+)', lambda m: f"{m.group(1)} {int(m.group(2)) + 1}", line)
                    if new_line == line:  # No change made
                        new_line = re.sub(r'(<|<=)\s*(\d+)', lambda m: f"{m.group(1)} {int(m.group(2)) - 1}", line)
                
                lines[idx] = new_line
            
            elif mod_type == "change_operator":
                # Change a mathematical operator
                operator_pairs = [('+', '-'), ('*', '/'), ('&', '|'), ('>>', '<<')]
                for op1, op2 in operator_pairs:
                    if op1 in line:
                        new_line = line.replace(op1, op2, 1)
                        lines[idx] = new_line
                        break
                    elif op2 in line:
                        new_line = line.replace(op2, op1, 1)
                        lines[idx] = new_line
                        break
            
            elif mod_type == "reverse_logic":
                # Reverse a logical condition
                if " and " in line:
                    parts = re.split(r'\band\b', line, 1)
                    new_line = parts[0] + " or " + parts[1]
                    lines[idx] = new_line
                elif " or " in line:
                    parts = re.split(r'\bor\b', line, 1)
                    new_line = parts[0] + " and " + parts[1]
                    lines[idx] = new_line
        
        else:
            # If no if statements, try to modify a variable assignment
            assignments = []
            for i, line in enumerate(lines):
                if "=" in line and "==" not in line and "!=" not in line:
                    assignments.append((i, line))
            
            if assignments:
                idx, line = random.choice(assignments)
                # Change a numeric value
                new_line = re.sub(r'=\s*(\d+)', lambda m: f"= {int(m.group(1)) + 1}", line)
                if new_line == line:  # No change made
                    # Try changing a variable name
                    var_match = re.match(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
                    if var_match:
                        indent, var_name = var_match.groups()
                        # Just add a typo to the variable name
                        new_var = var_name + '_'
                        new_line = line.replace(var_name, new_var, 1)
                        
                        # Also replace other occurrences of this variable
                        for i, other_line in enumerate(lines):
                            if i != idx and re.search(r'\b' + var_name + r'\b', other_line):
                                # Only replace some occurrences to make it tricky
                                if random.random() < 0.7:
                                    lines[i] = re.sub(r'\b' + var_name + r'\b', 
                                                    new_var, other_line, 1)
                
                lines[idx] = new_line
            else:
                # As a last resort, add a logical error at the end
                lines.append("# The following line introduces a subtle logical error")
                lines.append("result = None  # This will cause a failure")
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "logical",
            "line": idx + 1,
            "description": f"Logical error introduced in line {idx + 1} with modification type {mod_type}"
        })
    
    def _add_performance_issue(self, state: ProblemState, solution: str, parsed_solution: ast.Module) -> None:
        """
        Add a performance issue to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
            parsed_solution: The parsed AST of the solution
        """
        lines = solution.split('\n')
        
        # Look for loops that could be made inefficient
        loop_lines = []
        for i, line in enumerate(lines):
            if re.search(r'\bfor\b|\bwhile\b', line):
                loop_lines.append((i, line))
        
        if loop_lines:
            # Choose a loop to modify
            idx, line = random.choice(loop_lines)
            
            # Determine the indentation of the loop
            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ""
            
            # Choose a performance issue to introduce
            issue_type = random.choice([
                "add_unnecessary_operation",
                "nested_loop",
                "inefficient_data_structure"
            ])
            
            if issue_type == "add_unnecessary_operation":
                # Add unnecessary operations inside the loop
                next_line_idx = idx + 1
                while next_line_idx < len(lines) and (not lines[next_line_idx].strip() or 
                                                    lines[next_line_idx].startswith(indent + " ")):
                    next_line_idx += 1
                
                # Insert an unnecessary operation
                operation = random.choice([
                    indent + "    temp_list = []  # Unnecessary list creation",
                    indent + "    temp_dict = {}  # Unnecessary dictionary creation",
                    indent + "    sorted([x for x in range(100)])  # Unnecessary sorting operation",
                    indent + "    ''.join([str(i) for i in range(50)])  # Unnecessary string join"
                ])
                
                lines.insert(next_line_idx, operation)
            
            elif issue_type == "nested_loop":
                # Add an unnecessary nested loop
                next_line_idx = idx + 1
                nested_loop = indent + "    for _ in range(100):  # Unnecessary nested loop\n"
                nested_loop += indent + "        pass"
                
                lines.insert(next_line_idx, nested_loop)
            
            elif issue_type == "inefficient_data_structure":
                # Convert an efficient data structure to an inefficient one
                # For example, replace a set with a list for lookups
                for i, line in enumerate(lines):
                    if "set(" in line:
                        lines[i] = line.replace("set(", "list(")
                        break
                    elif "dict(" in line:
                        lines[i] = line.replace("dict(", "list(")
                        break
                
                # If no replacement was made, add a comment explaining the issue
                if idx < len(lines):
                    lines.insert(idx, f"{indent}# The following loop uses an inefficient data structure")
        
        else:
            # If no loops found, add a general performance issue
            for i, line in enumerate(lines):
                if "def " in line:
                    # Add a slow operation inside a function
                    func_indent_match = re.match(r'^(\s*)', line)
                    func_indent = func_indent_match.group(1) if func_indent_match else ""
                    
                    next_line_idx = i + 1
                    while next_line_idx < len(lines) and (not lines[next_line_idx].strip() or 
                                                        lines[next_line_idx].startswith(func_indent + " ")):
                        next_line_idx += 1
                    
                    # Insert a slow operation
                    slow_op = func_indent + "    import time; time.sleep(0.1)  # Unnecessary delay"
                    lines.insert(next_line_idx, slow_op)
                    break
            else:
                # As a last resort, append a slow function at the end
                lines.append("\n# The following function introduces a performance issue")
                lines.append("def slow_operation():")
                lines.append("    import time")
                lines.append("    time.sleep(0.1)  # Unnecessary delay")
                lines.append("    return True")
                
                # And call it somewhere in the code
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith("#"):
                        indent_match = re.match(r'^(\s*)', line)
                        indent = indent_match.group(1) if indent_match else ""
                        lines.insert(i + 1, f"{indent}slow_operation()  # Unnecessary slow operation")
                        break
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "performance",
            "description": f"Performance issue introduced with type {issue_type}"
        })
    
    def _add_edge_case_issue(self, state: ProblemState, solution: str, parsed_solution: ast.Module) -> None:
        """
        Add an edge case issue to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
            parsed_solution: The parsed AST of the solution
        """
        lines = solution.split('\n')
        
        # Find places where edge cases might be handled
        edge_case_candidates = []
        for i, line in enumerate(lines):
            # Look for conditionals that might be handling edge cases
            if (re.search(r'\bif\b', line) and 
                any(term in line for term in ["==", "!=", "<", ">", "<=", ">=", "None", "0", "empty", "[]", "{}", "len", "not"])):
                edge_case_candidates.append((i, line, "conditional"))
                
            # Look for function parameters that might need edge case handling
            elif re.search(r'\bdef\b', line):
                edge_case_candidates.append((i, line, "function"))
        
        if edge_case_candidates:
            idx, line, case_type = random.choice(edge_case_candidates)
            
            if case_type == "conditional":
                # Modify a conditional that handles edge cases
                if "== 0" in line or "== None" in line or "== []" in line or "== {}" in line:
                    # Remove the edge case check
                    new_line = re.sub(r'if\s+.*?(\b==\b|\bis\b)\s*(0|None|\[\]|\{\}|""|\'\').*?:', 
                                    "if False:  # Removed edge case check", line)
                    lines[idx] = new_line
                    
                elif "!= 0" in line or "!= None" in line or "is not None" in line:
                    # Invert the edge case check
                    new_line = line.replace("!=", "==").replace("is not", "is")
                    lines[idx] = new_line
                    
                elif "len(" in line:
                    # Modify length check
                    if "== 0" in line:
                        new_line = line.replace("== 0", "> 0")
                    elif "> 0" in line:
                        new_line = line.replace("> 0", "== 0")
                    else:
                        new_line = line  # No change
                    
                    lines[idx] = new_line
            
            elif case_type == "function":
                # Add a missed edge case in a function
                match = re.match(r'^(\s*def\s+\w+\s*\()(.*?)(\):)', line)
                if match:
                    # Look for the function body
                    func_indent_match = re.match(r'^(\s*)', line)
                    func_indent = func_indent_match.group(1) if func_indent_match else ""
                    body_indent = func_indent + "    "
                    
                    # Find where to insert the edge case handling
                    next_line_idx = idx + 1
                    while (next_line_idx < len(lines) and 
                          (not lines[next_line_idx].strip() or 
                           lines[next_line_idx].startswith(body_indent))):
                        next_line_idx += 1
                    
                    # Extract parameter names
                    params = match.group(2).split(',')
                    if params and params[0].strip():
                        param_name = params[0].strip().split('=')[0].strip()
                        
                        # Add missing edge case handling that will be wrong
                        edge_case_code = [
                            f"{body_indent}# The following edge case handling is incomplete",
                            f"{body_indent}if {param_name} is None:",
                            f"{body_indent}    return []  # This might not be the correct behavior"
                        ]
                        
                        for i, code_line in enumerate(edge_case_code):
                            lines.insert(next_line_idx + i, code_line)
        
        else:
            # If no suitable candidates, add a general edge case issue
            for i, line in enumerate(lines):
                if "def " in line:
                    # Extract the function name
                    match = re.search(r'def\s+(\w+)', line)
                    if match:
                        function_name = match.group(1)
                        
                        # Add an edge case issue at the end of the file
                        lines.append("\n# The following code has an edge case issue")
                        lines.append(f"# When calling {function_name} with an empty input")
                        lines.append(f"result = {function_name}([])")
                        lines.append("# The result will cause problems because edge cases aren't handled correctly")
                        break
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "edge_case",
            "description": "Edge case handling issue introduced"
        })
    
    def _generate_new_requirement(self, state: ProblemState, solution: str) -> Dict[str, Any]:
        """
        Generate a new requirement to challenge the current solution.
        
        Args:
            state: The current problem state
            solution: The current solution
            
        Returns:
            A new requirement dictionary
        """
        # Choose a requirement type
        req_types = [
            "performance_requirement",
            "edge_case_requirement",
            "validation_requirement",
            "error_handling_requirement"
        ]
        
        req_type = random.choice(req_types)
        
        if req_type == "performance_requirement":
            return {
                "id": str(uuid.uuid4()),
                "type": "performance",
                "description": "The solution must execute in O(n) time complexity.",
                "details": "Avoid nested loops and unnecessary operations that would increase time complexity.",
                "priority": "high"
            }
        
        elif req_type == "edge_case_requirement":
            edge_cases = [
                "empty inputs",
                "None values",
                "negative numbers",
                "extremely large inputs",
                "duplicate values"
            ]
            edge_case = random.choice(edge_cases)
            
            return {
                "id": str(uuid.uuid4()),
                "type": "edge_case",
                "description": f"The solution must correctly handle {edge_case}.",
                "details": f"Ensure proper validation and processing of {edge_case} without errors.",
                "priority": "medium"
            }
        
        elif req_type == "validation_requirement":
            return {
                "id": str(uuid.uuid4()),
                "type": "validation",
                "description": "The solution must validate all inputs before processing.",
                "details": "Check for invalid inputs and raise appropriate exceptions with descriptive messages.",
                "priority": "medium"
            }
        
        elif req_type == "error_handling_requirement":
            return {
                "id": str(uuid.uuid4()),
                "type": "error_handling",
                "description": "The solution must include comprehensive error handling.",
                "details": "Catch and handle exceptions appropriately, providing meaningful error messages.",
                "priority": "high"
            }
        
        # Default fallback
        return {
            "id": str(uuid.uuid4()),
            "type": "general",
            "description": "The solution must be well-documented with comments.",
            "details": "Add comments explaining the approach and any non-obvious code sections.",
            "priority": "low"
        }
    
    def _modify_requirement(self, requirement: Dict[str, Any], state: ProblemState, solution: str) -> Dict[str, Any]:
        """
        Modify an existing requirement to make it more challenging.
        
        Args:
            requirement: The requirement to modify
            state: The current problem state
            solution: The current solution
            
        Returns:
            The modified requirement
        """
        # Create a copy of the requirement to modify
        modified = copy.deepcopy(requirement)
        
        # Determine how to modify based on requirement type
        if requirement["type"] == "performance":
            # Make performance requirements more stringent
            if "O(n)" in requirement["description"]:
                modified["description"] = requirement["description"].replace("O(n)", "O(log n)")
                modified["details"] = "The previous O(n) solution is no longer acceptable. Optimize further."
            elif "O(log n)" in requirement["description"]:
                modified["description"] = requirement["description"].replace("O(log n)", "O(1)")
                modified["details"] = "The previous O(log n) solution is no longer acceptable. Optimize further."
        
        elif requirement["type"] == "edge_case":
            # Add more edge cases to handle
            edge_cases = ["null values", "empty strings", "zero values", "special characters"]
            new_case = next((case for case in edge_cases if case not in requirement["description"]), edge_cases[0])
            
            modified["description"] += f" Additionally, it must handle {new_case}."
            modified["details"] += f" The solution must now also account for {new_case}."
        
        elif requirement["type"] == "validation":
            # Make validation more specific
            modified["description"] = "The solution must perform strict validation on all inputs."
            modified["details"] = "Implement type checking, range validation, and format verification as appropriate."
        
        else:
            # Make general requirements more specific
            modified["description"] += " with increased rigor."
            modified["details"] += " The previous implementation does not fully satisfy this requirement."
        
        # Update the priority if it's not already high
        if modified["priority"] != "high":
            modified["priority"] = "high"
        
        return modified
    
    def _add_edge_case_requirement(self, state: ProblemState) -> None:
        """
        Add an edge case handling requirement.
        
        Args:
            state: The problem state to modify
        """
        edge_cases = [
            "empty collections",
            "null/None values",
            "negative numbers",
            "zero values",
            "extremely large values",
            "duplicate entries",
            "special characters",
            "whitespace-only strings",
            "maximum/minimum representable values"
        ]
        
        # Choose edge cases that aren't already covered
        existing_edge_cases = []
        for req in state.requirements:
            if req["type"] == "edge_case":
                for case in edge_cases:
                    if case in req["description"]:
                        existing_edge_cases.append(case)
        
        available_cases = [case for case in edge_cases if case not in existing_edge_cases]
        if not available_cases:
            available_cases = edge_cases  # If all are covered, allow duplicates
            
        selected_case = random.choice(available_cases)
        
        # Add the requirement
        state.requirements.append({
            "id": str(uuid.uuid4()),
            "type": "edge_case",
            "description": f"The solution must correctly handle {selected_case} as a special case.",
            "details": f"Ensure that {selected_case} are properly validated and processed without errors.",
            "priority": "high"
        })
        
        # Also add a test for this edge case
        if "tests" not in state.code_context:
            state.code_context["tests"] = []
        
        # Generate a test based on the edge case
        edge_case_test = self._generate_edge_case_test(selected_case, state.code_context.get("code", ""))
        if edge_case_test:
            state.code_context["tests"].append({
                "name": f"test_edge_case_{selected_case.replace(' ', '_')}",
                "content": edge_case_test
            })
    
    def _increase_data_volume(self, state: ProblemState, solution: str) -> None:
        """
        Modify the problem to require handling larger data volumes.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Update requirements to specify larger data volume
        state.requirements.append({
            "id": str(uuid.uuid4()),
            "type": "performance",
            "description": "The solution must efficiently handle large datasets (10,000+ elements).",
            "details": "Ensure the solution maintains performance with significantly larger inputs.",
            "priority": "high"
        })
        
        # Add a test with larger data volume
        if "tests" not in state.code_context:
            state.code_context["tests"] = []
        
        # Generate a test with large data
        large_data_test = self._generate_large_data_test(solution)
        if large_data_test:
            state.code_context["tests"].append({
                "name": "test_large_data_volume",
                "content": large_data_test
            })
    
    def _add_performance_constraint(self, state: ProblemState, solution: str) -> None:
        """
        Add a performance constraint to the problem.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Determine current complexity by analyzing the solution
        complexity = "O(n)"  # Default assumption
        if re.search(r'for\s+.*?\s+in\s+.*?for\s+', solution, re.DOTALL):
            complexity = "O(n²)"  # Nested loops
        
        target_complexity = "O(n)"
        if complexity == "O(n)":
            target_complexity = "O(log n)"
        elif complexity == "O(log n)":
            target_complexity = "O(1)"
        
        # Add a performance requirement
        state.requirements.append({
            "id": str(uuid.uuid4()),
            "type": "performance",
            "description": f"The solution must have {target_complexity} time complexity.",
            "details": f"Optimize the current solution to achieve {target_complexity} performance.",
            "priority": "high"
        })
        
        # Add a test that measures performance
        if "tests" not in state.code_context:
            state.code_context["tests"] = []
        
        # Generate a performance test
        performance_test = self._generate_performance_test(solution, target_complexity)
        if performance_test:
            state.code_context["tests"].append({
                "name": "test_performance_constraint",
                "content": performance_test
            })
    
    def _expand_functionality(self, state: ProblemState, solution: str) -> None:
        """
        Expand required functionality of the solution.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Analyze the solution to determine its function
        function_type = "unknown"
        if re.search(r'sort|sorted|order', solution, re.IGNORECASE):
            function_type = "sorting"
        elif re.search(r'find|search|match|index', solution, re.IGNORECASE):
            function_type = "searching"
        elif re.search(r'sum|total|add', solution, re.IGNORECASE):
            function_type = "calculation"
        elif re.search(r'merge|combine|join', solution, re.IGNORECASE):
            function_type = "merging"
        elif re.search(r'filter|select|where', solution, re.IGNORECASE):
            function_type = "filtering"
        
        # Generate an expanded functionality requirement based on the function type
        expansion = {
            "sorting": {
                "description": "The solution must support custom sorting keys and both ascending and descending order.",
                "details": "Extend the sorting functionality to allow the caller to specify a custom key function and sort direction."
            },
            "searching": {
                "description": "The solution must support fuzzy matching and return multiple potential matches.",
                "details": "Extend the search functionality to find not only exact matches but also similar items."
            },
            "calculation": {
                "description": "The solution must support weighted calculations and custom aggregation functions.",
                "details": "Extend the calculation to allow weights to be applied to values and custom aggregation methods."
            },
            "merging": {
                "description": "The solution must support conditional merging with conflict resolution strategies.",
                "details": "Extend the merging functionality to handle conflicts and apply conditional logic."
            },
            "filtering": {
                "description": "The solution must support complex filtering with compound conditions and custom predicates.",
                "details": "Extend the filtering to allow complex boolean logic and custom filter functions."
            },
            "unknown": {
                "description": "The solution must provide additional utility functions for working with the results.",
                "details": "Extend the functionality with helper methods that make the results more useful."
            }
        }
        
        # Add the expanded functionality requirement
        state.requirements.append({
            "id": str(uuid.uuid4()),
            "type": "functionality",
            "description": expansion[function_type]["description"],
            "details": expansion[function_type]["details"],
            "priority": "high"
        })
        
        # Add a test for the expanded functionality
        if "tests" not in state.code_context:
            state.code_context["tests"] = []
        
        # Generate a test for the expanded functionality
        expanded_test = self._generate_expanded_functionality_test(solution, function_type)
        if expanded_test:
            state.code_context["tests"].append({
                "name": f"test_expanded_{function_type}_functionality",
                "content": expanded_test
            })
    
    def _generate_default_test(self) -> str:
        """
        Generate a default test if no tests are specified.
        
        Returns:
            Default test code
        """
        return """
import unittest
import solution

class DefaultTest(unittest.TestCase):
    def test_basic_functionality(self):
        # A basic test to check if the solution runs without errors
        result = solution.main()
        self.assertIsNotNone(result)
        
if __name__ == '__main__':
    unittest.main()
"""
    
    def _generate_edge_case_test(self, edge_case: str, code: str) -> str:
        """
        Generate a test for a specific edge case.
        
        Args:
            edge_case: The edge case to test
            code: The current solution code
            
        Returns:
            Test code for the edge case
        """
        # Extract function names from the code
        function_match = re.search(r'def\s+(\w+)', code)
        if not function_match:
            return ""
            
        function_name = function_match.group(1)
        
        # Generate a test based on the edge case
        if edge_case == "empty collections":
            return f"""
import unittest
from solution import {function_name}

class EdgeCaseTest(unittest.TestCase):
    def test_empty_collection(self):
        # Test with empty input
        result = {function_name}([])
        # Add appropriate assertions based on expected behavior
        self.assertIsNotNone(result)
        
if __name__ == '__main__':
    unittest.main()
"""
        elif edge_case == "null/None values":
            return f"""
import unittest
from solution import {function_name}

class EdgeCaseTest(unittest.TestCase):
    def test_none_values(self):
        # Test with None input
        result = {function_name}(None)
        self.assertIsNotNone(result)
        
        # Test with collection containing None
        result = {function_name}([1, None, 3])
        self.assertIsNotNone(result)
        
if __name__ == '__main__':
    unittest.main()
"""
        elif edge_case == "negative numbers":
            return f"""
import unittest
from solution import {function_name}

class EdgeCaseTest(unittest.TestCase):
    def test_negative_numbers(self):
        # Test with negative numbers
        result = {function_name}([-1, -5, -10])
        self.assertIsNotNone(result)
        
        # Test with mixed positive and negative
        result = {function_name}([-1, 2, -3, 4])
        self.assertIsNotNone(result)
        
if __name__ == '__main__':
    unittest.main()
"""
        # recursive_swe_bench/task_generators/bug_fixing.py

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import uuid
import json
import re
import random
import ast
import copy
from pathlib import Path
import tempfile
import subprocess
import shutil
import os

from recursive_swe_bench.core.recursive_task import (
    RecursiveTask, ProblemState, EvaluationResult, Feedback, TaskStatus
)

class BugCategory:
    """Categories of bugs for classification and evolution."""
    SYNTAX = "syntax"
    LOGICAL = "logical"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONCURRENCY = "concurrency"
    EXCEPTION_HANDLING = "exception_handling"
    API_USAGE = "api_usage"
    MEMORY_MANAGEMENT = "memory_management"
    TYPE_ERROR = "type_error"
    EDGE_CASE = "edge_case"
    DATA_HANDLING = "data_handling"
    DEPENDENCY = "dependency"


class BugFixingTask(RecursiveTask):
    """
    A recursive task for evaluating how models fix bugs in code.
    
    The task presents a piece of code with one or more bugs, and evolves
    based on the model's fix attempts. As the model addresses issues,
    the task may introduce more subtle bugs, change requirements, or
    increase complexity to test adaptive problem-solving.
    """
    
    def __init__(
        self,
        initial_state: ProblemState,
        config: Dict[str, Any] = None,
        test_runner: Any = None
    ):
        """
        Initialize the bug fixing task.
        
        Args:
            initial_state: The initial problem state
            config: Configuration options
            test_runner: Custom test runner (optional)
        """
        super().__init__(initial_state, config)
        self.test_runner = test_runner or DefaultTestRunner()
        self.bug_categories: Set[str] = set(
            self.config.get("bug_categories", [BugCategory.LOGICAL, BugCategory.SYNTAX])
        )
        self.difficulty_progression = self.config.get(
            "difficulty_progression", [0.0, 0.15, 0.3, 0.5, 0.7]
        )
        self.evolution_strategies = self.config.get(
            "evolution_strategies", ["add_subtle_bug", "change_requirements", "increase_complexity"]
        )
        
    def _run_evaluation(self, solution: str) -> EvaluationResult:
        """
        Run tests to evaluate the solution.
        
        Args:
            solution: The solution code
            
        Returns:
            Evaluation results
        """
        # Create a temporary directory to run tests
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write solution code to file
            solution_file = temp_path / "solution.py"
            with open(solution_file, "w") as f:
                f.write(solution)
            
            # Create test files
            test_files = self._create_test_files(temp_path)
            
            # Run tests
            results = self.test_runner.run_tests(
                solution_file=solution_file,
                test_files=test_files,
                code_context=self.state.code_context
            )
            
            # Calculate score based on test results
            score = self._calculate_score(results)
            
            return EvaluationResult(
                success=results["all_passed"],
                score=score,
                execution_results=results["execution"],
                error_details=results.get("errors"),
                test_results=results["tests"],
                metrics={
                    "passed_tests": results["passed_tests"],
                    "total_tests": results["total_tests"],
                    "execution_time": results["execution_time"],
                    "memory_usage": results.get("memory_usage", 0),
                    "code_complexity": self._calculate_complexity(solution)
                }
            )
    
    def _generate_feedback(self, solution: str, result: EvaluationResult) -> Feedback:
        """
        Generate structured feedback based on evaluation results.
        
        Args:
            solution: The solution code
            result: The evaluation results
            
        Returns:
            Structured feedback
        """
        issues = []
        suggestions = []
        focus_areas = []
        
        # Add issues for failing tests
        if result.test_results:
            for test_name, test_result in result.test_results.items():
                if not test_result["passed"]:
                    issues.append({
                        "type": "test_failure",
                        "test": test_name,
                        "message": test_result.get("message", "Test failed"),
                        "expected": test_result.get("expected"),
                        "actual": test_result.get("actual")
                    })
        
        # Add issues for errors
        if result.error_details:
            for error_type, error_info in result.error_details.items():
                issues.append({
                    "type": "error",
                    "error_type": error_type,
                    "message": error_info.get("message", "An error occurred"),
                    "location": error_info.get("location")
                })
        
        # Generate suggestions based on issues
        for issue in issues:
            if issue["type"] == "test_failure":
                suggestion = self._generate_suggestion_for_test_failure(
                    issue, solution, result.test_results
                )
                if suggestion:
                    suggestions.append(suggestion)
            elif issue["type"] == "error":
                suggestion = self._generate_suggestion_for_error(
                    issue, solution
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        # Determine focus areas based on issues and task state
        focus_areas = self._determine_focus_areas(issues, solution, result)
        
        # Generate adaptation hints based on the current state and results
        adaptation_hints = self._generate_adaptation_hints(solution, result)
        
        # Create summary
        if result.success:
            summary = (
                f"Your solution passes all tests with a score of {result.score:.2f}. "
                f"The code successfully addresses the bugs in the original implementation."
            )
        else:
            passed = result.metrics.get("passed_tests", 0)
            total = result.metrics.get("total_tests", 0)
            summary = (
                f"Your solution passes {passed}/{total} tests with a score of {result.score:.2f}. "
                f"There are still issues that need to be addressed."
            )
        
        return Feedback(
            summary=summary,
            issues=issues,
            suggestions=suggestions,
            focus_areas=focus_areas,
            adaptation_hints=adaptation_hints
        )
    
    def _evolve_state(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Evolve the problem state based on the solution and feedback.
        
        This method implements the recursive nature of the benchmark by
        adapting the problem to challenge the model's understanding.
        
        Args:
            solution: The attempted solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state
        """
        # If the solution perfectly solved the problem, make it more challenging
        if result.success and result.score > 0.95:
            return self._increase_difficulty(solution, result, feedback)
        
        # If the solution was close but not perfect, focus on the remaining issues
        elif result.score > 0.7:
            return self._focus_remaining_issues(solution, result, feedback)
            
        # If the solution was not very good, provide more guidance
        else:
            return self._provide_more_guidance(solution, result, feedback)
    
    def _increase_difficulty(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Increase the difficulty of the problem for models that solved it well.
        
        Args:
            solution: The successful solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state with increased difficulty
        """
        # Create a new state based on the current state
        new_state = copy.deepcopy(self.state)
        
        # Increment evolution stage
        new_state.evolution_stage += 1
        
        # Increase difficulty based on progression schedule
        current_difficulty_idx = min(new_state.evolution_stage, 
                                    len(self.difficulty_progression) - 1)
        new_state.difficulty = self.difficulty_progression[current_difficulty_idx]
        
        # Select an evolution strategy based on the current state
        strategy = self._select_evolution_strategy(solution, result, feedback)
        
        # Apply the selected strategy
        if strategy == "add_subtle_bug":
            self._add_subtle_bug(new_state, solution)
        elif strategy == "change_requirements":
            self._change_requirements(new_state, solution)
        elif strategy == "increase_complexity":
            self._increase_complexity(new_state, solution)
        
        # Update the description to reflect the changes
        new_state.description = self._generate_description(new_state)
        
        # Update adaptation vector to guide future evolution
        new_state.adaptation_vector = self._calculate_adaptation_vector(
            solution, result, feedback
        )
        
        return new_state
    
    def _focus_remaining_issues(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Evolve the state to focus on remaining issues when the solution is close but not perfect.
        
        Args:
            solution: The nearly-successful solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state focusing on remaining issues
        """
        # Create a new state based on the current state
        new_state = copy.deepcopy(self.state)
        
        # Increment evolution stage
        new_state.evolution_stage += 1
        
        # Maintain the same difficulty level
        current_difficulty_idx = min(new_state.evolution_stage - 1, 
                                    len(self.difficulty_progression) - 1)
        new_state.difficulty = self.difficulty_progression[current_difficulty_idx]
        
        # Update the code context to focus on remaining issues
        new_state.code_context["focus_areas"] = feedback.focus_areas
        
        # Highlight failing tests in the code context
        if result.test_results:
            failing_tests = [
                test_name for test_name, test_result in result.test_results.items()
                if not test_result["passed"]
            ]
            new_state.code_context["failing_tests"] = failing_tests
        
        # Update the description to be more specific about remaining issues
        new_state.description = self._generate_focused_description(
            new_state, feedback.issues
        )
        
        # Update adaptation vector to guide future evolution
        new_state.adaptation_vector = self._calculate_adaptation_vector(
            solution, result, feedback
        )
        
        return new_state
    
    def _provide_more_guidance(self, solution: str, result: EvaluationResult, feedback: Feedback) -> ProblemState:
        """
        Evolve the state to provide more guidance when the solution was not very good.
        
        Args:
            solution: The unsuccessful solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The evolved problem state with more guidance
        """
        # Create a new state based on the current state
        new_state = copy.deepcopy(self.state)
        
        # Increment evolution stage
        new_state.evolution_stage += 1
        
        # Maintain or slightly decrease difficulty
        current_difficulty_idx = max(0, min(new_state.evolution_stage - 1, 
                                          len(self.difficulty_progression) - 1) - 1)
        new_state.difficulty = self.difficulty_progression[current_difficulty_idx]
        
        # Add more hints to the code context
        new_state.code_context["hints"] = self._generate_hints(
            solution, result, feedback
        )
        
        # Add more detailed information about failing tests
        if result.test_results:
            detailed_test_results = {}
            for test_name, test_result in result.test_results.items():
                if not test_result["passed"]:
                    detailed_test_results[test_name] = {
                        "message": test_result.get("message", "Test failed"),
                        "expected": test_result.get("expected"),
                        "actual": test_result.get("actual"),
                        "hint": self._generate_test_hint(test_name, test_result)
                    }
            new_state.code_context["detailed_test_results"] = detailed_test_results
        
        # Update the description to include more guidance
        new_state.description = self._generate_guided_description(
            new_state, feedback.issues, feedback.suggestions
        )
        
        # Update adaptation vector to guide future evolution
        new_state.adaptation_vector = self._calculate_adaptation_vector(
            solution, result, feedback
        )
        
        return new_state
    
    def _select_evolution_strategy(self, solution: str, result: EvaluationResult, feedback: Feedback) -> str:
        """
        Select an evolution strategy based on the current state and solution.
        
        Args:
            solution: The current solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            The selected evolution strategy
        """
        available_strategies = self.evolution_strategies.copy()
        
        # Weight the strategies based on the current state
        weights = {}
        
        # Prefer adding subtle bugs if the solution is very good
        if result.score > 0.95:
            weights["add_subtle_bug"] = 0.6
            weights["change_requirements"] = 0.3
            weights["increase_complexity"] = 0.1
        
        # Prefer changing requirements if we've already added several bugs
        elif self.state.evolution_stage >= 2 and "bug_count" in self.state.code_context and self.state.code_context["bug_count"] >= 3:
            weights["add_subtle_bug"] = 0.1
            weights["change_requirements"] = 0.7
            weights["increase_complexity"] = 0.2
            
        # Prefer increasing complexity if the solution is good but not perfect
        elif result.score > 0.85:
            weights["add_subtle_bug"] = 0.2
            weights["change_requirements"] = 0.2
            weights["increase_complexity"] = 0.6
            
        # Default to equal weights
        else:
            weights = {strategy: 1.0 / len(available_strategies) 
                      for strategy in available_strategies}
        
        # Normalize weights for available strategies
        total_weight = sum(weights.get(strategy, 0) for strategy in available_strategies)
        normalized_weights = [weights.get(strategy, 0) / total_weight 
                             for strategy in available_strategies]
        
        # Select a strategy based on weights
        return random.choices(available_strategies, weights=normalized_weights)[0]
    
    def _add_subtle_bug(self, state: ProblemState, solution: str) -> None:
        """
        Add a subtle bug to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Parse the solution to find potential bug insertion points
        try:
            parsed_solution = ast.parse(solution)
        except SyntaxError:
            # If we can't parse the solution, just add a syntax error
            self._add_syntax_error(state, solution)
            return
        
        # Choose a bug category based on available categories
        available_categories = list(self.bug_categories)
        if available_categories:
            bug_category = random.choice(available_categories)
        else:
            bug_category = BugCategory.LOGICAL
        
        # Add a bug based on the selected category
        if bug_category == BugCategory.SYNTAX:
            self._add_syntax_error(state, solution)
        elif bug_category == BugCategory.LOGICAL:
            self._add_logical_error(state, solution, parsed_solution)
        elif bug_category == BugCategory.PERFORMANCE:
            self._add_performance_issue(state, solution, parsed_solution)
        elif bug_category == BugCategory.EDGE_CASE:
            self._add_edge_case_issue(state, solution, parsed_solution)
        else:
            # Default to logical error
            self._add_logical_error(state, solution, parsed_solution)
        
        # Update bug count in code context
        if "bug_count" not in state.code_context:
            state.code_context["bug_count"] = 0
        state.code_context["bug_count"] += 1
        
        # Add the bug category to the context
        if "bug_categories" not in state.code_context:
            state.code_context["bug_categories"] = []
        state.code_context["bug_categories"].append(bug_category)
    
    def _change_requirements(self, state: ProblemState, solution: str) -> None:
        """
        Change the requirements to challenge the current solution.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Get the current requirements
        requirements = state.requirements
        
        # Add a new requirement
        new_requirement = self._generate_new_requirement(state, solution)
        if new_requirement:
            requirements.append(new_requirement)
        
        # Modify an existing requirement if possible
        if requirements and random.random() < 0.5:
            idx = random.randint(0, len(requirements) - 1)
            requirements[idx] = self._modify_requirement(requirements[idx], state, solution)
    
    def _increase_complexity(self, state: ProblemState, solution: str) -> None:
        """
        Increase the complexity of the task.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Parse the solution if possible
        try:
            parsed_solution = ast.parse(solution)
        except SyntaxError:
            # If we can't parse the solution, make a simpler change
            self._add_edge_case_requirement(state)
            return
        
        # Choose a complexity increase strategy
        strategies = [
            "add_edge_cases",
            "increase_data_volume",
            "add_performance_constraint",
            "expand_functionality"
        ]
        
        strategy = random.choice(strategies)
        
        if strategy == "add_edge_cases":
            self._add_edge_case_requirement(state)
        elif strategy == "increase_data_volume":
            self._increase_data_volume(state, solution)
        elif strategy == "add_performance_constraint":
            self._add_performance_constraint(state, solution)
        elif strategy == "expand_functionality":
            self._expand_functionality(state, solution)
    
    def _create_test_files(self, temp_path: Path) -> List[Path]:
        """
        Create test files based on the current problem state.
        
        Args:
            temp_path: The temporary directory path
            
        Returns:
            List of test file paths
        """
        test_files = []
        
        # Create test files from the code context
        if "tests" in self.state.code_context:
            for i, test in enumerate(self.state.code_context["tests"]):
                test_file = temp_path / f"test_{i}.py"
                with open(test_file, "w") as f:
                    f.write(test["content"])
                test_files.append(test_file)
        
        # Create a default test file if no tests are specified
        if not test_files:
            test_file = temp_path / "test_default.py"
            with open(test_file, "w") as f:
                f.write(self._generate_default_test())
            test_files.append(test_file)
        
        return test_files
    
    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate a score based on test results.
        
        Args:
            results: The test results
            
        Returns:
            A score between 0 and 1
        """
        # Base score on test results
        if results["total_tests"] == 0:
            test_score = 0.0
        else:
            test_score = results["passed_tests"] / results["total_tests"]
        
        # Adjust for execution success
        execution_score = 1.0 if results["execution"]["success"] else 0.0
        
        # Combine scores with weights
        weights = self.config.get("score_weights", {"test": 0.7, "execution": 0.3})
        score = (test_score * weights["test"] + execution_score * weights["execution"])
        
        # Apply difficulty modifier
        difficulty_modifier = 1.0 + (self.state.difficulty * 0.2)
        score = score / difficulty_modifier
        
        return max(0.0, min(1.0, score))
    
    def _calculate_complexity(self, code: str) -> float:
        """
        Calculate the complexity of code.
        
        Args:
            code: The code to analyze
            
        Returns:
            A complexity score
        """
        # Simple cyclomatic complexity estimation
        complexity = 1
        
        # Count control flow statements
        for pattern in ["if", "for", "while", "and", "or"]:
            complexity += code.count(f" {pattern} ")
        
        # Count function definitions
        complexity += code.count("def ")
        
        # Normalize to 0-1 range
        normalized = min(1.0, complexity / 50.0)
        
        return normalized
    
    def _generate_suggestion_for_test_failure(
        self,
        issue: Dict[str, Any],
        solution: str,
        test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a suggestion for a test failure.
        
        Args:
            issue: The issue data
            solution: The solution code
            test_results: The test results
            
        Returns:
            A suggestion dictionary
        """
        test_name = issue["test"]
        test_result = test_results[test_name]
        
        # Extract relevant parts of the test
        test_content = None
        for test in self.state.code_context.get("tests", []):
            if test.get("name") == test_name:
                test_content = test.get("content")
                break
        
        if test_content:
            # Try to extract the assertion that failed
            assertion_match = re.search(r"assert.*", test_content)
            assertion = assertion_match.group(0) if assertion_match else None
            
            # Look for function names in both test and solution
            test_funcs = re.findall(r"def\s+(\w+)", test_content)
            solution_funcs = re.findall(r"def\s+(\w+)", solution)
            
            # Find functions in test that aren't in solution
            missing_funcs = [f for f in test_funcs if f not in solution_funcs]
            
            if missing_funcs:
                return {
                    "type": "missing_function",
                    "message": f"Implement the missing function(s): {', '.join(missing_funcs)}",
                    "functions": missing_funcs
                }
            elif assertion:
                return {
                    "type": "fix_assertion_failure",
                    "message": f"Fix the code to pass the assertion: {assertion}",
                    "assertion": assertion,
                    "expected": test_result.get("expected"),
                    "actual": test_result.get("actual")
                }
            else:
                return {
                    "type": "fix_test_failure",
                    "message": f"Fix the code to pass the test: {test_name}",
                    "test_name": test_name
                }
        else:
            return {
                "type": "general_fix",
                "message": f"Fix the code to pass the failing test: {test_name}"
            }
    
    def _generate_suggestion_for_error(
        self,
        issue: Dict[str, Any],
        solution: str
    ) -> Dict[str, Any]:
        """
        Generate a suggestion for an error.
        
        Args:
            issue: The issue data
            solution: The solution code
            
        Returns:
            A suggestion dictionary
        """
        error_type = issue["error_type"]
        message = issue["message"]
        location = issue.get("location")
        
        if error_type == "syntax":
            return {
                "type": "fix_syntax",
                "message": f"Fix the syntax error: {message}",
                "location": location
            }
        elif error_type == "runtime":
            return {
                "type": "fix_runtime_error",
                "message": f"Fix the runtime error: {message}",
                "location": location
            }
        else:
            return {
                "type": "fix_error",
                "message": f"Fix the error: {message}",
                "error_type": error_type,
                "location": location
            }
    
    def _determine_focus_areas(
        self,
        issues: List[Dict[str, Any]],
        solution: str,
        result: EvaluationResult
    ) -> List[str]:
        """
        Determine focus areas based on issues and results.
        
        Args:
            issues: The identified issues
            solution: The solution code
            result: The evaluation results
            
        Returns:
            List of focus areas
        """
        focus_areas = []
        
        # Check for syntax issues
        syntax_issues = [i for i in issues if i.get("error_type") == "syntax"]
        if syntax_issues:
            focus_areas.append("syntax")
        
        # Check for failing tests
        test_issues = [i for i in issues if i["type"] == "test_failure"]
        if test_issues:
            if any("expected" in i and "actual" in i for i in test_issues):
                focus_areas.append("logic")
            else:
                focus_areas.append("functionality")
        
        # Check for performance issues
        if result.metrics and "execution_time" in result.metrics:
            if result.metrics["execution_time"] > self.config.get("performance_threshold", 1.0):
                focus_areas.append("performance")
        
        # Check for complexity issues
        if result.metrics and "code_complexity" in result.metrics:
            if result.metrics["code_complexity"] > self.config.get("complexity_threshold", 0.7):
                focus_areas.append("complexity")
        
        # Default focus area if none were identified
        if not focus_areas:
            focus_areas.append("general")
        
        return focus_areas
    
    def _generate_adaptation_hints(
        self,
        solution: str,
        result: EvaluationResult
    ) -> List[Dict[str, Any]]:
        """
        Generate hints about how the problem might adapt in the next iteration.
        
        Args:
            solution: The solution code
            result: The evaluation results
            
        Returns:
            List of adaptation hints
        """
        hints = []
        
        # Hint about potential complexity increases
        if result.score > 0.8:
            hints.append({
                "type": "complexity_increase",
                "message": "The problem may become more complex in the next iteration."
            })
        
        # Hint about potential requirement changes
        if result.score > 0.9 and self.state.evolution_stage >= 1:
            hints.append({
                "type": "requirement_change",
                "message": "The requirements may change in the next iteration."
            })
        
        # Hint about potential bug additions
        if result.score > 0.95:
            hints.append({
                "type": "new_bugs",
                "message": "New, more subtle bugs may be introduced in the next iteration."
            })
        
        # Hint about focus on specific areas
        if result.score > 0.7 and result.score < 0.95:
            focus_areas = result.metrics.get("focus_areas", [])
            if focus_areas:
                hints.append({
                    "type": "focus_shift",
                    "message": f"The next iteration may focus more on: {', '.join(focus_areas)}",
                    "areas": focus_areas
                })
        
        return hints
    
    def _generate_description(self, state: ProblemState) -> str:
        """
        Generate a description for the current problem state.
        
        Args:
            state: The problem state
            
        Returns:
            A descriptive prompt for the problem
        """
        # Base description
        base_desc = (
            f"Fix the bug(s) in the following code. "
            f"This is iteration {state.evolution_stage + 1} of the task."
        )
        
        # Add information about known bug categories
        if "bug_categories" in state.code_context:
            categories = state.code_context["bug_categories"]
            if categories:
                base_desc += f"\n\nThe code contains the following types of issues: {', '.join(categories)}."
        
        # Add requirements
        if state.requirements:
            base_desc += "\n\nRequirements:"
            for i, req in enumerate(state.requirements):
                base_desc += f"\n{i+1}. {req['description']}"
                
        # Add information about difficulty
        difficulty_desc = "easy"
        if state.difficulty > 0.3 and state.difficulty <= 0.6:
            difficulty_desc = "moderate"
        elif state.difficulty > 0.6 and state.difficulty <= 0.8:
            difficulty_desc = "challenging"
        elif state.difficulty > 0.8:
            difficulty_desc = "very challenging"
        
        base_desc += f"\n\nThis is a {difficulty_desc} bug fixing task."
        
        return base_desc
    
    def _generate_focused_description(self, state: ProblemState, issues: List[Dict[str, Any]]) -> str:
        """
        Generate a description focused on remaining issues.
        
        Args:
            state: The problem state
            issues: The identified issues
            
        Returns:
            A descriptive prompt focused on remaining issues
        """
        base_desc = self._generate_description(state)
        
        # Add focus on remaining issues
        if issues:
            base_desc += "\n\nFocus on the following issues:"
            for i, issue in enumerate(issues):
                if issue["type"] == "test_failure":
                    base_desc += f"\n{i+1}. Test failure in '{issue['test']}': {issue['message']}"
                else:
                    base_desc += f"\n{i+1}. {issue['error_type']} error: {issue['message']}"
        
        # Add focus areas if present
        if "focus_areas" in state.code_context:
            areas = state.code_context["focus_areas"]
            if areas:
                base_desc += f"\n\nPay particular attention to: {', '.join(areas)}."
        
        return base_desc
    
    def _generate_guided_description(
        self,
        state: ProblemState,
        issues: List[Dict[str, Any]],
        suggestions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a description with added guidance.
        
        Args:
            state: The problem state
            issues: The identified issues
            suggestions: The suggested fixes
            
        Returns:
            A descriptive prompt with added guidance
        """
        base_desc = self._generate_description(state)
        
        # Add detailed information about issues
        if issues:
            base_desc += "\n\nThe following issues were identified in your previous solution:"
            for i, issue in enumerate(issues):
                if issue["type"] == "test_failure":
                    base_desc += f"\n{i+1}. Test failure in '{issue['test']}': {issue['message']}"
                    if "expected" in issue and "actual" in issue:
                        base_desc += f"\n   Expected: {issue['expected']}"
                        base_desc += f"\n   Actual: {issue['actual']}"
                else:
                    base_desc += f"\n{i+1}. {issue['error_type']} error: {issue['message']}"
                    if "location" in issue:
                        base_desc += f"\n   Location: {issue['location']}"
        
        # Add suggestions
        if suggestions:
            base_desc += "\n\nConsider the following suggestions:"
            for i, suggestion in enumerate(suggestions):
                base_desc += f"\n{i+1}. {suggestion['message']}"
        
        # Add hints if present
        if "hints" in state.code_context:
            hints = state.code_context["hints"]
            if hints:
                base_desc += "\n\nHints:"
                for i, hint in enumerate(hints):
                    base_desc += f"\n{i+1}. {hint}"
        
        return base_desc
    
    def _generate_hints(
        self,
        solution: str,
        result: EvaluationResult,
        feedback: Feedback
    ) -> List[str]:
        """
        Generate hints based on the solution and feedback.
        
        Args:
            solution: The solution code
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            List of hints
        """
        hints = []
        
        # Add hints based on failing tests
        if result.test_results:
            failing_tests = [
                test_name for test_name, test_result in result.test_results.items()
                if not test_result["passed"]
            ]
            
            if failing_tests:
                test_hint = "Focus on fixing the failing tests"
                
                # Add specific information about test expectations if available
                for test_name in failing_tests[:2]:  # Limit to first two tests
                    test_result = result.test_results[test_name]
                    if "expected" in test_result and "actual" in test_result:
                        test_hint += f". For test '{test_name}', expected '{test_result['expected']}' but got '{test_result['actual']}'"
                
                hints.append(test_hint + ".")
        
        # Add hints based on errors
        if result.error_details:
            for error_type, error_info in result.error_details.items():
                hints.append(f"Fix the {error_type} error: {error_info.get('message', 'Unknown error')}.")
        
        # Add hints based on focus areas
        for area in feedback.focus_areas:
            if area == "syntax":
                hints.append("Check your syntax carefully, especially parentheses, indentation, and function definitions.")
            elif area == "logic":
                hints.append("Review the logic of your solution, especially conditional statements and loop conditions.")
            elif area == "functionality":
                hints.append("Ensure your solution implements all required functionality specified in the tests.")
            elif area == "performance":
                hints.append("Consider optimizing your solution for better performance, avoid unnecessary operations.")
            elif area == "complexity":
                hints.append("Try to simplify your solution, it may be more complex than necessary.")
        
        return hints
    
    def _generate_test_hint(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """
        Generate a hint for a specific failing test.
        
        Args:
            test_name: The name of the test
            test_result: The test result
            
        Returns:
            A hint for the test
        """
        if "expected" in test_result and "actual" in test_result:
            return f"The test expected '{test_result['expected']}' but got '{test_result['actual']}'"
        elif "message" in test_result:
            return test_result["message"]
        else:
            return "The test failed, but no detailed information is available."
    
    def _add_syntax_error(self, state: ProblemState, solution: str) -> None:
        """
        Add a syntax error to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        lines = solution.split('\n')
        if not lines:
            return
        
        # Choose a line to modify
        idx = random.randint(0, len(lines) - 1)
        line = lines[idx]
        
        # Skip empty lines or comment lines
        while not line.strip() or line.strip().startswith('#'):
            idx = random.randint(0, len(lines) - 1)
            line = lines[idx]
        
        # Choose a modification type
        mod_type = random.choice([
            "remove_character",
            "add_character",
            "swap_characters",
            "change_indent"
        ])
        
        if mod_type == "remove_character" and line:
            char_idx = random.randint(0, len(line) - 1)
            lines[idx] = line[:char_idx] + line[char_idx+1:]
        
        elif mod_type == "add_character":
            char_idx = random.randint(0, len(line))
            char = random.choice(["(", ")", "{", "}", "[", "]", ":", ";", ",", "."])
            lines[idx] = line[:char_idx] + char + line[char_idx:]
        
        elif mod_type == "swap_characters" and len(line) >= 2:
            char_idx = random.randint(0, len(line) - 2)
            lines[idx] = (line[:char_idx] + line[char_idx+1] + 
                         line[char_idx] + line[char_idx+2:])
        
        elif mod_type == "change_indent":
            # Either add or remove indentation
            if line.startswith("    "):
                lines[idx] = line[2:]  # Remove some indent
            else:
                lines[idx] = "  " + line  # Add inconsistent indent
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "syntax",
            "line": idx + 1,
            "description": f"Syntax error introduced in line {idx + 1}"
        })
    
    def _add_logical_error(self, state: ProblemState, solution: str, parsed_solution: ast.Module) -> None:
        """
        Add a logical error to the solution code.
        
        Args:
            state: The problem state to modify
            solution: The current solution
            parsed_solution: The parsed AST of the solution
        """
        modification_types = [
            "change_comparison",
            "invert_condition",
            "off_by_one",
            "change_operator",
            "reverse_logic"
        ]
        
        mod_type = random.choice(modification_types)
        lines = solution.split('\n')
        
        # Find all if statements and loops
        if_statements = []
        for i, line in enumerate(lines):
            if re.search(r'\bif\b|\bwhile\b|\bfor\b', line):
                if_statements.append((i, line))
        
        if if_statements:
            # Choose an if statement to modify
            idx, line = random.choice(if_                          
