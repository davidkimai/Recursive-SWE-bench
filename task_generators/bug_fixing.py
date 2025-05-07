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
# recursive_swe_bench/task_generators/bug_fixing.py (continued)

        if if_statements:
            # Choose an if statement to modify
            idx, line = random.choice(if_statements)
            
            if mod_type == "change_comparison":
                # Change comparison operators
                comparisons = {"==": "!=", "!=": "==", ">": "<", "<": ">", ">=": "<=", "<=": ">="}
                for op, new_op in comparisons.items():
                    if op in line:
                        lines[idx] = line.replace(op, new_op, 1)
                        break
            
            elif mod_type == "invert_condition":
                # Add or remove a "not" to invert the condition
                if "not" in line:
                    lines[idx] = line.replace("not ", "", 1)
                else:
                    match = re.search(r'(if|while)\s+([^:]+):', line)
                    if match:
                        condition = match.group(2)
                        lines[idx] = line.replace(condition, f"not ({condition})", 1)
            
            elif mod_type == "off_by_one":
                # Introduce an off-by-one error
                for op in ["+", "-"]:
                    if op in line:
                        # If there's a number after the operator, change it
                        match = re.search(f'\\{op}\\s*(\\d+)', line)
                        if match:
                            num = int(match.group(1))
                            new_num = num + 1 if op == "+" else max(0, num - 1)
                            lines[idx] = line.replace(f"{op} {num}", f"{op} {new_num}", 1)
                            break
            
            elif mod_type == "change_operator":
                # Change arithmetic or logical operators
                operators = {"+": "-", "-": "+", "*": "/", "/": "*", "and": "or", "or": "and"}
                for op, new_op in operators.items():
                    if f" {op} " in line:
                        lines[idx] = line.replace(f" {op} ", f" {new_op} ", 1)
                        break
            
            elif mod_type == "reverse_logic":
                # Reverse the logic of a compound condition
                if " and " in line:
                    parts = line.split(" and ")
                    lines[idx] = line.replace(" and ".join(parts), " or ".join(parts), 1)
                elif " or " in line:
                    parts = line.split(" or ")
                    lines[idx] = line.replace(" or ".join(parts), " and ".join(parts), 1)
        
        else:
            # If no if statements found, introduce a different kind of logical error
            # Find variable assignments
            assignments = []
            for i, line in enumerate(lines):
                if "=" in line and "==" not in line and "!=" not in line:
                    assignments.append((i, line))
            
            if assignments:
                # Choose an assignment to modify
                idx, line = random.choice(assignments)
                
                # Modify the assignment
                if "+" in line:
                    lines[idx] = line.replace("+", "-", 1)
                elif "-" in line:
                    lines[idx] = line.replace("-", "+", 1)
                elif "*" in line:
                    lines[idx] = line.replace("*", "/", 1)
                elif "/" in line:
                    lines[idx] = line.replace("/", "*", 1)
                else:
                    # If no arithmetic operator, change the value
                    match = re.search(r'=\s*(\d+)', line)
                    if match:
                        num = int(match.group(1))
                        new_num = num + random.choice([-1, 1]) * random.randint(1, 3)
                        lines[idx] = line.replace(f"= {num}", f"= {new_num}", 1)
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "logical",
            "line": idx + 1,
            "description": f"Logical error introduced in line {idx + 1}: {mod_type}"
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
        
        # Find loops in the code
        loops = []
        for i, line in enumerate(lines):
            if re.search(r'\bfor\b|\bwhile\b', line):
                loops.append((i, line))
        
        if loops:
            # Choose a loop to modify
            idx, line = random.choice(loops)
            
            # Choose a modification type
            mod_type = random.choice([
                "add_nested_loop",
                "replace_efficient_operation",
                "add_redundant_computation"
            ])
            
            if mod_type == "add_nested_loop":
                # Add a nested loop
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                loop_body_indent = indent_str + '    '
                
                # Find the next line with the same indentation or less
                end_idx = idx + 1
                while end_idx < len(lines) and (not lines[end_idx].strip() or len(lines[end_idx]) - len(lines[end_idx].lstrip()) > indent):
                    end_idx += 1
                
                # Insert a nested loop before the end of the current loop
                insert_pos = end_idx
                lines.insert(insert_pos, f"{loop_body_indent}for _ in range(100):  # Unnecessary loop")
                lines.insert(insert_pos + 1, f"{loop_body_indent}    pass")
            
            elif mod_type == "replace_efficient_operation":
                # Replace an efficient operation with a less efficient one
                # Look for list comprehensions or efficient operations
                for i in range(idx + 1, min(idx + 10, len(lines))):
                    if "append" in lines[i] or "extend" in lines[i]:
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        indent_str = ' ' * indent
                        match = re.search(r'(\w+)\.(append|extend)', lines[i])
                        if match:
                            list_name = match.group(1)
                            operation = match.group(2)
                            item = lines[i].split(f"{list_name}.{operation}(")[1].split(")")[0]
                            
                            if operation == "append":
                                # Replace append with concatenation
                                lines[i] = f"{indent_str}{list_name} = {list_name} + [{item}]  # Less efficient than append"
                            elif operation == "extend":
                                # Replace extend with concatenation
                                lines[i] = f"{indent_str}{list_name} = {list_name} + {item}  # Less efficient than extend"
                            break
            
            elif mod_type == "add_redundant_computation":
                # Add redundant computation inside the loop
                # Find the indentation level of the loop body
                if idx + 1 < len(lines):
                    body_indent = len(lines[idx + 1]) - len(lines[idx + 1].lstrip())
                    body_indent_str = ' ' * body_indent
                    
                    # Add redundant computation
                    lines.insert(idx + 1, f"{body_indent_str}temp = []  # Redundant computation")
                    lines.insert(idx + 2, f"{body_indent_str}for i in range(1000):")
                    lines.insert(idx + 3, f"{body_indent_str}    temp.append(i)")
                    lines.insert(idx + 4, f"{body_indent_str}    temp.sort()  # Unnecessary sort in each iteration")
        
        else:
            # If no loops found, introduce inefficient data structure or algorithm
            function_defs = []
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    function_defs.append((i, line))
            
            if function_defs:
                # Choose a function to modify
                idx, line = random.choice(function_defs)
                
                # Find the indentation level of the function body
                if idx + 1 < len(lines):
                    body_indent = len(lines[idx + 1]) - len(lines[idx + 1].lstrip())
                    body_indent_str = ' ' * body_indent
                    
                    # Add inefficient code at the beginning of the function
                    lines.insert(idx + 1, f"{body_indent_str}# Inefficient data structure usage")
                    lines.insert(idx + 2, f"{body_indent_str}data = []")
                    lines.insert(idx + 3, f"{body_indent_str}for i in range(1000):")
                    lines.insert(idx + 4, f"{body_indent_str}    data.append(i)")
                    lines.insert(idx + 5, f"{body_indent_str}    # Inefficient search operation")
                    lines.insert(idx + 6, f"{body_indent_str}    if i in data:  # Linear search instead of using a set")
                    lines.insert(idx + 7, f"{body_indent_str}        pass")
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "performance",
            "line": idx + 1,
            "description": f"Performance issue introduced around line {idx + 1}"
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
        
        # Find functions in the code
        functions = []
        current_func = None
        func_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if current_func:
                    functions.append((func_start, i - 1, current_func))
                current_func = line.strip()[4:].split("(")[0]
                func_start = i
            elif i == len(lines) - 1 and current_func:
                functions.append((func_start, i, current_func))
        
        if functions:
            # Choose a function to modify
            start_idx, end_idx, func_name = random.choice(functions)
            
            # Choose a modification type
            mod_type = random.choice([
                "remove_boundary_check",
                "introduce_zero_division",
                "handling_empty_input",
                "type_assumption"
            ])
            
            if mod_type == "remove_boundary_check":
                # Find and remove or modify boundary checks
                for i in range(start_idx, end_idx + 1):
                    if re.search(r'if\s+.*(?:len|count|size|length|empty|<=|>=|<|>|\!=)', lines[i]):
                        # Comment out the boundary check
                        lines[i] = f"# {lines[i]}  # Boundary check removed"
                        # Skip the body of the if statement
                        j = i + 1
                        indent = len(lines[i]) - len(lines[i].lstrip())
                        body_indent = indent + 4
                        while j <= end_idx and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) >= body_indent):
                            lines[j] = f"# {lines[j]}"
                            j += 1
                        break
            
            elif mod_type == "introduce_zero_division":
                # Find division operations and modify them
                for i in range(start_idx, end_idx + 1):
                    if "/" in lines[i] and "try" not in lines[i] and "except" not in lines[i]:
                        # Remove denominator check if it exists
                        if re.search(r'if\s+.*(?:!=\s*0|>\s*0)', lines[i]):
                            lines[i] = f"# {lines[i]}  # Denominator check removed"
                        else:
                            # Or modify a division to potentially cause zero division
                            match = re.search(r'(\w+)\s*/\s*(\w+)', lines[i])
                            if match:
                                denominator = match.group(2)
                                # Add a potential zero value for the denominator
                                indent = len(lines[i]) - len(lines[i].lstrip())
                                indent_str = ' ' * indent
                                lines.insert(i, f"{indent_str}if random.random() < 0.1:  # Introduce potential zero division")
                                lines.insert(i + 1, f"{indent_str}    {denominator} = 0")
                                break
            
            elif mod_type == "handling_empty_input":
                # Modify parameter handling to not handle empty inputs correctly
                params = re.search(r'def\s+\w+\s*\((.*?)\)', lines[start_idx])
                if params and params.group(1):
                    param_list = [p.strip() for p in params.group(1).split(",")]
                    if param_list:
                        param = param_list[0].split("=")[0].strip()
                        # Find checks for the parameter
                        for i in range(start_idx + 1, end_idx + 1):
                            if re.search(rf'if\s+.*(?:not\s+{param}|len\s*\(\s*{param}\s*\)\s*==\s*0)', lines[i]):
                                # Comment out the empty check
                                lines[i] = f"# {lines[i]}  # Empty input check removed"
                                # Skip the body of the if statement
                                j = i + 1
                                indent = len(lines[i]) - len(lines[i].lstrip())
                                body_indent = indent + 4
                                while j <= end_idx and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) >= body_indent):
                                    lines[j] = f"# {lines[j]}"
                                    j += 1
                                break
            
            elif mod_type == "type_assumption":
                # Introduce assumptions about parameter types
                params = re.search(r'def\s+\w+\s*\((.*?)\)', lines[start_idx])
                if params and params.group(1):
                    param_list = [p.strip() for p in params.group(1).split(",")]
                    if param_list:
                        param = param_list[0].split("=")[0].strip()
                        # Find type checks for the parameter
                        type_check_found = False
                        for i in range(start_idx + 1, end_idx + 1):
                            if re.search(rf'(?:isinstance|type)\s*\(\s*{param}\s*,', lines[i]):
                                # Comment out the type check
                                lines[i] = f"# {lines[i]}  # Type check removed"
                                type_check_found = True
                                break
                        
                        if not type_check_found:
                            # Add a problematic type assumption
                            indent = 4  # Assume basic indentation
                            for i in range(start_idx + 1, min(start_idx + 5, end_idx + 1)):
                                if lines[i].strip():
                                    indent = len(lines[i]) - len(lines[i].lstrip())
                                    break
                            
                            indent_str = ' ' * indent
                            # Add code that assumes a specific type
                            lines.insert(start_idx + 1, f"{indent_str}# Assuming {param} is a specific type without checking")
                            lines.insert(start_idx + 2, f"{indent_str}{param}_length = len({param})  # Will fail if {param} doesn't support len()")
        
        # Update the code
        modified_code = '\n'.join(lines)
        state.code_context["code"] = modified_code
        
        # Add information about the modification
        if "bugs" not in state.code_context:
            state.code_context["bugs"] = []
        
        state.code_context["bugs"].append({
            "type": "edge_case",
            "line": start_idx + 1,
            "description": f"Edge case issue introduced in function '{func_name}': {mod_type}"
        })
    
    def _generate_new_requirement(self, state: ProblemState, solution: str) -> Dict[str, Any]:
        """
        Generate a new requirement based on the current state and solution.
        
        Args:
            state: The current problem state
            solution: The current solution
            
        Returns:
            A new requirement dictionary
        """
        # Parse the solution to find functions and variables
        function_names = re.findall(r'def\s+(\w+)', solution)
        variable_names = re.findall(r'(\w+)\s*=', solution)
        
        # Choose a requirement type
        req_type = random.choice([
            "edge_case_handling",
            "performance_improvement",
            "error_handling",
            "type_checking",
            "feature_addition"
        ])
        
        if req_type == "edge_case_handling":
            if function_names:
                func_name = random.choice(function_names)
                edge_cases = [
                    "empty input",
                    "negative values",
                    "zero values",
                    "extremely large values",
                    "special characters",
                    "duplicate values"
                ]
                edge_case = random.choice(edge_cases)
                return {
                    "type": "edge_case_handling",
                    "description": f"The function '{func_name}' should handle {edge_case} correctly.",
                    "difficulty": random.uniform(0.3, 0.7)
                }
            
        elif req_type == "performance_improvement":
            return {
                "type": "performance_improvement",
                "description": "The solution should be optimized to run in O(n) time or better.",
                "difficulty": random.uniform(0.4, 0.8)
            }
            
        elif req_type == "error_handling":
            error_types = [
                "invalid input",
                "division by zero",
                "file not found",
                "network timeout",
                "permission denied"
            ]
            error_type = random.choice(error_types)
            return {
                "type": "error_handling",
                "description": f"The code should handle {error_type} errors gracefully.",
                "difficulty": random.uniform(0.2, 0.6)
            }
            
        elif req_type == "type_checking":
            if function_names:
                func_name = random.choice(function_names)
                return {
                    "type": "type_checking",
                    "description": f"The function '{func_name}' should validate input types before processing.",
                    "difficulty": random.uniform(0.1, 0.5)
                }
            
        elif req_type == "feature_addition":
            features = [
                "logging capability",
                "progress tracking",
                "caching for repeated operations",
                "parameter validation",
                "configuration options"
            ]
            feature = random.choice(features)
            return {
                "type": "feature_addition",
                "description": f"Add {feature} to the solution.",
                "difficulty": random.uniform(0.3, 0.7)
            }
        
        # Default requirement if none of the above were applicable
        return {
            "type": "general_improvement",
            "description": "Improve the overall code quality and readability.",
            "difficulty": random.uniform(0.1, 0.4)
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
        # Make a copy of the requirement
        modified_req = copy.deepcopy(requirement)
        
        # Increase the difficulty
        modified_req["difficulty"] = min(1.0, requirement.get("difficulty", 0.3) + random.uniform(0.1, 0.3))
        
        # Modify the description based on the requirement type
        if requirement["type"] == "edge_case_handling":
            modified_req["description"] += " Additionally, it should handle very large inputs efficiently."
        
        elif requirement["type"] == "performance_improvement":
            modified_req["description"] = modified_req["description"].replace("O(n)", "O(log n)")
        
        elif requirement["type"] == "error_handling":
            modified_req["description"] += " And provide detailed error messages for debugging."
        
        elif requirement["type"] == "type_checking":
            modified_req["description"] += " And automatically convert types when possible."
        
        elif requirement["type"] == "feature_addition":
            modified_req["description"] += " Ensure this feature is configurable via parameters."
        
        else:
            modified_req["description"] += " The code should also be well-documented with comments."
        
        return modified_req
    
    def _add_edge_case_requirement(self, state: ProblemState) -> None:
        """
        Add a requirement for handling edge cases.
        
        Args:
            state: The problem state to modify
        """
        edge_cases = [
            "empty collections",
            "null/None values",
            "boundary values (min/max)",
            "negative numbers",
            "special characters",
            "Unicode characters",
            "very large inputs",
            "malformed input"
        ]
        
        edge_case = random.choice(edge_cases)
        
        # Add a new requirement
        state.requirements.append({
            "type": "edge_case_handling",
            "description": f"The solution must correctly handle {edge_case}.",
            "difficulty": random.uniform(0.3, 0.7)
        })
        
        # Add test cases for the edge case if tests exist
        if "tests" in state.code_context:
            # Create a new test for the edge case
            test_template = self._generate_edge_case_test(edge_case, state.code_context)
            if test_template:
                state.code_context["tests"].append({
                    "name": f"test_edge_case_{len(state.code_context['tests'])}",
                    "content": test_template,
                    "description": f"Test handling of {edge_case}"
                })
    
    def _increase_data_volume(self, state: ProblemState, solution: str) -> None:
        """
        Modify the problem to require handling larger data volumes.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Add a requirement for handling large data
        state.requirements.append({
            "type": "scalability",
            "description": "The solution must efficiently handle large datasets (10,000+ items).",
            "difficulty": random.uniform(0.5, 0.8)
        })
        
        # Modify existing tests to use larger data if tests exist
        if "tests" in state.code_context:
            for i, test in enumerate(state.code_context["tests"]):
                content = test["content"]
                
                # Look for small lists or arrays in tests
                for pattern, replacement in [
                    (r'\[[^\]]{0,50}\]', '[random.randint(0, 1000) for _ in range(10000)]'),
                    (r'range\(\d+\)', 'range(10000)'),
                    (r'"[^"]{0,20}"', '"' + 'a' * 10000 + '"')
                ]:
                    match = re.search(pattern, content)
                    if match and random.random() < 0.3:  # Only replace some instances
                        content = content.replace(match.group(0), replacement, 1)
                        break
                
                state.code_context["tests"][i]["content"] = content
                state.code_context["tests"][i]["description"] = f"{test.get('description', 'Test')} (with large data)"
    
    def _add_performance_constraint(self, state: ProblemState, solution: str) -> None:
        """
        Add a performance constraint to the problem.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Choose a performance constraint
        constraints = [
            "linear time complexity (O(n))",
            "logarithmic time complexity (O(log n))",
            "constant memory usage (O(1) space)",
            "execution time under 100ms for large inputs",
            "minimal function calls"
        ]
        
        constraint = random.choice(constraints)
        
        # Add a new requirement
        state.requirements.append({
            "type": "performance",
            "description": f"The solution must achieve {constraint}.",
            "difficulty": random.uniform(0.6, 0.9)
        })
        
        # Add performance testing code if tests exist
        if "tests" in state.code_context:
            # Add a performance test
            perf_test = self._generate_performance_test(constraint, state.code_context)
            if perf_test:
                state.code_context["tests"].append({
                    "name": f"test_performance_{len(state.code_context['tests'])}",
                    "content": perf_test,
                    "description": f"Test {constraint}"
                })
    
    def _expand_functionality(self, state: ProblemState, solution: str) -> None:
        """
        Expand the required functionality of the solution.
        
        Args:
            state: The problem state to modify
            solution: The current solution
        """
        # Choose a functionality expansion
        expansions = [
            "support for different input types",
            "parameterized behavior",
            "additional output formats",
            "flexible error handling",
            "integration with external systems"
        ]
        
        expansion = random.choice(expansions)
        
        # Add a new requirement
        state.requirements.append({
            "type": "functionality",
            "description": f"Expand the solution to include {expansion}.",
            "difficulty": random.uniform(0.4, 0.8)
        })
        
        # Add test cases for the new functionality if tests exist
        if "tests" in state.code_context:
            # Create a new test for the expanded functionality
            test_template = self._generate_functionality_test(expansion, state.code_context)
            if test_template:
                state.code_context["tests"].append({
                    "name": f"test_expanded_functionality_{len(state.code_context['tests'])}",
                    "content": test_template,
                    "description": f"Test {expansion}"
                })
    
    def _generate_default_test(self) -> str:
        """
        Generate a default test based on the current problem state.
        
        Returns:
            A default test script
        """
        # Generate a basic test script
        return """
import unittest
import sys
import os

# Add the directory containing the solution to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the solution
from solution import *

class DefaultTest(unittest.TestCase):
    def test_basic_functionality(self):
        # A basic test that should pass if the solution is correct
        self.assertTrue(True, "Basic assertion failed")
        
    def test_expected_output(self):
        # Test expected output of main functions
        # This will need to be updated based on the specific problem
        pass
        
if __name__ == '__main__':
    unittest.main()
"""
    
    def _generate_edge_case_test(self, edge_case: str, code_context: Dict[str, Any]) -> str:
        """
        Generate a test for an edge case.
        
        Args:
            edge_case: The edge case to test
            code_context: The code context containing information about the problem
            
        Returns:
            A test script for the edge case
        """
        # Extract function names from the code context
        function_names = []
        if "code" in code_context:
            function_names = re.findall(r'def\s+(\w+)', code_context["code"])
        
        if not function_names:
            return None
        
        # Choose a function to test
        function_name = random.choice(function_names)
        
        # Generate test code based on the edge case
        if edge_case == "empty collections":
            return f"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class EmptyCollectionTest(unittest.TestCase):
    def test_empty_input(self):
        # Test with empty list
        result = {function_name}([])
        self.assertIsNotNone(result, "Function should handle empty list")
        
        # Test with empty string
        result = {function_name}("")
        self.assertIsNotNone(result, "Function should handle empty string")
        
        # Test with empty dict
        result = {function_name}({{}})
        self.assertIsNotNone(result, "Function should handle empty dict")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif edge_case == "null/None values":
            return f"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class NoneValueTest(unittest.TestCase):
    def test_none_input(self):
        # Test with None as input
        result = {function_name}(None)
        self.assertIsNotNone(result, "Function should handle None input")
        
        # Test with list containing None
        result = {function_name}([1, None, 3])
        self.assertIsNotNone(result, "Function should handle list with None values")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif edge_case == "boundary values (min/max)":
            return f"""
# recursive_swe_bench/task_generators/bug_fixing.py (completion)

import unittest
import sys
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class BoundaryValueTest(unittest.TestCase):
    def test_min_max_values(self):
        # Test with minimum integer
        min_int = -sys.maxsize - 1
        result = {function_name}(min_int)
        self.assertIsNotNone(result, "Function should handle minimum integer")
        
        # Test with maximum integer
        max_int = sys.maxsize
        result = {function_name}(max_int)
        self.assertIsNotNone(result, "Function should handle maximum integer")
        
        # Test with very large list
        large_list = list(range(10000))
        result = {function_name}(large_list)
        self.assertIsNotNone(result, "Function should handle very large inputs")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif edge_case == "negative numbers":
            return f"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class NegativeNumberTest(unittest.TestCase):
    def test_negative_numbers(self):
        # Test with negative number
        result = {function_name}(-1)
        self.assertIsNotNone(result, "Function should handle negative numbers")
        
        # Test with list of negative numbers
        result = {function_name}([-1, -2, -3])
        self.assertIsNotNone(result, "Function should handle lists of negative numbers")
        
        # Test with mixed positive and negative
        result = {function_name}([-1, 0, 1])
        self.assertIsNotNone(result, "Function should handle mixed positive and negative")
        
if __name__ == '__main__':
    unittest.main()
"""
        else:
            # Generic edge case test
            return f"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class EdgeCaseTest(unittest.TestCase):
    def test_edge_case_{edge_case.replace(' ', '_')}(self):
        # Test edge case: {edge_case}
        # This is a placeholder test that needs to be customized for the specific edge case
        self.assertTrue(True, "Edge case test not implemented")
        
if __name__ == '__main__':
    unittest.main()
"""
    
    def _generate_performance_test(self, constraint: str, code_context: Dict[str, Any]) -> str:
        """
        Generate a performance test based on a constraint.
        
        Args:
            constraint: The performance constraint
            code_context: The code context containing information about the problem
            
        Returns:
            A test script for the performance constraint
        """
        # Extract function names from the code context
        function_names = []
        if "code" in code_context:
            function_names = re.findall(r'def\s+(\w+)', code_context["code"])
        
        if not function_names:
            return None
        
        # Choose a function to test
        function_name = random.choice(function_names)
        
        if "time complexity" in constraint:
            return f"""
import unittest
import sys
import os
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class PerformanceTest(unittest.TestCase):
    def test_time_complexity(self):
        # Test for {constraint}
        sizes = [100, 1000, 10000]
        times = []
        
        for size in sizes:
            # Generate input of the given size
            input_data = [random.randint(0, 1000) for _ in range(size)]
            
            # Measure execution time
            start_time = time.time()
            {function_name}(input_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check if time grows appropriately
        # For O(n), time should grow linearly with input size
        # For O(log n), time should grow logarithmically
        # This is a simplified check and might need adjustment
        if "log n" in "{constraint}":
            # For logarithmic time, the ratio of times should decrease
            ratio1 = times[1] / times[0]
            ratio2 = times[2] / times[1]
            self.assertLess(ratio2, ratio1 * 1.5, 
                           f"Growth rate appears super-logarithmic: {times}")
        else:  # Assume linear or better
            # For linear time, the ratio of times should be roughly equal to ratio of sizes
            ratio1 = times[1] / times[0]
            size_ratio1 = sizes[1] / sizes[0]
            
            ratio2 = times[2] / times[1]
            size_ratio2 = sizes[2] / sizes[1]
            
            self.assertLess(ratio1, size_ratio1 * 1.5, 
                           f"First growth rate appears super-linear: {times}")
            self.assertLess(ratio2, size_ratio2 * 1.5, 
                           f"Second growth rate appears super-linear: {times}")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif "execution time" in constraint:
            return f"""
import unittest
import sys
import os
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class PerformanceTest(unittest.TestCase):
    def test_execution_time(self):
        # Test for {constraint}
        # Generate a large input
        input_data = [random.randint(0, 1000) for _ in range(10000)]
        
        # Measure execution time
        start_time = time.time()
        {function_name}(input_data)
        end_time = time.time()
        
        execution_time = (end_time - start_time) * 1000  # Convert to ms
        
        self.assertLess(execution_time, 100, 
                       f"Execution time exceeded 100ms: {execution_time:.2f}ms")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif "memory usage" in constraint:
            return f"""
import unittest
import sys
import os
import psutil
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class MemoryUsageTest(unittest.TestCase):
    def test_memory_usage(self):
        # Test for {constraint}
        # Note: This is an approximate test and may not be accurate in all environments
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate a large input
        input_data = [random.randint(0, 1000) for _ in range(100000)]
        
        # Run function
        {function_name}(input_data)
        
        # Measure memory after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate memory usage
        memory_used = memory_after - memory_before
        
        # A crude approximation, adjust as needed
        self.assertLess(memory_used, 10, 
                       f"Memory usage seems high: {memory_used:.2f}MB")
        
if __name__ == '__main__':
    unittest.main()
"""
        else:
            # Generic performance test
            return f"""
import unittest
import sys
import os
import time
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class PerformanceTest(unittest.TestCase):
    def test_performance(self):
        # Test for {constraint}
        # This is a placeholder test that needs to be customized for the specific constraint
        
        # Generate a large input
        input_data = [random.randint(0, 1000) for _ in range(10000)]
        
        # Measure execution time
        start_time = time.time()
        {function_name}(input_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Just log the time for now
        print(f"Execution time: {execution_time:.4f} seconds")
        self.assertTrue(True, "Performance test completed")
        
if __name__ == '__main__':
    unittest.main()
"""
    
    def _generate_functionality_test(self, expansion: str, code_context: Dict[str, Any]) -> str:
        """
        Generate a test for expanded functionality.
        
        Args:
            expansion: The functionality expansion
            code_context: The code context containing information about the problem
            
        Returns:
            A test script for the expanded functionality
        """
        # Extract function names from the code context
        function_names = []
        if "code" in code_context:
            function_names = re.findall(r'def\s+(\w+)', code_context["code"])
        
        if not function_names:
            return None
        
        # Choose a function to test
        function_name = random.choice(function_names)
        
        if "different input types" in expansion:
            return f"""
import unittest
import sys
import os
import json
from collections import namedtuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class InputTypesTest(unittest.TestCase):
    def test_different_input_types(self):
        # Test with different types of inputs
        
        # Test with list
        list_input = [1, 2, 3]
        list_result = {function_name}(list_input)
        self.assertIsNotNone(list_result, "Function should handle list input")
        
        # Test with tuple
        tuple_input = (1, 2, 3)
        tuple_result = {function_name}(tuple_input)
        self.assertIsNotNone(tuple_result, "Function should handle tuple input")
        
        # Test with set
        set_input = {{1, 2, 3}}
        set_result = {function_name}(set_input)
        self.assertIsNotNone(set_result, "Function should handle set input")
        
        # Test with dictionary
        dict_input = {{"a": 1, "b": 2, "c": 3}}
        dict_result = {function_name}(dict_input)
        self.assertIsNotNone(dict_result, "Function should handle dictionary input")
        
        # Test with JSON string
        json_input = '{{"data": [1, 2, 3]}}'
        json_result = {function_name}(json_input)
        self.assertIsNotNone(json_result, "Function should handle JSON string")
        
        # Test with custom object
        Point = namedtuple('Point', ['x', 'y'])
        obj_input = Point(1, 2)
        obj_result = {function_name}(obj_input)
        self.assertIsNotNone(obj_result, "Function should handle custom object")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif "parameterized behavior" in expansion:
            return f"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class ParameterizedTest(unittest.TestCase):
    def test_parameterized_behavior(self):
        # Test function with different parameters
        
        # Base case with default parameters
        base_input = [1, 2, 3]
        base_result = {function_name}(base_input)
        
        # The function should now accept additional parameters
        # These are example parameters, adjust based on the specific function
        
        # With sorting parameter
        try:
            sorted_result = {function_name}(base_input, sort=True)
            self.assertIsNotNone(sorted_result, "Function should handle sort parameter")
        except TypeError as e:
            self.fail(f"Function does not support sort parameter: {{e}}")
        
        # With filtering parameter
        try:
            filtered_result = {function_name}(base_input, filter_fn=lambda x: x > 1)
            self.assertIsNotNone(filtered_result, "Function should handle filter_fn parameter")
        except TypeError as e:
            self.fail(f"Function does not support filter_fn parameter: {{e}}")
        
        # With formatting parameter
        try:
            formatted_result = {function_name}(base_input, format="json")
            self.assertIsNotNone(formatted_result, "Function should handle format parameter")
        except TypeError as e:
            self.fail(f"Function does not support format parameter: {{e}}")
        
if __name__ == '__main__':
    unittest.main()
"""
        elif "additional output formats" in expansion:
            return f"""
import unittest
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class OutputFormatsTest(unittest.TestCase):
    def test_output_formats(self):
        # Test function with different output formats
        input_data = [1, 2, 3]
        
        # Original format
        original_result = {function_name}(input_data)
        
        # The function should now support different output formats
        # These are example formats, adjust based on the specific function
        
        # JSON format
        try:
            json_result = {function_name}(input_data, format="json")
            # Check if it's valid JSON
            try:
                json_obj = json.loads(json_result) if isinstance(json_result, str) else json_result
                self.assertIsNotNone(json_obj, "JSON result should be valid")
            except json.JSONDecodeError:
                self.fail("JSON result is not valid")
        except TypeError as e:
            self.fail(f"Function does not support JSON format: {{e}}")
        
        # CSV format
        try:
            csv_result = {function_name}(input_data, format="csv")
            self.assertIsNotNone(csv_result, "CSV result should not be None")
            if isinstance(csv_result, str):
                self.assertIn(",", csv_result, "CSV result should contain commas")
        except TypeError as e:
            self.fail(f"Function does not support CSV format: {{e}}")
        
        # XML format
        try:
            xml_result = {function_name}(input_data, format="xml")
            self.assertIsNotNone(xml_result, "XML result should not be None")
            if isinstance(xml_result, str):
                self.assertIn("<", xml_result, "XML result should contain tags")
                self.assertIn(">", xml_result, "XML result should contain tags")
        except TypeError as e:
            self.fail(f"Function does not support XML format: {{e}}")
        
if __name__ == '__main__':
    unittest.main()
"""
        else:
            # Generic functionality expansion test
            return f"""
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import {function_name}

class ExpandedFunctionalityTest(unittest.TestCase):
    def test_expanded_functionality(self):
        # Test for {expansion}
        # This is a placeholder test that needs to be customized for the specific expansion
        
        # Basic test to verify the function exists
        input_data = [1, 2, 3]
        result = {function_name}(input_data)
        self.assertIsNotNone(result, "Function should return a result")
        
        # You need to add specific tests for the expanded functionality
        
if __name__ == '__main__':
    unittest.main()
"""
    
    def _calculate_adaptation_vector(self, solution: str, result: EvaluationResult, feedback: Feedback) -> List[float]:
        """
        Calculate an adaptation vector based on the solution, result, and feedback.
        
        The adaptation vector encodes how the problem should evolve in future iterations,
        capturing dimensions like difficulty, bug type emphasis, and feedback focus.
        
        Args:
            solution: The current solution
            result: The evaluation results
            feedback: The feedback provided
            
        Returns:
            An adaptation vector (list of floats)
        """
        # Initialize adaptation vector with zeros
        # Dimensions:
        # [0] - difficulty adjustment
        # [1] - syntax vs logical bug emphasis
        # [2] - performance focus
        # [3] - edge case focus
        # [4] - requirement expansion
        adaptation_vector = [0.0] * 5
        
        # Adjust difficulty based on score
        if result.score > 0.95:
            adaptation_vector[0] = 0.2  # Increase difficulty significantly
        elif result.score > 0.8:
            adaptation_vector[0] = 0.1  # Increase difficulty moderately
        elif result.score > 0.6:
            adaptation_vector[0] = 0.0  # Maintain current difficulty
        elif result.score > 0.4:
            adaptation_vector[0] = -0.1  # Decrease difficulty moderately
        else:
            adaptation_vector[0] = -0.2  # Decrease difficulty significantly
        
        # Adjust bug type emphasis based on error types
        syntax_issues = sum(1 for issue in feedback.issues if issue.get("error_type") == "syntax")
        logical_issues = sum(1 for issue in feedback.issues if issue.get("type") == "test_failure")
        
        if syntax_issues > logical_issues:
            adaptation_vector[1] = -0.1  # Move toward more logical bugs
        elif logical_issues > syntax_issues:
            adaptation_vector[1] = 0.1  # Move toward more syntax bugs
        
        # Adjust performance focus based on execution time and metrics
        if result.metrics and "execution_time" in result.metrics:
            if result.metrics["execution_time"] > self.config.get("performance_threshold", 1.0):
                adaptation_vector[2] = 0.2  # Increase performance focus
            else:
                adaptation_vector[2] = -0.1  # Decrease performance focus
        
        # Adjust edge case focus based on test failures
        if result.test_results:
            edge_case_failures = sum(1 for test_name, test_result in result.test_results.items()
                                    if not test_result["passed"] and "edge" in test_name.lower())
            if edge_case_failures > 0:
                adaptation_vector[3] = 0.2  # Increase edge case focus
            else:
                adaptation_vector[3] = 0.0  # Maintain current edge case focus
        
        # Adjust requirement expansion based on current state
        current_requirements = len(self.state.requirements)
        if current_requirements < 3:
            adaptation_vector[4] = 0.1  # Increase likelihood of adding requirements
        elif current_requirements >= 5:
            adaptation_vector[4] = -0.1  # Decrease likelihood of adding requirements
        
        return adaptation_vector


class DefaultTestRunner:
    """Default test runner for evaluating bug fixes."""
    
    def run_tests(self, solution_file: Path, test_files: List[Path], code_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run tests against a solution file.
        
        Args:
            solution_file: Path to the solution file
            test_files: List of test file paths
            code_context: Context information about the code
            
        Returns:
            Dictionary of test results
        """
        # Initialize results
        results = {
            "all_passed": True,
            "passed_tests": 0,
            "total_tests": 0,
            "tests": {},
            "execution": {
                "success": True,
                "error": None,
                "stdout": None,
                "stderr": None
            },
            "execution_time": 0.0
        }
        
        # Import the solution to check for syntax errors
        try:
            # Check if the solution file exists
            if not solution_file.exists():
                results["execution"]["success"] = False
                results["execution"]["error"] = "Solution file not found"
                results["all_passed"] = False
                return results
            
            # Try to import the module to test for syntax errors
            sys.path.insert(0, str(solution_file.parent))
            import importlib.util
            spec = importlib.util.spec_from_file_location("solution", solution_file)
            solution_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(solution_module)
            
            # Check for required functions
            if "required_functions" in code_context:
                for func_name in code_context["required_functions"]:
                    if not hasattr(solution_module, func_name):
                        results["execution"]["success"] = False
                        results["execution"]["error"] = f"Required function '{func_name}' not found"
                        results["all_passed"] = False
                        return results
            
        except Exception as e:
            results["execution"]["success"] = False
            results["execution"]["error"] = str(e)
            results["all_passed"] = False
            return results
        
        # Run each test file
        for test_file in test_files:
            # Skip if the test file doesn't exist
            if not test_file.exists():
                continue
            
            # Run the test file
            import unittest
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            # Create a test loader and find tests in the file
            loader = unittest.TestLoader()
            try:
                tests = loader.discover(str(test_file.parent), pattern=test_file.name)
                
                # Count the number of test cases
                test_cases = 0
                for suite in tests:
                    for test_case in suite:
                        test_cases += test_case.countTestCases()
                
                results["total_tests"] += test_cases
                
                # Run the tests
                runner = unittest.TextTestRunner(verbosity=2)
                
                # Capture stdout and stderr
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    test_result = runner.run(tests)
                
                stdout = stdout_buffer.getvalue()
                stderr = stderr_buffer.getvalue()
                
                # Check if all tests passed
                if not test_result.wasSuccessful():
                    results["all_passed"] = False
                
                # Count passed tests
                passed_tests = test_cases - len(test_result.failures) - len(test_result.errors)
                results["passed_tests"] += passed_tests
                
                # Store individual test results
                test_name = test_file.stem
                results["tests"][test_name] = {
                    "passed": test_result.wasSuccessful(),
                    "failures": len(test_result.failures),
                    "errors": len(test_result.errors),
                    "skipped": len(test_result.skipped),
                    "total": test_cases,
                    "passed_count": passed_tests,
                    "stdout": stdout,
                    "stderr": stderr
                }
                
                # Extract more detailed information about failures
                for failure in test_result.failures:
                    test_id = failure[0].id()
                    failure_message = failure[1]
                    
                    # Extract expected and actual values if available
                    import re
                    expected_match = re.search(r'Expected\s*:(.+)', failure_message)
                    actual_match = re.search(r'Actual\s*:(.+)', failure_message)
                    
                    expected = expected_match.group(1).strip() if expected_match else None
                    actual = actual_match.group(1).strip() if actual_match else None
                    
                    if test_id not in results["tests"]:
                        results["tests"][test_id] = {}
                    
                    results["tests"][test_id].update({
                        "passed": False,
                        "message": failure_message,
                        "expected": expected,
                        "actual": actual
                    })
                
            except Exception as e:
                # If the test file itself has errors
                results["all_passed"] = False
                results["tests"][test_file.stem] = {
                    "passed": False,
                    "error": str(e),
                    "failures": 1,
                    "errors": 1,
                    "skipped": 0,
                    "total": 1,
                    "passed_count": 0
                }
                results["total_tests"] += 1
        
        return results


class BugFixingTaskGenerator:
    """Generator for bug fixing tasks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the bug fixing task generator.
        
        Args:
            config: Configuration options
        """
        self.config = config or {}
        self.difficulty_levels = self.config.get(
            "difficulty_levels", 
            ["easy", "medium", "hard", "expert"]
        )
        self.bug_categories = self.config.get(
            "bug_categories",
            [
                BugCategory.SYNTAX,
                BugCategory.LOGICAL,
                BugCategory.EDGE_CASE,
                BugCategory.PERFORMANCE
            ]
        )
        self.test_templates = self._load_test_templates()
    
    def generate_task(self, difficulty: str = None, bug_categories: List[str] = None) -> BugFixingTask:
        """
        Generate a new bug fixing task.
        
        Args:
            difficulty: The difficulty level (easy, medium, hard, expert)
            bug_categories: List of bug categories to include
            
        Returns:
            A new bug fixing task
        """
        # Choose difficulty if not specified
        if difficulty is None:
            difficulty = random.choice(self.difficulty_levels)
        
        # Choose bug categories if not specified
        if bug_categories is None:
            num_categories = random.randint(1, 3)
            bug_categories = random.sample(self.bug_categories, num_categories)
        
        # Generate a problem based on difficulty and bug categories
        problem_state = self._generate_problem_state(difficulty, bug_categories)
        
        # Create config for the task
        task_config = {
            "difficulty": difficulty,
            "bug_categories": bug_categories,
            "convergence_criteria": {
                "score_threshold": 0.95,
                "min_iterations": 1,
                "max_iterations": self.config.get("max_iterations", 5),
                "score_delta_threshold": 0.05,
                "consecutive_plateau_limit": 2
            },
            "score_weights": {
                "test": 0.7,
                "execution": 0.3
            },
            "performance_threshold": 1.0,
            "complexity_threshold": 0.7
        }
        
        # Create and return the task
        return BugFixingTask(problem_state, task_config)
    
    def _generate_problem_state(self, difficulty: str, bug_categories: List[str]) -> ProblemState:
        """
        Generate a problem state for the given difficulty and bug categories.
        
        Args:
            difficulty: The difficulty level
            bug_categories: List of bug categories
            
        Returns:
            A problem state for the task
        """
        # Choose a template based on difficulty and bug categories
        template = self._choose_template(difficulty, bug_categories)
        
        # Create a copy of the template
        problem_state = copy.deepcopy(template)
        
        # Generate a unique ID
        problem_state.problem_id = str(uuid.uuid4())
        
        # Initialize evolution stage and adaptation vector
        problem_state.evolution_stage = 0
        problem_state.adaptation_vector = [0.0] * 5
        
        # Adjust difficulty value based on level
        difficulty_values = {
            "easy": 0.25,
            "medium": 0.5,
            "hard": 0.75,
            "expert": 0.9
        }
        problem_state.difficulty = difficulty_values.get(difficulty, 0.5)
        
        # Insert bugs based on categories
        for category in bug_categories:
            self._insert_bug(problem_state, category)
        
        # Update description to reflect the current state
        problem_state.description = self._generate_description(problem_state)
        
        return problem_state
    
    def _choose_template(self, difficulty: str, bug_categories: List[str]) -> ProblemState:
        """
        Choose a template that matches the difficulty and bug categories.
        
        Args:
            difficulty: The difficulty level
            bug_categories: List of bug categories
            
        Returns:
            A template problem state
        """
        # In a real implementation, this would load from a database of templates
        # For now, we'll generate a simple template
        
        # Generate code context with a sample function
        code = self._generate_template_code(difficulty, bug_categories)
        tests = self._generate_template_tests(code)
        
        # Create a basic problem state
        return ProblemState(
            problem_id="template",
            description="Fix the bugs in the given code.",
            code_context={
                "code": code,
                "tests": tests,
                "bug_count": 0,
                "bug_categories": []
            },
            requirements=[
                {
                    "type": "functional",
                    "description": "The code should pass all the provided tests.",
                    "difficulty": 0.3
                }
            ],
            difficulty=0.5,  # Will be overridden
            evolution_stage=0,
            adaptation_vector=[0.0] * 5
        )
    
    def _generate_template_code(self, difficulty: str, bug_categories: List[str]) -> str:
        """
        Generate template code based on difficulty and bug categories.
        
        Args:
            difficulty: The difficulty level
            bug_categories: List of bug categories
            
        Returns:
            Template code
        """
        # For demonstration, we'll use a few predefined templates
        templates = {
            "easy": """
def calculate_sum(numbers):
    \"\"\"Calculate the sum of a list of numbers.\"\"\"
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    if not numbers:
        return 0
    return calculate_sum(numbers) / len(numbers)
""",
            "medium": """
def find_most_frequent(items):
    \"\"\"Find the most frequently occurring item in
