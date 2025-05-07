# recursive_swe_bench/models/base_model.py

from typing import Any, Dict, List, Optional, Union
import logging
import time
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """
    Base interface for models that can be evaluated using Recursive-SWE-bench.
    
    This abstract class defines the core functionality required for a model to
    be evaluated using the recursive evaluation framework. Concrete implementations
    must provide the actual model-specific logic.
    """
    
    def __init__(self, model_identifier: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model interface.
        
        Args:
            model_identifier: Identifier for the model
            config: Configuration options
        """
        self.model_identifier = model_identifier
        self.config = config or {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the model."""
        logger = logging.getLogger(f"Model.{self.model_identifier}")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.config.get("log_level", logging.INFO))
        return logger
    
    @abstractmethod
    def solve(self, problem: Dict[str, Any], history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a solution for the given problem.
        
        Args:
            problem: The problem to solve
            history: Optional history of previous solution attempts
            
        Returns:
            The generated solution
        """
        pass
    
    @abstractmethod
    def get_meta_information(self) -> Dict[str, Any]:
        """
        Get meta information about the model.
        
        Returns:
            Dictionary containing model information
        """
        pass


# recursive_swe_bench/models/openai.py

import openai
import json
import backoff
from typing import Any, Dict, List, Optional, Union

from recursive_swe_bench.models.base_model import ModelInterface

class OpenAIModel(ModelInterface):
    """
    Integration with OpenAI models (GPT-3.5, GPT-4, etc.).
    
    This class provides integration with OpenAI's API for evaluating
    models like GPT-3.5 and GPT-4 with Recursive-SWE-bench.
    """
    
    def __init__(
        self, 
        model_identifier: str, 
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the OpenAI model interface.
        
        Args:
            model_identifier: OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key (optional if set in environment)
            config: Additional configuration options
        """
        super().__init__(model_identifier, config)
        
        # Set API key if provided
        if api_key:
            openai.api_key = api_key
        
        # Load default prompts or use config-provided ones
        self.prompts = self.config.get("prompts", {
            "system": "You are an expert programmer tasked with fixing bugs in code. Fix the code based on the description and tests.",
            "user_template": "# Bug Fixing Task\n\n{description}\n\n# Code\n```python\n{code}\n```\n\n{tests_description}\n\n# Your task\nFix the bugs in the code above. Provide only the corrected code without any explanations.",
        })
        
        # Configure API parameters
        self.api_params = self.config.get("api_params", {
            "temperature": 0.2,
            "max_tokens": 2000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        })
        
        self.logger.info(f"Initialized OpenAI model: {model_identifier}")
    
    @backoff.on_exception(
        backoff.expo,
        (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError),
        max_tries=5
    )
    def solve(
        self, 
        problem: Dict[str, Any], 
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Generate a solution using the OpenAI model.
        
        Args:
            problem: The problem to solve
            history: Optional history of previous solution attempts
            
        Returns:
            The generated solution
        """
        self.logger.info(f"Solving problem with OpenAI model: {self.model_identifier}")
        start_time = time.time()
        
        # Format the problem for the model
        messages = self._format_messages(problem, history)
        
        # Make API call
        response = openai.ChatCompletion.create(
            model=self.model_identifier,
            messages=messages,
            **self.api_params
        )
        
        # Extract the solution from the response
        solution = response.choices[0].message.content.strip()
        
        end_time = time.time()
        self.logger.info(f"Solution generated in {end_time - start_time:.2f} seconds")
        
        return self._extract_code(solution)
    
    def _format_messages(
        self, 
        problem: Dict[str, Any], 
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """
        Format the problem and history into messages for the OpenAI API.
        
        Args:
            problem: The problem to solve
            history: Optional history of previous solution attempts
            
        Returns:
            List of formatted messages
        """
        messages = [
            {"role": "system", "content": self.prompts["system"]}
        ]
        
        # Format the user message
        code = problem["code_context"]["code"]
        
        # Prepare tests description
        tests_description = "# Tests\n"
        if "tests" in problem["code_context"]:
            tests_description += "The code must pass the following tests:\n\n"
            for i, test in enumerate(problem["code_context"]["tests"]):
                tests_description += f"## Test {i+1}: {test['name']}\n```python\n{test['content']}\n```\n\n"
        else:
            tests_description += "The code must work correctly according to its intended functionality."
        
        # Create the user message using the template
        user_content = self.prompts["user_template"].format(
            description=problem["description"],
            code=code,
            tests_description=tests_description
        )
        
        messages.append({"role": "user", "content": user_content})
        
        # Add history if available
        if history and self.config.get("include_history", True):
            for entry in history:
                # Add previous attempt
                messages.append({
                    "role": "assistant", 
                    "content": entry["solution"]
                })
                
                # Add feedback on previous attempt
                feedback_content = f"Your solution has the following issues:\n"
                for issue in entry["feedback"]["issues"]:
                    feedback_content += f"- {issue['message']}\n"
                
                feedback_content += "\nPlease try again with these improvements:\n"
                for suggestion in entry["feedback"]["suggestions"]:
                    feedback_content += f"- {suggestion['message']}\n"
                
                messages.append({
                    "role": "user", 
                    "content": feedback_content
                })
        
        return messages
    
    def _extract_code(self, text: str) -> str:
        """
        Extract code from the model's response.
        
        Args:
            text: The model's response
            
        Returns:
            Extracted code
        """
        # Try to extract code from markdown code blocks
        import re
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0].strip()
        
        # If no code blocks, return the full text (it might be just code)
        return text.strip()
    
    def get_meta_information(self) -> Dict[str, Any]:
        """
        Get meta information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_identifier,
            "provider": "OpenAI",
            "type": "API",
            "parameters": self.api_params,
            "system_prompt": self.prompts["system"]
        }


