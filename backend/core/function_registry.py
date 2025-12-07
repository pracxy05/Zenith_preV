"""
Function Registry Module
Manages all available tools and function calling
"""

import json
from typing import Dict, List, Any, Callable

class FunctionRegistry:
    """Registry for all available tools and functions"""
    
    def __init__(self):
        self.functions: Dict[str, Dict] = {}
        self.executors: Dict[str, Callable] = {}
    
    def register_function(self, name: str, executor: Callable, description: str = "", parameters: Dict = None) -> None:
        """
        Register a new function with its schema
        Compatible signature: register_function(name, executor, description)
        """
        self.functions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters or {}
        }
        self.executors[name] = executor
    
    def get_function_schemas(self) -> List[Dict]:
        """Get all function schemas for LLM context"""
        return list(self.functions.values())
    
    def execute(self, name: str, **kwargs) -> Any:
        """Execute a function by name with given arguments"""
        if name not in self.executors:
            raise ValueError(f"Function '{name}' not found")
        
        try:
            result = self.executors[name](**kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f"Error executing {name}: {str(e)}")
    
    def execute_function(self, name: str, arguments: Dict) -> str:
        """Execute a function by name with given arguments (legacy method)"""
        if name not in self.executors:
            return f"Error: Function '{name}' not found"
        
        try:
            result = self.executors[name](**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
    
    def parse_function_call(self, llm_output: str) -> tuple:
        """Parse LLM output to extract function calls"""
        # Simple parsing - in production, use structured output
        if "{" in llm_output and "}" in llm_output:
            try:
                # Try to extract JSON-like function call
                start = llm_output.find("{")
                end = llm_output.rfind("}") + 1
                func_json = llm_output[start:end]
                func_data = json.loads(func_json)
                if "function" in func_data and "arguments" in func_data:
                    return func_data["function"], func_data["arguments"]
            except:
                pass
        
        return None, None
